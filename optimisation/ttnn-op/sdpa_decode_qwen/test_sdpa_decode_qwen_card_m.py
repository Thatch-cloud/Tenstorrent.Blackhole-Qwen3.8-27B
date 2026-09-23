"""Card-M unit test for the [QWEN-SDPA] stage-1 decode SDPA: the tail-only mask.

WHAT IT PROVES. With the K64e graft mounted (its _ttnncpp.so carries the factory branch,
its sdpa_decode op directory the two qwen kernels), a paged_scaled_dot_product_attention_decode
call whose SDPAProgramConfig.q_chunk_size is the sentinel 0x51DEC001 ('tail') returns the
SAME BYTES as the legacy call (q_chunk_size=0) on replay-shaped inputs, and is faster:

  - Q (1, B, 48, 256) bf16: 4 draft rows x 12 heads folded to 48 rows, 2 KV heads per chip
    (attention_head_fold.fold_query), one 16-row block = a batch-3 bundle (group offsets
    0, 4, 8) and a batch-1 bundle (offset 12) - the T16 plan at four-row groups;
  - K/V: one bf8 paged pool (capacity // 64 + 64 blocks, 2 KV heads, 64-token pages, 256
    dims), K ~ 2 N(0,1), V ~ N(0,1); ONE random page table (a permutation of the pool's
    blocks) repeated to both bundles' rows, as the replay reader does;
  - mask (B, 1, 48, capacity) bf16: zeros everywhere but the last 256 columns, which hold
    the refresh kernel's formula (-inf where cache_position > start + offset + b*rows + t);
  - capacities 33,024 and 131,328 (513 chunks: 16 cores per head, 4 tree rounds);
    is_causal=False, k_chunk_size=256, full grid, exp_approx_mode=False, scale 1/16.

Checks (all byte-for-byte, torch.equal on int16 views of every logical output element):
  equal      tail == legacy for every capacity x seed x bundle x query variant x start
             (variants: normal; peaky - q aligned with 8 cached keys per head so the running
             max moves across chunks and cores; zeroq - every fourth folded row zero, +-0 scores)
  planted    -inf planted at mask chunk 0, column 0: the legacy output changes, the tail
             output does not (the non-final mask chunks are really neither read nor added)
  narrow     tail with the narrow (B,1,48,256) mask == tail with the wide mask (the factory
             already admits it; stage 1b's Python is not in this build)
  refused    TT_FATAL for flags 0x2 (share, stage 3), 0x4 (unknown), a 512-wide mask under
             tail, a narrow mask without tail, and the sentinel on a causal call
  cache      alternating legacy / tail calls on identical shapes: each always equals its first
  log        the factory's '[QWEN-SDPA] flags=0x1 ...' lines: one per new tail program
             (wide and narrow, B=3 and B=1, per capacity), scratch_slots=4, cb_bytes 790,592
             at 33,024 and 796,736 at 131,328 (the spec's L1 table)
  timing     median host-timed call (synchronised) legacy vs tail, per bundle and per
             user-layer (B3 + B1)

The padded query rows 48..63 of the host-built mask are the tilizer's zeros, where the
device refresh kernel writes -inf in the last chunk; the ops are row-independent and only
the 48 logical rows are compared, so this does not bear on the equality.

--legacy-only runs just the legacy calls (for the stock image, no graft) and records each
output's sha256; --reference <that report> makes a graft run assert its legacy outputs are
the stock binary's bytes (the graft's legacy branch did not regress).

RUN on card M only (never the serving cards), inside the serving image with the graft
mounted exactly as the arm mounts it and a FRESH kernel cache (run_card_m.sh does this):

    python3 -B test_sdpa_decode_qwen_card_m.py --out /results/card-m.json

Everything the C++ side prints (tt-logger, including the [QWEN-SDPA] lines) goes to
<out>.native.log; the helpers above Case import no ttnn and are unit-tested on CPU in
test_sdpa_decode_qwen_sources.py.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

HEADS, KV_HEADS, HEAD_DIM, PAGE, K_CHUNK, TILE = 48, 2, 256, 64, 256, 32
ROWS = 4                                   # draft rows per fold group (4 x 12 heads = 48)
BUNDLES = ((0, 4, 8), (12,))              # group offsets per bundle: batch 3, then batch 1
CAPACITIES = (33024, 131328)
STARTS = (0, 7, 240)                       # block start = capacity - 256 + this
SEEDS = (0, 1, 2, 3, 4)
VARIANTS = ('normal', 'peaky', 'zeroq')
LEGACY, TAIL = 0, 0x51DEC001
SHARE_STAGE3, UNKNOWN_FLAG, NO_FLAGS = 0x51DEC002, 0x51DEC004, 0x51DEC000
SCALE = 1.0 / 16
CB_BYTES = {33024: 790592, 131328: 796736}
SCRATCH_SLOTS = 4
BINARY_MARKER = b'[QWEN-SDPA] flags='
FACTORY_LINE = re.compile(r'\[QWEN-SDPA\] flags=(0x[0-9a-f]+) B=([0-9]+) PNHt=([0-9]+) St=([0-9]+) '
                          r'mask_width_t=([0-9]+) kv_share=([a-z0-9]+) scratch_slots=([0-9]+) cb_bytes=([0-9]+)')


def num_blocks(capacity):
    if type(capacity) is not int or capacity <= 0 or capacity % K_CHUNK:
        raise ValueError('Positive chunk-aligned capacity required')
    return capacity // PAGE + 64


def mask_positions(start, offsets, rows=ROWS):
    """Per bundle entry b and folded row r, the last visible cache position - the pinned
    refresh kernel's `start + offset + batch * rows + (head % (rows * 6)) / 6`, where the
    kernel's offset is the bundle's first group offset and batch the entry index."""
    first = offsets[0]
    return [[start + first + b * rows + (head % (rows * 6)) // 6 for head in range(rows * 12)]
            for b in range(len(offsets))]


def build_mask(torch, capacity, start, offsets, *, width=None, plant=False):
    """(B, 1, 48, width) bf16: zeros, the last 256 columns masked per mask_positions. width
    defaults to the full capacity; 256 gives the narrow mask (the same last 256 columns),
    512 a two-chunk mask the factory must refuse. plant puts -inf at column 0 of every
    row: chunk 0, which only the legacy op reads."""
    width = capacity if width is None else width
    positions = torch.tensor(mask_positions(start, offsets), dtype=torch.int64)   # (B, 48)
    cache = torch.arange(capacity - 256, capacity, dtype=torch.int64)             # (256,)
    tail = torch.where(cache[None, None, :] > positions[:, :, None], float('-inf'), 0.0)
    mask = torch.zeros(len(offsets), 1, ROWS * 12, width, dtype=torch.float32)
    mask[:, 0, :, width - 256:] = tail
    if plant:
        if width < capacity:
            raise ValueError('Planting needs the full-width mask')
        mask[:, 0, :, 0] = float('-inf')
    return mask.to(torch.bfloat16)


def build_query(torch, batches, seed, variant, keys=None, table=None):
    """(1, B, 48, 256) bf16, N(0,1). 'peaky': each folded row is 6 x the unit vectors of 8
    random cached keys of its KV head plus small noise, so those scores dominate wherever
    they fall (their chunks, cores and tree order move with the seed); 'zeroq': every
    fourth folded row is zero, so its scores are +-0."""
    generator = torch.Generator().manual_seed(1000 + seed)
    query = torch.randn(1, batches, HEADS, HEAD_DIM, generator=generator)
    if variant == 'peaky':
        if keys is None or table is None:
            raise ValueError('Peaky queries need the host keys and page table')
        query *= 0.1
        for head in range(HEADS):
            kv = head // (HEADS // KV_HEADS)
            for token in torch.randint(0, len(table) * PAGE, (8,), generator=generator).tolist():
                vector = keys[int(table[token // PAGE]), kv, token % PAGE].float()
                query[0, :, head] += 6 * vector / vector.norm().clamp_min(1e-3)
    elif variant == 'zeroq':
        query[0, :, ::4] = 0.0
    elif variant != 'normal':
        raise ValueError('Unknown query variant %r' % (variant,))
    return query.to(torch.bfloat16)


def int16_view(torch, tensor):
    return tensor.to(torch.bfloat16).contiguous().view(torch.int16)


def digest(torch, tensor):
    return hashlib.sha256(int16_view(torch, tensor).numpy().tobytes()).hexdigest()


def differing(torch, left, right):
    return int((int16_view(torch, left) != int16_view(torch, right)).sum())


def factory_lines(text):
    return [dict(flags=m.group(1), B=int(m.group(2)), PNHt=int(m.group(3)), St=int(m.group(4)),
                 mask_width_t=int(m.group(5)), kv_share=m.group(6), scratch_slots=int(m.group(7)),
                 cb_bytes=int(m.group(8))) for m in FACTORY_LINE.finditer(text)]


def check_factory_lines(lines, capacities, narrow=True):
    """A tail program per capacity x batch x mask width (wide St; narrow Sk_chunk_t when the
    narrow calls ran) was built, so at least one line each - one per program with the
    program cache on - every one with the spec's scratch slots and CB bytes."""
    problems = []
    for capacity in capacities:
        st = capacity // TILE
        for batches in sorted({len(offsets) for offsets in BUNDLES}):
            for width in (st, K_CHUNK // TILE) if narrow else (st,):
                found = [line for line in lines if (line['St'], line['B'], line['mask_width_t']) == (st, batches, width)]
                label = 'St=%d B=%d mask_width_t=%d' % (st, batches, width)
                if not found:
                    problems.append('%s: no [QWEN-SDPA] line' % label)
                expected = dict(flags='0x1', PNHt=2, kv_share='false', scratch_slots=SCRATCH_SLOTS,
                                cb_bytes=CB_BYTES.get(capacity))
                for line in found:
                    wrong = {key: (line[key], value) for key, value in expected.items()
                             if value is not None and line[key] != value}
                    if wrong:
                        problems.append('%s: (found, expected) %r' % (label, wrong))
                        break
    stray = [line for line in lines if line['St'] not in {c // TILE for c in capacities}]
    if stray:
        problems.append('%d [QWEN-SDPA] lines for no tested capacity' % len(stray))
    return problems


def loaded_binary(maps='/proc/self/maps'):
    """The _ttnncpp.so this process mapped, and whether it carries the factory branch."""
    import mmap

    paths = sorted({line.split()[-1] for line in Path(maps).read_text().splitlines()
                    if line.rstrip().endswith('_ttnncpp.so')})
    if len(paths) != 1:
        raise RuntimeError('Expected exactly one mapped _ttnncpp.so, found %r' % (paths,))
    with open(paths[0], 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
        return paths[0], view.find(BINARY_MARKER) >= 0


class NativeLog:
    """Send fds 1 and 2 (tt-logger's sinks) to a file for the whole run, so the factory's
    [QWEN-SDPA] lines can be counted even when the logger flushes late; Python's own
    progress keeps going to the original stdout."""

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.saved = [os.dup(1), os.dup(2)]
        self.handle = open(self.path, 'wb')
        os.dup2(self.handle.fileno(), 1)
        os.dup2(self.handle.fileno(), 2)
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout = os.fdopen(os.dup(self.saved[0]), 'w', buffering=1)
        sys.stderr = os.fdopen(os.dup(self.saved[1]), 'w', buffering=1)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout, sys.stderr = self.stdout, self.stderr
        os.dup2(self.saved[0], 1)
        os.dup2(self.saved[1], 2)
        for fd in self.saved:
            os.close(fd)
        self.handle.close()
        return False

    def text(self):
        return self.path.read_bytes().decode('utf-8', 'replace') if self.path.is_file() else ''


# ---------------------------------------------------------------------------------------
# Device part: card M only.
# ---------------------------------------------------------------------------------------

class Case:
    """One capacity and seed: the K/V pool and page tables, and the calls on them."""

    def __init__(self, ttnn, torch, device, capacity, seed):
        self.ttnn, self.torch, self.device, self.capacity = ttnn, torch, device, capacity
        generator = torch.Generator().manual_seed(seed)
        blocks = num_blocks(capacity)
        self.keys = (torch.randn(blocks, KV_HEADS, PAGE, HEAD_DIM, generator=generator) * 2).to(torch.bfloat16)
        values = torch.randn(blocks, KV_HEADS, PAGE, HEAD_DIM, generator=generator).to(torch.bfloat16)
        self.table = torch.randperm(blocks, generator=generator)[:capacity // PAGE].to(torch.int32)
        self.owned = []
        self.k = self.upload(self.keys, ttnn.bfloat8_b)
        self.v = self.upload(values, ttnn.bfloat8_b)
        # One user's page table, repeated to each bundle's rows (attention_replay.py).
        self.tables = {batches: self.upload(self.table[None, :].repeat(batches, 1).contiguous(), ttnn.int32)
                       for batches in sorted({len(offsets) for offsets in BUNDLES})}

    def upload(self, host, dtype=None, *, keep=True):
        ttnn = self.ttnn
        dtype = ttnn.bfloat16 if dtype is None else dtype
        layout = ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT
        tensor = ttnn.from_torch(host, device=self.device, dtype=dtype, layout=layout,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if keep:
            self.owned.append(tensor)
        return tensor

    def config(self, sentinel):
        grid = self.device.compute_with_storage_grid_size()
        return self.ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
            exp_approx_mode=False, q_chunk_size=sentinel, k_chunk_size=K_CHUNK)

    def call(self, query, mask, sentinel, *, is_causal=False, cur_pos_tensor=None):
        # The paged binding (sdpa_decode_nanobind.cpp:82-106) takes page_table_tensor positionally
        # and only these keywords after kw_only(); it has no list-valued cur_pos, and a causal
        # paged call must carry an int32 row-major cur_pos TENSOR or validate refuses it first.
        ttnn = self.ttnn
        options = dict(is_causal=is_causal, scale=SCALE,
                       program_config=self.config(sentinel), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if mask is not None:
            options['attn_mask'] = mask
        if cur_pos_tensor is not None:
            options['cur_pos_tensor'] = cur_pos_tensor
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            query, self.k, self.v, self.tables[query.shape[1]], **options)

    def host(self, tensor):
        result = self.ttnn.to_torch(tensor)
        self.ttnn.deallocate(tensor)
        return result

    def close(self):
        for tensor in self.owned:
            self.ttnn.deallocate(tensor)
        self.owned.clear()


def timed(ttnn, device, once, warmup, iters):
    for _ in range(warmup):
        ttnn.deallocate(once())
    ttnn.synchronize_device(device)
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        out = once()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - started) * 1e6)
        ttnn.deallocate(out)
    return dict(median_us=statistics.median(samples), min_us=min(samples), mean_us=statistics.mean(samples))


def expect_refusal(case, query, mask, sentinel, needle, **options):
    try:
        out = case.call(query, mask, sentinel, **options)
    except Exception as error:  # noqa: BLE001 - TT_FATAL surfaces as RuntimeError
        text = str(error)
        return dict(refused=True, matched=needle in text, message=' '.join(text.split())[:300])
    case.ttnn.deallocate(out)
    return dict(refused=False, matched=False, message='call was accepted')


def equality_case(ttnn, torch, case, capacity, seed, offsets, variant, start, args, report):
    """One replay-shaped call pair; every tensor it uploads is freed before it returns."""
    failures = report['failures']
    batches = len(offsets)
    label = 'cap%d/seed%d/B%d/%s/start+%d' % (capacity, seed, batches, variant, start)
    begin = capacity - 256 + start
    local = []

    def upload(host):
        tensor = case.upload(host, keep=False)
        local.append(tensor)
        return tensor

    try:
        query = upload(build_query(torch, batches, seed, variant, case.keys, case.table))
        mask = upload(build_mask(torch, capacity, begin, offsets))
        legacy = case.host(case.call(query, mask, LEGACY))
        entry = dict(label=label, legacy_sha256=digest(torch, legacy))
        report['legacy_sha256'][label] = entry['legacy_sha256']
        if args.reference_shas is not None:
            entry['matches_reference'] = args.reference_shas.get(label) == entry['legacy_sha256']
            if not entry['matches_reference']:
                failures.append('%s: graft legacy output differs from the reference binary' % label)
        if not bool(torch.isfinite(legacy.float()).all()):
            failures.append('%s: legacy output is not finite' % label)
        if not args.legacy_only:
            tail = case.host(case.call(query, mask, TAIL))
            entry['tail_differing_elements'] = differing(torch, legacy, tail)
            if entry['tail_differing_elements']:
                failures.append('%s: tail differs from legacy in %d elements' % (label, entry['tail_differing_elements']))
            if variant == 'normal' and start == args.starts[0]:
                planted = upload(build_mask(torch, capacity, begin, offsets, plant=True))
                legacy_planted = case.host(case.call(query, planted, LEGACY))
                tail_planted = case.host(case.call(query, planted, TAIL))
                entry['planted_legacy_changed'] = differing(torch, legacy, legacy_planted) > 0
                entry['planted_tail_differing'] = differing(torch, tail, tail_planted)
                if not entry['planted_legacy_changed']:
                    failures.append('%s: the planted chunk-0 -inf did not change the legacy output '
                                    '(the liveness control is dead)' % label)
                if entry['planted_tail_differing']:
                    failures.append('%s: tail read a non-final mask chunk (%d elements moved)'
                                    % (label, entry['planted_tail_differing']))
                narrow = upload(build_mask(torch, capacity, begin, offsets, width=256))
                entry['narrow_differing'] = differing(torch, tail, case.host(case.call(query, narrow, TAIL)))
                if entry['narrow_differing']:
                    failures.append('%s: the narrow tail mask differs from the wide one in %d elements'
                                    % (label, entry['narrow_differing']))
    finally:
        for tensor in local:
            ttnn.deallocate(tensor)
    report['cases'].append(entry)
    print('%-44s %s' % (label, {key: value for key, value in entry.items() if key not in ('label', 'legacy_sha256')}),
          flush=True)


def timing_and_controls(ttnn, torch, device, case, capacity, args, report):
    failures = report['failures']
    begin = capacity - 256 + args.starts[0]
    per_bundle = {}
    for offsets in BUNDLES:
        batches = len(offsets)
        query = case.upload(build_query(torch, batches, 0, 'normal'))
        mask = case.upload(build_mask(torch, capacity, begin, offsets))
        for name, sentinel in (('legacy', LEGACY), ('tail', TAIL)):
            per_bundle[(batches, name)] = timed(ttnn, device, lambda s=sentinel: case.call(query, mask, s),
                                                args.warmup, args.iters)
        row = dict(capacity=capacity, batches=batches, legacy=per_bundle[(batches, 'legacy')],
                   tail=per_bundle[(batches, 'tail')])
        row['tail_over_legacy'] = row['tail']['median_us'] / row['legacy']['median_us']
        report['timing'].append(row)
        print('timing cap=%d B=%d legacy %.1f us tail %.1f us ratio %.3f' % (
            capacity, batches, row['legacy']['median_us'], row['tail']['median_us'], row['tail_over_legacy']), flush=True)
        if batches == 3 and args.alternations:
            first, drift = {}, 0
            for index in range(2 * args.alternations):
                sentinel = (LEGACY, TAIL)[index % 2]
                out = case.host(case.call(query, mask, sentinel))
                if sentinel not in first:
                    first[sentinel] = out
                elif differing(torch, first[sentinel], out):
                    drift += 1
            report['alternation'].append(dict(capacity=capacity, calls=2 * args.alternations, drifted=drift,
                                              modes_equal=differing(torch, first[LEGACY], first[TAIL]) == 0))
            if drift:
                failures.append("cap%d: %d alternating calls drifted from their mode's first output" % (capacity, drift))
    user_layer = {name: per_bundle[(3, name)]['median_us'] + per_bundle[(1, name)]['median_us'] for name in ('legacy', 'tail')}
    report['user_layer_us'][capacity] = dict(user_layer, ratio=user_layer['tail'] / user_layer['legacy'])
    print('timing cap=%d per user-layer (B3+B1) legacy %.1f us tail %.1f us ratio %.3f' % (
        capacity, user_layer['legacy'], user_layer['tail'], user_layer['tail'] / user_layer['legacy']), flush=True)
    # Refusals: the factory's TT_FATALs fire before any program is built or launched.
    query = case.upload(build_query(torch, 3, 0, 'normal'))
    wide = case.upload(build_mask(torch, capacity, begin, BUNDLES[0]))
    narrow = case.upload(build_mask(torch, capacity, begin, BUNDLES[0], width=256))
    half = case.upload(build_mask(torch, capacity, begin, BUNDLES[0], width=512))
    # validate() needs an INT32 ROW_MAJOR cur_pos tensor of length B on a causal paged call;
    # with it validate passes and the factory's [QWEN-SDPA] non-causal TT_FATAL is what fires.
    positions = case.upload(torch.full((len(BUNDLES[0]),), capacity - 1, dtype=torch.int32), case.ttnn.int32)
    refusals = {
        'share flag (stage 3)': expect_refusal(case, query, wide, SHARE_STAGE3, 'KV share is not in this build'),
        'unknown flag 0x4': expect_refusal(case, query, wide, UNKNOWN_FLAG, 'unknown flags'),
        'tail with a 512-wide mask': expect_refusal(case, query, half, TAIL, 'tail mask must be full width'),
        'narrow mask without tail': expect_refusal(case, query, narrow, NO_FLAGS, 'a narrow mask needs the tail flag'),
        'sentinel on a causal call': expect_refusal(case, query, None, TAIL, 'modes are non-causal',
                                                    is_causal=True, cur_pos_tensor=positions),
    }
    report['refusals'][capacity] = refusals
    for name, result in refusals.items():
        print('refusal cap=%d %-28s %s' % (capacity, name, result), flush=True)
        if not (result['refused'] and result['matched']):
            failures.append('cap%d: %s was not refused by its [QWEN-SDPA] TT_FATAL: %s' % (capacity, name, result['message']))


def run(args, report):
    import torch
    import ttnn

    failures = report['failures']
    device = ttnn.open_device(device_id=args.device_id, l1_small_size=24576)
    try:
        path, present = loaded_binary()
        report['binary'] = dict(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), qwen_branch=present)
        print('binary %s sha256 %s qwen_branch=%s' % (path, report['binary']['sha256'][:16], present), flush=True)
        if not args.legacy_only and not present:
            failures.append('the loaded _ttnncpp.so lacks the [QWEN-SDPA] branch: the graft is not mounted')
            return
        for capacity in args.capacities:
            for seed in args.seeds:
                case = Case(ttnn, torch, device, capacity, seed)
                try:
                    for offsets in BUNDLES:
                        for variant in args.variants:
                            for start in args.starts:
                                equality_case(ttnn, torch, case, capacity, seed, offsets, variant, start, args, report)
                    if not args.legacy_only and seed == args.seeds[0]:
                        timing_and_controls(ttnn, torch, device, case, capacity, args, report)
                finally:
                    case.close()
    finally:
        ttnn.close_device(device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--capacities', default=','.join(map(str, CAPACITIES)))
    parser.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--starts', default=','.join(map(str, STARTS)))
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--alternations', type=int, default=50, help='legacy/tail pairs in the program-cache check')
    parser.add_argument('--legacy-only', action='store_true', help='stock binary: legacy calls only, record shas')
    parser.add_argument('--reference', help='a --legacy-only report whose legacy shas the graft must reproduce')
    args = parser.parse_args(argv)
    args.capacities = [int(value) for value in args.capacities.split(',')]
    args.seeds = [int(value) for value in args.seeds.split(',')]
    args.variants = [value for value in args.variants.split(',') if value]
    args.starts = [int(value) for value in args.starts.split(',')]
    for capacity in args.capacities:
        num_blocks(capacity)
    if any(variant not in VARIANTS for variant in args.variants) or any(not 0 <= s <= 240 for s in args.starts):
        parser.error('unknown variant or start outside the family')
    args.reference_shas = json.loads(Path(args.reference).read_text())['legacy_sha256'] if args.reference else None
    report = dict(passed=False, capacities=args.capacities, seeds=args.seeds, variants=args.variants,
                  starts=args.starts, legacy_only=args.legacy_only, reference=args.reference, failures=[], cases=[],
                  timing=[], alternation=[], user_layer_us={}, refusals={}, legacy_sha256={})
    native = NativeLog(args.out.with_name(args.out.name + '.native.log'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with native:
            run(args, report)
        if not args.legacy_only and report.get('binary', {}).get('qwen_branch'):
            report['factory_lines'] = factory_lines(native.text())
            report['factory_line_count'] = len(report['factory_lines'])
            for problem in check_factory_lines(report['factory_lines'], args.capacities, narrow='normal' in args.variants):
                report['failures'].append('factory log: ' + problem)
        report['passed'] = not report['failures'] and bool(report['cases'])
    except Exception as error:  # noqa: BLE001
        report['error'] = '%s: %s' % (type(error).__name__, error)
    finally:
        args.out.write_text(json.dumps(report, indent=2, default=str))
    for failure in report['failures']:
        print('FAIL', failure)
    if report.get('error'):
        print('ERROR', report['error'])
    print('SDPA_QWEN_CARD_M passed=%s cases=%d failures=%d report=%s native_log=%s' % (
        report['passed'], len(report['cases']), len(report['failures']), args.out, native.path))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
