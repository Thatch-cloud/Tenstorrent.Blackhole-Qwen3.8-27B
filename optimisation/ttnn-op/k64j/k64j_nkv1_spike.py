#!/usr/bin/env python3
"""K64j at ONE KV head per chip (S2T-01, the first hardware question of the four-card S2 port): does the runtime-extent
program (flag 0x20) build and run bit-exact against its compile-time twin when a chip holds six query heads on a single KV
head, and how fast is it?

WHAT CHANGES AT FOUR CARDS. A chip's attention is 6 query heads on 1 KV head of 256 (24 / 4 and 4 / 4), not 12 on 2. A
token's folded query is 6 rows (token-major: at one KV head the fold is a reshape), a group of R tokens is (1, 1, 6R, 256),
the paged cache is (blocks, 1, 64, 256), and the narrow tail mask is (B, 1, 6R, 256). The K64j factory takes the head
counts from the tensors and its q-slice (flag 0x4) needs a second KV head to slice between (optimisation/ttnn-op/
sdpa_decode_slice, F15), so the flag sets are tail 0x1 | share 0x2 | extent 0x20 = 0x23 at G8B2 (two 8-row groups, 48 folded
rows per entry, PNHt 2: the pair's 96 rows were 3 tiles) and tail | extent = 0x21 at G16B1 (one 16-row group, 96 rows, 3
tiles). Nothing in the pair's CB1 / CB2a / CB2b evidence says the kernels are exact there; this is the card question before
any more four-card attention work.

WHAT IT RUNS, per geometry, per family E (--extents), per position within the family (--starts), per seed:
  X   the extent call - cur_pos = E - 1 per entry, the full C-wide page table POISONED past E (K = 0, V = +16,384: a read past
      E moves every row), the narrow tail mask, sentinel MAGIC | flags - against the compile-time call at capacity E (the same
      flags without 0x20, no cur_pos, the table truncated to E / 64, the wide mask): every entry BIT-EQUAL (the same comparison
      CB1 makes at two KV heads).
  L   the legacy stock decode (sentinel 0, the full causal mask, the same truncated table): a NUMERICS report (max / mean
      absolute difference against the extent output), never a verdict - it is a different program.
  F   the outputs are finite; the [QWEN-SDPA] factory line for every requested program appears in the native log
      (captured for the whole run) with the geometry it should have (B, PNHt = ceil(6R / 32)).
  T   eager per-call medians (--iterations) of the extent, compile-time and legacy calls at the first and last family.

A refusal or a hang is DATA: an arm that raises records the error and the run continues; a device fault ends the container
and the partial report (rewritten after every arm) names the last arm. Run through optimisation/ttnn-op/k64j/run_card_b.sh
with K64J_HARNESS=nkv1_spike on the card M window (the graft K64j mounted and verified by the runner), watcher pass first.

The verdict line: 'K64J_NKV1 verdict=PASS|FAIL|NO-DECISION scope=full|reduced ...'. PASS: every geometry ran, every extent
call bit-equal to its compile-time twin on finite outputs with its factory lines; scope=full when both geometries ran at
--extents covering the pair's K1 set. FAIL: a decisive comparison differs. NO-DECISION: a refusal, a missing factory line,
an unwritten output, or nothing decisive.

The helpers above the device part import no ttnn or torch at module level and are tested on CPU by test_k64j_nkv1_spike.py,
which also runs the whole flow on a fake device whose attention honours cur_pos and the mask (so a call that ignored either
one is caught as FAIL).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / 'sdpa_decode_qwen',):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

VERDICT = 'K64J_NKV1'
HEAD_ROWS = 6                      # folded query rows per token at one KV head (tp_shapes.geometry(4).attn_fold_rows)
HEAD_DIM = 256
PAGE = 64
K_CHUNK = 256
TILE = 32
CAPACITY = 131328
SCALE = 1.0 / 16
POISON_K, POISON_V, POISON_BLOCKS = 0.0, 16384.0, 64
TAIL, SHARE, EXTENT = 0x1, 0x2, 0x20
LEGACY = 0
MAGIC = 0x51DEC000
# geometry -> (rows per group, groups per entry bundle, the flags the extent call carries)
GEOMETRIES = {'G8B2': (8, 2, TAIL | SHARE | EXTENT), 'G16B1': (16, 1, TAIL | EXTENT)}
DEFAULT_EXTENTS = (2304, 16896, 33024, 65792, 98560, 131328)     # K1's six families
DEFAULT_STARTS = (128, 200, 240, 255)                             # position inside the family (the last 256 keys)
FULL_EXTENTS = frozenset(DEFAULT_EXTENTS)


# ---------------------------------------------------------------------------------------------
# Pure helpers.
# ---------------------------------------------------------------------------------------------

def extent_of(position):
    """E = (position // 256 + 1) * 256: the 256-key family a row at `position` reads."""
    return (position // K_CHUNK + 1) * K_CHUNK


def bundle_positions(extent, start, batch, rows):
    """Entry b's first position in family E: E - 256 + min(255, start + b * rows) (a bundle never crosses its family)."""
    return [extent - K_CHUNK + min(K_CHUNK - 1, start + slot * rows) for slot in range(batch)]


def words_for(positions, share):
    """The cur_pos words: E - 1 per entry, every slot carrying slot 0's word under share (the one the kernels read)."""
    words = [extent_of(position) - 1 for position in positions]
    return [words[0]] * len(words) if share else words


def narrow_mask(torch, positions, extent, rows):
    """(B, 1, rows * 6, 256) bf16: entry b's folded row h is token h // 6 at positions[b] + token; the tail chunk's column c
    is the key at position extent - 256 + c, masked (-inf) where it is past the row's position, +0.0 elsewhere."""
    batch = len(positions)
    mask = torch.zeros(batch, 1, rows * HEAD_ROWS, K_CHUNK, dtype=torch.float32)
    columns = torch.arange(K_CHUNK) + (extent - K_CHUNK)
    for entry, position in enumerate(positions):
        for head in range(rows * HEAD_ROWS):
            row_position = position + head // HEAD_ROWS
            mask[entry, 0, head] = torch.where(columns > row_position, float('-inf'), 0.0)
    return mask.to(torch.bfloat16)


def wide_mask(torch, positions, extent, rows):
    """(B, 1, rows * 6, E): +0.0 before the last 256 columns, the narrow mask after (tail mode reads the last chunk only)."""
    narrow = narrow_mask(torch, positions, extent, rows)
    wide = torch.zeros(narrow.shape[0], 1, narrow.shape[2], extent, dtype=torch.bfloat16)
    wide[..., extent - K_CHUNK:] = narrow
    return wide


def causal_mask(torch, positions, extent, rows):
    """(B, 1, rows * 6, E): the legacy program's full causal mask - every column past a row's position is -inf."""
    batch = len(positions)
    mask = torch.zeros(batch, 1, rows * HEAD_ROWS, extent, dtype=torch.float32)
    columns = torch.arange(extent)
    for entry, position in enumerate(positions):
        for head in range(rows * HEAD_ROWS):
            mask[entry, 0, head] = torch.where(columns > position + head // HEAD_ROWS, float('-inf'), 0.0)
    return mask.to(torch.bfloat16)


def poisoned_row(torch, table, extent, capacity, poison):
    """A C / 64 page-table row: the user's first E / 64 pages, then the poison blocks (cycled) - a read past E reads them."""
    row = table[:capacity // PAGE].clone()
    tail = list(range(extent // PAGE, capacity // PAGE))
    for index, slot in enumerate(tail):
        row[slot] = poison[index % len(poison)]
    return row


def query_rows(torch, batch, rows, seed, salt=0):
    """(1, B, rows * 6, 256) bf16 N(0, 1): one folded query per entry."""
    generator = torch.Generator().manual_seed(1000 + seed + 31 * salt)
    return torch.randn(1, batch, rows * HEAD_ROWS, HEAD_DIM, generator=generator).to(torch.bfloat16)


def bits(torch, tensor):
    return tensor.to(torch.bfloat16).contiguous().view(torch.int16)


def differing(torch, left, right):
    if tuple(left.shape) != tuple(right.shape):
        return left.numel() + right.numel()
    return int((bits(torch, left) != bits(torch, right)).sum())


def numerics(torch, reference, candidate):
    gap = (reference.float() - candidate.float()).abs()
    scale = float(reference.float().abs().max()) or 1.0
    return dict(max_abs=float(gap.max()), mean_abs=float(gap.mean()), relative_max=float(gap.max()) / scale)


def program_key(capacity, batch, rows, mask_width, flags):
    """The factory's view of one call: (capacity, B, PNHt, mask_width_t, flags)."""
    heads = rows * HEAD_ROWS
    return (capacity, batch, (heads + TILE - 1) // TILE, mask_width // TILE, flags)


def factory_keys(lines):
    return {(line['St'] * TILE, line['B'], line['PNHt'], line['mask_width_t'], int(line['flags'], 16)) for line in lines}


def comparison(section, kind, label, differing_count, decisive, **details):
    return dict(section=section, kind=kind, label=label, differing=int(differing_count), decisive=bool(decisive),
                equal=differing_count == 0, **details)


def tally(comparisons):
    decisive = [entry for entry in comparisons if entry['decisive']]
    return dict(comparisons=len(comparisons), decisive=len(decisive),
                differing=sum(1 for entry in decisive if not entry['equal']))


def decide(report):
    """PASS / FAIL / NO-DECISION. A raised arm, a missing factory line, a non-finite output or a run with nothing decisive
    is NO-DECISION; a differing decisive comparison is FAIL."""
    problems = list(report.get('failures', []))
    for name, state in sorted(report.get('geometries', {}).items()):
        if state.get('error'):
            problems.append('%s: %s' % (name, state['error']))
    comparisons = report.get('comparisons', [])
    counted = tally(comparisons)
    if problems or not counted['decisive']:
        return dict(verdict='NO-DECISION', problems=problems or ['nothing decisive ran'])
    if counted['differing']:
        return dict(verdict='FAIL', problems=['%d of %d decisive comparisons differ' % (counted['differing'],
                                                                                      counted['decisive'])])
    return dict(verdict='PASS', problems=[])


def scope_of(report):
    ran = {name for name, state in report.get('geometries', {}).items() if state.get('ran')}
    covered = set(report.get('extents', [])) >= FULL_EXTENTS
    return 'full' if ran == set(GEOMETRIES) and covered and len(report.get('seeds', [])) >= 1 else 'reduced'


def verdict_line(report):
    counted = tally(report.get('comparisons', []))
    timings = report.get('timings') or {}
    return '%s verdict=%s scope=%s flags=%s comparisons=%d differing=%d geometries=%s timed=%s' % (
        VERDICT, report['decision']['verdict'], scope_of(report),
        ','.join('%s:0x%x' % (name, GEOMETRIES[name][2]) for name in sorted(report.get('geometries', {}))),
        counted['comparisons'], counted['differing'], ','.join(sorted(report.get('geometries', {}))),
        'yes' if timings else 'no')


def parse_geometries(text):
    names = [part.strip() for part in text.split(',') if part.strip()]
    unknown = [name for name in names if name not in GEOMETRIES]
    if unknown or not names:
        raise ValueError('geometries are %s, got %r' % (', '.join(sorted(GEOMETRIES)), text))
    return names


def parse_ints(text, name, multiple=None, low=1, high=None):
    values = [int(part) for part in text.split(',') if part.strip()]
    for value in values:
        if value < low or (multiple and value % multiple) or (high is not None and value > high):
            raise ValueError('%s: %d is not an accepted value' % (name, value))
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--capacity', type=int, default=CAPACITY)
    parser.add_argument('--extents', default=','.join(map(str, DEFAULT_EXTENTS)))
    parser.add_argument('--starts', default=','.join(map(str, DEFAULT_STARTS)))
    parser.add_argument('--geometries', default=','.join(sorted(GEOMETRIES)))
    parser.add_argument('--seeds', default='0,1')
    parser.add_argument('--iterations', type=int, default=20, help='timing calls per arm (0: no timing)')
    parser.add_argument('--no-legacy', action='store_true', help='skip the legacy numerics arm')
    parser.add_argument('--device-id', type=int, default=int(os.environ.get('K64J_DEVICE_ID', '0')))
    parser.add_argument('--deadline-s', type=float, default=0.0, help='stop cleanly between arms after this long (0: none)')
    parser.add_argument('--watchdog', type=float, default=0.0, help='accepted for the runner; per-call watchdogs are the runner\'s')
    parser.add_argument('--expect-binary-sha256', default='')
    args = parser.parse_args(argv)
    args.geometries = parse_geometries(args.geometries)
    args.extents = parse_ints(args.extents, '--extents', multiple=K_CHUNK, low=K_CHUNK, high=args.capacity)
    args.starts = parse_ints(args.starts, '--starts', low=0, high=K_CHUNK - 1)
    args.seeds = parse_ints(args.seeds, '--seeds', low=0)
    if args.capacity % PAGE:
        parser.error('--capacity must be a multiple of %d' % PAGE)
    return args


# ---------------------------------------------------------------------------------------------
# The device part.
# ---------------------------------------------------------------------------------------------

class Deadline:
    def __init__(self, seconds):
        self.started, self.seconds = time.time(), seconds

    def reached(self):
        return bool(self.seconds) and time.time() - self.started > self.seconds


class Pool:
    """One seed's bf8 paged K/V (C / 64 clean blocks + 64 poison blocks, ONE KV head, 64-key pages) and one page table."""

    def __init__(self, ttnn, torch, device, capacity, seed):
        self.ttnn, self.torch, self.device, self.capacity = ttnn, torch, device, capacity
        clean = capacity // PAGE
        generator = torch.Generator().manual_seed(4000 + seed)
        shape = (1, PAGE, HEAD_DIM)
        keys = torch.cat([torch.randn((clean,) + shape, generator=generator) * 2,
                          torch.full((POISON_BLOCKS,) + shape, POISON_K)])
        values = torch.cat([torch.randn((clean,) + shape, generator=generator),
                            torch.full((POISON_BLOCKS,) + shape, POISON_V)])
        self.poison = list(range(clean, clean + POISON_BLOCKS))
        self.table = torch.randperm(clean, generator=generator).to(torch.int32)
        self.k = self.upload(keys.to(torch.bfloat16), ttnn.bfloat8_b)
        self.v = self.upload(values.to(torch.bfloat16), ttnn.bfloat8_b)

    def upload(self, host, dtype=None):
        ttnn = self.ttnn
        dtype = ttnn.bfloat16 if dtype is None else dtype
        layout = ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT
        return ttnn.from_torch(host, device=self.device, dtype=dtype, layout=layout,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def config(self, sentinel):
        grid = self.device.compute_with_storage_grid_size()
        return self.ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y), exp_approx_mode=False,
                                           q_chunk_size=sentinel, k_chunk_size=K_CHUNK)

    def launch(self, query, pages, *, mask, sentinel, cur_pos=None):
        ttnn = self.ttnn
        options = dict(is_causal=False, attn_mask=mask, scale=SCALE, program_config=self.config(sentinel),
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if cur_pos is not None:
            options['cur_pos_tensor'] = cur_pos
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(query, self.k, self.v, page_table_tensor=pages,
                                                                          **options)

    def run(self, *arguments, **keywords):
        output = self.launch(*arguments, **keywords)
        host = self.ttnn.to_torch(output)
        self.ttnn.deallocate(output)
        return host

    def close(self):
        for tensor in (self.k, self.v):
            self.ttnn.deallocate(tensor)


def timed(ttnn, device, call, iterations):
    for _ in range(2):
        ttnn.deallocate(call())
    ttnn.synchronize_device(device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = call()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - start) * 1000)
        ttnn.deallocate(output)
    samples.sort()
    return dict(median_ms=statistics.median(samples), best_ms=samples[0], worst_ms=samples[-1], samples=len(samples))


def run_geometry(ttnn, torch, device, args, report, name, seed, deadline):
    rows, batch, flags = GEOMETRIES[name]
    share = bool(flags & SHARE)
    state = report['geometries'].setdefault(name, dict(ran=False, flags='0x%x' % flags, rows=rows, batch=batch))
    pool = Pool(ttnn, torch, device, args.capacity, seed)
    handles = []
    try:
        for extent in args.extents:
            if deadline.reached():
                report['deadline'] = 'stopped before %s E=%d' % (name, extent)
                return
            pages = pool.upload(torch.stack([poisoned_row(torch, pool.table, extent, args.capacity, pool.poison)
                                             for _ in range(batch)]).contiguous(), ttnn.int32)
            reference_pages = pool.upload(pool.table[:extent // PAGE].repeat(batch, 1).contiguous(), ttnn.int32)
            handles += [pages, reference_pages]
            for start in args.starts:
                positions = bundle_positions(extent, start, batch, rows)
                words = pool.upload(torch.tensor(words_for(positions, share), dtype=torch.int32), ttnn.int32)
                narrow = pool.upload(narrow_mask(torch, positions, extent, rows))
                wide = pool.upload(wide_mask(torch, positions, extent, rows))
                handles += [words, narrow, wide]
                query = pool.upload(query_rows(torch, batch, rows, seed, salt=extent))
                handles.append(query)
                label = '%s/seed%d/E%d+%d' % (name, seed, extent, start)
                got = pool.run(query, pages, mask=narrow, sentinel=MAGIC | flags, cur_pos=words)
                reference = pool.run(query, reference_pages, mask=wide, sentinel=MAGIC | (flags & ~EXTENT))
                report['requested'].add(program_key(args.capacity, batch, rows, K_CHUNK, flags))
                report['requested'].add(program_key(extent, batch, rows, extent, flags & ~EXTENT))
                finite = bool(torch.isfinite(got.float()).all()) and bool(torch.isfinite(reference.float()).all())
                if not finite:
                    report['failures'].append('%s: a non-finite output' % label)
                for index in range(batch):
                    entry = comparison('X', 'extent_vs_compile_time', '%s/entry%d' % (label, index),
                                       differing(torch, got[:, index], reference[:, index]), True, geometry=name,
                                       extent=extent, start=start, seed=seed, position=positions[index])
                    report['comparisons'].append(entry)
                    if not entry['equal']:
                        print(json.dumps(entry), flush=True)
                if not args.no_legacy:
                    full = pool.upload(causal_mask(torch, positions, extent, rows))
                    handles.append(full)
                    legacy = pool.run(query, reference_pages, mask=full, sentinel=LEGACY)
                    report['numerics'].append(dict(label=label, **numerics(torch, legacy, got)))
                for tensor in (words, narrow, wide, query):
                    ttnn.deallocate(tensor)
                    handles.remove(tensor)
                write(args, report)
            for tensor in (pages, reference_pages):
                ttnn.deallocate(tensor)
                handles.remove(tensor)
        if args.iterations and seed == args.seeds[0]:
            for extent in (min(args.extents), max(args.extents)):
                positions = bundle_positions(extent, args.starts[-1], batch, rows)
                arms = dict(
                    pages=pool.upload(torch.stack([poisoned_row(torch, pool.table, extent, args.capacity, pool.poison)
                                                   for _ in range(batch)]).contiguous(), ttnn.int32),
                    reference_pages=pool.upload(pool.table[:extent // PAGE].repeat(batch, 1).contiguous(), ttnn.int32),
                    words=pool.upload(torch.tensor(words_for(positions, share), dtype=torch.int32), ttnn.int32),
                    narrow=pool.upload(narrow_mask(torch, positions, extent, rows)),
                    wide=pool.upload(wide_mask(torch, positions, extent, rows)),
                    query=pool.upload(query_rows(torch, batch, rows, seed, salt=extent)),
                    full=pool.upload(causal_mask(torch, positions, extent, rows)))
                try:
                    report['timings']['%s/E%d' % (name, extent)] = dict(
                        extent=timed(ttnn, device, lambda: pool.launch(
                            arms['query'], arms['pages'], mask=arms['narrow'], sentinel=MAGIC | flags,
                            cur_pos=arms['words']), args.iterations),
                        compile_time=timed(ttnn, device, lambda: pool.launch(
                            arms['query'], arms['reference_pages'], mask=arms['wide'],
                            sentinel=MAGIC | (flags & ~EXTENT)), args.iterations),
                        legacy=timed(ttnn, device, lambda: pool.launch(
                            arms['query'], arms['reference_pages'], mask=arms['full'], sentinel=LEGACY),
                            args.iterations))
                finally:
                    for tensor in arms.values():
                        ttnn.deallocate(tensor)
        state['ran'] = True
    finally:
        for tensor in handles:
            try:
                ttnn.deallocate(tensor)
            except BaseException:  # noqa: BLE001 - best effort on the way out
                pass
        pool.close()


def write(args, report):
    payload = {key: value for key, value in report.items() if not key.startswith('_') and key != 'requested'}
    payload['requested_programs'] = sorted(list(key) for key in report['requested'])
    payload['tally'] = tally(report['comparisons'])
    args.out.write_text(json.dumps(payload, indent=2, default=str))


def check_factory_lines(report, text, card):
    """Every requested program's [QWEN-SDPA] factory line is in the native log."""
    seen = factory_keys(card.factory_lines(text))
    report['factory_lines'] = len(seen)
    missing = sorted(key for key in report['requested'] if key not in seen)
    for key in missing:
        report['failures'].append('no [QWEN-SDPA] factory line for capacity=%d B=%d PNHt=%d mask_width_t=%d flags=0x%x' % key)


def run(args, report):
    import torch
    import ttnn

    import test_sdpa_decode_qwen_card_m as card

    if os.environ.get(card.SCRATCH_ENV) != '1':
        report['failures'].append('%s=1 is required: the G8 programs do not fit L1 without the compact scratch'
                                  % card.SCRATCH_ENV)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    native = card.NativeLog(args.out.with_name(args.out.name + '.native.log'))
    deadline = Deadline(args.deadline_s)
    with native:
        device = ttnn.open_device(device_id=args.device_id, l1_small_size=24576)
        try:
            try:
                device.enable_program_cache()
            except Exception:  # noqa: BLE001 - default-on in newer runtimes
                pass
            grid = device.compute_with_storage_grid_size()
            report['grid'] = [grid.x, grid.y]
            for name in args.geometries:
                for seed in args.seeds:
                    try:
                        run_geometry(ttnn, torch, device, args, report, name, seed, deadline)
                    except BaseException as error:  # noqa: BLE001 - a refusal is data; the run continues
                        state = report['geometries'].setdefault(name, dict(ran=False))
                        state['error'] = '%s: %s' % (type(error).__name__, str(error)[:400])
                        state['traceback'] = traceback.format_exc()
                        print(state['traceback'], flush=True)
                        write(args, report)
                        break
        finally:
            ttnn.close_device(device)
    check_factory_lines(report, native.text(), card)


def main(argv=None):
    args = parse_args(argv)
    report = dict(plan='S2T-01: K64j at one KV head per chip', argv=list(sys.argv[1:] if argv is None else argv),
                  capacity=args.capacity, extents=args.extents, starts=args.starts, seeds=args.seeds,
                  geometries={}, comparisons=[], numerics=[], timings={}, failures=[], requested=set(),
                  env={name: os.environ.get(name) for name in ('QWEN_SDPA_TREE_SCRATCH_ROUNDS', 'TT_METAL_WATCHER')},
                  expect_binary_sha256=args.expect_binary_sha256)
    try:
        run(args, report)
    except BaseException as error:  # noqa: BLE001
        report['error'] = '%s: %s' % (type(error).__name__, error)
        report['traceback'] = traceback.format_exc()
    report['decision'] = decide(report)
    if report.get('error'):
        report['decision'] = dict(verdict='NO-DECISION', problems=[report['error']] + report['decision']['problems'])
    report['verdict_line'] = verdict_line(report)
    write(args, report)
    for failure in report['failures']:
        print('FAIL', failure)
    print(report['verdict_line'], flush=True)
    return 0 if report['decision']['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
