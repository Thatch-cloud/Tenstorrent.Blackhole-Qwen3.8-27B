"""Card-M unit test for the [QWEN-SDPA] decode SDPA: stage 1 (tail-only mask) and stage 3 (KV share).

WHAT IT PROVES. With a graft mounted (its _ttnncpp.so carries the factory branch, its
sdpa_decode op directory the two qwen kernels), a paged_scaled_dot_product_attention_decode
call whose SDPAProgramConfig.q_chunk_size is a 0x51DEC0xx sentinel returns the SAME BYTES as
the legacy call (q_chunk_size=0) on replay-shaped inputs, and is faster. The stage is read off
the loaded binary: K64e (stage 1) refuses flag 0x2; K64f (stage 3) carries the KV share.

  - Q (1, B, rows*12, 256) bf16: `rows` draft rows x 12 heads folded per entry, 2 KV heads per
    chip (attention_head_fold.fold_query). G4 (four-row groups, PNHt=2): one 16-row block is a
    batch-3 bundle (group offsets 0, 4, 8) and a batch-1 bundle (offset 12). G8 (eight-row
    groups, PNHt=3): the same 16 tokens as ONE batch-2 bundle (offsets 0, 8);
  - K/V: one bf8 paged pool (capacity // 64 + 64 blocks, 2 KV heads, 64-token pages, 256
    dims), K ~ 2 N(0,1), V ~ N(0,1); ONE random page table (a permutation of the pool's
    blocks) repeated to every bundle row, as the replay reader does;
  - mask (B, 1, rows*12, capacity) bf16: zeros everywhere but the last 256 columns, which hold
    the refresh kernel's formula (-inf where cache_position > start + offset + b*rows + t);
  - capacities 2,304 (9 chunks for 16 cores per head: the early-return path), 33,024 and
    131,328 (513 chunks, 4 tree rounds); is_causal=False, k_chunk_size=256, full grid,
    exp_approx_mode=False, scale 1/16.

Stage-1 checks (G4; all byte-for-byte, torch.equal on int16 views of every output element):
  equal      tail == legacy for every capacity x seed x bundle x query variant x start
             (variants: normal; peaky - q aligned with 8 cached keys per head so the running
             max moves across chunks and cores; zeroq - every fourth folded row zero, +-0 scores)
  planted    -inf planted at mask chunk 0, column 0: the legacy output changes, the tail
             output does not (the non-final mask chunks are really neither read nor added)
  narrow     tail with the narrow (B,1,rows*12,256) mask == tail with the wide mask
  cache      alternating legacy / tail calls on identical shapes: each always equals its first

Stage-3 checks (spec section 8; skipped on a stage-1 binary, legacy halves recorded by the
reference run):
  E5         for G8 (B=2) and G4 (B=3, and B=1 where the share flag must be a no-op), per case:
             0x51DEC000 (qwen reader, no flags), 0x2 share, 0x1 tail, 0x3 tail+share, and tail /
             tail+share with the narrow mask, each == legacy
  E4         (stage 2, recorded, not a failure) G8 unfolded per token == G4 unfolded, legacy and tail
  N1         twins read no K/V: page-table rows 1.. set to OTHER permutations. share output ==
             legacy with every row equal (the leader's row); entries 1.. != legacy with the
             distinct rows. Wrong attention by design, never a hang (the watchdog is the check)
  N2         the planted chunk-0 -inf does not move tail+share (the share path's mask reads)
  N3         refusals, per stage: B=4 twin bands, unknown flag, 512-wide mask under tail, narrow
             mask without tail, the sentinel on a causal call (stage 1: flag 0x2 itself)
  N4         1,000 eager calls alternating legacy and tail+share on identical shapes and the
             distinct page table, so the two modes' outputs differ: each always equals its first
  N5         one trace of legacy/tail+share G8, legacy/tail+share G4 B3, legacy/tail G4 B1,
             replayed 200 times, every replay bit-equal to the eager outputs
  log        one factory '[QWEN-SDPA] flags=...' line for every qwen program the run requested
             (key: capacity, B, PNHt, mask width, flags), none it did not, kv_share true exactly
             for flag 0x2 with B>1, scratch_slots=4, cb_bytes 790,592 / 796,736 (PNHt 2) and
             1,042,496 / 1,048,640 (PNHt 3) at 33,024 / 131,328
  timing     median synchronised host timing per bundle; the spec's acceptance ratios recorded

The padded query rows 48..63 of the host-built G4 mask are the tilizer's zeros, where the
device refresh kernel writes -inf in the last chunk; the ops are row-independent and only
the logical rows are compared, so this does not bear on the equality.

--legacy-only runs just the legacy calls (for the stock image, no graft) and records each
output's sha256; --reference <that report> makes a graft run assert its legacy outputs are
the stock binary's bytes (the graft's legacy branch did not regress).

--watchdog S arms a per-device-call watchdog: a call (or read-back, or trace replay) that has
not returned within S seconds prints WATCHDOG, writes the partial report and os._exit(3)s, so
a hung multicast handshake ends the container instead of wedging it (run_card_m.sh WATCHER=1).

RUN on the qualification card only (run_card_m.sh: QUAL_CARD, default card B; never the serving
pair without ALLOW_SERVING_CARD=1), inside the serving image with the graft
mounted exactly as the arm mounts it and a FRESH kernel cache (run_card_m.sh does this):

    python3 -B test_sdpa_decode_qwen_card_m.py --out /results/card-m.json

Everything the C++ side prints (tt-logger, including the [QWEN-SDPA] lines) goes to
<out>.native.log; the helpers above Case import no ttnn and are unit-tested on CPU in
test_sdpa_decode_qwen_sources.py.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import threading
import time

HEADS, KV_HEADS, HEAD_DIM, PAGE, K_CHUNK, TILE = 48, 2, 256, 64, 256, 32
ROWS = 4                                   # draft rows per four-row fold group (4 x 12 heads = 48)
ROWS8 = 8                                  # eight-row groups (8 x 12 = 96 folded rows, PNHt=3)
BUNDLES = ((0, 4, 8), (12,))              # G4 group offsets per bundle: batch 3, then batch 1
G8_OFFSETS = (0, 8)                        # G8: the same 16 tokens as one batch-2 bundle
CAPACITIES = (2304, 33024, 131328)
STARTS = (0, 7, 240)                       # block start = capacity - 256 + this
SEEDS = (0, 1, 2, 3, 4)
VARIANTS = ('normal', 'peaky', 'zeroq')
LEGACY = 0
MAGIC = 0x51DEC000
QWEN_PLAIN, TAIL, SHARE, TAIL_SHARE = MAGIC, MAGIC | 0x1, MAGIC | 0x2, MAGIC | 0x3
UNKNOWN_FLAG = MAGIC | 0x4
SHARE_STAGE3, NO_FLAGS = SHARE, QWEN_PLAIN   # stage-1 names
SCALE = 1.0 / 16
# Per (capacity, PNHt): the spec's L1 table (compact c_19; KV share adds no CB bytes).
CB_BYTES = {(33024, 2): 790592, (131328, 2): 796736, (33024, 3): 1042496, (131328, 3): 1048640}
SCRATCH_SLOTS = 4
SCRATCH_ENV = 'QWEN_SDPA_TREE_SCRATCH_ROUNDS'   # the legacy factory's compact-scratch switch
BINARY_MARKER = b'[QWEN-SDPA] flags='                         # both stages (factory F4)
SHARE_BINARY_MARKER = b'[QWEN-SDPA] KV-share twin bands'      # stage 3 only (factory F9)
STAGE1_BINARY_MARKER = b'[QWEN-SDPA] KV share is not in this build'  # stage 1 only
FACTORY_LINE = re.compile(r'\[QWEN-SDPA\] flags=(0x[0-9a-f]+) B=([0-9]+) PNHt=([0-9]+) St=([0-9]+) '
                          r'mask_width_t=([0-9]+) kv_share=([a-z0-9]+) scratch_slots=([0-9]+) cb_bytes=([0-9]+)')
# N3: (name, sentinel, factory text, mask, batches, causal, stages it applies to).
REFUSALS = (
    ('share flag (stage 1)', SHARE, 'KV share is not in this build', 'wide', 3, False, (1,)),
    ('twin bands for B=4', SHARE, 'twin bands do not fit', 'wide', 4, False, (3,)),
    ('unknown flag 0x4', UNKNOWN_FLAG, 'unknown flags', 'wide', 3, False, (1, 3)),
    ('tail with a 512-wide mask', TAIL, 'tail mask must be full width', 'half', 3, False, (1, 3)),
    ('narrow mask without tail', QWEN_PLAIN, 'a narrow mask needs the tail flag', 'narrow', 3, False, (1, 3)),
    ('sentinel on a causal call', TAIL, 'modes are non-causal', None, 3, True, (1, 3)),
)
# E5: the modes each share case runs against legacy, (name, sentinel, narrow mask).
SHARE_MODES = (('plain', QWEN_PLAIN, False), ('share', SHARE, False), ('tail', TAIL, False),
               ('tail_share', TAIL_SHARE, False), ('tail_narrow', TAIL, True), ('tail_share_narrow', TAIL_SHARE, True))


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


def build_mask(torch, capacity, start, offsets, *, width=None, plant=False, rows=ROWS):
    """(B, 1, rows*12, width) bf16: zeros, the last 256 columns masked per mask_positions.
    width defaults to the full capacity; 256 gives the narrow mask (the same last 256 columns),
    512 a two-chunk mask the factory must refuse. plant puts -inf at column 0 of every
    row: chunk 0, which only the non-tail paths read."""
    width = capacity if width is None else width
    positions = torch.tensor(mask_positions(start, offsets, rows), dtype=torch.int64)   # (B, rows*12)
    cache = torch.arange(capacity - 256, capacity, dtype=torch.int64)                   # (256,)
    tail = torch.where(cache[None, None, :] > positions[:, :, None], float('-inf'), 0.0)
    mask = torch.zeros(len(offsets), 1, rows * 12, width, dtype=torch.float32)
    mask[:, 0, :, width - 256:] = tail
    if plant:
        if width < capacity:
            raise ValueError('Planting needs the full-width mask')
        mask[:, 0, :, 0] = float('-inf')
    return mask.to(torch.bfloat16)


def build_query(torch, batches, seed, variant, keys=None, table=None, rows=ROWS):
    """(1, B, rows*12, 256) bf16, N(0,1). 'peaky': each folded row is 6 x the unit vectors of 8
    random cached keys of its KV head plus small noise, so those scores dominate wherever
    they fall (their chunks, cores and tree order move with the seed); 'zeroq': every
    fourth folded row is zero, so its scores are +-0. rows=4 draws exactly the stage-1 bytes."""
    heads = rows * 12
    generator = torch.Generator().manual_seed(1000 + seed)
    query = torch.randn(1, batches, heads, HEAD_DIM, generator=generator)
    if variant == 'peaky':
        if keys is None or table is None:
            raise ValueError('Peaky queries need the host keys and page table')
        query *= 0.1
        for head in range(heads):
            kv = head // (heads // KV_HEADS)
            for token in torch.randint(0, len(table) * PAGE, (8,), generator=generator).tolist():
                vector = keys[int(table[token // PAGE]), kv, token % PAGE].float()
                query[0, :, head] += 6 * vector / vector.norm().clamp_min(1e-3)
    elif variant == 'zeroq':
        query[0, :, ::4] = 0.0
    elif variant != 'normal':
        raise ValueError('Unknown query variant %r' % (variant,))
    return query.to(torch.bfloat16)


def fold_tokens(tokens):
    """(1, rows, 12, 256) -> (1, 1, rows*12, 256): attention_head_fold.fold_query's layout
    (KV head major, then token, then the head's six query heads)."""
    rows = tokens.shape[1]
    return tokens.reshape(rows, KV_HEADS, 6, HEAD_DIM).permute(1, 0, 2, 3).reshape(1, 1, rows * 12, HEAD_DIM).contiguous()


def unfold_rows(folded, rows):
    """(1, 1, >= rows*12, 256) -> (1, rows, 12, 256): attention_head_fold.unfold_output."""
    folded = folded[:, :, :rows * 12]
    return folded.reshape(KV_HEADS, rows, 6, HEAD_DIM).permute(1, 0, 2, 3).reshape(1, rows, 12, HEAD_DIM).contiguous()


def fold_entries(torch, tokens, offsets, rows):
    """A bundle's (1, B, rows*12, 256) query from token-level (1, T, 12, 256) queries."""
    return torch.cat([fold_tokens(tokens[:, offset:offset + rows]) for offset in offsets], dim=1)


def unfold_entries(torch, output, rows):
    """A bundle's (1, B, >= rows*12, 256) output back to (1, B*rows, 12, 256), entry order."""
    return torch.cat([unfold_rows(output[:, entry:entry + 1], rows) for entry in range(output.shape[1])], dim=1)


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


def program_key(capacity, batches, query_rows, mask_width, sentinel):
    """The factory's view of one qwen call: (capacity, B, PNHt, mask_width_t, flags)."""
    return (capacity, batches, (query_rows + TILE - 1) // TILE,
            (capacity if mask_width is None else mask_width) // TILE, sentinel & 0xFF)


def line_key(line):
    return (line['St'] * TILE, line['B'], line['PNHt'], line['mask_width_t'], int(line['flags'], 16))


def describe(key):
    return 'cap=%d B=%d PNHt=%d mask_width_t=%d flags=0x%x' % key


def check_program_lines(lines, requested):
    """Every qwen program the run requested (program_key) has a factory line, every line was
    requested, and each carries kv_share (true exactly for flag 0x2 with B>1), the compact
    scratch and, where the spec tabulates it, the CB bytes. With the program cache on there is
    one line per program; repeats (cache off) are allowed."""
    problems = []
    found = {}
    for line in lines:
        found.setdefault(line_key(line), []).append(line)
    for key in sorted(requested):
        capacity, batches, pnht, _width, flags = key
        if key not in found:
            problems.append('%s: no [QWEN-SDPA] line' % describe(key))
            continue
        expected = dict(kv_share='true' if flags & 0x2 and batches > 1 else 'false', scratch_slots=SCRATCH_SLOTS,
                        cb_bytes=CB_BYTES.get((capacity, pnht)))
        for line in found[key]:
            wrong = {name: (line[name], value) for name, value in expected.items()
                     if value is not None and line[name] != value}
            if wrong:
                problems.append('%s: (found, expected) %r' % (describe(key), wrong))
                break
    stray = sorted(set(found) - set(requested))
    if stray:
        problems.append('%d [QWEN-SDPA] programs nobody requested: %s' % (len(stray), '; '.join(map(describe, stray))))
    return problems


def check_factory_lines(lines, capacities, narrow=True):
    """Stage-1 form: a tail program per capacity x G4 batch x mask width was built."""
    requested = {program_key(capacity, len(offsets), ROWS * 12, width, TAIL)
                 for capacity in capacities for offsets in BUNDLES
                 for width in ((None, K_CHUNK) if narrow else (None,))}
    return check_program_lines(lines, requested)


def binary_markers(path):
    """Which [QWEN-SDPA] format literals the _ttnncpp.so at `path` carries."""
    import mmap

    with open(path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
        return dict(flags=view.find(BINARY_MARKER) >= 0, share=view.find(SHARE_BINARY_MARKER) >= 0,
                    stage1=view.find(STAGE1_BINARY_MARKER) >= 0)


def binary_stage(markers):
    """3: the KV-share build (K64f); 1: tail only (K64e); 0: no [QWEN-SDPA] branch (stock)."""
    if markers['flags'] and markers['share'] and not markers['stage1']:
        return 3
    if markers['flags'] and markers['stage1'] and not markers['share']:
        return 1
    if not any(markers.values()):
        return 0
    raise RuntimeError('Inconsistent [QWEN-SDPA] markers in the loaded binary: %r' % (markers,))


def loaded_binary(maps='/proc/self/maps'):
    """The _ttnncpp.so this process mapped, and the markers it carries."""
    paths = sorted({line.split()[-1] for line in Path(maps).read_text().splitlines()
                    if line.rstrip().endswith('_ttnncpp.so')})
    if len(paths) != 1:
        raise RuntimeError('Expected exactly one mapped _ttnncpp.so, found %r' % (paths,))
    return paths[0], binary_markers(paths[0])


def refusals_for(stage):
    return [entry for entry in REFUSALS if stage in entry[-1]]


class Watchdog:
    """A per-device-call deadline. op(label) arms it; if the call has not returned within
    `seconds`, the poller thread prints WATCHDOG, calls on_fire (the partial report) and
    os._exit(3)s - a hung NoC handshake cannot be interrupted from Python otherwise."""

    def __init__(self, seconds, *, on_fire=None, stream=None, exit=os._exit, clock=time.monotonic):
        self.seconds, self.on_fire, self.stream, self.exit, self.clock = seconds, on_fire, stream, exit, clock
        self.label, self.deadline, self.fired = None, None, False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.seconds and self.thread is None:
            self.thread = threading.Thread(target=self.poll, name='sdpa-watchdog', daemon=True)
            self.thread.start()
        return self

    @contextmanager
    def op(self, label):
        if not self.seconds:
            yield
            return
        with self.lock:
            outer = (self.label, self.deadline)
            self.label, self.deadline = label, self.clock() + self.seconds
        try:
            yield
        finally:
            with self.lock:
                self.label, self.deadline = outer

    def check(self):
        """One poll: fire if the armed call is past its deadline. Returns whether it fired."""
        with self.lock:
            label, deadline = self.label, self.deadline
        if label is None or self.clock() < deadline:
            return False
        self.fired = True
        stream = self.stream or sys.stdout
        try:
            stream.write('WATCHDOG: %r did not return within %ss; exiting 3 (docker rm -f, then tt-smi -r '
                         'this card only, by the runner\'s printed reset command)\n' % (label, self.seconds))
            stream.flush()
            if self.on_fire is not None:
                self.on_fire(label)
        finally:
            self.exit(3)
        return True

    def poll(self):
        while not self.check():
            time.sleep(1.0)


WATCHDOG = Watchdog(0)


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
# Device part: the qualification card only.
# ---------------------------------------------------------------------------------------

class Case:
    """One capacity and seed: the K/V pool and page tables, and the calls on them."""

    def __init__(self, ttnn, torch, device, capacity, seed, *, requested=None):
        self.ttnn, self.torch, self.device, self.capacity = ttnn, torch, device, capacity
        self.requested = requested
        generator = torch.Generator().manual_seed(seed)
        blocks = num_blocks(capacity)
        self.keys = (torch.randn(blocks, KV_HEADS, PAGE, HEAD_DIM, generator=generator) * 2).to(torch.bfloat16)
        values = torch.randn(blocks, KV_HEADS, PAGE, HEAD_DIM, generator=generator).to(torch.bfloat16)
        self.table = torch.randperm(blocks, generator=generator)[:capacity // PAGE].to(torch.int32)
        # N1's other rows: other permutations of the same pool, from their own generator so the
        # stage-1 bytes above are unchanged.
        others = torch.Generator().manual_seed(5000 + seed)
        self.alt_tables = [torch.randperm(blocks, generator=others)[:capacity // PAGE].to(torch.int32) for _ in range(3)]
        self.owned = []
        self.k = self.upload(self.keys, ttnn.bfloat8_b)
        self.v = self.upload(values, ttnn.bfloat8_b)
        # One user's page table, repeated to each bundle's rows (attention_replay.py).
        self.tables = {}
        for batches in (1, 2, 3):
            self.pages(batches)

    def pages(self, batches, distinct=False):
        """(batches, capacity // 64) int32: the one table repeated, or (distinct) row 0 the
        table and rows 1.. other permutations."""
        key = (batches, distinct)
        if key not in self.tables:
            rows = [self.table] + (self.alt_tables if distinct else [self.table] * 3)
            host = self.torch.stack(rows[:batches]).contiguous()
            self.tables[key] = self.upload(host, self.ttnn.int32)
        return self.tables[key]

    def upload(self, host, dtype=None, *, keep=True):
        ttnn = self.ttnn
        dtype = ttnn.bfloat16 if dtype is None else dtype
        layout = ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT
        with WATCHDOG.op('upload %s' % (tuple(host.shape),)):
            tensor = ttnn.from_torch(host, device=self.device, dtype=dtype, layout=layout,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if keep:
            self.owned.append(tensor)
        return tensor

    def config(self, sentinel):
        grid = self.device.compute_with_storage_grid_size()
        return self.ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
            exp_approx_mode=False, q_chunk_size=sentinel, k_chunk_size=K_CHUNK)

    def call(self, query, mask, sentinel, *, is_causal=False, cur_pos_tensor=None, pages=None, record=True):
        # The paged binding (sdpa_decode_nanobind.cpp:82-106) takes page_table_tensor positionally
        # and only these keywords after kw_only(); it has no list-valued cur_pos, and a causal
        # paged call must carry an int32 row-major cur_pos TENSOR or validate refuses it first.
        ttnn = self.ttnn
        batches = query.shape[1]
        if record and sentinel != LEGACY and self.requested is not None:
            self.requested.add(program_key(self.capacity, batches, query.shape[2],
                                           None if mask is None else mask.shape[3], sentinel))
        options = dict(is_causal=is_causal, scale=SCALE,
                       program_config=self.config(sentinel), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if mask is not None:
            options['attn_mask'] = mask
        if cur_pos_tensor is not None:
            options['cur_pos_tensor'] = cur_pos_tensor
        with WATCHDOG.op('sdpa B=%d sentinel=0x%x cap=%d' % (batches, sentinel, self.capacity)):
            return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                query, self.k, self.v, self.tables[(batches, False)] if pages is None else pages, **options)

    def host(self, tensor):
        with WATCHDOG.op('read back'):
            result = self.ttnn.to_torch(tensor)
        self.ttnn.deallocate(tensor)
        return result

    def close(self):
        for tensor in self.owned:
            self.ttnn.deallocate(tensor)
        self.owned.clear()
        self.tables.clear()


def timed(ttnn, device, once, warmup, iters):
    for _ in range(warmup):
        ttnn.deallocate(once())
    with WATCHDOG.op('synchronize'):
        ttnn.synchronize_device(device)
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        out = once()
        with WATCHDOG.op('synchronize'):
            ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - started) * 1e6)
        ttnn.deallocate(out)
    return dict(median_us=statistics.median(samples), min_us=min(samples), mean_us=statistics.mean(samples))


def expect_refusal(case, query, mask, sentinel, needle, **options):
    try:
        out = case.call(query, mask, sentinel, record=False, **options)
    except Exception as error:  # noqa: BLE001 - TT_FATAL surfaces as RuntimeError
        text = str(error)
        return dict(refused=True, matched=needle in text, message=' '.join(text.split())[:300])
    case.ttnn.deallocate(out)
    return dict(refused=False, matched=False, message='call was accepted')


def record_legacy(torch, label, output, args, report):
    """The legacy output's sha, and (with --reference) whether the stock binary produced it."""
    sha = digest(torch, output)
    report['legacy_sha256'][label] = sha
    entry = dict(legacy_sha256=sha)
    if args.reference_shas is not None:
        if label not in args.reference_shas:
            entry['matches_reference'] = None
            report['failures'].append('%s: not in the reference report (rerun the reference with this script)' % label)
        else:
            entry['matches_reference'] = args.reference_shas[label] == sha
            if not entry['matches_reference']:
                report['failures'].append('%s: graft legacy output differs from the reference binary' % label)
    if not bool(torch.isfinite(output.float()).all()):
        report['failures'].append('%s: legacy output is not finite' % label)
    return entry


def equality_case(ttnn, torch, case, capacity, seed, offsets, variant, start, args, report):
    """Stage 1, G4: one replay-shaped call pair; every tensor it uploads is freed before it returns."""
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
        entry = dict(label=label, **record_legacy(torch, label, legacy, args, report))
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


def share_geometries(torch, seed, variant, keys, table):
    """G8 (one batch-2 bundle) and G4 (batch 3 + batch 1) over the SAME 16 tokens: the G4
    queries are folded from the G8 query's tokens, so the outputs compare per token (E4)."""
    q8 = build_query(torch, 2, seed, variant, keys, table, rows=ROWS8)
    tokens = unfold_entries(torch, q8, ROWS8)                                      # (1, 16, 12, 256)
    return (('G8', q8, G8_OFFSETS, ROWS8),
            ('G4B3', fold_entries(torch, tokens, BUNDLES[0], ROWS), BUNDLES[0], ROWS),
            ('G4B1', fold_entries(torch, tokens, BUNDLES[1], ROWS), BUNDLES[1], ROWS))


def share_case(ttnn, torch, case, capacity, seed, variant, start, args, report):
    """Stage 3 (E5, E4, N1, N2) for one capacity x seed x variant x start. On a reference
    (stock) run only the legacy calls run, to record their shas."""
    failures = report['failures']
    base = 'share/cap%d/seed%d/%s/start+%d' % (capacity, seed, variant, start)
    begin = capacity - 256 + start
    share = not args.legacy_only
    local = []
    entry = dict(label=base)

    def upload(host, dtype=None):
        tensor = case.upload(host, dtype, keep=False)
        local.append(tensor)
        return tensor

    try:
        inputs, outputs = {}, {}
        for name, host_query, offsets, rows in share_geometries(torch, seed, variant, case.keys, case.table):
            query = upload(host_query)
            wide = upload(build_mask(torch, capacity, begin, offsets, rows=rows))
            narrow = upload(build_mask(torch, capacity, begin, offsets, rows=rows, width=256)) if share else None
            inputs[name] = (query, wide, narrow, offsets, rows)
            legacy = case.host(case.call(query, wide, LEGACY))
            entry[name] = record_legacy(torch, '%s/%s/legacy' % (base, name), legacy, args, report)
            outputs[name] = {'legacy': legacy}
            if share:
                for mode, sentinel, use_narrow in SHARE_MODES:
                    out = case.host(case.call(query, narrow if use_narrow else wide, sentinel))
                    outputs[name][mode] = out
                    moved = differing(torch, legacy, out)
                    entry[name][mode + '_differing'] = moved
                    if moved:
                        failures.append('%s/%s: %s differs from legacy in %d elements' % (base, name, mode, moved))
        # E4 (stage 2's question, recorded here): the 8-row fold computes the same per-token bytes.
        modes = ('legacy', 'tail') if share else ('legacy',)
        e4 = {}
        for mode in modes:
            g8 = unfold_entries(torch, outputs['G8'][mode], ROWS8)
            g4 = torch.cat([unfold_entries(torch, outputs['G4B3'][mode], ROWS),
                            unfold_entries(torch, outputs['G4B1'][mode], ROWS)], dim=1)
            e4[mode] = differing(torch, g8, g4)
        entry['e4_differing'] = e4
        report['e4'].append(dict(label=base, differing=e4))
        if variant == 'normal' and start == args.starts[0]:
            for name in ('G8', 'G4B3'):
                query, wide, narrow, offsets, rows = inputs[name]
                batches = len(offsets)
                # N1: rows 1.. of the page table point at other blocks.
                distinct = case.pages(batches, distinct=True)
                legacy = outputs[name]['legacy']
                legacy_distinct = case.host(case.call(query, wide, LEGACY, pages=distinct))
                entry[name]['distinct'] = record_legacy(torch, '%s/%s/legacy-distinct' % (base, name),
                                                        legacy_distinct, args, report)
                if differing(torch, legacy_distinct[:, :1], legacy[:, :1]):
                    failures.append('%s/%s: legacy entry 0 moved with the other rows of the page table' % (base, name))
                if not share:
                    continue
                for mode, sentinel in (('share', SHARE), ('tail_share', TAIL_SHARE)):
                    out = case.host(case.call(query, wide, sentinel, pages=distinct))
                    same_as_leader_row = differing(torch, out, legacy)
                    twins_moved = [differing(torch, out[:, b:b + 1], legacy_distinct[:, b:b + 1]) > 0
                                   for b in range(1, batches)]
                    entry[name]['n1_' + mode] = dict(differing_from_leader_row=same_as_leader_row, twins_differ=twins_moved)
                    if same_as_leader_row:
                        failures.append('%s/%s: %s with distinct page-table rows is not legacy on the leader row '
                                        '(%d elements)' % (base, name, mode, same_as_leader_row))
                    if not all(twins_moved):
                        failures.append('%s/%s: %s twin output equals legacy on its own row - the twin read its own '
                                        'K/V or the control is dead (%r)' % (base, name, mode, twins_moved))
                # N2: the share path reads only the final mask chunk.
                planted = upload(build_mask(torch, capacity, begin, offsets, rows=rows, plant=True))
                legacy_planted = case.host(case.call(query, planted, LEGACY))
                if not differing(torch, legacy, legacy_planted):
                    failures.append('%s/%s: the planted chunk-0 -inf did not change legacy (control dead)' % (base, name))
                if share:
                    moved = differing(torch, outputs[name]['tail_share'],
                                      case.host(case.call(query, planted, TAIL_SHARE)))
                    entry[name]['planted_tail_share_differing'] = moved
                    if moved:
                        failures.append('%s/%s: tail+share read a non-final mask chunk (%d elements)' % (base, name, moved))
    finally:
        for tensor in local:
            ttnn.deallocate(tensor)
    report['share_cases'].append(entry)
    print('%-44s e4=%s' % (base, e4), flush=True)


def controls(ttnn, torch, device, case, capacity, stage, args, report):
    """Stage-1 program-cache alternation and the refusals of this binary's stage."""
    failures = report['failures']
    begin = capacity - 256 + args.starts[0]
    if args.alternations:
        query = case.upload(build_query(torch, 3, 0, 'normal'))
        mask = case.upload(build_mask(torch, capacity, begin, BUNDLES[0]))
        first, drift = {}, 0
        for index in range(2 * args.alternations):
            sentinel = (LEGACY, TAIL)[index % 2]
            out = case.host(case.call(query, mask, sentinel))
            if sentinel not in first:
                first[sentinel] = out
            elif differing(torch, first[sentinel], out):
                drift += 1
        report['alternation'].append(dict(capacity=capacity, modes='legacy/tail', calls=2 * args.alternations,
                                          drifted=drift, modes_equal=differing(torch, first[LEGACY], first[TAIL]) == 0))
        if drift:
            failures.append("cap%d: %d alternating legacy/tail calls drifted from their mode's first output" % (capacity, drift))
    # Refusals: the factory's TT_FATALs fire before any program is built or launched.
    results = {}
    for name, sentinel, needle, mask_kind, batches, causal, _stages in refusals_for(stage):
        offsets = tuple(ROWS * index for index in range(batches))
        query = case.upload(build_query(torch, batches, 0, 'normal'))
        width = {'wide': None, 'narrow': 256, 'half': 512, None: None}[mask_kind]
        mask = None if mask_kind is None else case.upload(build_mask(torch, capacity, begin, offsets, width=width))
        options = dict(pages=case.pages(batches))
        if causal:
            # validate() needs an INT32 ROW_MAJOR cur_pos tensor of length B on a causal paged call;
            # with it validate passes and the factory's [QWEN-SDPA] non-causal TT_FATAL is what fires.
            positions = case.upload(torch.full((batches,), capacity - 1, dtype=torch.int32), case.ttnn.int32)
            options.update(is_causal=True, cur_pos_tensor=positions)
        results[name] = expect_refusal(case, query, mask, sentinel, needle, **options)
        print('refusal cap=%d %-28s %s' % (capacity, name, results[name]), flush=True)
        if not (results[name]['refused'] and results[name]['matched']):
            failures.append('cap%d: %s was not refused by its [QWEN-SDPA] TT_FATAL: %s'
                            % (capacity, name, results[name]['message']))
    report['refusals'][capacity] = results


def capture(ttnn, device, operation):
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    ended = False
    try:
        try:
            result = operation()
        finally:
            ttnn.end_trace_capture(device, trace, cq_id=0)
            ended = True
    except BaseException:
        if ended:
            ttnn.release_trace(device, trace)
        raise
    return trace, result


def share_controls(ttnn, torch, device, case, capacity, args, report):
    """N4 (program-cache alternation) and N5 (trace-replay soak) on inputs where the modes'
    outputs differ (rows 1.. of the page table distinct), so a cache or replay mix-up shows."""
    failures = report['failures']
    begin = capacity - 256 + args.starts[0]
    q8 = case.upload(build_query(torch, 2, 0, 'normal', rows=ROWS8))
    m8 = case.upload(build_mask(torch, capacity, begin, G8_OFFSETS, rows=ROWS8))
    d2 = case.pages(2, distinct=True)
    if args.share_alternations:
        first, drift = {}, 0
        for index in range(2 * args.share_alternations):
            sentinel = (LEGACY, TAIL_SHARE)[index % 2]
            out = case.host(case.call(q8, m8, sentinel, pages=d2))
            if sentinel not in first:
                first[sentinel] = out
            elif differing(torch, first[sentinel], out):
                drift += 1
        distinguishable = differing(torch, first[LEGACY], first[TAIL_SHARE]) > 0
        report['alternation'].append(dict(capacity=capacity, modes='legacy/tail+share', calls=2 * args.share_alternations,
                                          drifted=drift, modes_distinguishable=distinguishable))
        print('alternation cap=%d legacy/tail+share calls=%d drifted=%d distinguishable=%s'
              % (capacity, 2 * args.share_alternations, drift, distinguishable), flush=True)
        if drift:
            failures.append("cap%d: %d alternating legacy/tail+share calls drifted" % (capacity, drift))
        if not distinguishable:
            failures.append('cap%d: legacy and tail+share agree on distinct page-table rows (control dead)' % capacity)
    if not args.trace_replays:
        return
    q3 = case.upload(build_query(torch, 3, 0, 'normal'))
    m3 = case.upload(build_mask(torch, capacity, begin, BUNDLES[0]))
    q1 = case.upload(build_query(torch, 1, 0, 'normal'))
    m1 = case.upload(build_mask(torch, capacity, begin, BUNDLES[1]))
    d3 = case.pages(3, distinct=True)
    ops = (('legacy G8', q8, m8, LEGACY, d2), ('tail+share G8', q8, m8, TAIL_SHARE, d2),
           ('legacy G4B3', q3, m3, LEGACY, d3), ('tail+share G4B3', q3, m3, TAIL_SHARE, d3),
           ('legacy G4B1', q1, m1, LEGACY, None), ('tail G4B1', q1, m1, TAIL, None))

    def launch():
        return [case.call(query, mask, sentinel, pages=pages) for _name, query, mask, sentinel, pages in ops]

    eager = [case.host(tensor) for tensor in launch()]          # every program compiled before capture
    distinguishable = [differing(torch, eager[0], eager[1]) > 0, differing(torch, eager[2], eager[3]) > 0]
    trace, outputs = capture(ttnn, device, launch)
    mismatches = []
    try:
        for replay in range(args.trace_replays):
            with WATCHDOG.op('trace replay %d cap=%d' % (replay, capacity)):
                ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            for (name, *_rest), output, expected in zip(ops, outputs, eager):
                with WATCHDOG.op('trace read back %s' % name):
                    got = ttnn.to_torch(output)
                if differing(torch, got, expected):
                    mismatches.append((replay, name))
    finally:
        ttnn.release_trace(device, trace)
        for output in outputs:
            ttnn.deallocate(output)
    report['trace'].append(dict(capacity=capacity, replays=args.trace_replays, ops=[op[0] for op in ops],
                                mismatches=mismatches[:20], mismatch_count=len(mismatches),
                                modes_distinguishable=distinguishable))
    print('trace cap=%d replays=%d mismatches=%d distinguishable=%s'
          % (capacity, args.trace_replays, len(mismatches), distinguishable), flush=True)
    if mismatches:
        failures.append('cap%d: %d trace replay outputs differ from eager (first %r)' % (capacity, len(mismatches), mismatches[0]))
    if not all(distinguishable):
        failures.append('cap%d: trace modes are not distinguishable (control dead): %r' % (capacity, distinguishable))


def timing(ttnn, torch, device, case, capacity, stage, args, report):
    begin = capacity - 256 + args.starts[0]
    runs = []
    for offsets in BUNDLES:
        runs.append(('G4B%d' % len(offsets), len(offsets), ROWS, offsets, (('legacy', LEGACY), ('tail', TAIL))
                     + ((('tail_share', TAIL_SHARE),) if stage == 3 and len(offsets) > 1 else ())))
    if stage == 3:
        runs.append(('G8', 2, ROWS8, G8_OFFSETS, (('legacy', LEGACY), ('tail', TAIL), ('tail_share', TAIL_SHARE))))
    medians = {}
    for name, batches, rows, offsets, modes in runs:
        query = case.upload(build_query(torch, batches, 0, 'normal', rows=rows))
        mask = case.upload(build_mask(torch, capacity, begin, offsets, rows=rows))
        row = dict(capacity=capacity, bundle=name, batches=batches)
        for mode, sentinel in modes:
            row[mode] = timed(ttnn, device, lambda s=sentinel: case.call(query, mask, s), args.warmup, args.iters)
            medians[(name, mode)] = row[mode]['median_us']
        row['tail_over_legacy'] = row['tail']['median_us'] / row['legacy']['median_us']
        report['timing'].append(row)
        print('timing cap=%d %s %s' % (capacity, name, ' '.join('%s %.1f us' % (mode, row[mode]['median_us'])
                                                                for mode, _s in modes)), flush=True)
    legacy_layer = medians[('G4B3', 'legacy')] + medians[('G4B1', 'legacy')]
    tail_layer = medians[('G4B3', 'tail')] + medians[('G4B1', 'tail')]
    summary = dict(legacy=legacy_layer, tail=tail_layer, ratio=tail_layer / legacy_layer,
                   b1_tail_over_legacy=medians[('G4B1', 'tail')] / medians[('G4B1', 'legacy')])
    if stage == 3:
        # Spec section 8's acceptance: G8+tail B2 at 131k <= 0.6 x legacy (B3+B1); share+tail B3
        # <= 1.6 x tail B1 (the upper end of f). Recorded, not failed on: card M's CI load skews.
        summary.update(
            g8_tail=medians[('G8', 'tail')], g8_tail_share=medians[('G8', 'tail_share')],
            g8_tail_over_g4_legacy_layer=medians[('G8', 'tail')] / legacy_layer,
            g8_tail_share_over_g4_legacy_layer=medians[('G8', 'tail_share')] / legacy_layer,
            g8_tail_share_over_g8_tail=medians[('G8', 'tail_share')] / medians[('G8', 'tail')],
            b3_tail_share_over_b1_tail=medians[('G4B3', 'tail_share')] / medians[('G4B1', 'tail')])
        summary['accept_g8_tail'] = summary['g8_tail_over_g4_legacy_layer'] <= 0.6
        summary['accept_share_f'] = summary['b3_tail_share_over_b1_tail'] <= 1.6
    report['user_layer_us'][capacity] = summary
    print('timing cap=%d per user-layer %s' % (capacity, json.dumps(summary)), flush=True)


def run(args, report):
    import torch
    import ttnn

    failures = report['failures']
    options = dict(device_id=args.device_id, l1_small_size=24576)
    if args.trace_replays and not args.legacy_only:
        options['trace_region_size'] = args.trace_region_bytes
    device = ttnn.open_device(**options)
    try:
        path, markers = loaded_binary()
        stage = binary_stage(markers)
        report['binary'] = dict(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                qwen_branch=markers['flags'], markers=markers, stage=stage)
        print('binary %s sha256 %s stage=%d markers=%s' % (path, report['binary']['sha256'][:16], stage, markers),
              flush=True)
        if not args.legacy_only and stage == 0:
            failures.append('the loaded _ttnncpp.so lacks the [QWEN-SDPA] branch: the graft is not mounted')
            return
        if args.require_stage is not None and not args.legacy_only and stage != args.require_stage:
            failures.append('the loaded binary is stage %d, --require-stage %d' % (stage, args.require_stage))
            return
        share_section = not args.no_share and (args.legacy_only or stage == 3)
        report['share_section'] = share_section
        report['tree_scratch_rounds_env'] = os.environ.get(SCRATCH_ENV)
        if share_section and os.environ.get(SCRATCH_ENV) != '1':
            # Legacy G8 (PNHt=3) with the full tree scratch needs ~1.83 MB of L1 (spec section 1);
            # the arm sets this for every run (lever_n_m3native_run_arm.sh), so the reference does too.
            failures.append('%s=1 is required for the G8 legacy calls (the arm sets it; run_card_m.sh does)' % SCRATCH_ENV)
            return
        requested = set()
        report['_requested'] = requested
        for capacity in args.capacities:
            for seed in args.seeds:
                case = Case(ttnn, torch, device, capacity, seed, requested=requested)
                try:
                    if not args.skip_stage1:
                        for offsets in BUNDLES:
                            for variant in args.variants:
                                for start in args.starts:
                                    equality_case(ttnn, torch, case, capacity, seed, offsets, variant, start, args, report)
                    if share_section:
                        for variant in args.variants:
                            for start in args.starts:
                                share_case(ttnn, torch, case, capacity, seed, variant, start, args, report)
                    if not args.legacy_only and seed == args.seeds[0]:
                        controls(ttnn, torch, device, case, capacity, stage, args, report)
                        if share_section:
                            share_controls(ttnn, torch, device, case, capacity, args, report)
                        if not args.no_timing:
                            timing(ttnn, torch, device, case, capacity, stage, args, report)
                finally:
                    case.close()
    finally:
        with WATCHDOG.op('close device'):
            ttnn.close_device(device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--capacities', default=','.join(map(str, CAPACITIES)))
    parser.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--starts', default=','.join(map(str, STARTS)))
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--alternations', type=int, default=50, help='legacy/tail pairs in the stage-1 cache check')
    parser.add_argument('--share-alternations', type=int, default=500,
                        help='legacy/tail+share pairs in N4 (500 pairs = the spec\'s 1,000 calls)')
    parser.add_argument('--trace-replays', type=int, default=200, help='N5 replays per capacity (0: no trace)')
    parser.add_argument('--trace-region-bytes', type=int, default=16 << 20)
    parser.add_argument('--watchdog', type=float, default=0, help='seconds per device call before os._exit(3); 0 off')
    parser.add_argument('--no-timing', action='store_true', help='skip the timing (e.g. under the watcher)')
    parser.add_argument('--no-share', action='store_true', help='skip the stage-3 section even on a stage-3 binary')
    parser.add_argument('--skip-stage1', action='store_true', help='skip the stage-1 G4 equality sweep')
    parser.add_argument('--require-stage', type=int, choices=(1, 3), help='fail unless the loaded binary is this stage')
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
    if any(capacity < 768 for capacity in args.capacities):
        parser.error('capacities below 768 leave no room for the tail, a planted chunk 0 and a refusable 512-wide mask')
    args.reference_shas = json.loads(Path(args.reference).read_text())['legacy_sha256'] if args.reference else None
    return args


def main(argv=None):
    global WATCHDOG
    args = parse_args(argv)
    report = dict(passed=False, capacities=args.capacities, seeds=args.seeds, variants=args.variants,
                  starts=args.starts, legacy_only=args.legacy_only, reference=args.reference, watchdog=args.watchdog,
                  failures=[], cases=[], share_cases=[], e4=[], timing=[], alternation=[], trace=[],
                  user_layer_us={}, refusals={}, legacy_sha256={})
    native = NativeLog(args.out.with_name(args.out.name + '.native.log'))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write_report(extra=None):
        payload = {key: value for key, value in report.items() if not key.startswith('_')}
        payload['requested_programs'] = sorted(describe(key) for key in report.get('_requested', ()))
        if extra:
            payload.update(extra)
        args.out.write_text(json.dumps(payload, indent=2, default=str))

    def on_fire(label):
        try:
            write_report(dict(error='watchdog: %r exceeded %ss' % (label, args.watchdog), passed=False))
        except Exception:  # noqa: BLE001 - the main thread may be mid-update; the WATCHDOG line stands
            pass

    WATCHDOG = Watchdog(args.watchdog, on_fire=on_fire).start()
    try:
        with native:
            run(args, report)
        if not args.legacy_only and report.get('binary', {}).get('qwen_branch'):
            report['factory_lines'] = factory_lines(native.text())
            report['factory_line_count'] = len(report['factory_lines'])
            counts = {}
            for line in report['factory_lines']:
                counts[describe(line_key(line))] = counts.get(describe(line_key(line)), 0) + 1
            report['factory_line_counts'] = counts
            for problem in check_program_lines(report['factory_lines'], report.get('_requested', set())):
                report['failures'].append('factory log: ' + problem)
        report['passed'] = not report['failures'] and bool(report['cases'] or report['share_cases'])
    except Exception as error:  # noqa: BLE001
        report['error'] = '%s: %s' % (type(error).__name__, error)
    finally:
        write_report()
    for failure in report['failures']:
        print('FAIL', failure)
    if report.get('error'):
        print('ERROR', report['error'])
    e4 = [entry['differing'] for entry in report['e4']]
    print('SDPA_QWEN_CARD_M passed=%s stage=%s cases=%d share_cases=%d e4_exact=%s failures=%d report=%s native_log=%s' % (
        report['passed'], report.get('binary', {}).get('stage'), len(report['cases']), len(report['share_cases']),
        all(not any(value.values()) for value in e4) if e4 else None, len(report['failures']), args.out, native.path))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
