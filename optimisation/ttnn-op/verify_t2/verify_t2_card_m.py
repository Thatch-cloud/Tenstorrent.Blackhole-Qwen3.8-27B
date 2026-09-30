#!/usr/bin/env python3
"""Card-M unit test and microbenches for verify-trace T2 (verify_trace_t2.py): cut #1, the packed
GDN conv windows (gdn_conv_windows_packed), and cut #2, the per-user chained ordered K/V write
(packed_ordered_cache).

WHAT IT PROVES. On one p150a, in the serving image the T1 arms ran (A5), each new launch leaves the
SAME RAW PAGES as the served launches it replaces - every page of every output (or of the whole
K/V cache), padding rows, -0, NaN payloads and bf8 exponent sections included:

  windows   R1  the image's gdn_conv_windows.cpp re-driven on this chip with build_windows' exact
                descriptor (8x6, 48 workers, addresses + [rows, worker], CB 6 x 2048), once per user;
            R2  gdn_conv_windows_packed.reference_windows, the served loop transcribed in torch on
                the raw input pages;
            the candidate at every setting (VTW_PORT, hist_row x copy_noc x nbuf).
  kv        R1  per 32-row tile, the served slice and ordered_cache.update's descriptor re-driven on
                this chip (hash-verified kernels, the served args, 32-row chains);
            R2  for BF8-exact payloads, the host prediction (written rows exact, every other row
                its initial content);
            the candidate: update_chained at launch rows 64 and 32, CB16 at 16, 256 and 512 pages,
            one chain per user - or, for g4 (the served placeholders the warm forward writes:
            every user on page 0 at the capture position), ONE chain over the block.

The served drivers hard-require two chips (build_windows: 'Both chips required'; update: 'Two
chips required') and card M is one chip, so R1 re-drives the served kernel with the served
descriptor for range(1) chips; test_verify_t2_card_m pins both re-drives against the served
drivers' own descriptors on CPU.

  sections  selftest          raw_pages round-trips known bytes, and a bf16 tile tensor's raw
                              pages untile to its to_torch rows
            windows           the equality matrix (users 4/1/2/3, widths 8240/8256, pieces direct
                              or cut from a 64-row block as the served path cuts them, histories
                              DRAM or L1, seven data kinds each side, padding poisoned); inputs
                              byte-unchanged; slice_canonicalises recorded (users 1 and 3); rows
                              8 / 32 and a mixed history placement refused before any device work
            windows_negative  VTW_NEG_SLOT / _USER / _HIST / _PAD must each DIFFER from R1
            windows_trace     TWO launches over independent input sets captured in one trace (one
                              cached program, two common-arg sets - the model captures 48),
                              replayed after restaging new contents in place, each launch against
                              R1: 3 checked replays and a 20-replay soak
            windows_cache     three calls, every earlier call's inputs and outputs still held, so
                              each cache hit MUST take new addresses (checked): exact, program-
                              cache deltas [n, 0, 0]
            kv                the cache matrix (page widths, geometries g1-g4, payloads, seeds)
            kv_negative       nochain / conflict must differ in 1 of 5 repeats; index every time
            kv_trace          the wide-2052-trace pattern: restaged metadata (two tables
                              alternating) and inputs, each replay against R1 (both caches start
                              from the same pristine pages)
            kv_cache          program-cache deltas equal R1's pattern; and on a CB16 no launch
                              used before, the warm forward's one chain compiles and the
                              capture's per-user chains are then a program-cache HIT (a capture
                              runs inside a trace, where nothing may compile)
            timing            trace-timed launches: R1 x 4 against the packed windows at every
                              setting (gates: VTW_PORT <= 115 us, DEFAULTS <= 60 us, DEFAULTS within
                              2 us of the fastest exact setting); R1 2 x (slice + update) against
                              the chained launch (gate <= 60 us) and the 32-row mode

RUN on the qualification card only (run_card_m.sh: QUAL_CARD, default card B; never the serving
pair without ALLOW_SERVING_CARD=1), inside the image, with a fresh kernel cache. The
helpers above `Device part` import no ttnn and are unit-tested on CPU (test_verify_t2_card_m.py),
and every section is driven there against a torch-only fake bench. Every device call - launches,
uploads, readbacks, restaging - runs under the per-call watchdog.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

CHANNELS = 5120
PAGES = 160
ROWS = 16
WIDTHS = (8240, 8256)
USERS = (4, 1, 2, 3)
DATA = ('randn', 'small', 'large', 'zeros', 'denormal', 'specials', 'nearmax')
SEEDS = (0, 1, 2)
NEGATIVES = ('slot', 'user', 'hist', 'pad')
KV_NEGATIVES = ('nochain', 'conflict', 'index')
KV_WIDTHS = (516, 1024, 2052)        # the 4 x 32k arm's page width (33024 / 64), a control, 131k
KV_GEOMETRIES = ('g1', 'g2', 'g3', 'g4')
KV_PAYLOADS = ('bf8', 'randn', 'zeros', 'denormal', 'nearmax')
KV_OFFSETS = (0, 5, 17, 31, 48, 60)
KV_ROWS_PER_USER = 16
KV_USERS = 4
CB16_VARIANTS = (16, 256, 512)
# kv_cache's warm-then-capture check: a CB16 no other launch uses, so the warm forward's one
# chain is the first program of its kind and the per-user chains after it must be a cache hit.
CB16_FRESH = 128
POISON = 0x7FA5             # a NaN payload no data kind produces: any read of a padding row shows
GATES_US = dict(port=115.0, defaults=60.0, chained=60.0, settings_margin=2.0)
SECTIONS = ('selftest', 'windows', 'windows_negative', 'windows_trace', 'windows_cache', 'kv', 'kv_negative',
            'kv_trace', 'kv_cache', 'timing')
# The image files the references are, and every other image module the code under test imports,
# compared with the checkout (run_card_m.sh passes the checkout's sha256 of each as --expect
# name=sha): 77d6995a's gdn_conv_windows.* and gdn_multitoken_conv.py (validate_projected), the
# frozen ordered-cache writer evidence, gdn_user_batch.py (MAX_USERS), gdn_device_loop_state.py
# (resident_piece, the served cut) and verify_trace_t1.py (rectangle_set, update_chained's core
# set) - so the card runs exactly what an image built from this checkout ships.
REFERENCE_FILES = ('gdn_conv_windows.py', 'gdn_conv_windows.cpp', 'ordered_cache.py', 'packed_cache_writer.py',
                   'gdn_multitoken_conv.py', 'gdn_user_batch.py', 'gdn_device_loop_state.py', 'verify_trace_t1.py')


# ---------------------------------------------------------------------------------------------
# Pure helpers (CPU-tested).
# ---------------------------------------------------------------------------------------------

def windows_matrix(seeds=SEEDS, quick=False):
    """The windows equality cases: dict(users, width, source, placement, piece, history, seed)."""
    def case(users=4, width=8240, source='served', placement='dram', piece='randn', history='randn', seed=0):
        return dict(users=users, width=width, source=source, placement=placement, piece=piece, history=history,
                    seed=seed)

    cases = [case(seed=seed) for seed in (seeds if not quick else seeds[:1])]
    cases += [case(users=users) for users in USERS if users != 4]
    cases += [case(width=8256), case(source='direct'), case(placement='l1'), case(source='direct', placement='l1')]
    for kind in DATA[1:]:
        cases += [case(piece=kind), case(history=kind)]
        if not quick:
            cases += [case(piece=kind, history=kind, source='direct', seed=seeds[-1])]
    out = []
    for item in cases:
        if item not in out:
            out.append(item)
    if quick:
        out = [item for item in out if item['piece'] in ('randn', 'zeros', 'denormal', 'specials')
               and item['history'] in ('randn', 'specials')]
    return out


def windows_case_name(case):
    return 'u%(users)d_w%(width)d_%(source)s_%(placement)s_p%(piece)s_h%(history)s_s%(seed)d' % case


def refusal_cases():
    """Inputs the op must refuse before any device work (the program cache must not grow)."""
    return [dict(name='rows8', rows=8), dict(name='rows32', rows=32), dict(name='mixed_history', rows=16, mixed=True)]


def make_rows(torch, kind, rows, cols, seed):
    """(rows, cols) bf16 of one data kind."""
    generator = torch.Generator().manual_seed(4000 + seed)
    base = torch.randn(rows, cols, generator=generator)
    if kind == 'randn':
        return base.to(torch.bfloat16)
    if kind == 'small':
        return (base * 1e-3).to(torch.bfloat16)
    if kind == 'large':
        return (base * 30).to(torch.bfloat16)
    if kind == 'zeros':
        out = base.to(torch.bfloat16)
        mask = torch.rand(rows, cols, generator=generator)
        out[mask < 0.25] = 0.0
        out[(mask >= 0.25) & (mask < 0.5)] = -0.0
        return out
    if kind == 'denormal':
        out = base.to(torch.bfloat16)
        mask = torch.rand(rows, cols, generator=generator) < 0.3
        out[mask] = (base * 1e-39).to(torch.bfloat16)[mask]
        return out
    if kind == 'specials':
        bits = base.to(torch.bfloat16).view(torch.int16).clone()
        choice = torch.randint(0, 12, (rows, cols), generator=generator)
        for index, value in enumerate((0x7FC1, 0x7F81, 0xFFC3, 0x7F80, 0xFF80, 0x8000)):
            bits[choice == index] = value - 0x10000 if value >= 0x8000 else value
        return bits.view(torch.bfloat16)
    if kind == 'nearmax':
        return (base.sign() * 3.0e38 * (0.5 + base.abs().clamp(max=1) / 2)).to(torch.bfloat16)
    raise ValueError(kind)


def tile_pages(torch, matrix, poison=None):
    """A (rows <= 32 * k, cols) bf16 matrix as the raw int16 pages of its TILE tensor: tiles in
    row-major tile order, each 1024 values face-ordered (faces 0,1 top, 2,3 bottom, 16x16
    row-major). Padding rows (past `rows`) and columns hold `poison` (default 0)."""
    rows, cols = matrix.shape
    tile_rows, tile_cols = -(-rows // 32), -(-cols // 32)
    padded = torch.full((tile_rows * 32, tile_cols * 32), 0 if poison is None else poison, dtype=torch.int16)
    padded[:rows, :cols] = matrix.contiguous().view(torch.int16)
    tiles = padded.reshape(tile_rows, 32, tile_cols, 32).permute(0, 2, 1, 3)
    faces = tiles.reshape(tile_rows, tile_cols, 2, 16, 2, 16).permute(0, 1, 2, 4, 3, 5)
    return faces.reshape(tile_rows * tile_cols, 1024).contiguous()


def untile_pages(torch, pages, rows, cols):
    """The inverse of tile_pages: the logical (rows, cols) int16 values of raw tile pages."""
    tile_rows, tile_cols = -(-rows // 32), -(-cols // 32)
    faces = pages.reshape(tile_rows, tile_cols, 2, 2, 16, 16).permute(0, 1, 2, 4, 3, 5)
    padded = faces.reshape(tile_rows, tile_cols, 32, 32).permute(0, 2, 1, 3).reshape(tile_rows * 32, tile_cols * 32)
    return padded[:rows, :cols].contiguous()


def special_counts(torch, values):
    """-0 and bf16 subnormals in int16 bits."""
    word = values.to(torch.int32) & 0xFFFF
    return dict(minus_zero=int((word == 0x8000).sum()),
                denormal=int((((word & 0x7F80) == 0) & ((word & 0x7F) != 0)).sum()))


def compare_pages(torch, candidate, reference):
    """Raw-page comparison of two int16 page stacks: exact, the differing values and pages."""
    if tuple(candidate.shape) != tuple(reference.shape):
        return dict(exact=False, shape=[list(candidate.shape), list(reference.shape)])
    differ = candidate != reference
    count = int(differ.sum())
    result = dict(exact=count == 0, differing=count)
    if count:
        pages = differ.reshape(differ.shape[0], -1).any(dim=1).nonzero().flatten().tolist()
        result['pages'] = pages[:8]
        result['differing_pages'] = len(pages)
    return result


def settings_verdict(exact_by_setting, timings_us, defaults_name, margin_us=GATES_US['settings_margin']):
    """Whether the module DEFAULTS may stay: every setting must be exact on every case, and
    DEFAULTS must be exact and within `margin_us` of the fastest exact setting. timings_us: name ->
    per-launch us (None when untimed)."""
    inexact = sorted(name for name, exact in exact_by_setting.items() if not exact)
    result = dict(inexact=inexact, defaults=defaults_name)
    timed = {name: value for name, value in (timings_us or {}).items() if value is not None and exact_by_setting.get(name)}
    if inexact:
        result['verdict'] = 'inexact'
    elif not timed:
        result['verdict'] = 'untimed'
    else:
        fastest = min(timed, key=timed.get)
        result.update(fastest=fastest, fastest_us=timed[fastest], defaults_us=timed.get(defaults_name))
        within = defaults_name in timed and timed[defaults_name] <= timed[fastest] + margin_us
        result['verdict'] = 'keep' if within else 'change'
    return result


def kv_pages_total(width):
    return width + 12


def kv_geometry(name, width, seed):
    """Per user: (start position, page-table row) for the kv cases.

    g1  served: 4 users, distinct random page rows, starts at offset s_u % 64 in KV_OFFSETS (rows
        span one or two tile rows; some cross into the next virtual block, another physical page);
    g2  the table tail: positions >= 131072 through entries 2048-2051 (width 2052 only);
    g3  users 0 and 1 on one physical page at DIFFERENT tile rows (the guard passes);
    g4  the served placeholders the warm forward writes: every user on page 0 at the capture
        position - written as ONE chain over the block (kv_case_spans), the served order.
    """
    from ordered_cache_hw_plan import page_table

    pages_total = kv_pages_total(width)
    table = page_table(20000 + seed * 17 + width, KV_USERS, width, pages_total)
    capacity = width * 64
    if name == 'g1':
        starts = []
        for user in range(KV_USERS):
            block = (seed * 131 + user * 977 + 3) % max(1, (capacity - 128) // 64)
            starts.append(block * 64 + KV_OFFSETS[(seed + user * 2) % len(KV_OFFSETS)])
        return deconflict([(start, row) for start, row in zip(starts, table)], pages_total)
    if name == 'g2':
        if width != 2052:
            return None
        starts = [131072 + 5, 131072 + 64 + 48, 131072 + 128 + 17, 131072 + 192 + 40]
        return deconflict([(start, row) for start, row in zip(starts, table)], pages_total)
    if name == 'g3':
        rows = [list(row) for row in table]
        start0, start1 = 64 * 10 + 0, 64 * 20 + 32
        rows[1][start1 // 64] = rows[0][start0 // 64]           # same physical page, tile rows 0 and 1
        others = {rows[0][start0 // 64]}
        for user in (2, 3):
            rows[user] = [value for value in rows[user]]
        starts = [start0, start1, 64 * 30 + 5, 64 * 40 + 17]
        for user in (2, 3):
            if rows[user][starts[user] // 64] in others:
                rows[user][starts[user] // 64] = pages_total - 1 - user
        return deconflict([(start, row) for start, row in zip(starts, rows)], pages_total)
    if name == 'g4':
        capture = capacity - 256
        return [(capture, [0] * width) for user in range(KV_USERS)]
    raise ValueError(name)


def touched_blocks(start):
    return range(start // 64, (start + KV_ROWS_PER_USER - 1) // 64 + 1)


def deconflict(geometry, pages_total):
    """The page-table rows are partial permutations, so two users' random rows can name one
    physical page at one tile row by chance: remap the later user's entry to a page nobody's
    rows touch, until verify_trace_t2.kv_conflict passes. Deliberate same-page geometries (g3:
    different tile rows) are not conflicts and stay as built."""
    import verify_trace_t2

    geometry = [(start, list(table)) for start, table in geometry]
    used = {table[block] for start, table in geometry for block in touched_blocks(start)}
    spare = [page for page in range(pages_total - 1, -1, -1) if page not in used]
    while True:
        conflict = verify_trace_t2.kv_conflict([(range(start, start + KV_ROWS_PER_USER), table)
                                                for start, table in geometry])
        if conflict is None:
            return geometry
        start, table = geometry[conflict['users'][1]]
        for block in touched_blocks(start):
            if table[block] == conflict['page']:
                table[block] = spare.pop(0)


def kv_conflict_geometry(width, seed):
    """Two users forced onto one tile row: user 2 writes user 0's first tile row."""
    geometry = [list(pair) for pair in kv_geometry('g1', width, seed)]
    start0, table0 = geometry[0]
    start2 = start0 - (start0 % 32) + 3 if start0 % 32 < 13 else start0 - (start0 % 32)
    table2 = list(geometry[2][1])
    table2[start2 // 64] = table0[start0 // 64]
    geometry[2] = [start2, table2]
    return [(start, table) for start, table in geometry]


def kv_block_host(geometry):
    """(positions (64,), pages (64, width)) host lists, segment by segment."""
    positions, pages = [], []
    for start, table in geometry:
        positions += list(range(start, start + KV_ROWS_PER_USER))
        pages += [list(table)] * KV_ROWS_PER_USER
    return positions, pages


def kv_matrix(widths=KV_WIDTHS, seeds=SEEDS, quick=False):
    cases = []
    for width in widths:
        for geometry in KV_GEOMETRIES:
            if geometry == 'g2' and width != 2052:
                continue
            for payload in KV_PAYLOADS:
                if quick and payload not in ('bf8', 'zeros'):
                    continue
                for seed in (seeds if payload == 'bf8' and not quick else seeds[:1]):
                    cases.append(dict(width=width, geometry=geometry, payload=payload, seed=seed))
    return cases


def kv_case_name(case):
    return 'w%(width)d_%(geometry)s_%(payload)s_s%(seed)d' % case


def kv_variants(quick=False):
    """(launch_rows, cb16_pages) the candidate runs per case; the CB16 sweep on 64 rows."""
    if quick:
        return [(64, 256), (32, 256)]
    return [(64, 256), (32, 256), (64, 16), (64, 512)]


def kv_payload(torch, kind, rows, seed):
    """(rows, 32, 256) bf16 K/V block rows: heads 0-1 are what the cache takes, heads 2-31 a poison
    pattern a writer that copied a padding head would leave in the cache."""
    from ordered_cache_hw_plan import payload

    if kind == 'bf8':
        values = torch.stack([payload(1000 * seed + row + 1) for row in range(rows)])
    else:
        values = make_rows(torch, 'randn' if kind == 'bf8' else kind, rows * 32, 256, seed).reshape(rows, 32, 256)
    values = values.clone()
    values[:, 2:, :] = (torch.full((rows, 30, 256), POISON, dtype=torch.int16)).view(torch.bfloat16)
    return values


def kv_initial_values(torch, width, seed, payload):
    """The whole cache's initial (pages, 2, 64, 256) bf16 values: 128 distinct BF8-exact rows
    (ordered_cache_hw_plan.payload heads 0-1, one shared exponent per 16 values) cycled, so every
    row is nonzero and survives the bf8 upload exactly."""
    rows = torch.stack([payload(900000 + seed * 7919 + index) for index in range(64)])[:, :2, :].reshape(128, 256)
    pages_total = kv_pages_total(width)
    return rows[torch.arange(pages_total * 2 * 64) % 128].reshape(pages_total, 2, 64, 256)


def predict_cache(torch, initial, geometry, values):
    """The cache the kernel must leave behind, from the initial (pages, 2, 64, 256) values and the
    host metadata alone: row r's heads 0-1 at [table[pos // 64], :, pos % 64]. Exact only for
    BF8-exact payloads (bf8 re-rounding of a non-exact row is the kernel's, not the host's)."""
    cache = initial.clone()
    positions, pages = kv_block_host(geometry)
    for row, (position, table) in enumerate(zip(positions, pages)):
        cache[table[position // 64], :, position % 64, :] = values[row, :2, :]
    return cache


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance_failures(image_shas, expected):
    """Every reference file whose image sha differs from the checkout's, could not be read, or
    has no checkout sha to compare with (run_card_m.sh passes all four)."""
    failures = ['no checkout sha256 for %s (--expect)' % name for name in REFERENCE_FILES if name not in expected]
    for name, sha in sorted(expected.items()):
        found = image_shas.get(name)
        if found is None:
            failures.append('the image has no %s' % name)
        elif found != sha:
            failures.append('the image %s (%s) is not the checkout\'s (%s)' % (name, found[:12], sha[:12]))
    return failures


def summarise_timing(samples):
    return dict(median=statistics.median(samples), min=min(samples), samples=len(samples))


def verdict(report):
    """Pass iff no failure was recorded and at least one section ran to the end."""
    return not report['failures'] and bool(report.get('sections_completed'))


class Watchdog:
    """A per-device-call deadline: a hung NoC handshake or semaphore chain cannot be interrupted from
    Python, so the poller prints WATCHDOG, writes the partial report and os._exit(3)s."""

    def __init__(self, seconds, on_fire=None):
        self.seconds, self.on_fire = seconds, on_fire
        self.label, self.deadline = None, None
        self.lock = threading.Lock()

    def start(self):
        if self.seconds:
            threading.Thread(target=self.poll, name='vt2-watchdog', daemon=True).start()
        return self

    @contextmanager
    def op(self, label):
        if not self.seconds:
            yield
            return
        with self.lock:
            outer = (self.label, self.deadline)
            self.label, self.deadline = label, time.monotonic() + self.seconds
        try:
            yield
        finally:
            with self.lock:
                self.label, self.deadline = outer

    def poll(self):
        while True:
            time.sleep(1.0)
            with self.lock:
                label, deadline = self.label, self.deadline
            if label is not None and time.monotonic() >= deadline:
                sys.stdout.write('WATCHDOG: %r did not return within %ss; exiting 3 (docker rm -f, then '
                                 'reset this card only, by the runner\'s printed reset command)\n' % (label, self.seconds))
                sys.stdout.flush()
                try:
                    if self.on_fire is not None:
                        self.on_fire(label)
                finally:
                    os._exit(3)


WATCHDOG = Watchdog(0)


# ---------------------------------------------------------------------------------------------
# The served drivers, re-driven for range(chips). CPU-pinned against the drivers themselves
# (test_verify_t2_card_m.RedriveTests): same kernels, compile args, runtime args, CBs, cores.
# ---------------------------------------------------------------------------------------------

def served_windows_redrive(ttnn, mesh, projected, history, chips, kernel_path):
    """gdn_conv_windows.build_windows with `range(chips)` for its two chips."""
    from gdn_multitoken_conv import validate_projected

    rows = validate_projected(tuple(projected.shape), history)
    inputs = [projected, *history]
    if any(value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT or
           value.memory_config() not in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG) for value in inputs):
        raise ValueError('Interleaved BF16 tiled inputs required')
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 6:
        raise ValueError('Window DMA requires the audited 8x6 worker grid')
    outputs = []
    try:
        for slot in range(4):
            outputs.append(ttnn.empty((1, rows, 5120), device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        tensors = inputs + outputs
        shards = [ttnn.get_device_tensors(value) for value in tensors]
        if any(len(parts) != chips for parts in shards):
            raise ValueError('One shard per re-driven chip required')
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))])
        buffer = ttnn.CBDescriptor(total_size=6 * 2048, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
                page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
        program = ttnn.MeshProgramDescriptor()
        for chip in range(chips):
            local = [parts[chip] for parts in shards]
            addresses = [value.buffer_address() for value in local]
            if len(set(addresses)) != len(addresses):
                raise ValueError('Immutable input and mutable windows must not alias')
            descriptor = ttnn.KernelDescriptor(kernel_source=str(kernel_path),
                core_ranges=cores,
                compile_time_args=[argument for value in local for argument in ttnn.TensorAccessorArgs(value).get_compile_time_args()],
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                         noc=ttnn.NOC.RISCV_0_default))
            runtime = ttnn.RuntimeArgs()
            for worker in range(48):
                runtime[worker % 8][worker // 8] = addresses + [rows, worker]
            descriptor.runtime_args = runtime
            coordinate = ttnn.MeshCoordinate(0, chip)
            program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[descriptor], cbs=[buffer])
        ttnn.generic_op(tensors, program)
        return outputs
    except BaseException:
        for value in outputs:
            ttnn.deallocate(value)
        raise


def served_kv_redrive(ttnn, mesh, cache, packed, positions, pages, kernels, chips):
    """ordered_cache.update with `range(chips)` for its two chips."""
    from ordered_cache import validate_shapes

    rows = validate_shapes(tuple(cache.shape), tuple(packed.shape), tuple(positions.shape), tuple(pages.shape))
    tensors = [cache, packed, positions, pages]
    if any(value.memory_config() != ttnn.DRAM_MEMORY_CONFIG for value in tensors):
        raise ValueError('Interleaved DRAM buffers required')
    if cache.dtype != ttnn.bfloat8_b or packed.dtype != ttnn.bfloat16 or any(
            value.layout != ttnn.TILE_LAYOUT for value in tensors[:2]):
        raise ValueError('Native BF8 cache and BF16 input tiles required')
    if any(value.dtype != ttnn.int32 or value.layout != ttnn.ROW_MAJOR_LAYOUT for value in tensors[2:]):
        raise ValueError('Int32 row-major metadata required')
    shards = [ttnn.get_device_tensors(value) for value in tensors]
    if any(len(parts) != chips for parts in shards):
        raise ValueError('One shard per re-driven chip required')
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 2:
        raise ValueError('Audited 8x2 worker subset required')
    coordinates = [ttnn.CoreCoord(index % 8, index // 8) for index in range(rows)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in coordinates])
    buffers = []

    def buffer(indices, count, dtype, page, tiled=True):
        formats = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page,
            **(dict(tile=ttnn.TileDescriptor(ttnn.Tile([32, 32]))) if tiled else {})) for index in indices]
        buffers.append(ttnn.CBDescriptor(total_size=count * page, core_ranges=cores, format_descriptors=formats))

    buffer([0], 16, ttnn.bfloat8_b, 1088)
    buffer([1], 8, ttnn.bfloat16, 2048)
    buffer([24, 25], 16, ttnn.bfloat16, 2048)
    buffer([26], 16, ttnn.bfloat16, 2048)
    buffer([16], rows * 8, ttnn.bfloat8_b, 1088)
    buffer([2], 1, ttnn.int32, 4096, False)
    page_bytes = pages.padded_shape[-1] * 4
    buffer([3], 1, ttnn.int32, page_bytes, False)
    semaphore = ttnn.SemaphoreDescriptor(id=0, core_ranges=cores, initial_value=0)
    program = ttnn.MeshProgramDescriptor()
    for chip in range(chips):
        local = [parts[chip] for parts in shards]
        addresses = [value.buffer_address() for value in local]
        if len(set(addresses)) != 4:
            raise ValueError('Cache, packed input and metadata must not alias')
        reader_args = [0, 1, 1, 2, 0, 8, 0, rows * 4, 1, 2, 64, 2, pages.padded_shape[-1], 0, page_bytes, 3, 2, 0, 0]
        for index in (0, 2, 3, 1):
            reader_args.extend(ttnn.TensorAccessorArgs(local[index]).get_compile_time_args())
        writer_args = [16, 24, 25, 26, 1, 2, 0, 8, 512, 1, 2, 64, 2, pages.padded_shape[-1], 3, 2, 0, 0]
        writer_args.extend(ttnn.TensorAccessorArgs(local[0]).get_compile_time_args())
        descriptors = []
        for role, args, config in (
            ('reader', reader_args, ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1,
                                                                     noc=ttnn.NOC.RISCV_1_default)),
            ('writer', writer_args, ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                     noc=ttnn.NOC.RISCV_0_default)),
            ('compute', [0, 1, 24, 25, 26, 16, 8, 2], ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False)),
        ):
            runtime = ttnn.RuntimeArgs()
            for index, core in enumerate(coordinates):
                next_core = local[0].device().worker_core_from_logical_core(coordinates[min(index + 1, rows - 1)])
                runtime[core.x][core.y] = ([addresses[0], 0, addresses[2], index, addresses[3], int(index > 0), addresses[1]]
                    if role == 'reader' else [addresses[0], 0, 0, index, int(index < rows - 1), next_core.x, next_core.y]
                    if role == 'writer' else [])
            descriptor = ttnn.KernelDescriptor(kernel_source=kernels[role],
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                compile_time_args=args, config=config)
            descriptor.runtime_args = runtime
            descriptors.append(descriptor)
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(
            kernels=descriptors, cbs=buffers, semaphores=[semaphore])
    ttnn.generic_op(tensors, program)


def copy_pages_program(ttnn, mesh, source, destination, page_bytes, pages, kernel_path, chips):
    """The raw_pages.cpp launch: page i of `source` to page i of `destination`, i < pages, spread
    over every core. Only for page sizes that are DRAM-aligned (2048, 1088), so page i of either
    buffer is page i whatever its layout."""
    if page_bytes % 64:
        raise ValueError('raw page copies need 64-byte multiples; %d is not' % page_bytes)
    grid = mesh.compute_with_storage_grid_size()
    cores = grid.x * grid.y
    source_shards, destination_shards = ttnn.get_device_tensors(source), ttnn.get_device_tensors(destination)
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    buffer = ttnn.CBDescriptor(total_size=page_bytes, core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.int32, page_size=page_bytes)])
    program = ttnn.MeshProgramDescriptor()
    for chip in range(chips):
        source_local, destination_local = source_shards[chip], destination_shards[chip]
        base, extra = divmod(pages, cores)
        runtime = ttnn.RuntimeArgs()
        start = 0
        for worker in range(cores):
            count = base + (1 if worker < extra else 0)
            runtime[worker % grid.x][worker // grid.x] = [start, count]
            start += count
        descriptor = ttnn.KernelDescriptor(kernel_source=str(kernel_path), core_ranges=core_set,
            compile_time_args=[page_bytes] + list(ttnn.TensorAccessorArgs(source_local).get_compile_time_args())
                              + list(ttnn.TensorAccessorArgs(destination_local).get_compile_time_args()),
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                     noc=ttnn.NOC.RISCV_0_default))
        descriptor.runtime_args = runtime
        descriptor.common_runtime_args = [source_local.buffer_address(), destination_local.buffer_address()]
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[descriptor],
                                                                                          cbs=[buffer])
    return program


def page_count(shape, tiled):
    """Pages of an interleaved tensor of this padded shape: 32x32 tiles, or rows."""
    shape = tuple(shape)
    volume = 1
    for value in shape[:-2 if tiled else -1]:
        volume *= value
    if tiled:
        return volume * (shape[-2] // 32) * (shape[-1] // 32)
    return volume


# ---------------------------------------------------------------------------------------------
# Device part: the qualification card only.
# ---------------------------------------------------------------------------------------------

class Bench:
    """Everything the sections do on the device, behind one interface (the CPU tests drive the
    sections through a torch-only fake of it)."""

    def __init__(self, ttnn, torch, vtw, poc, device, *, op_dir, image_ci, harness_dir, kernels):
        self.ttnn, self.torch, self.vtw, self.poc, self.device = ttnn, torch, vtw, poc, device
        self.op_dir, self.image_ci, self.harness_dir, self.kernels = Path(op_dir), Path(image_ci), Path(harness_dir), kernels
        self.copy_kernel = self.harness_dir / 'raw_pages.cpp'
        self.windows_kernel = self.image_ci / 'gdn_conv_windows.cpp'

    def io(self, label):
        """The per-call watchdog around a host<->device transfer or any other device call."""
        return WATCHDOG.op(label)

    def address(self, tensor):
        """Chip 0's buffer address: what a cached program's common args must follow."""
        return self.ttnn.get_device_tensors(tensor)[0].buffer_address()

    # --- raw pages -----------------------------------------------------------------------
    def pages_of(self, tensor):
        return page_count(tensor.padded_shape, tensor.layout == self.ttnn.TILE_LAYOUT)

    def raw(self, tensor, page_bytes):
        """The raw pages of an interleaved tensor as int16 (pages, page_bytes / 2)."""
        ttnn, torch = self.ttnn, self.torch
        pages = self.pages_of(tensor)
        out = ttnn.empty((pages, page_bytes // 4), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
        try:
            with WATCHDOG.op('raw_pages'):
                ttnn.generic_op([tensor, out], copy_pages_program(ttnn, self.device, tensor, out, page_bytes, pages,
                                                                  self.copy_kernel, 1))
                host = ttnn.to_torch(ttnn.get_device_tensors(out)[0])
            return host.to(torch.int32).contiguous().view(torch.int16).reshape(pages, page_bytes // 2)
        finally:
            ttnn.deallocate(out)

    def put(self, pages_int16, tensor, page_bytes):
        """Write host-built raw pages into an existing interleaved tensor, page for page."""
        ttnn, torch = self.ttnn, self.torch
        pages = self.pages_of(tensor)
        if pages_int16.shape[0] != pages:
            raise ValueError('%d host pages for a %d-page tensor' % (pages_int16.shape[0], pages))
        words = pages_int16.contiguous().view(torch.int32).reshape(pages, page_bytes // 4)
        with self.io('put upload'):
            staged = ttnn.from_torch(words, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
        try:
            with WATCHDOG.op('put_pages'):
                ttnn.generic_op([staged, tensor], copy_pages_program(ttnn, self.device, staged, tensor, page_bytes, pages,
                                                                     self.copy_kernel, 1))
                ttnn.synchronize_device(self.device)
        finally:
            ttnn.deallocate(staged)

    def release(self, *tensors):
        for tensor in tensors:
            if tensor is None:
                continue
            if isinstance(tensor, (list, tuple)):
                self.release(*tensor)
            else:
                self.ttnn.deallocate(tensor)

    def memory(self, name):
        return self.ttnn.L1_MEMORY_CONFIG if name == 'l1' else self.ttnn.DRAM_MEMORY_CONFIG

    # --- windows -------------------------------------------------------------------------
    def tile_tensor(self, matrix, memory, poison=None):
        """A (1, rows, cols) bf16 TILE tensor whose raw pages are exactly tile_pages(matrix, poison)."""
        ttnn, torch = self.ttnn, self.torch
        rows, cols = matrix.shape
        with self.io('tile tensor'):
            tensor = ttnn.from_torch(torch.zeros(1, rows, cols, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=self.memory(memory))
        self.put(tile_pages(torch, matrix, poison), tensor, 2048)
        return tensor

    def windows_inputs(self, case):
        """Per user (piece, history4) on the device, and the host matrices they hold."""
        from gdn_device_loop_state import resident_piece

        ttnn, torch = self.ttnn, self.torch
        width, users = case['width'], case['users']
        hosts = dict(pieces=[], block=None)
        tensors, owned = [], []
        if case['source'] == 'served':
            block = torch.cat([make_rows(torch, case['piece'], ROWS, width, case['seed'] * 10 + user)
                               for user in range(4)])
            hosts['block'] = block
            with self.io('windows block upload'):
                device_block = ttnn.from_torch(block.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                               device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG)
            owned.append(device_block)
            for user in range(users):
                with self.io('served piece cut'):
                    piece = resident_piece(ttnn, ttnn.slice(device_block, (0, ROWS * user, 0),
                                                            (1, ROWS * user + ROWS, width)), owned)
                tensors.append([piece])
                hosts['pieces'].append(block[ROWS * user:ROWS * user + ROWS])
        else:
            for user in range(users):
                matrix = make_rows(torch, case['piece'], ROWS, width, case['seed'] * 10 + user)
                piece = self.tile_tensor(matrix, 'l1', poison=POISON)
                owned.append(piece)
                tensors.append([piece])
                hosts['pieces'].append(matrix)
        for user in range(users):
            own = []
            for index in range(4):
                row = make_rows(torch, case['history'], 1, CHANNELS, case['seed'] * 100 + user * 4 + index)
                history = self.tile_tensor(row, case['placement'], poison=POISON)
                owned.append(history)
                own.append(history)
            tensors[user].append(own)
        return [(piece, history) for piece, history in tensors], owned, hosts

    def served_windows(self, piece, history):
        with WATCHDOG.op('served windows'):
            return served_windows_redrive(self.ttnn, self.device, piece, history, 1, self.windows_kernel)

    def packed_windows(self, users, settings=None, negative=None):
        with WATCHDOG.op('packed windows %s %s' % (settings, negative)):
            return self.vtw.build_windows_packed(self.device, users, settings=settings, negative=negative,
                                                 directory=self.op_dir)

    def window_pages(self, tensor):
        return self.raw(tensor, 2048)[:PAGES]

    def input_pages(self, tensor):
        return self.raw(tensor, 2048)

    def program_entries(self):
        return self.device.num_program_cache_entries()

    def synchronize(self):
        with WATCHDOG.op('synchronize'):
            self.ttnn.synchronize_device(self.device)

    # --- traces --------------------------------------------------------------------------
    def capture(self, call):
        ttnn = self.ttnn
        with WATCHDOG.op('capture'):
            trace = ttnn.begin_trace_capture(self.device, cq_id=0)
            result = call()
            ttnn.end_trace_capture(self.device, trace, cq_id=0)
        return trace, result

    def replay(self, trace):
        with WATCHDOG.op('replay'):
            self.ttnn.execute_trace(self.device, trace, cq_id=0, blocking=True)

    def release_trace(self, trace):
        self.ttnn.release_trace(self.device, trace)

    def restage(self, tensor, matrix, poison=None):
        """New contents into the SAME buffer (the addresses a trace baked)."""
        self.put(tile_pages(self.torch, matrix, poison), tensor, 2048)

    # --- kv ------------------------------------------------------------------------------
    def kv_initial(self, width, seed):
        from ordered_cache_hw_plan import payload

        return kv_initial_values(self.torch, width, seed, payload)

    def kv_cache(self, values):
        ttnn = self.ttnn
        with self.io('kv cache upload'):
            return ttnn.from_torch(values, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def kv_empty(self, shape):
        ttnn = self.ttnn
        return ttnn.empty(tuple(shape), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def int_values(self, tensor):
        with self.io('int readback'):
            return self.ttnn.to_torch(self.ttnn.get_device_tensors(tensor)[0]).to(self.torch.int32)

    def kv_metadata(self, positions, pages):
        ttnn, torch = self.ttnn, self.torch

        def upload(value):
            with self.io('kv metadata upload'):
                return ttnn.from_torch(torch.tensor(value, dtype=torch.int32), dtype=ttnn.int32,
                                       layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device,
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG)

        block = (upload(positions), upload(pages))
        tiles = [(upload(positions[first:first + 32]), upload(pages[first:first + 32])) for first in (0, 32)]
        return block, tiles

    def kv_input(self, values):
        ttnn = self.ttnn
        with self.io('kv input upload'):
            return ttnn.from_torch(values.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def served_kv(self, cache, packed, tiles):
        """R1: the served segmented write - per 32-row tile, the whole-tile slice and the ordered
        update re-driven, in tile order on one command queue."""
        ttnn = self.ttnn
        with WATCHDOG.op('served kv'):
            for (first, last), (positions, pages) in zip(((0, 32), (32, 64)), tiles):
                piece = ttnn.slice(packed, (0, first, 0, 0), (1, last, 32, 256), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                try:
                    served_kv_redrive(ttnn, self.device, cache, piece, positions, pages, self.kernels, 1)
                finally:
                    ttnn.deallocate(piece)

    def chained_kv(self, cache, packed, block, tiles, spans, launch_rows=64, cb16_pages=256, negative=None):
        ttnn, poc = self.ttnn, self.poc
        with WATCHDOG.op('chained kv %d %d %s' % (launch_rows, cb16_pages, negative)):
            if launch_rows == 64:
                poc.update_chained(self.device, cache, packed, block[0], block[1], self.kernels, spans,
                                   cb16_pages=cb16_pages, negative=negative)
                return
            for (first, last), (positions, pages) in zip(((0, 32), (32, 64)), tiles):
                piece = ttnn.slice(packed, (0, first, 0, 0), (1, last, 32, 256), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                try:
                    poc.update_chained(self.device, cache, piece, positions, pages, self.kernels,
                                       poc.tile_spans(spans, first, last), cb16_pages=cb16_pages, negative=negative)
                finally:
                    ttnn.deallocate(piece)

    def cache_pages(self, cache):
        return self.raw(cache, 1088)

    def reset_cache(self, pristine, cache):
        """cache <- pristine, page for page (both bf8 tiles of one geometry)."""
        ttnn = self.ttnn
        with WATCHDOG.op('reset cache'):
            ttnn.generic_op([pristine, cache], copy_pages_program(ttnn, self.device, pristine, cache, 1088,
                                                                  self.pages_of(cache), self.copy_kernel, 1))
            ttnn.synchronize_device(self.device)

    def cache_values(self, cache):
        with self.io('cache readback'):
            return self.ttnn.to_torch(self.ttnn.get_device_tensors(cache)[0]).to(self.torch.bfloat16)

    def restage_kv(self, block, tiles, packed, positions, pages, values):
        """New positions, pages and K/V rows into the SAME buffers (the addresses a trace baked),
        one fence."""
        ttnn, torch = self.ttnn, self.torch
        host = [(block[0], torch.tensor(positions, dtype=torch.int32)), (block[1], torch.tensor(pages, dtype=torch.int32))]
        for (first, last), (tile_positions, tile_pages_) in zip(((0, 32), (32, 64)), tiles):
            host.append((tile_positions, torch.tensor(positions[first:last], dtype=torch.int32)))
            host.append((tile_pages_, torch.tensor(pages[first:last], dtype=torch.int32)))
        with self.io('restage kv'):
            for destination, value in host:
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(value, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
                                                destination)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(values.unsqueeze(0), dtype=ttnn.bfloat16,
                                                            layout=ttnn.TILE_LAYOUT), packed)
        self.synchronize()


# ---------------------------------------------------------------------------------------------
# Sections.
# ---------------------------------------------------------------------------------------------

def selftest(bench, args, report):
    torch = bench.torch
    known = (torch.arange(64 * 1024, dtype=torch.int64) * 2654435761 % 65536 - 32768).to(torch.int16).reshape(64, 1024)
    with bench.io('selftest upload'):
        staged = bench.ttnn.from_torch(known.contiguous().view(torch.int32).reshape(64, 512), dtype=bench.ttnn.int32,
                                       layout=bench.ttnn.ROW_MAJOR_LAYOUT, device=bench.device,
                                       memory_config=bench.ttnn.DRAM_MEMORY_CONFIG)
    try:
        roundtrip = bench.raw(staged, 2048)
    finally:
        bench.release(staged)
    special = make_rows(torch, 'specials', 16, 8240, 0)
    ordinary = make_rows(torch, 'randn', 16, 8240, 1)
    tensors = [bench.tile_tensor(special, 'l1', poison=POISON), bench.tile_tensor(ordinary, 'dram', poison=POISON)]
    try:
        pages = [bench.input_pages(tensor) for tensor in tensors]
        with bench.io('selftest readback'):
            logical = bench.ttnn.to_torch(bench.ttnn.get_device_tensors(tensors[1])[0]).to(torch.bfloat16)
    finally:
        bench.release(tensors)
    results = dict(int32_roundtrip=bool(torch.equal(roundtrip, known)),
                   tile_rows=bool(torch.equal(untile_pages(torch, pages[0], 16, 8240), special.view(torch.int16))),
                   padding_poisoned=bool((untile_pages(torch, pages[0], 32, 8256)[16:] == POISON).all()),
                   logical_rows=bool(torch.equal(logical.reshape(16, 8240).view(torch.int16),
                                                 untile_pages(torch, pages[1], 16, 8240))))
    report['selftest'] = results
    print('selftest %s' % results, flush=True)
    for name, ok in results.items():
        if not ok:
            report['failures'].append('selftest %s failed: raw pages cannot be trusted' % name)


def windows_equality(bench, args, report):
    torch, vtw = bench.torch, bench.vtw
    failures = report['failures']
    settings = vtw.settings_matrix() if not args.quick else [dict(port=True), vtw.resolve_settings()]
    exact_by_setting = {vtw.settings_name(setting): True for setting in settings}
    canonical = report.setdefault('slice_canonicalises', [])
    for case in windows_matrix(args.seeds, quick=args.quick):
        name = windows_case_name(case)
        users, owned, hosts = bench.windows_inputs(case)
        try:
            before = [bench.input_pages(tensor) for piece, history in users for tensor in [piece] + history]
            served = [bench.served_windows(piece, history) for piece, history in users]
            r1 = [[bench.window_pages(window) for window in own] for own in served]
            bench.release(served)
            r2 = []
            for user, (piece, history) in enumerate(users):
                piece_pages = before[5 * user][:PAGES]
                history_pages = [before[5 * user + 1 + index][:PAGES] for index in range(4)]
                r2.append(vtw.reference_windows(piece_pages, history_pages))
            entry = dict(case, name=name, r1_r2=all(compare_pages(torch, a, b)['exact']
                                                    for own1, own2 in zip(r1, r2) for a, b in zip(own1, own2)))
            if not entry['r1_r2']:
                failures.append('%s: the served kernel disagrees with its torch transcription' % name)
            entry['settings'] = {}
            for setting in settings:
                label = vtw.settings_name(setting)
                outputs = bench.packed_windows(users, settings=setting)
                results = [compare_pages(torch, bench.window_pages(window), r1[user][slot])
                           for user, own in enumerate(outputs) for slot, window in enumerate(own)]
                bench.release(outputs)
                exact = all(result['exact'] for result in results)
                entry['settings'][label] = exact if exact else [r for r in results if not r['exact']][:2]
                if not exact:
                    exact_by_setting[label] = False
                    failures.append('%s %s: %s' % (name, label, [r for r in results if not r['exact']][:2]))
            after = [bench.input_pages(tensor) for piece, history in users for tensor in [piece] + history]
            entry['inputs_unchanged'] = all(torch.equal(a, b) for a, b in zip(before, after))
            if not entry['inputs_unchanged']:
                failures.append('%s: an input changed' % name)
            if case['source'] == 'served':
                for user in (1, 3):
                    if user < case['users']:
                        block_rows = hosts['block'][ROWS * user:ROWS * user + ROWS].contiguous().view(torch.int16)
                        piece_rows = untile_pages(torch, before[5 * user], ROWS, case['width'])
                        canonical.append(dict(case=name, user=user, block=special_counts(torch, block_rows),
                                              piece=special_counts(torch, piece_rows),
                                              identical=bool(torch.equal(block_rows, piece_rows))))
            report['windows_cases'].append(entry)
            report['cases_run'] = report.get('cases_run', 0) + 1
            print('%-60s %s' % (name, 'exact' if all(v is True for v in entry['settings'].values()) else entry['settings']),
                  flush=True)
        finally:
            bench.release(owned)
    report['windows_exact_by_setting'] = exact_by_setting
    windows_refusals(bench, args, report)


def windows_refusals(bench, args, report):
    """rows 8 / 32 and a mixed history placement: Unsupported before any device work."""
    torch, vtw = bench.torch, bench.vtw
    before = bench.program_entries()
    for case in refusal_cases():
        tensors = []
        try:
            piece = bench.tile_tensor(make_rows(torch, 'randn', case['rows'], 8240, 0), 'l1')
            tensors.append(piece)
            history = [bench.tile_tensor(make_rows(torch, 'randn', 1, CHANNELS, index),
                                         'l1' if case.get('mixed') and index == 0 else 'dram') for index in range(4)]
            tensors.extend(history)
            entries = bench.program_entries()
            try:
                outputs = bench.packed_windows([(piece, history)])
            except vtw.Unsupported as reason:
                refused = str(reason)
            else:
                bench.release(outputs)
                refused = None
            grew = bench.program_entries() - entries
            report['windows_refusals'].append(dict(case=case['name'], refused=refused, program_cache_delta=grew))
            if refused is None or grew:
                report['failures'].append('refusal %s: refused=%r program cache +%d' % (case['name'], refused, grew))
        finally:
            bench.release(tensors)
    report['windows_refusal_entries'] = bench.program_entries() - before


def windows_negative(bench, args, report):
    torch = bench.torch
    case = windows_matrix(args.seeds, quick=True)[0]
    users, owned, hosts = bench.windows_inputs(case)
    try:
        served = [bench.served_windows(piece, history) for piece, history in users]
        r1 = [[bench.window_pages(window) for window in own] for own in served]
        bench.release(served)
        for negative in NEGATIVES:
            outputs = bench.packed_windows(users, negative=negative)
            same = all(compare_pages(torch, bench.window_pages(window), r1[user][slot])['exact']
                       for user, own in enumerate(outputs) for slot, window in enumerate(own))
            bench.release(outputs)
            report['windows_negative'][negative] = dict(differs=not same)
            print('negative %-6s %s' % (negative, 'differs (good)' if not same else 'EXACT (bad)'), flush=True)
            if same:
                report['failures'].append('negative control %s did not change the windows: the test cannot see that fault'
                                          % negative)
    finally:
        bench.release(owned)


def windows_trace(bench, args, report):
    """Two launches over independent input sets in ONE trace - one cached program whose common args
    differ per launch, as the model captures 48 per trace - replayed after restaging new contents
    in place; each launch against R1 on the restaged inputs."""
    torch = bench.torch
    base = dict(windows_matrix(args.seeds, quick=True)[0], source='direct')
    sets = [bench.windows_inputs(dict(base, seed=seed)) for seed in (0, 1)]
    inputs = [users for users, owned, hosts in sets]
    owned = [value for users, own, hosts in sets for value in own]
    trace = outputs = None
    try:
        for users in inputs:
            bench.release(bench.packed_windows(users))        # compile outside the capture
        bench.synchronize()
        trace, outputs = bench.capture(lambda: [bench.packed_windows(users) for users in inputs])
        checked = []
        for replay in range(3 + args.soak):
            seed = 50 + replay
            for index, users in enumerate(inputs):
                for user, (piece, history) in enumerate(users):
                    bench.restage(piece, make_rows(torch, 'randn', ROWS, base['width'], seed * 10 + user + 1000 * index),
                                  POISON)
                    for slot, tensor in enumerate(history):
                        bench.restage(tensor, make_rows(torch, 'specials' if replay % 2 else 'randn', 1, CHANNELS,
                                                        seed * 100 + user * 4 + slot + 1000 * index), POISON)
            bench.replay(trace)
            if replay < 3 or replay == 3 + args.soak - 1:
                launches = []
                for users, launch in zip(inputs, outputs):
                    served = [bench.served_windows(piece, history) for piece, history in users]
                    launches.append(all(compare_pages(torch, bench.window_pages(window),
                                                      bench.window_pages(served[user][slot]))['exact']
                                        for user, own in enumerate(launch) for slot, window in enumerate(own)))
                    bench.release(served)
                checked.append(dict(replay=replay, exact=all(launches), launches=launches))
                if not all(launches):
                    report['failures'].append('windows trace replay %d: launch %s differs from the served launch'
                                              % (replay, [index for index, ok in enumerate(launches) if not ok]))
        report['windows_trace'] = checked
        print('windows trace %s' % checked, flush=True)
    finally:
        if trace is not None:
            bench.release_trace(trace)
        if outputs is not None:
            bench.release(outputs)
        bench.release(owned)


def windows_cache(bench, args, report):
    """Three calls with every earlier call's inputs AND outputs still held, so the allocator cannot
    hand a cache hit the addresses the first call baked: each call's common-arg addresses are
    checked fresh, each call's windows exact against R1, and only the first may add programs."""
    torch, vtw = bench.torch, bench.vtw
    deltas, held, kept, seen, fresh = [], [], [], [], []
    try:
        for index in range(3):
            case = dict(windows_matrix(args.seeds, quick=True)[0], seed=30 + index)
            users, owned, hosts = bench.windows_inputs(case)
            held.extend(owned)
            kept.append(users)
            served = [bench.served_windows(piece, history) for piece, history in users]
            held.extend(served)
            entries = bench.program_entries()
            outputs = bench.packed_windows(users)
            held.extend(outputs)
            deltas.append(bench.program_entries() - entries)
            addresses = {bench.address(tensor) for piece, history in users for tensor in [piece] + list(history)}
            addresses |= {bench.address(window) for own in outputs for window in own}
            fresh.append(all(not addresses & earlier for earlier in seen))
            seen.append(addresses)
            exact = all(compare_pages(torch, bench.window_pages(window), bench.window_pages(served[user][slot]))['exact']
                        for user, own in enumerate(outputs) for slot, window in enumerate(own))
            if not exact:
                report['failures'].append('windows cache call %d is not exact' % index)
    finally:
        bench.release(held)
    report['windows_cache'] = dict(new_entries_per_call=deltas, descriptor_cache=vtw.cache_size(),
                                   fresh_addresses=fresh)
    print('windows program cache %s' % report['windows_cache'], flush=True)
    if any(deltas[1:]):
        report['failures'].append('a cached windows call added program-cache entries: %s' % deltas)
    if not all(fresh[1:]):
        report['failures'].append('a cached windows call reused an earlier call\'s addresses %s: the section cannot '
                                  'show that a cache hit takes the new common args' % fresh)


def kv_case_inputs(bench, case):
    torch = bench.torch
    if case.get('conflict'):
        geometry = kv_conflict_geometry(case['width'], case['seed'])
    else:
        geometry = kv_geometry(case['geometry'], case['width'], case['seed'])
    positions, pages = kv_block_host(geometry)
    values = kv_payload(torch, case['payload'], 64, case['seed'])
    return geometry, positions, pages, values


def kv_state(bench, width, seed):
    """The initial BF8-exact cache values, a pristine device copy, and OLD / NEW - reset from it
    page for page HERE (ttnn.empty is uninitialised: two fresh caches never start equal) and
    again before each case, so the two always start from identical raw pages."""
    initial = bench.kv_initial(width, seed)
    pristine = bench.kv_cache(initial)
    old, new = bench.kv_empty(initial.shape), bench.kv_empty(initial.shape)
    try:
        bench.reset_cache(pristine, old)
        bench.reset_cache(pristine, new)
    except BaseException:
        bench.release(pristine, old, new)
        raise
    return initial, pristine, old, new


def kv_case_spans(case):
    """g4 (the warm forward's served placeholders): one chain over the block; else one per user."""
    if case.get('geometry') == 'g4':
        return kv_single_span()
    return kv_spans()


def kv_single_span():
    return ((0, KV_USERS * KV_ROWS_PER_USER),)


def kv_equality(bench, args, report):
    torch = bench.torch
    import verify_trace_t2 as t2

    failures = report['failures']
    state, state_width = None, None
    try:
        for case in kv_matrix(args.kv_widths, args.seeds, quick=args.quick):
            name = kv_case_name(case)
            geometry, positions, pages, values = kv_case_inputs(bench, case)
            if geometry is None:
                continue
            spans = kv_case_spans(case)
            guard = t2.kv_conflict([(range(start, start + KV_ROWS_PER_USER), table) for start, table in geometry])
            entry = dict(case, name=name, chains=len(spans), guard_disjoint=guard is None)
            if guard is not None and len(spans) > 1:
                failures.append('%s: the geometry is meant to be disjoint; the guard says %s' % (name, guard))
                continue
            if state_width != case['width']:
                if state is not None:
                    bench.release(state[1:])
                state, state_width = kv_state(bench, case['width'], 0), case['width']
            initial, pristine, old, new = state
            block, tiles = bench.kv_metadata(positions, pages)
            packed = bench.kv_input(values)
            try:
                metadata = [bench.int_values(value) for value in (block[0], block[1])]
                kv_before = bench.raw(packed, 2048)
                bench.reset_cache(pristine, old)
                bench.served_kv(old, packed, tiles)
                r1 = bench.cache_pages(old)
                entry['variants'] = {}
                for launch_rows, cb16 in kv_variants(args.quick):
                    bench.reset_cache(pristine, new)
                    bench.chained_kv(new, packed, block, tiles, spans, launch_rows, cb16)
                    result = compare_pages(torch, bench.cache_pages(new), r1)
                    label = 'rows%d_cb%d' % (launch_rows, cb16)
                    entry['variants'][label] = True if result['exact'] else result
                    if not result['exact']:
                        failures.append('%s %s: %s' % (name, label, result))
                    if case['payload'] == 'bf8' and (launch_rows, cb16) == (64, 256):
                        predicted = predict_cache(torch, initial, geometry, values)
                        entry['r2'] = bool(torch.equal(bench.cache_values(new).view(torch.int16),
                                                       predicted.view(torch.int16)))
                        if not entry['r2']:
                            failures.append('%s: the chained cache is not the host prediction' % name)
                unchanged = all(torch.equal(before, bench.int_values(value))
                                for before, value in zip(metadata, block)) and torch.equal(kv_before, bench.raw(packed, 2048))
                entry['inputs_unchanged'] = unchanged
                if not unchanged:
                    failures.append('%s: the metadata or the K/V input changed' % name)
                report['kv_cases'].append(entry)
                report['cases_run'] = report.get('cases_run', 0) + 1
                print('%-40s %s' % (name, entry['variants']), flush=True)
            finally:
                bench.release(packed, block, tiles)
    finally:
        if state is not None:
            bench.release(state[1:])


def kv_spans():
    return tuple((user * KV_ROWS_PER_USER, (user + 1) * KV_ROWS_PER_USER) for user in range(KV_USERS))


def kv_negative(bench, args, report):
    torch = bench.torch
    spans = kv_spans()
    width = 2052 if 2052 in args.kv_widths else args.kv_widths[-1]
    initial, pristine, old, new = kv_state(bench, width, 0)
    try:
        for negative in KV_NEGATIVES:
            repeats = 1 if negative == 'index' else 5
            outcomes = []
            for repeat in range(repeats):
                case = dict(width=width, geometry='g1', payload='randn', seed=repeat, conflict=negative == 'conflict')
                geometry, positions, pages, values = kv_case_inputs(bench, case)
                block, tiles = bench.kv_metadata(positions, pages)
                packed = bench.kv_input(values)
                try:
                    bench.reset_cache(pristine, old)
                    bench.reset_cache(pristine, new)
                    bench.served_kv(old, packed, tiles)
                    bench.chained_kv(new, packed, block, tiles, spans,
                                     negative=None if negative == 'conflict' else negative)
                    outcomes.append(not compare_pages(torch, bench.cache_pages(new), bench.cache_pages(old))['exact'])
                finally:
                    bench.release(packed, block, tiles)
            differs = all(outcomes) if negative == 'index' else any(outcomes)
            report['kv_negative'][negative] = dict(differs=differs, repeats=outcomes)
            print('kv negative %-8s %s %s' % (negative, 'differs (good)' if differs else 'EXACT (bad)', outcomes), flush=True)
            if not differs:
                report['failures'].append('kv negative control %s never changed the cache: the test cannot see that fault'
                                          % negative)
    finally:
        bench.release(pristine, old, new)


def kv_trace(bench, args, report):
    torch = bench.torch
    spans = kv_spans()
    width = 2052 if 2052 in args.kv_widths else args.kv_widths[-1]
    initial, pristine, old, new = kv_state(bench, width, 0)
    tables = [kv_case_inputs(bench, dict(width=width, geometry='g1', payload='bf8', seed=seed)) for seed in (5, 6)]
    geometry, positions, pages, values = tables[0]
    block, tiles = bench.kv_metadata(positions, pages)
    packed = bench.kv_input(values)
    trace = None
    try:
        bench.chained_kv(new, packed, block, tiles, spans)          # compile outside the capture
        bench.served_kv(old, packed, tiles)
        bench.synchronize()
        trace, unused = bench.capture(lambda: bench.chained_kv(new, packed, block, tiles, spans))
        checked = []
        for replay in range(3):
            geometry, positions, pages, values = tables[replay % 2]
            host_values = kv_payload(torch, 'bf8', 64, 40 + replay)
            bench.restage_kv(block, tiles, packed, positions, pages, host_values)
            bench.replay(trace)
            bench.served_kv(old, packed, tiles)
            exact = compare_pages(torch, bench.cache_pages(new), bench.cache_pages(old))['exact']
            checked.append(dict(replay=replay, table=replay % 2, exact=exact))
            if not exact:
                report['failures'].append('kv trace replay %d differs from the served write' % replay)
        report['kv_trace'] = checked
        print('kv trace %s' % checked, flush=True)
    finally:
        if trace is not None:
            bench.release_trace(trace)
        bench.release(packed, block, tiles, pristine, old, new)


def kv_cache(bench, args, report):
    torch = bench.torch
    spans = kv_spans()
    width = args.kv_widths[0]
    initial, pristine, old, new = kv_state(bench, width, 0)
    served_deltas, chained_deltas = [], []
    try:
        for index in range(3):
            geometry, positions, pages, values = kv_case_inputs(bench, dict(width=width, geometry='g1', payload='bf8',
                                                                            seed=60 + index))
            block, tiles = bench.kv_metadata(positions, pages)
            packed = bench.kv_input(values)
            try:
                entries = bench.program_entries()
                bench.served_kv(old, packed, tiles)
                served_deltas.append(bench.program_entries() - entries)
                entries = bench.program_entries()
                bench.chained_kv(new, packed, block, tiles, spans)
                chained_deltas.append(bench.program_entries() - entries)
                if not compare_pages(torch, bench.cache_pages(new), bench.cache_pages(old))['exact']:
                    report['failures'].append('kv cache call %d is not exact' % index)
            finally:
                bench.release(packed, block, tiles)
        # The warm forward's ONE chain, then the capture's per-user chains, on a CB16 no launch
        # used before: the first must compile, the second must be a cache hit - the per-user
        # chains differ from the warm forward's only in runtime args, and the capture that runs
        # them is recorded inside a trace, where nothing may compile.
        geometry, positions, pages, values = kv_case_inputs(bench, dict(width=width, geometry='g1', payload='bf8',
                                                                        seed=63))
        block, tiles = bench.kv_metadata(positions, pages)
        packed = bench.kv_input(values)
        try:
            entries = bench.program_entries()
            bench.chained_kv(new, packed, block, tiles, kv_single_span(), cb16_pages=CB16_FRESH)
            warm_delta = bench.program_entries() - entries
            entries = bench.program_entries()
            bench.chained_kv(new, packed, block, tiles, spans, cb16_pages=CB16_FRESH)
            capture_delta = bench.program_entries() - entries
        finally:
            bench.release(packed, block, tiles)
    finally:
        bench.release(pristine, old, new)
    report['kv_cache'] = dict(served=served_deltas, chained=chained_deltas, warm_then_capture=[warm_delta, capture_delta])
    print('kv program cache %s' % report['kv_cache'], flush=True)
    if [delta > 0 for delta in served_deltas[1:]] != [delta > 0 for delta in chained_deltas[1:]]:
        report['failures'].append('the chained write\'s program-cache pattern %s is not the served %s'
                                  % (chained_deltas, served_deltas))
    if warm_delta < 1:
        report['failures'].append('the warm chain on a fresh CB16 (%d pages) added no program: the check cannot see '
                                  'a miss' % CB16_FRESH)
    if capture_delta:
        report['failures'].append('the per-user chains after the warm forward\'s one chain added %d program(s): '
                                  'the capture would compile inside its trace' % capture_delta)


def trace_timed(bench, call, launches, iterations):
    """us per launch: `launches` calls captured in one trace, replayed `iterations` times."""
    held = []

    def captured():
        for index in range(launches):
            held.append(call())

    captured()                      # compile and warm outside the capture
    bench.release(held)
    held.clear()
    bench.synchronize()
    trace, unused = bench.capture(captured)
    try:
        bench.replay(trace)
        samples = []
        for index in range(iterations):
            started = time.perf_counter()
            bench.replay(trace)
            samples.append((time.perf_counter() - started) * 1e6 / launches)
        return summarise_timing(samples)
    finally:
        bench.release_trace(trace)
        bench.release(held)


def timing(bench, args, report):
    torch, vtw = bench.torch, bench.vtw
    timing = report['timing']
    case = windows_matrix(args.seeds, quick=True)[0]
    users, owned, hosts = bench.windows_inputs(case)
    try:
        timing['served_windows_x4'] = trace_timed(bench, lambda: [bench.served_windows(piece, history)
                                                                  for piece, history in users], 4, args.iters)
        per_setting = {}
        for setting in vtw.settings_matrix():
            label = vtw.settings_name(setting)
            timing[label] = trace_timed(bench, lambda setting=setting: bench.packed_windows(users, settings=setting),
                                        16, args.iters)
            per_setting[label] = timing[label]['median']
            print('windows %-18s %.1f us per launch' % (label, per_setting[label]), flush=True)
        host = []
        for index in range(args.host_calls):
            started = time.perf_counter()
            outputs = bench.packed_windows(users)
            host.append((time.perf_counter() - started) * 1e6)
            bench.release(outputs)
        bench.synchronize()
        # Host time matters only at capture (a replay runs no Python); recorded, not gated.
        timing['host_us_per_call'] = dict(median=statistics.median(host), p90=sorted(host)[int(0.9 * len(host))],
                                          calls=len(host))
        exact = report.get('windows_exact_by_setting') or {label: True for label in per_setting}
        verdict_ = settings_verdict(exact, per_setting, vtw.settings_name(vtw.resolve_settings()))
        timing['settings_verdict'] = verdict_
        timing['port_gate_met'] = per_setting['port'] <= GATES_US['port']
        defaults = per_setting[vtw.settings_name(vtw.resolve_settings())]
        timing['defaults_gate_met'] = defaults <= GATES_US['defaults']
        if not timing['port_gate_met']:
            report['failures'].append('VTW_PORT %.1f us per launch > %.0f us' % (per_setting['port'], GATES_US['port']))
        if not timing['defaults_gate_met']:
            report['failures'].append('DEFAULTS %.1f us per launch > %.0f us' % (defaults, GATES_US['defaults']))
        if verdict_['verdict'] == 'change':
            report['failures'].append('set gdn_conv_windows_packed.DEFAULTS to %s (%.1f us; DEFAULTS %.1f us)'
                                      % (verdict_['fastest'], verdict_['fastest_us'], verdict_['defaults_us']))
    finally:
        bench.release(owned)
    spans = kv_spans()
    width = 2052 if 2052 in args.kv_widths else args.kv_widths[-1]
    initial, pristine, old, new = kv_state(bench, width, 0)
    geometry, positions, pages, values = kv_case_inputs(bench, dict(width=width, geometry='g1', payload='bf8', seed=0))
    block, tiles = bench.kv_metadata(positions, pages)
    packed = bench.kv_input(values)
    try:
        timing['served_kv'] = trace_timed(bench, lambda: bench.served_kv(old, packed, tiles), 4, args.iters)
        timing['chained_kv_64'] = trace_timed(bench, lambda: bench.chained_kv(new, packed, block, tiles, spans), 8,
                                              args.iters)
        timing['chained_kv_32'] = trace_timed(bench, lambda: bench.chained_kv(new, packed, block, tiles, spans, 32), 4,
                                              args.iters)
        timing['chained_gate_met'] = timing['chained_kv_64']['median'] <= GATES_US['chained']
        print('kv served %.1f chained64 %.1f chained32 %.1f us per call' % (
            timing['served_kv']['median'], timing['chained_kv_64']['median'], timing['chained_kv_32']['median']), flush=True)
        if not timing['chained_gate_met']:
            report['failures'].append('the chained 64-row launch %.1f us > %.0f us'
                                      % (timing['chained_kv_64']['median'], GATES_US['chained']))
    finally:
        bench.release(packed, block, tiles, pristine, old, new)
    if os.environ.get('TT_METAL_DEVICE_PROFILER') == '1':
        with bench.io('read device profiler'):
            bench.ttnn.ReadDeviceProfiler(bench.device)
        timing['device_profiler'] = 'read: generated/profiler (the run script mounts it into the results)'


SECTION_DRIVERS = dict(selftest=selftest, windows=windows_equality, windows_negative=windows_negative,
                       windows_trace=windows_trace, windows_cache=windows_cache, kv=kv_equality,
                       kv_negative=kv_negative, kv_trace=kv_trace, kv_cache=kv_cache, timing=timing)


def provenance(args, report):
    """sha256 of every mounted file and every image reference file, against the checkout's."""
    report['mounted'] = {name: sha256(Path(args.op_dir) / name) for name in args.runtime_files}
    image = Path(args.image_ci)
    report['image_references'] = {name: sha256(image / name) if (image / name).is_file() else None
                                  for name in REFERENCE_FILES}
    report['failures'].extend(provenance_failures(report['image_references'], args.expect))
    return not report['failures']


def run(args, report):
    import torch
    import ttnn

    sys.path[:0] = [str(args.op_dir), str(args.harness_dir), str(args.image_ci)]
    sys.path.append(os.environ.get('TT_METAL_HOME', '/opt/tt-metal'))
    import gdn_conv_windows_packed as vtw
    import ordered_cache
    import packed_ordered_cache as poc

    provenance(args, report)
    try:
        kernels = ordered_cache.load_kernels(os.environ.get('TT_METAL_HOME', '/opt/tt-metal'))
    except ValueError as error:
        report['failures'].append('ordered_cache.load_kernels refused the native sources: %s' % error)
        return
    report['op_source_sha'] = vtw.source_sha(args.op_dir)
    print('provenance %s op kernel %s' % ('ok' if not report['failures'] else report['failures'], report['op_source_sha']),
          flush=True)
    if report['failures']:
        return
    device = ttnn.open_device(device_id=args.device_id, trace_region_size=args.trace_region)
    try:
        grid = device.compute_with_storage_grid_size()
        report['grid'] = [grid.x, grid.y]
        bench = Bench(ttnn, torch, vtw, poc, device, op_dir=args.op_dir, image_ci=args.image_ci,
                      harness_dir=args.harness_dir, kernels=kernels)
        for name in ['selftest'] + [section for section in args.sections if section != 'selftest']:
            if name == 'timing' and args.no_timing:
                continue
            SECTION_DRIVERS[name](bench, args, report)
            report['sections_completed'].append(name)
            if name == 'selftest' and report['failures']:
                return
    finally:
        with WATCHDOG.op('close device'):
            ttnn.close_device(device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--op-dir', type=Path, default=Path('/bench/vt2'))
    parser.add_argument('--harness-dir', type=Path, default=Path('/bench/vt2-harness'))
    parser.add_argument('--image-ci', type=Path, default=Path('/experiment-scripts/ci'))
    parser.add_argument('--runtime-files', default='verify_trace_t2.py,gdn_conv_windows_packed.py,'
                                                   'gdn_conv_windows_packed.cpp,packed_ordered_cache.py')
    parser.add_argument('--expect', action='append', default=[], help='name=sha256 of the checkout copy')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    parser.add_argument('--kv-widths', default=','.join(map(str, KV_WIDTHS)))
    parser.add_argument('--sections', default=','.join(SECTIONS))
    parser.add_argument('--quick', action='store_true', help='the thinner matrices (the watcher pass)')
    parser.add_argument('--no-timing', action='store_true')
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--soak', type=int, default=20)
    parser.add_argument('--host-calls', type=int, default=100)
    parser.add_argument('--trace-region', type=int, default=32 * 1024 * 1024)
    parser.add_argument('--watchdog', type=float, default=0)
    args = parser.parse_args(argv)
    args.seeds = [int(value) for value in args.seeds.split(',')]
    args.kv_widths = [int(value) for value in args.kv_widths.split(',')]
    args.sections = [value for value in args.sections.split(',') if value]
    args.runtime_files = [value for value in args.runtime_files.split(',') if value]
    unknown = set(args.sections) - set(SECTIONS)
    if unknown or any(width not in KV_WIDTHS for width in args.kv_widths):
        parser.error('unknown section %s or a page width outside %s' % (sorted(unknown), KV_WIDTHS))
    expected = {}
    for item in args.expect:
        name, _, sha = item.partition('=')
        if name not in REFERENCE_FILES or len(sha) != 64:
            parser.error('--expect takes name=sha256 for one of %s' % (REFERENCE_FILES,))
        expected[name] = sha
    args.expect = expected
    return args


def main(argv=None):
    global WATCHDOG
    args = parse_args(argv)
    report = dict(passed=False, failures=[], sections_completed=[], windows_cases=[], windows_refusals=[],
                  windows_negative={}, kv_cases=[], kv_negative={}, timing={},
                  args={k: str(v) for k, v in vars(args).items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write(extra=None):
        payload = dict(report)
        if extra:
            payload.update(extra)
        args.out.write_text(json.dumps(payload, indent=2, default=str))

    WATCHDOG = Watchdog(args.watchdog, on_fire=lambda label: write(dict(error='watchdog: %r' % label))).start()
    try:
        run(args, report)
        report['passed'] = verdict(report)
    except Exception as error:  # noqa: BLE001 - recorded, then re-raised for the exit status
        report['error'] = repr(error)
        write()
        raise
    write()
    print('PASSED' if report['passed'] else 'FAILED: %d failures' % len(report['failures']), flush=True)
    for failure in report['failures'][:40]:
        print('  ' + failure, flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
