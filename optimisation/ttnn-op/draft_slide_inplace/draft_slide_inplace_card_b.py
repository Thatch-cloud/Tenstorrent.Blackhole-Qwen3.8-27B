"""M-F0 card-B harness (round-fence plan S0.1): is the served draft K/V slide byte-identical IN PLACE?

H1b (the fused commit) wants each drafter K/V publication to slide the committed history bank in place
(output aliased to the input bank), so the bank never swaps and the pair traces can bind the live banks
(F4). This harness runs the SERVED slide kernel both ways on one card and compares bytes:

  reference  out of place, exactly as served: draft_kv_slide.prepare's per-chip program (16 workers, one
             per (head, 32-column tile), io [active, delta, spare]) writing a poisoned spare;
  candidate  in place: the same kernel file and runtime args, except that the output address and the
             output accessor are the bank's own, with the io list in one of three forms (see forms).

Which kernel is served. draft_kv_slide.{py,cpp} are in neither image copy list, so every fast-serving
image (v40 through A5, A7, P1 and P2) carries serving-bundle 35489235797's copies. Its driver is the
checkout's (8ea58ae4), but /experiment-scripts/ci/draft_kv_slide.cpp is the DIRECT-DMA kernel (1679bbd7,
byte-identical to the checkout's draft_kv_slide_direct.cpp, staged under the served name by
draft_kv_slide_hardware_stage --direct-dma for cumulative-t16-full-v20), NOT the checkout's scalar
draft_kv_slide.cpp (bc45d472). The harness hashes the image's files, names the kind, and by default also
runs the checkout's scalar kernel (mounted by run_card_b.sh), so the verdict covers both.

Sections (all on by default):
  forms      does generic_op take the in-place io list? 'aliased' [bank, delta, bank] (the served arity,
             output == input), 'pair' [bank, delta], 'pair_out_last' [delta, bank]. Each on fresh
             tensors: refused (the exception is recorded), accepted and wrong, or accepted and exact. The
             first exact form in that order (or --form) is used by every section below.
  cases      per kernel: history 2048 x prefixes 1..16 x the two chips' data (card B is one chip: chip c
             is data seed c, run on it in turn) - the plan's 32 of 32 - plus the fused path's other
             rows == 2048 entries (history 2032-2047, drop 0 included). Random finite bf16 bit patterns
             everywhere; the delta's rows past the
             prefix and the spare are random too (poison). Out of place == host oracle; in place == out of
             place == oracle; the delta (and, out of place, the bank) unchanged; the kernels agree.
  multibank  one user's 10 banks (5 layers x k/v) in programs of 1, 2 and 5 banks (and 10 when the grid
             has 160 cores): 16 disjoint workers per bank, every bank == its out-of-place reference.
  cache      program cache: a repeat call, a call on FRESH banks (other addresses) and a new prefix must
             add no entry and must be honoured (the fresh bank slid, the old bank untouched); the same for
             a 5-bank program; the served out-of-place path's entries are recorded.
  trace      the 10-bank in-place programs, per layout, captured once and replayed on new contents
             written into the same banks: exact on every replay, no cache entry added.
  timing     one user's 10 banks at prefix 16: 10 x 16-worker launches (today's served layout, with and
             without rebuilding the descriptor per call as the served driver does), 5 x 32, 2 x 80 (and
             1 x 160 when it fits), each out of place and in place; eager (per-iteration synchronised,
             host enqueue, pipelined) and traced (blocking replay, pipelined replays).

Decision (report['decision'], see decide): go = no failure, every byte comparison exact, the full served
matrix run, an accepted exact io form, the cache verified, and the best in-place traced layout that was
itself verified exact (multibank and trace) at or under 0.5 ms per user (pipelined replay); otherwise the
parity-keyed out-of-place fallback, with why_not. Timing never fails the run: it is the input to that
decision.

Nothing is read from the model and nothing is written outside --out. The served driver refuses a one-chip
mesh (two shards) and aliasing, so the harness builds its per-chip program itself;
test_draft_slide_inplace_card_b.py pins that program, field for field, to draft_kv_slide.prepare's chip-0
program, and pins the kernel mirrors (the S0.4 argument) to both kernel sources by hash.
"""

import argparse
import faulthandler
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

KV_SHAPE = (1, 4, 2048, 128)
DELTA_SHAPE = (1, 4, 32, 128)
HEADS, CAPACITY, HEAD_DIM, TILE = 4, 2048, 128, 32
TILES = CAPACITY // TILE            # 64 history tiles per (head, column)
COLUMNS = HEAD_DIM // TILE          # 4
WORKERS = HEADS * COLUMNS           # 16 per bank: worker w is (head w // 4, column w % 4)
BANKS = 10                          # one user: 5 learned draft layers x (k, v)
HISTORY = 2048                      # the fused path's guard: rows == 2048 (dflash_traced_publish)
PREFIXES = tuple(range(1, 17))
CHIPS = (0, 1)
# The fused path's other entries (its guard is rows == min(2048, history + prefix) == 2048, so a history
# under 2048 reaches it): drop < prefix, including drop 0 (history + prefix == 2048 exactly, the window's
# first fill: every tile rewrites itself). drop == rows cannot happen there (drop <= prefix <= 32 < 2048).
EDGES = ((2047, 2), (2040, 16), (2033, 16), (2047, 16), (2047, 1), (2032, 16))
LAYOUTS = (1, 2, 5, 10)             # banks per program: 10 x 16, 5 x 32, 2 x 80, 1 x 160 workers
MULTIBANK_PREFIXES = (1, 7, 16)
OPEN_EXTRA_S = 300                  # open_device's extra watchdog budget: firmware JIT into a fresh cache
TIMING_SPAN_S = 600                 # one timing variant's whole loop (warmup, iters x 2, capture, replays)
# --quick (the watcher pass) defaults; explicit arguments (CARD_B_ARGS) still override them.
QUICK = dict(prefixes='1,2,15,16', chips='0', edges='2047:2,2047:1', layouts='1,5', multibank_prefixes='16',
             trace_layouts='5', trace_replays=1, iters=5, warmup=1)
FORM_PREFIX = 7
CACHE_PREFIXES = (5, 11)
TRACE_PREFIX = 16
TIMING_PREFIX = 16
CB_BYTES, PAGE_BYTES = 8192, 2048   # draft_kv_slide.prepare's circular buffer
STALE = 0x7FC1                      # stale-L1 marker in the mirrors (a NaN pattern the data never holds)
GO_MS_PER_USER = 0.5                # the plan's in-place threshold (S0.1)

SERVED_DIR = Path('/experiment-scripts/ci')
CHECKOUT_DIR = Path('/bench/slide')  # run_card_b.sh: scalar/draft_kv_slide.cpp, direct/draft_kv_slide.cpp
SCALAR_SHA256 = 'bc45d47257c844aff4bf17f478b536a48763578e544597884b7d1620083b6ba1'
DIRECT_SHA256 = '1679bbd779add56b4bd445a6b4c51bd3e49c39a8a520dfaddfc3bd9f36667d47'
KERNEL_KINDS = {SCALAR_SHA256: 'scalar', DIRECT_SHA256: 'direct'}   # draft_kv_slide_gate.QUALIFIED_KERNELS
# serving-bundle 35489235797 (serving-bundle.json): experiment-scripts/ci/draft_kv_slide.{cpp,py}.
BUNDLE_KERNEL_SHA256 = DIRECT_SHA256
BUNDLE_DRIVER_SHA256 = '8ea58ae42260cb03f37e224d18572c4efe467ff556912a64ebcb1d15aed9c7fc'
KERNEL_LABELS = ('served', 'scalar', 'direct')
FORMS = ('aliased', 'pair', 'pair_out_last')
FORM_IO = {'aliased': '[bank, delta, bank] per bank', 'pair': '[bank, delta] per bank',
           'pair_out_last': '[delta, bank] per bank'}
SECTIONS = ('forms', 'cases', 'multibank', 'cache', 'trace', 'timing')

WATCHDOG = None


class Unsupported(ValueError):
    """A program this harness refuses to build (the served driver's refusals, and grid limits)."""


# ---------------------------------------------------------------------------------------------
# Pure helpers: the oracle, the kernel mirrors and their access orders (CPU-tested).
# ---------------------------------------------------------------------------------------------

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(history_rows, prefix):
    """draft_kv_slide.geometry, restated for the pure helpers (the device path uses the served driver's)."""
    if (type(history_rows) is not int or not 1 <= history_rows <= CAPACITY
            or type(prefix) is not int or not 1 <= prefix <= 32):
        raise ValueError('One committed history and bounded accepted prefix required')
    rows = min(CAPACITY, history_rows + prefix)
    return dict(history_rows=history_rows, prefix=prefix, rows=rows, drop=history_rows + prefix - rows)


def random_bits(torch, shape, seed):
    """int16 bf16 bit patterns: random sign and mantissa, exponent 1..254 (finite, normal, never 0 or
    denormal), so a tilize or copy that rounds, flushes or canonicalises anything shows up as a change."""
    generator = torch.Generator().manual_seed(int(seed))
    sign = torch.randint(0, 2, shape, generator=generator, dtype=torch.int32) << 15
    exponent = torch.randint(1, 255, shape, generator=generator, dtype=torch.int32) << 7
    mantissa = torch.randint(0, 128, shape, generator=generator, dtype=torch.int32)
    value = sign | exponent | mantissa
    return torch.where(value >= 32768, value - 65536, value).to(torch.int16)


def oracle(torch, active, delta, history_rows, prefix):
    """The slide, on bit patterns: [committed history | accepted prefix], its last `rows` rows at the top,
    zero after (draft-kv-slide-probe.py's expectation)."""
    shape = geometry(history_rows, prefix)
    combined = torch.cat((active[:, :, :history_rows], delta[:, :, :prefix]), dim=2)
    out = torch.zeros(tuple(active.shape), dtype=torch.int16)
    out[:, :, :shape['rows']] = combined[:, :, shape['drop']:shape['drop'] + shape['rows']]
    return out


def direct_segments(history_rows, prefix, drop, rows, tile):
    """The direct kernel's row loop for one output tile: [(source, source_row, row, count)], source in
    active / delta / zero. Raises where the kernel's unsigned count would not advance."""
    result, row = [], 0
    while row < TILE:
        destination = tile * TILE + row
        logical = destination + drop
        if destination >= rows:
            result.append(('zero', 0, row, TILE - row))
            break
        historical = logical < history_rows
        source = logical if historical else logical - history_rows
        remaining = history_rows - source if historical else prefix - source
        count = min(16 - row % 16, 16 - source % 16, remaining, rows - destination)
        if count <= 0:
            raise ValueError('direct segment would not advance: tile %d row %d' % (tile, row))
        result.append(('active' if historical else 'delta', source, row, count))
        row += count
    return result


def tile_order(order):
    if order == 'ascending':
        return range(TILES)
    if order == 'descending':             # the negative control: not the kernel's order
        return range(TILES - 1, -1, -1)
    raise ValueError(order)


def mirror_slide(torch, kind, active, delta, out, *, history_rows, prefix, workers=None, order='ascending'):
    """Run a slide kernel's per-worker tile loop on int16 bit tensors ([1, 4, 2048, 128] banks, a
    [1, 4, 32, 128] delta); `out` may BE `active` (in place). Workers run one after another: their pages are
    disjoint. Each iteration reads (behind its read barrier) and only then writes its tile (behind its write
    barrier), so a sequential model of one worker is exact. The kernel's L1 (scalar: the 2-tile scratch and
    the output tile; direct: the output tile) persists across iterations as a STALE fill, so any row a
    kernel never refreshes shows up."""
    shape = geometry(history_rows, prefix)
    drop, rows = shape['drop'], shape['rows']
    for worker in (range(WORKERS) if workers is None else workers):
        head, column = divmod(worker, COLUMNS)
        cols = slice(column * TILE, (column + 1) * TILE)
        if kind == 'scalar':
            _mirror_scalar(torch, active, delta, out, head, cols, history_rows, prefix, drop, rows, order)
        elif kind == 'direct':
            _mirror_direct(torch, active, delta, out, head, cols, history_rows, prefix, drop, rows, order)
        else:
            raise ValueError('no mirror for kernel kind %r' % (kind,))


def _mirror_scalar(torch, active, delta, out, head, cols, history_rows, prefix, drop, rows, order):
    # draft_kv_slide.cpp (bc45d472): the delta tile once into `added`; per tile, source tiles t' and t'+1
    # into scratch (read barrier), rows assembled word by word, the tile written (write barrier).
    added = delta[0, head, 0:TILE, cols].clone()
    scratch = torch.full((2 * TILE, TILE), STALE, dtype=torch.int16)
    row = torch.arange(TILE)
    for tile in tile_order(order):
        source_start = tile * TILE + drop
        source_tile = source_start // TILE
        if source_start < history_rows:
            scratch[:TILE] = active[0, head, source_tile * TILE:(source_tile + 1) * TILE, cols]
            if source_start % TILE != 0 and (source_tile + 1) * TILE < history_rows:
                scratch[TILE:] = active[0, head, (source_tile + 1) * TILE:(source_tile + 2) * TILE, cols]
        destination = tile * TILE + row
        source = destination + drop
        from_history = (destination < rows) & (source < history_rows)
        from_delta = (destination < rows) & ~from_history & (source - history_rows < prefix)
        history_index = ((source // TILE - source_tile) * TILE + source % TILE).clamp(0, 2 * TILE - 1)
        delta_index = (source - history_rows).clamp(0, TILE - 1)
        output = torch.zeros((TILE, TILE), dtype=torch.int16)
        output[from_history] = scratch[history_index[from_history]]
        output[from_delta] = added[delta_index[from_delta]]
        out[0, head, tile * TILE:(tile + 1) * TILE, cols] = output


def _mirror_direct(torch, active, delta, out, head, cols, history_rows, prefix, drop, rows, order):
    # draft_kv_slide_direct.cpp (1679bbd7, the served one): per tile, face-row segments read from the bank
    # or the delta into the output tile (staged through scratch when misaligned: same bytes, same
    # iteration), invalid rows zeroed, read barrier, the tile written, write barrier.
    output = torch.full((TILE, TILE), STALE, dtype=torch.int16)
    for tile in tile_order(order):
        for source, source_row, row, count in direct_segments(history_rows, prefix, drop, rows, tile):
            if source == 'zero':
                output[row:] = 0
            elif source == 'active':
                output[row:row + count] = active[0, head, source_row:source_row + count, cols]
            else:
                output[row:row + count] = delta[0, head, source_row:source_row + count, cols]
        out[0, head, tile * TILE:(tile + 1) * TILE, cols] = output


def worker_accesses(kind, history_rows, prefix, worker, order='ascending'):
    """The pages one worker touches, in issue order: [(tile, reads, write)], a read being ('bank', page) or
    ('delta', page) and the write the output page. Pages are the TensorAccessor page indices the kernels
    compute: (head * 64 + tile) * 4 + column for a bank, head * 4 + column for the delta."""
    shape = geometry(history_rows, prefix)
    drop, rows = shape['drop'], shape['rows']
    head, column = divmod(worker, COLUMNS)

    def page(tile):
        return (head * TILES + tile) * COLUMNS + column

    steps = []
    if kind == 'scalar':
        steps.append((None, {('delta', head * COLUMNS + column)}, None))
    for tile in tile_order(order):
        reads = set()
        if kind == 'scalar':
            source_start = tile * TILE + drop
            source_tile = source_start // TILE
            if source_start < history_rows:
                reads.add(('bank', page(source_tile)))
                if source_start % TILE != 0 and (source_tile + 1) * TILE < history_rows:
                    reads.add(('bank', page(source_tile + 1)))
        elif kind == 'direct':
            for source, source_row, _row, _count in direct_segments(history_rows, prefix, drop, rows, tile):
                if source == 'active':
                    reads.add(('bank', page(source_row // TILE)))
                elif source == 'delta':
                    reads.add(('delta', head * COLUMNS + column))
        else:
            raise ValueError(kind)
        steps.append((tile, reads, page(tile)))
    return steps


def in_place_hazards(kind, history_rows, prefix, worker, order='ascending'):
    """Bank pages read after this worker has already overwritten them, when the output IS the bank: the
    whole in-place condition, given the kernels' per-iteration read and write barriers. [] means safe."""
    written, hazards = set(), []
    for tile, reads, write in worker_accesses(kind, history_rows, prefix, worker, order):
        hazards.extend((tile, page) for source, page in sorted(reads) if source == 'bank' and page in written)
        if write is not None:
            written.add(write)
    return hazards


def io_list(form, banks):
    """generic_op's tensor list for in-place banks [(bank, delta)] in one of FORMS."""
    tensors = []
    for bank, delta in banks:
        if form == 'aliased':
            tensors += [bank, delta, bank]
        elif form == 'pair':
            tensors += [bank, delta]
        elif form == 'pair_out_last':
            tensors += [delta, bank]
        else:
            raise ValueError('unknown io form %r' % (form,))
    return tensors


def served_io(triples):
    """The served driver's list, flattened per bank: [active, delta, spare]."""
    return [value for triple in triples for value in triple]


def groups(items, per_program):
    return [items[index:index + per_program] for index in range(0, len(items), per_program)]


def layout_name(per_program, banks=BANKS):
    return '%dx%d' % (-(-banks // per_program), WORKERS * per_program)


def compare(torch, got, expected):
    """Bit comparison of two int16 tensors of one shape: counts, the first differing index and the first
    differing (head, tile) pairs (where an in-place hazard would show)."""
    if tuple(got.shape) != tuple(expected.shape):
        return dict(exact=False, mismatches=-1, first=None, tiles=[],
                    shape=[list(got.shape), list(expected.shape)])
    differ = got != expected
    count = int(differ.sum())
    first, tiles = None, []
    if count:
        index = int(differ.reshape(-1).nonzero()[0])
        first = []
        for size in reversed(tuple(got.shape)):
            first.insert(0, index % size)
            index //= size
        if tuple(got.shape) == KV_SHAPE:
            hit = differ.reshape(HEADS, TILES, TILE, HEAD_DIM).any(dim=3).any(dim=2)
            tiles = [[int(h), int(t)] for h, t in hit.nonzero()[:8]]
    return dict(exact=count == 0, mismatches=count, first=first, tiles=tiles)


def equal(torch, left, right):
    return bool(tuple(left.shape) == tuple(right.shape) and torch.equal(left, right))


def ms(seconds):
    return round(seconds * 1e3, 4)


def summary(samples):
    if not samples:
        return None
    ordered = sorted(samples)
    return dict(median_ms=ms(statistics.median(ordered)), min_ms=ms(ordered[0]),
                p90_ms=ms(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]), n=len(ordered))


def exact_layouts(entries):
    """Layouts whose served-kernel entries (every prefix and replay run) were all byte-exact."""
    seen = {}
    for entry in entries:
        if entry.get('kernel') == 'served' and 'all_exact' in entry:
            seen[entry['layout']] = seen.get(entry['layout'], True) and bool(entry['all_exact'])
    return {layout for layout, exact in seen.items() if exact}


def decide(report):
    """The S0.1 decision from the sections that ran. go needs: no failure at all; every byte comparison
    exact; the plan's full matrix on the SERVED kernel (history 2048 x prefixes 1..16 x both chips' data,
    so a --quick or narrowed run never says go); an accepted exact io form; the program cache verified;
    and the fastest in-place traced layout among those verified exact both eagerly (multibank) and in
    trace replay at or under GO_MS_PER_USER. A faster layout that was not verified is reported, not used."""
    checks = [c for c in report['cases'] if 'inplace_vs_oop' in c]
    identical = [c for c in checks if c['all_exact']]
    multibank = [m for m in report['multibank'] if 'all_exact' in m]
    traced = [t for t in report['trace'] if 'all_exact' in t]
    bytes_ok = (bool(checks) and len(identical) == len(checks) and all(m['all_exact'] for m in multibank)
                and all(t['all_exact'] for t in traced))
    covered = {(c['prefix'], c['chip']) for c in identical if c.get('kernel') == 'served' and c.get('history') == HISTORY}
    missing = sorted({(prefix, chip) for prefix in PREFIXES for chip in CHIPS} - covered)
    verified = exact_layouts(report['multibank']) & exact_layouts(report['trace'])
    cache = report.get('cache') or {}
    per_user, eligible = {}, {}
    for name, entry in (report.get('timing') or {}).get('variants', {}).items():
        traced_ms = (entry.get('traced') or {}).get('pipelined_ms')
        if entry.get('mode') == 'inplace' and traced_ms is not None:
            per_user[name] = traced_ms
            if entry.get('layout') in verified:
                eligible[name] = traced_ms
    best = min(eligible.items(), key=lambda item: item[1]) if eligible else None
    fastest = min(per_user.items(), key=lambda item: item[1]) if per_user else None
    form = report.get('form')
    reasons = [reason for reason, bad in (
        ('%d failures' % len(report['failures']), bool(report['failures'])),
        ('bytes not identical everywhere', not bytes_ok),
        ('served matrix incomplete: missing (prefix, chip) %s' % missing[:8], bool(missing)),
        ('no in-place io form accepted and exact', not form),
        ('program cache not verified', not cache.get('ok')),
        ('no in-place traced timing on a layout verified exact (multibank and trace)', best is None),
        ('best verified in-place layout over %s ms per user' % GO_MS_PER_USER,
         best is not None and best[1] > GO_MS_PER_USER)) if bad]
    go = not reasons
    return dict(
        served_kind=(report.get('served') or {}).get('kind'),
        bytes_identical=bytes_ok, cases=len(checks), cases_identical=len(identical),
        full_matrix=not missing, missing=[list(pair) for pair in missing],
        multibank_programs=len(multibank), trace_replays=sum(t.get('replays', 0) for t in traced),
        verified_layouts=sorted(verified),
        form=form, forms={name: dict(accepted=entry.get('accepted'), exact=entry.get('exact'))
                          for name, entry in report.get('forms', {}).items()},
        program_cache_ok=cache.get('ok'),
        inplace_traced_ms_per_user=per_user,
        best_inplace=None if best is None else dict(variant=best[0], ms_per_user=best[1]),
        fastest_inplace=None if fastest is None else dict(variant=fastest[0], ms_per_user=fastest[1],
                                                          verified=fastest[0] in eligible),
        go_threshold_ms=GO_MS_PER_USER, go=go, why_not=reasons,
        h1b=('in-place slide traces (F-I) and pairs bound to the live banks (F4)' if go else
             'fallback: out-of-place slides with parity-keyed traces (keep copy_cache); see why_not'))


def verdict(report):
    return not report['failures'] and report.get('cases_run', 0) > 0


# ---------------------------------------------------------------------------------------------
# Programs: draft_kv_slide.prepare's per-chip program, for any number of banks, in or out of place.
# ---------------------------------------------------------------------------------------------

def load_driver(directory):
    """The slide driver module from `directory` (the image's by default), under its own name."""
    path = Path(directory) / 'draft_kv_slide.py'
    spec = importlib.util.spec_from_file_location('served_draft_kv_slide', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(ttnn, value, expected):
    """draft_kv_slide.prepare's tensor check, per shard."""
    if (tuple(value.shape) != expected or value.dtype != ttnn.bfloat16
            or value.layout != ttnn.TILE_LAYOUT or value.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            or tuple(value.tile.tile_shape) != (32, 32)
            or value.tile.transpose_of_faces or value.tile.transpose_within_face):
        raise Unsupported('Exact non-transposed interleaved BF16 cache tiles required')


def single_shard(ttnn, value):
    parts = ttnn.get_device_tensors(value)
    if len(parts) != 1:
        raise Unsupported('one device shard per tensor required (this harness opens one chip)')
    return parts[0]


def build_program(ttnn, grid, kernel_path, banks, *, history_rows, prefix, geometry_of, in_place):
    """One MeshProgramDescriptor at mesh coordinate (0, 0): `banks` is [(active, delta, out)], out being
    active itself in place. Bank i's 16 workers take cores 16 i .. 16 i + 15 in row-major order over the
    grid, with the served runtime args [active, delta, out, history_rows, prefix, drop, rows, worker].
    With one bank out of place this is draft_kv_slide.prepare's program for one chip, field for field."""
    shape = geometry_of(history_rows, prefix)
    needed = WORKERS * len(banks)
    if not banks or grid[0] * grid[1] < needed:
        raise Unsupported('%d banks need %d transport workers; the grid has %d cores'
                          % (len(banks), needed, grid[0] * grid[1]))
    coordinates = [ttnn.CoreCoord(index % grid[0], index // grid[0]) for index in range(needed)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in coordinates])
    buffer = ttnn.CBDescriptor(total_size=CB_BYTES, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=PAGE_BYTES, tile=ttnn.TileDescriptor(ttnn.Tile((32, 32))))])
    compile_args, rows = None, []
    for active, delta, out in banks:
        local = [single_shard(ttnn, value) for value in (active, delta, out)]
        for value, expected in zip(local, (KV_SHAPE, DELTA_SHAPE, KV_SHAPE)):
            validate(ttnn, value, expected)
        addresses = [value.buffer_address() for value in local]
        if in_place:
            if addresses[2] != addresses[0] or addresses[1] == addresses[0]:
                raise Unsupported('In place: the output is the bank, and the delta is not')
        elif len(set(addresses)) != 3:
            raise Unsupported('Active, delta and spare storage must not alias')
        arguments = [argument for value in local for argument in ttnn.TensorAccessorArgs(value).get_compile_time_args()]
        if compile_args is None:
            compile_args = arguments
        elif arguments != compile_args:
            raise Unsupported('banks with different accessor layouts cannot share one kernel')
        rows.append(addresses)
    kernel = ttnn.KernelDescriptor(kernel_source=str(kernel_path), core_ranges=cores,
        compile_time_args=compile_args,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
            noc=ttnn.NOC.RISCV_0_default))
    runtime = ttnn.RuntimeArgs()
    for index, addresses in enumerate(rows):
        for worker in range(WORKERS):
            core = coordinates[index * WORKERS + worker]
            runtime[core.x][core.y] = addresses + [history_rows, prefix, shape['drop'], shape['rows'], worker]
    kernel.runtime_args = runtime
    program = ttnn.MeshProgramDescriptor()
    coordinate = ttnn.MeshCoordinate(0, 0)
    program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[kernel], cbs=[buffer])
    return program


# ---------------------------------------------------------------------------------------------
# Device harness.
# ---------------------------------------------------------------------------------------------

class Watchdog:
    """A per-device-call deadline: a hung NoC handshake cannot be interrupted from Python, so the poller
    prints WATCHDOG, writes the partial report and os._exit(3)s.

    The poll thread needs the GIL, and a blocking ttnn call (a read or a synchronize on a hung program) may
    hold it. So each op also arms a faulthandler backstop, a C thread (as sdpa_prefill_bench and
    sdpa_prefill_chain do): at the op's budget + `grace` it dumps every thread's stack ('Timeout (h:mm:ss)!')
    and exits 1, which run_card_b.sh reads, with 'Timeout (' in the log, as a hang. Leaving a nested op
    re-arms the backstop for the outer op's remaining time; leaving the outermost cancels it.

    Arming restarts faulthandler's thread (~75 us an arm and cancel), which would be charged to every
    launch of a timed loop, so a timed loop runs inside one span: one deadline for the whole loop, and the
    per-call ops inside it cost nothing."""

    def __init__(self, seconds, on_fire=None, grace=60.0, backstop=True):
        self.seconds, self.on_fire, self.grace = seconds, on_fire, grace
        self.backstop = bool(backstop and seconds)
        self.label, self.deadline = None, None
        self.coarse = False
        self.lock = threading.Lock()

    def start(self):
        if self.seconds:
            threading.Thread(target=self.poll, name='slide-watchdog', daemon=True).start()
        return self

    def arm(self, seconds):
        if self.backstop:
            try:
                faulthandler.dump_traceback_later(seconds + self.grace, exit=True, file=sys.stdout)
            except (AttributeError, OSError, RuntimeError, ValueError):
                pass

    def cancel(self):
        if self.backstop:
            try:
                faulthandler.cancel_dump_traceback_later()
            except (AttributeError, OSError, RuntimeError, ValueError):
                pass

    @contextmanager
    def span(self, label, seconds):
        """One deadline of `seconds` (at least the per-call one) over a timed loop."""
        if not self.seconds or self.coarse:
            yield
            return
        with self.op(label, extra=max(0.0, seconds - self.seconds)):
            self.coarse = True
            try:
                yield
            finally:
                self.coarse = False

    @contextmanager
    def op(self, label, extra=0.0):
        if not self.seconds or self.coarse:
            yield
            return
        budget = self.seconds + extra
        with self.lock:
            outer = (self.label, self.deadline)
            self.label, self.deadline = label, time.monotonic() + budget
        self.arm(budget)
        try:
            yield
        finally:
            with self.lock:
                self.label, self.deadline = outer
                remaining = None if outer[1] is None else max(outer[1] - time.monotonic(), 0.0)
            if remaining is None:
                self.cancel()
            else:
                self.arm(remaining)

    def poll(self):
        while True:
            time.sleep(1.0)
            with self.lock:
                label, deadline = self.label, self.deadline
            if label is not None and time.monotonic() >= deadline:
                sys.stdout.write('WATCHDOG: %r did not return within %ss; exiting 3 (docker rm -f, then reset '
                                 'this card only, by the runner\'s printed reset command)\n' % (label, self.seconds))
                sys.stdout.flush()
                try:
                    if self.on_fire is not None:
                        self.on_fire(label)
                finally:
                    os._exit(3)


class Bench:
    def __init__(self, ttnn, torch, device, driver):
        self.ttnn, self.torch, self.device, self.driver = ttnn, torch, device, driver
        grid = device.compute_with_storage_grid_size()
        self.grid = (int(grid.x), int(grid.y))

    # tensors ---------------------------------------------------------------------------------
    def upload(self, bits):
        ttnn = self.ttnn
        with WATCHDOG.op('upload'):
            return ttnn.from_torch(bits.view(self.torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def write(self, tensor, bits):
        """New contents into the SAME buffer (the addresses a program or a trace baked)."""
        ttnn = self.ttnn
        host = ttnn.from_torch(bits.view(self.torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        with WATCHDOG.op('write'):
            ttnn.copy_host_to_device_tensor(host, tensor)

    def read(self, tensor):
        with WATCHDOG.op('read'):
            value = self.ttnn.to_torch(tensor)
        return value.contiguous().view(self.torch.int16).reshape(tuple(tensor.shape))

    def address(self, tensor):
        return single_shard(self.ttnn, tensor).buffer_address()

    def release(self, tensors):
        for value in tensors:
            try:
                self.ttnn.deallocate(value)
            except Exception:  # noqa: BLE001 - teardown
                pass

    # programs ------------------------------------------------------------------------------------
    def program(self, kernel, triples, history_rows, prefix, in_place):
        return build_program(self.ttnn, self.grid, kernel['path'], triples, history_rows=history_rows,
                             prefix=prefix, geometry_of=self.driver.geometry, in_place=in_place)

    def generic(self, tensors, program, label):
        with WATCHDOG.op(label):
            self.ttnn.generic_op(tensors, program)

    def sync(self):
        with WATCHDOG.op('synchronize'):
            self.ttnn.synchronize_device(self.device)

    def entries(self):
        try:
            return int(self.device.num_program_cache_entries())
        except Exception:  # noqa: BLE001 - recorded as unknown
            return None

    # traces ----------------------------------------------------------------------------------------
    def capture(self, enqueue):
        ttnn = self.ttnn
        with WATCHDOG.op('capture'):
            trace = ttnn.begin_trace_capture(self.device, cq_id=0)
            try:
                enqueue()
            finally:
                ttnn.end_trace_capture(self.device, trace, cq_id=0)
        return trace

    def replay(self, trace, blocking=True):
        with WATCHDOG.op('replay'):
            self.ttnn.execute_trace(self.device, trace, cq_id=0, blocking=blocking)

    def release_trace(self, trace):
        with WATCHDOG.op('release trace'):
            self.ttnn.release_trace(self.device, trace)


def added(after, before):
    return None if after is None or before is None else after - before


def seed_of(*parts):
    return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:8], 16)


def resolve_kernels(args, report):
    """The kernels to run, deduplicated by content: the image's served kernel first."""
    kernels, seen = [], {}
    for label in args.kernels:
        if label == 'served':
            path = Path(args.served_dir) / 'draft_kv_slide.cpp'
        else:
            path = Path(args.checkout_dir) / label / 'draft_kv_slide.cpp'
        entry = dict(label=label, path=str(path))
        if not path.is_file():
            entry['skipped'] = 'missing'
            if label == 'served':
                report['failures'].append('the served kernel %s is missing from the image' % path)
            report['kernels'].append(entry)
            continue
        digest = sha256(path)
        entry.update(sha256=digest, kind=KERNEL_KINDS.get(digest, 'unknown'))
        if entry['kind'] == 'unknown':
            report['failures'].append('%s kernel %s (%s) is neither qualified slide kernel: the in-place '
                                      'argument and the mirrors were reviewed for bc45d472 and 1679bbd7 only'
                                      % (label, path, digest[:8]))
        if digest in seen:
            entry['skipped'] = 'same bytes as %s' % seen[digest]
        else:
            seen[digest] = label
            kernels.append(entry)
        report['kernels'].append(entry)
    return kernels


def served_record(args):
    kernel = Path(args.served_dir) / 'draft_kv_slide.cpp'
    driver = Path(args.served_dir) / 'draft_kv_slide.py'
    record = dict(kernel_path=str(kernel), driver_path=str(driver))
    if kernel.is_file():
        digest = sha256(kernel)
        record.update(kernel_sha256=digest, kind=KERNEL_KINDS.get(digest, 'unknown'),
                      kernel_matches_bundle=digest == BUNDLE_KERNEL_SHA256)
    if driver.is_file():
        record.update(driver_sha256=sha256(driver), driver_matches_bundle=sha256(driver) == BUNDLE_DRIVER_SHA256)
    checkout_driver = Path(args.checkout_dir) / 'draft_kv_slide.py'
    if checkout_driver.is_file():
        record['checkout_driver_sha256'] = sha256(checkout_driver)
    return record


# --- sections ------------------------------------------------------------------------------------

def section_forms(bench, kernel, report, args):
    torch = bench.torch
    results, owned = {}, []
    try:
        for index, form in enumerate(FORMS):
            a, d = random_bits(torch, KV_SHAPE, seed_of('form', index)), random_bits(torch, DELTA_SHAPE, seed_of('form-d', index))
            bank, delta = bench.upload(a), bench.upload(d)
            owned += [bank, delta]
            truth_a, truth_d = bench.read(bank), bench.read(delta)
            expected = oracle(torch, truth_a, truth_d, HISTORY, FORM_PREFIX)
            entry = dict(form=form, io=FORM_IO[form], prefix=FORM_PREFIX, kernel=kernel['label'],
                         aliased_address=bench.address(bank))
            # Built outside the try: a refusal here is the harness's (Unsupported), not generic_op's.
            program = bench.program(kernel, [(bank, delta, bank)], HISTORY, FORM_PREFIX, in_place=True)
            before = bench.entries()
            try:
                bench.generic(io_list(form, [(bank, delta)]), program, 'forms %s' % form)
                bench.sync()
            except Exception as error:  # noqa: BLE001 - the refusal is the finding
                entry.update(accepted=False, exact=False, error=repr(error)[:800])
            else:
                result = compare(torch, bench.read(bank), expected)
                entry.update(accepted=True, exact=result['exact'], result=result,
                             delta_unchanged=equal(torch, bench.read(delta), truth_d))
                entry['exact'] = entry['exact'] and entry['delta_unchanged']
            entry['entries_added'] = added(bench.entries(), before)
            results[form] = entry
            print('form %-14s %-28s accepted=%s exact=%s entries+%s %s' % (
                form, FORM_IO[form], entry['accepted'], entry['exact'], entry['entries_added'],
                entry.get('error', '')[:160]), flush=True)
    finally:
        bench.release(owned)
    report['forms'] = results
    if args.form:
        chosen = args.form if results.get(args.form, {}).get('exact') else None
        if chosen is None:
            report['failures'].append('--form %s was not accepted and exact: %s' % (args.form, results.get(args.form)))
    else:
        chosen = next((form for form in FORMS if results[form].get('exact')), None)
        if chosen is None:
            report['failures'].append('generic_op took no in-place io form exactly: %s'
                                      % {k: (v.get('accepted'), v.get('exact')) for k, v in results.items()})
    for form, entry in results.items():
        if entry.get('accepted') and not entry.get('exact'):
            report['failures'].append('io form %s was accepted but the bank is wrong: %s' % (form, entry.get('result')))
    report['form'] = chosen
    return chosen


def case_matrix(args):
    matrix = [(HISTORY, prefix) for prefix in args.prefixes] + list(args.edges)
    return [(history, prefix, chip) for history, prefix in matrix for chip in args.chips]


def section_cases(bench, kernels, form, report, args):
    torch = bench.torch
    zero = random_bits(torch, KV_SHAPE, 1)
    zero_d = random_bits(torch, DELTA_SHAPE, 2)
    active, delta, spare, bank = bench.upload(zero), bench.upload(zero_d), bench.upload(zero), bench.upload(zero)
    report['case_addresses'] = dict(active=bench.address(active), delta=bench.address(delta),
                                    spare=bench.address(spare), bank=bench.address(bank))
    outputs = {}
    try:
        for kernel in kernels:
            for history, prefix, chip in case_matrix(args):
                a = random_bits(torch, KV_SHAPE, seed_of('case', history, prefix, chip))
                d = random_bits(torch, DELTA_SHAPE, seed_of('case-d', history, prefix, chip))
                p = random_bits(torch, KV_SHAPE, seed_of('case-p', history, prefix, chip))
                for tensor, bits in ((active, a), (delta, d), (spare, p), (bank, a)):
                    bench.write(tensor, bits)
                truth_a, truth_d, truth_bank = bench.read(active), bench.read(delta), bench.read(bank)
                entry = dict(kernel=kernel['label'], kind=kernel['kind'], history=history, prefix=prefix, chip=chip,
                             upload_faithful=equal(torch, truth_a, a) and equal(torch, truth_d, d),
                             bank_equals_active=equal(torch, truth_bank, truth_a))
                expected = oracle(torch, truth_a, truth_d, history, prefix)
                oop = bench.program(kernel, [(active, delta, spare)], history, prefix, in_place=False)
                bench.generic([active, delta, spare], oop, 'case oop %d/%d' % (history, prefix))
                inplace = bench.program(kernel, [(bank, delta, bank)], history, prefix, in_place=True)
                bench.generic(io_list(form, [(bank, delta)]), inplace, 'case inplace %d/%d' % (history, prefix))
                bench.sync()
                out_oop, out_in = bench.read(spare), bench.read(bank)
                entry.update(oop_vs_oracle=compare(torch, out_oop, expected),
                             inplace_vs_oop=compare(torch, out_in, out_oop),
                             inplace_vs_oracle=compare(torch, out_in, expected),
                             active_unchanged=equal(torch, bench.read(active), truth_a),
                             delta_unchanged=equal(torch, bench.read(delta), truth_d))
                key = (history, prefix, chip)
                if key in outputs:
                    entry['kernels_agree'] = equal(torch, outputs[key][1], out_oop)
                    entry['agrees_with'] = outputs[key][0]
                else:
                    outputs[key] = (kernel['label'], out_oop)
                entry['all_exact'] = bool(entry['oop_vs_oracle']['exact'] and entry['inplace_vs_oop']['exact']
                                          and entry['inplace_vs_oracle']['exact'] and entry['active_unchanged']
                                          and entry['delta_unchanged'] and entry['bank_equals_active']
                                          and entry.get('kernels_agree', True))
                report['cases'].append(entry)
                report['cases_run'] = report.get('cases_run', 0) + 1
                print('case %-6s h=%4d p=%2d chip=%d oop=%s inplace==oop=%s inplace==oracle=%s delta=%s active=%s%s' % (
                    kernel['label'], history, prefix, chip, entry['oop_vs_oracle']['exact'],
                    entry['inplace_vs_oop']['exact'], entry['inplace_vs_oracle']['exact'], entry['delta_unchanged'],
                    entry['active_unchanged'],
                    ' agrees-with-%s=%s' % (entry['agrees_with'], entry['kernels_agree']) if 'kernels_agree' in entry else ''),
                    flush=True)
                if not entry['all_exact']:
                    report['failures'].append('case %s h=%d p=%d chip=%d: %s' % (
                        kernel['label'], history, prefix, chip,
                        {k: entry[k] for k in ('oop_vs_oracle', 'inplace_vs_oop', 'inplace_vs_oracle', 'active_unchanged',
                                               'delta_unchanged', 'bank_equals_active') if k in entry}))
    finally:
        bench.release([active, delta, spare, bank])


class UserBanks:
    """One user's 10 (bank, delta, spare) sets, allocated once."""

    def __init__(self, bench, tag):
        torch = bench.torch
        self.bench = bench
        self.banks = [bench.upload(random_bits(torch, KV_SHAPE, seed_of(tag, 'b', i))) for i in range(BANKS)]
        self.deltas = [bench.upload(random_bits(torch, DELTA_SHAPE, seed_of(tag, 'd', i))) for i in range(BANKS)]
        self.spares = [bench.upload(random_bits(torch, KV_SHAPE, seed_of(tag, 's', i))) for i in range(BANKS)]

    def fill(self, tag):
        """Fresh random contents written into the same buffers; returns the device truth (bank, delta)."""
        torch, bench = self.bench.torch, self.bench
        truth = []
        for index in range(BANKS):
            a = random_bits(torch, KV_SHAPE, seed_of(tag, 'fb', index))
            d = random_bits(torch, DELTA_SHAPE, seed_of(tag, 'fd', index))
            bench.write(self.banks[index], a)
            bench.write(self.deltas[index], d)
            truth.append((a, d))
        return [(bench.read(b), bench.read(d)) for b, d in zip(self.banks, self.deltas)]

    def restore(self, truth):
        for (a, _), bank in zip(truth, self.banks):
            self.bench.write(bank, a)

    def oop_triples(self):
        return list(zip(self.banks, self.deltas, self.spares))

    def inplace_pairs(self):
        return list(zip(self.banks, self.deltas))

    def close(self):
        self.bench.release(self.banks + self.deltas + self.spares)


def inplace_programs(bench, kernel, pairs, per_program, history, prefix):
    return [(chunk, bench.program(kernel, [(b, d, b) for b, d in chunk], history, prefix, in_place=True))
            for chunk in groups(pairs, per_program)]


def oop_programs(bench, kernel, triples, per_program, history, prefix):
    return [(chunk, bench.program(kernel, chunk, history, prefix, in_place=False))
            for chunk in groups(triples, per_program)]


def section_multibank(bench, kernels, form, report, args):
    torch = bench.torch
    cores = bench.grid[0] * bench.grid[1]
    user = UserBanks(bench, 'multibank')
    try:
        for kernel in kernels:
            for prefix in args.multibank_prefixes:
                truth = user.fill(('multibank', kernel['label'], prefix))
                for chunk, program in oop_programs(bench, kernel, user.oop_triples(), 1, HISTORY, prefix):
                    bench.generic(served_io(chunk), program, 'multibank oop')
                bench.sync()
                references = [bench.read(spare) for spare in user.spares]
                expected = [oracle(torch, a, d, HISTORY, prefix) for a, d in truth]
                for per_program in args.layouts:
                    name = layout_name(per_program)
                    entry = dict(kernel=kernel['label'], prefix=prefix, layout=name, banks_per_program=per_program)
                    if WORKERS * per_program > cores:
                        entry['skipped'] = 'needs %d cores; the grid has %d' % (WORKERS * per_program, cores)
                        report['multibank'].append(entry)
                        continue
                    user.restore(truth)
                    for chunk, program in inplace_programs(bench, kernel, user.inplace_pairs(), per_program, HISTORY, prefix):
                        bench.generic(io_list(form, chunk), program, 'multibank %s' % name)
                    bench.sync()
                    results = []
                    for index, bank in enumerate(user.banks):
                        got = bench.read(bank)
                        results.append(dict(bank=index, vs_oop=compare(torch, got, references[index]),
                                            vs_oracle=compare(torch, got, expected[index])))
                    entry['references_exact'] = all(equal(torch, r, e) for r, e in zip(references, expected))
                    entry['all_exact'] = bool(entry['references_exact'] and all(
                        r['vs_oop']['exact'] and r['vs_oracle']['exact'] for r in results))
                    entry['failed_banks'] = [r for r in results if not (r['vs_oop']['exact'] and r['vs_oracle']['exact'])]
                    report['multibank'].append(entry)
                    print('multibank %-6s p=%2d %-6s exact=%s' % (kernel['label'], prefix, name, entry['all_exact']), flush=True)
                    if not entry['all_exact']:
                        report['failures'].append('multibank %s p=%d %s: %s' % (kernel['label'], prefix, name, entry['failed_banks'][:3]))
    finally:
        user.close()


def section_cache(bench, kernel, form, report, args):
    """Program-cache behaviour of the in-place program (single bank and 5-bank), and the served path's."""
    torch = bench.torch
    steps, owned = [], []
    first_prefix, second_prefix = CACHE_PREFIXES

    def pair(tag):
        a, d = random_bits(torch, KV_SHAPE, seed_of('cache', tag)), random_bits(torch, DELTA_SHAPE, seed_of('cache-d', tag))
        bank, delta = bench.upload(a), bench.upload(d)
        owned.extend([bank, delta])
        return bank, delta

    def refill(bank, delta, tag):
        bench.write(bank, random_bits(torch, KV_SHAPE, seed_of('refill', tag)))
        bench.write(delta, random_bits(torch, DELTA_SHAPE, seed_of('refill-d', tag)))
        return bench.read(bank), bench.read(delta)

    def step(label, chunks, prefix, program=None, untouched=()):
        """Run the in-place program over `chunks` [(bank, delta)]; `untouched` banks must keep their bytes."""
        truths = [refill(bank, delta, (label, index)) for index, (bank, delta) in enumerate(chunks)]
        guards = [(value, bench.read(value)) for value in untouched]
        if program is None:
            program = bench.program(kernel, [(b, d, b) for b, d in chunks], HISTORY, prefix, in_place=True)
        before = bench.entries()
        bench.generic(io_list(form, chunks), program, 'cache %s' % label)
        bench.sync()
        grew = added(bench.entries(), before)
        exact = all(equal(torch, bench.read(bank), oracle(torch, a, d, HISTORY, prefix))
                    for (bank, _), (a, d) in zip(chunks, truths))
        kept = all(equal(torch, bench.read(value), bits) for value, bits in guards)
        entry = dict(step=label, prefix=prefix, banks=len(chunks), entries_added=grew, exact=exact,
                     untouched_kept=kept, addresses=[bench.address(bank) for bank, _ in chunks])
        steps.append(entry)
        print('cache %-34s entries+%s exact=%s untouched=%s' % (label, grew, exact, kept), flush=True)
        return program

    try:
        x = pair('x')
        program = step('first call (bank X)', [x], first_prefix)
        step('repeat: same program object', [x], first_prefix, program=program)
        step('repeat: rebuilt descriptor, same bank', [x], first_prefix)
        y = pair('y')
        if bench.address(y[0]) == bench.address(x[0]):
            report['failures'].append('cache: bank Y did not get a fresh address')
        step('fresh addresses (bank Y)', [y], first_prefix, untouched=[x[0]])
        step('new prefix (bank Y)', [y], second_prefix, untouched=[x[0]])
        step('back to bank X', [x], second_prefix, untouched=[y[0]])
        if WORKERS * 5 <= bench.grid[0] * bench.grid[1]:
            set_a = [pair(('a', i)) for i in range(5)]
            step('5-bank program, first (set A)', set_a, first_prefix)
            set_b = [pair(('b', i)) for i in range(5)]
            step('5-bank program, fresh set B', set_b, first_prefix, untouched=[bank for bank, _ in set_a])
            step('5-bank program, new prefix', set_b, second_prefix, untouched=[bank for bank, _ in set_a])
        # The served out-of-place path, for the record: a descriptor per call, fresh spares.
        s1, s2 = bench.upload(random_bits(torch, KV_SHAPE, 71)), bench.upload(random_bits(torch, KV_SHAPE, 72))
        owned.extend([s1, s2])
        served = []
        for label, spare, prefix in (('served first', s1, first_prefix), ('served fresh spare', s2, first_prefix),
                                     ('served new prefix', s2, second_prefix)):
            before = bench.entries()
            bench.generic([x[0], x[1], spare], bench.program(kernel, [(x[0], x[1], spare)], HISTORY, prefix, in_place=False),
                          'cache %s' % label)
            bench.sync()
            served.append(dict(step=label, entries_added=added(bench.entries(), before)))
    finally:
        bench.release(owned)
    repeats = [s for s in steps if not s['step'].startswith(('first call', '5-bank program, first'))]
    known = all(s['entries_added'] is not None for s in steps)
    # An unknown entry count leaves "the second call adds no entry" unverified, so it is not ok.
    ok = (all(s['exact'] and s['untouched_kept'] for s in steps) and known
          and all(s['entries_added'] == 0 for s in repeats))
    if not known:
        report['failures'].append('program cache: num_program_cache_entries() is unavailable, so "a repeat adds '
                                  'no entry" is unverified')
    report['cache'] = dict(kernel=kernel['label'], form=form, steps=steps, served=served, ok=ok,
                           entries_known=known,
                           fresh_addresses_honoured=all(s['exact'] and s['untouched_kept'] for s in steps
                                                        if 'fresh' in s['step'] or 'back to' in s['step']),
                           prefix_honoured=all(s['exact'] for s in steps if 'new prefix' in s['step']),
                           repeat_adds_no_entry=all(s['entries_added'] in (0, None) for s in repeats))
    bad = [s for s in steps if not (s['exact'] and s['untouched_kept'])
           or (s in repeats and s['entries_added'] not in (0, None))]
    if bad:
        report['failures'].append('program cache: %s' % bad)


def section_trace(bench, kernel, form, report, args):
    torch = bench.torch
    cores = bench.grid[0] * bench.grid[1]
    user = UserBanks(bench, 'trace')
    try:
        for per_program in args.trace_layouts:
            name = layout_name(per_program)
            entry = dict(kernel=kernel['label'], layout=name, prefix=TRACE_PREFIX)
            if WORKERS * per_program > cores:
                entry['skipped'] = 'needs %d cores; the grid has %d' % (WORKERS * per_program, cores)
                report['trace'].append(entry)
                continue
            programs = inplace_programs(bench, kernel, user.inplace_pairs(), per_program, HISTORY, TRACE_PREFIX)

            def enqueue():
                for chunk, program in programs:
                    bench.generic(io_list(form, chunk), program, 'trace %s' % name)

            user.fill(('trace-warm', name))
            enqueue()                       # compile before capture
            bench.sync()
            trace = bench.capture(enqueue)
            try:
                before = bench.entries()
                replays = []
                for replay in range(args.trace_replays):
                    truth = user.fill(('trace', name, replay))
                    bench.replay(trace, blocking=True)
                    exact = [equal(torch, bench.read(bank), oracle(torch, a, d, HISTORY, TRACE_PREFIX))
                             for bank, (a, d) in zip(user.banks, truth)]
                    replays.append(dict(replay=replay, exact=all(exact), failed_banks=[i for i, e in enumerate(exact) if not e]))
                entry.update(replays=len(replays), results=replays, entries_added=added(bench.entries(), before),
                             all_exact=all(r['exact'] for r in replays))
            finally:
                bench.release_trace(trace)
            report['trace'].append(entry)
            print('trace %-6s %-6s replays=%d exact=%s entries+%s' % (kernel['label'], name, entry['replays'],
                                                                      entry['all_exact'], entry['entries_added']), flush=True)
            if not entry['all_exact'] or entry['entries_added'] not in (0, None):
                report['failures'].append('trace %s: %s' % (name, entry))
    finally:
        user.close()


def section_timing(bench, kernel, form, report, args):
    cores = bench.grid[0] * bench.grid[1]
    prefix = args.timing_prefix
    user = UserBanks(bench, 'timing')
    variants = {}
    load = {}
    try:
        load['start'] = os.getloadavg() if hasattr(os, 'getloadavg') else None
        user.fill('timing')
        plans = [('served_rebuild_%s' % layout_name(1), 'oop', 1, True)]
        for mode in ('oop', 'inplace'):
            for per_program in args.timing_layouts:
                plans.append(('%s_%s' % (mode, layout_name(per_program)), mode, per_program, False))
        for name, mode, per_program, rebuild in plans:
            entry = dict(mode=mode, layout=layout_name(per_program), programs=-(-BANKS // per_program),
                         workers_per_program=WORKERS * per_program, rebuild_per_call=rebuild, prefix=prefix)
            if WORKERS * per_program > cores:
                entry['skipped'] = 'needs %d cores; the grid has %d' % (WORKERS * per_program, cores)
                variants[name] = entry
                continue
            if mode == 'oop':
                programs = [(served_io(chunk), program) for chunk, program in
                            oop_programs(bench, kernel, user.oop_triples(), per_program, HISTORY, prefix)]
            else:
                programs = [(io_list(form, chunk), program) for chunk, program in
                            inplace_programs(bench, kernel, user.inplace_pairs(), per_program, HISTORY, prefix)]

            if rebuild:
                def enqueue():
                    for active, delta, spare in user.oop_triples():
                        bench.generic([active, delta, spare],
                                      bench.program(kernel, [(active, delta, spare)], HISTORY, prefix, in_place=False),
                                      'timing %s' % name)
            else:
                def enqueue(programs=programs):
                    for tensors, program in programs:
                        bench.generic(tensors, program, 'timing %s' % name)

            # One deadline for the whole measured loop (Watchdog.span): no per-launch watchdog cost.
            with WATCHDOG.span('timing %s' % name, TIMING_SPAN_S):
                for _ in range(args.warmup):
                    enqueue()
                    bench.sync()
                synced, enqueued = [], []
                for _ in range(args.iters):
                    start = time.perf_counter()
                    enqueue()
                    middle = time.perf_counter()
                    bench.sync()
                    synced.append(time.perf_counter() - start)
                    enqueued.append(middle - start)
                start = time.perf_counter()
                for _ in range(args.iters):
                    enqueue()
                bench.sync()
                pipelined = (time.perf_counter() - start) / max(1, args.iters)
                entry['eager'] = dict(synced=summary(synced), enqueue=summary(enqueued), pipelined_ms=ms(pipelined))
                if not rebuild:
                    trace = bench.capture(enqueue)
                    try:
                        for _ in range(max(1, args.warmup)):
                            bench.replay(trace, blocking=True)
                        blocking = []
                        for _ in range(args.iters):
                            start = time.perf_counter()
                            bench.replay(trace, blocking=True)
                            blocking.append(time.perf_counter() - start)
                        start = time.perf_counter()
                        for _ in range(args.iters):
                            bench.replay(trace, blocking=False)
                        bench.sync()
                        replayed = (time.perf_counter() - start) / max(1, args.iters)
                        entry['traced'] = dict(blocking=summary(blocking), pipelined_ms=ms(replayed))
                    finally:
                        bench.release_trace(trace)
            variants[name] = entry
            print('timing %-22s eager synced %s ms, enqueue %s ms, pipelined %s ms; traced blocking %s ms, pipelined %s ms' % (
                name, entry['eager']['synced']['median_ms'], entry['eager']['enqueue']['median_ms'],
                entry['eager']['pipelined_ms'], (entry.get('traced') or {}).get('blocking', {}).get('median_ms'),
                (entry.get('traced') or {}).get('pipelined_ms')), flush=True)
        load['end'] = os.getloadavg() if hasattr(os, 'getloadavg') else None
    finally:
        user.close()
    report['timing'] = dict(kernel=kernel['label'], form=form, prefix=prefix, banks=BANKS, iters=args.iters,
                            warmup=args.warmup, loadavg=load, variants=variants,
                            note='ms per user (10 banks); card B shares the rig host with CI (loadavg recorded)')


def run(args, report):
    import torch
    import ttnn

    report['served'] = served_record(args)
    served = report['served']
    print('served kernel %s (%s) bundle=%s; driver %s bundle=%s' % (
        served.get('kernel_sha256', 'missing')[:8], served.get('kind'), served.get('kernel_matches_bundle'),
        served.get('driver_sha256', 'missing')[:8], served.get('driver_matches_bundle')), flush=True)
    if served.get('kernel_sha256') and not served.get('kernel_matches_bundle'):
        print('WARNING: the image kernel is not serving-bundle 35489235797\'s (%s)' % BUNDLE_KERNEL_SHA256[:8], flush=True)
    kernels = resolve_kernels(args, report)
    if not kernels:
        report['failures'].append('no kernel to run')
        return
    driver = load_driver(args.served_dir)
    with WATCHDOG.op('open device', extra=OPEN_EXTRA_S):      # firmware JITs into the fresh kernel cache
        device = ttnn.open_device(device_id=args.device_id, trace_region_size=args.trace_region)
    try:
        try:
            device.enable_program_cache()
            report['program_cache_enabled_call'] = True
        except Exception as error:  # noqa: BLE001 - default-on in newer runtimes
            report['program_cache_enabled_call'] = repr(error)[:200]
        bench = Bench(ttnn, torch, device, driver)
        report['grid'] = list(bench.grid)
        print('grid %s = %d cores; kernels %s' % (bench.grid, bench.grid[0] * bench.grid[1],
                                                   [(k['label'], k['kind']) for k in kernels]), flush=True)
        served_kernel = kernels[0]
        if 'forms' in args.sections:
            form = section_forms(bench, served_kernel, report, args)
        else:
            form = report['form'] = args.form or 'aliased'     # unprobed: the served arity
        if form is None:
            return
        if 'cases' in args.sections:
            section_cases(bench, kernels, form, report, args)
        if 'multibank' in args.sections:
            section_multibank(bench, kernels, form, report, args)
        if 'cache' in args.sections:
            section_cache(bench, served_kernel, form, report, args)
        if 'trace' in args.sections:
            section_trace(bench, served_kernel, form, report, args)
        if 'timing' in args.sections and not args.no_timing:
            section_timing(bench, served_kernel, form, report, args)
    finally:
        with WATCHDOG.op('close device'):
            ttnn.close_device(device)


def parse_pairs(text):
    pairs = []
    for item in text.split(','):
        if item:
            history, prefix = item.split(':')
            pairs.append((int(history), int(prefix)))
    return pairs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--served-dir', type=Path, default=SERVED_DIR, help='the image\'s draft_kv_slide.{py,cpp}')
    parser.add_argument('--checkout-dir', type=Path, default=CHECKOUT_DIR,
                        help='scalar/draft_kv_slide.cpp, direct/draft_kv_slide.cpp, draft_kv_slide.py (mounted)')
    parser.add_argument('--kernels', default=','.join(KERNEL_LABELS))
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--trace-region', type=int, default=32 << 20)
    parser.add_argument('--sections', default=','.join(SECTIONS))
    parser.add_argument('--form', choices=FORMS, help='force the in-place io form (default: the first exact one)')
    parser.add_argument('--prefixes', default=','.join(map(str, PREFIXES)))
    parser.add_argument('--chips', default=','.join(map(str, CHIPS)))
    parser.add_argument('--edges', default=','.join('%d:%d' % pair for pair in EDGES), help='history:prefix,...')
    parser.add_argument('--layouts', default=','.join(map(str, LAYOUTS)), help='banks per program')
    parser.add_argument('--multibank-prefixes', default=','.join(map(str, MULTIBANK_PREFIXES)))
    parser.add_argument('--trace-layouts', default=','.join(map(str, LAYOUTS)),
                        help='banks per program; a timed in-place layout counts for the decision only if traced here')
    parser.add_argument('--trace-replays', type=int, default=2)
    parser.add_argument('--timing-layouts', default='1,2,5,10')
    parser.add_argument('--timing-prefix', type=int, default=TIMING_PREFIX)
    parser.add_argument('--quick', action='store_true', help='the watcher pass: a thin matrix')
    parser.add_argument('--no-timing', action='store_true')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--watchdog', type=float, default=0)
    args = parser.parse_args(argv)
    if args.quick:                  # thin defaults, re-parsed so explicit arguments still win
        parser.set_defaults(**QUICK)
        args = parser.parse_args(argv)
    args.kernels = [value for value in args.kernels.split(',') if value]
    args.sections = [value for value in args.sections.split(',') if value]
    args.prefixes = [int(value) for value in args.prefixes.split(',') if value]
    args.chips = [int(value) for value in args.chips.split(',') if value]
    args.edges = parse_pairs(args.edges)
    for name in ('layouts', 'multibank_prefixes', 'trace_layouts', 'timing_layouts'):
        setattr(args, name, [int(value) for value in getattr(args, name).split(',') if value])
    unknown = (set(args.sections) - set(SECTIONS)) | (set(args.kernels) - set(KERNEL_LABELS))
    bad = ([p for p in args.prefixes + args.multibank_prefixes + [args.timing_prefix] if not 1 <= p <= 16]
           + [pair for pair in args.edges if min(CAPACITY, pair[0] + pair[1]) != CAPACITY or not 1 <= pair[1] <= 16]
           + [n for n in args.layouts + args.trace_layouts + args.timing_layouts if n not in LAYOUTS])
    if unknown or bad or 'served' not in args.kernels[:1]:
        parser.error('unknown %s, or out-of-range %s (prefixes 1..16, edges with rows == 2048, layouts in %s, '
                     'served first in --kernels)' % (sorted(unknown), bad, LAYOUTS))
    return args


def main(argv=None):
    global WATCHDOG
    args = parse_args(argv)
    report = dict(harness='draft_slide_inplace_card_b', plan='round-fence plan S0.1 (M-F0)', passed=False,
                  failures=[], kernels=[], forms={}, form=None, cases=[], multibank=[], cache={}, trace=[],
                  timing={}, args={k: str(v) for k, v in vars(args).items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write(extra=None):
        payload = dict(report)
        payload['decision'] = decide(report)
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
    decision = decide(report)
    print('DECISION go=%s bytes_identical=%s (%d/%d cases, full matrix %s) form=%s cache_ok=%s best_inplace=%s -> %s' % (
        decision['go'], decision['bytes_identical'], decision['cases_identical'], decision['cases'],
        decision['full_matrix'], decision['form'], decision['program_cache_ok'], decision['best_inplace'],
        decision['h1b']), flush=True)
    for reason in decision['why_not']:
        print('  not go: %s' % reason, flush=True)
    print('PASSED' if report['passed'] else 'FAILED: %d failures' % len(report['failures']), flush=True)
    for failure in report['failures'][:40]:
        print('  - %s' % failure, flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
