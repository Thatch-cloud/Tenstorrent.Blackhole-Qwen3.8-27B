"""S1 on ONE card: the tile-native shard argmax against the served Untilize + ArgMax + Gather, byte for byte, and timed.

Run with QWEN_FAST_TP=4 in the environment (the geometry is the four-card shard's: 62,080 vocabulary columns a chip) and QWEN_FAST_TP4_SHARD_VALUES=1 (the
served arm is the shipped one, with the V4a gather). One p150a, a 1x1 mesh, no model and no collective: the launches are per chip, so one chip proves the
kernels compile, run and move the same bytes; whether four chips of a mesh agree is the audited attach's job (every chip, every round).

BYTES. For each rows (64: the packed block; 32 and 16: the lone and ramp shapes), regime and seed the SAME logits (1, 1, rows, 62,080) bf16 TILE are
uploaded once and read back once (the reference is what the DEVICE holds, so a bit the upload altered cannot fake a difference), and then answered by

  - torch.argmax / the winning element's bits on the read-back logits (the reference);
  - the served composition (verify_trace_t1.served_shards: untilize, ttnn.argmax, the V4a gather);
  - S1 with the one-core fold, and S1 with the two-level tree fold (tp4_shard_argmax.launch);

and every arm's ids are compared with the reference exactly, its values bit for bit (the served arm as numbers: -0 equals +0, NaN equals NaN), and the
joined words of the S1 arms against their own ids and value bits. Regimes: random (real logit scale), peaked (one clear maximum a row), ties (equal maxima
planted at the columns where a tile face, a tile, a worker's run, a RISC-V's half and a fold group each end), negatives (all-negative rows with ties),
zeros (rows whose maximum is -0 and +0), special (infinities and NaN: against torch only, the served arm's answer is recorded). A regime the upload or
the read-back altered is reported and left out of the verdict.

TIMING. Per rows and arm a captured trace of LAUNCHES launches (eager timing would measure the host's program-descriptor build, not the card), over
serpentine rounds of replays; the per-launch median and quartiles of: served (untilize + argmax + gather), scan (the scan launch alone), fold1 (scan + the
one-core fold) and fold2 (scan + the tree fold). The folds' own costs are fold1 - scan and fold2 - scan; what S1 saves is served - fold1 or served - fold2.
Verdicts on the 64-row case: S1-WIN when the best S1 arm is at least WIN_US faster than the served composition; FOLD2-WIN / NEUTRAL / LOSS by the
tree's gain over the one-core fold against FOLD2_US.

Exit: 0 PASS (every compare of the verdict exact); 1 FAIL; 3 the watchdog; 4 NOT-RUN (a section raised). The last stdout line is one JSON object (kind
shard-argmax-card-m) and the line above it 'SHARD_ARGMAX verdict=...'.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time

KIND = 'shard-argmax-card-m'
ROWS = (64, 32, 16)
REGIMES = ('random', 'peaked', 'ties', 'negatives', 'zeros', 'special')
INFORMATIONAL = ('special',)             # compared with torch and recorded, but a served-arm difference is not a failure
FOLDS = ('single', 'tree')
LAUNCHES = 24
ROUNDS = 25
WIN_US = 600.0
FOLD2_US = 40.0
WATCHDOG_S = 2400
TRACE_REGION = 64 * 1024 * 1024
SHARD = 62080
KERNELS = ('tp4_shard_argmax_scan.cpp', 'tp4_shard_argmax_fold.cpp', 'tp4_shard_argmax_fold2.cpp')


def kernel_sha256(directory):
    """sha256 of each kernel source beside the module (the card's first compile of the fold2 text is this run's)."""
    found = {}
    for name in KERNELS:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            with open(path, 'rb') as handle:
                found[name] = hashlib.sha256(handle.read()).hexdigest()
    return found


def tie_pairs(sarg, workers):
    """(a, b) column pairs, a < b adjacent across a boundary the kernels cut at: a face (15/16), a tile (31/32), every worker's run, the middle
    of every run (the RISC-V halves up to 32 rows) and every fold group's block. Columns are shard-local."""
    tile_columns = SHARD // 32
    pairs = [(15, 16), (31, 32), (47, 48), (SHARD - 33, SHARD - 32), (SHARD - 2, SHARD - 1)]
    runs = sarg.column_runs(tile_columns, workers)
    for first, last in runs:
        if first:
            pairs.append((32 * first - 1, 32 * first))
        middle = first + (last - first + 1) // 2
        pairs.append((32 * middle - 1, 32 * middle))
    for per_tile_row in (workers, 2 * workers):
        for _group, first_task, _last_task in sarg.group_plan(per_tile_row):
            # the first column of a group's first task: tasks of one tile row follow the runs in order (64 rows) or the halves (up to 32)
            column = 32 * runs[min(first_task, workers - 1)][0]
            if column:
                pairs.append((column - 1, column))
    return sorted(set(pairs))


def host_logits(torch, rows, regime, seed, pairs):
    """The seeded (rows, 62,080) bfloat16 host logits of one case."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(rows, SHARD, generator=generator) * 3.0
    if regime == 'random':
        matrix = base
    elif regime == 'peaked':
        matrix = base * 0.5
        peaks = torch.randint(0, SHARD, (rows,), generator=generator)
        for row in range(rows):
            matrix[row, int(peaks[row])] = float(matrix[row].max()) + 8.0
    elif regime == 'ties':
        matrix = base.clamp(max=20.0)
        for row in range(rows):
            a, b = pairs[(row * 13 + seed) % len(pairs)]
            matrix[row, a] = 25.0
            matrix[row, b] = 25.0
            if row % 3 == 0:
                matrix[row, (a * 7 + 11) % SHARD] = 25.0                  # a third, far copy
    elif regime == 'negatives':
        matrix = -(base.abs()) - 0.01
        for row in range(rows):
            a, b = pairs[(row * 5 + seed) % len(pairs)]
            matrix[row, a] = -0.0009765625
            matrix[row, b] = -0.0009765625
    elif regime == 'zeros':
        matrix = -(base.abs()) - 1.0
        for row in range(rows):
            a, b = pairs[(row * 3 + seed) % len(pairs)]
            first, second = (-0.0, 0.0) if row % 2 else (0.0, -0.0)
            matrix[row, a] = first
            matrix[row, b] = second
    elif regime == 'special':
        matrix = base.clone()
        for row in range(rows):
            a, b = pairs[(row * 11 + seed) % len(pairs)]
            kind = row % 4
            if kind == 0:
                matrix[row, a] = float('inf')
            elif kind == 1:
                matrix[row, b] = float('inf')
                matrix[row, a] = float('nan')
            elif kind == 2:
                matrix[row, a] = float('nan')
                matrix[row, b] = float('nan')
            else:
                matrix[row] = float('-inf')
                matrix[row, b] = float('-inf')
    else:
        raise ValueError('unknown regime %r' % (regime,))
    return matrix.to(torch.bfloat16).reshape(1, 1, rows, SHARD)


def bits16(torch, tensor):
    """int16 bit patterns (as int32 0..65535) of a bfloat16 tensor."""
    return tensor.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def reference_of(torch, readback):
    """(ids, value bits) of torch.argmax on the (rows, SHARD) bf16 read-back logits."""
    matrix = readback.reshape(-1, SHARD)
    ids = torch.argmax(matrix, dim=1)
    chosen = matrix[torch.arange(matrix.shape[0]), ids]
    return ids.to(torch.int64), bits16(torch, chosen).to(torch.int64)


def numbers_equal(torch, left_bits, right_bits):
    """Rows whose two bit patterns are the same number as bf16 (-0 equals +0) or both NaN: a bool tensor."""
    left = left_bits.to(torch.int32).to(torch.int16).view(torch.bfloat16).to(torch.float32)
    right = right_bits.to(torch.int32).to(torch.int16).view(torch.bfloat16).to(torch.float32)
    return (left == right) | (torch.isnan(left) & torch.isnan(right))


def summarize(values):
    ordered = sorted(values)
    quarter = lambda fraction: ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
    return dict(n=len(ordered), median_us=round(statistics.median(ordered), 2), q1_us=round(quarter(0.25), 2), q3_us=round(quarter(0.75), 2),
                min_us=round(ordered[0], 2))


def verdict(sections):
    """(PASS | FAIL | NOT-RUN, exit status) from the compare sections. A section that raised is NOT-RUN; an informational or uncertain section never
    fails the run on the served arm's answer, but a differing S1 answer against torch fails it in every regime the upload left alone."""
    if not sections or any(section.get('error') for section in sections):
        return 'NOT-RUN', 4
    for section in sections:
        if section.get('uncertain'):
            continue
        if section.get('differing', 1) or section.get('words_differing', 1) or section.get('fell_back'):
            return 'FAIL', 1
        if section.get('served_differing', 0) and not section.get('informational'):
            return 'FAIL', 1
    return 'PASS', 0


def timing_verdict(timing):
    """(S1 verdict, fold verdict) from the 64-row timing entry: S1-WIN / S1-NO-WIN and FOLD2-WIN / FOLD2-NEUTRAL / FOLD2-LOSS."""
    entry = next((item for item in timing if item['rows'] == 64), None)
    if entry is None:
        return 'NOT-TIMED', 'NOT-TIMED'
    served = entry['served']['median_us']
    one, two = entry['fold1']['median_us'], entry['fold2']['median_us']
    win = 'S1-WIN' if served - min(one, two) >= WIN_US else 'S1-NO-WIN'
    gain = one - two
    fold = 'FOLD2-WIN' if gain >= FOLD2_US else ('FOLD2-LOSS' if gain <= -FOLD2_US else 'FOLD2-NEUTRAL')
    return win, fold


class Rig(object):
    """The device side of one case. `ttnn` is the module (a fake in the tests), `sarg` tp4_shard_argmax, `t1` verify_trace_t1."""

    def __init__(self, ttnn, mesh, torch, sarg, t1, grid):
        self.ttnn, self.mesh, self.torch, self.sarg, self.t1, self.grid = ttnn, mesh, torch, sarg, t1, grid
        self.workers = grid[0] * grid[1]
        self.partials = self.groups = None

    def reserve(self):
        ttnn, mesh = self.ttnn, self.mesh
        make = lambda shape: ttnn.empty(shape, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.partials = make((1, 1, 2 * self.workers, self.sarg.PAGE_WORDS))
        self.groups = make((1, 1, self.sarg.FOLD2_GROUPS, self.sarg.GROUP_PAGE_WORDS))

    def upload(self, host):
        ttnn = self.ttnn
        return ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))

    def outputs(self):
        ttnn, mesh = self.ttnn, self.mesh
        ids = ttnn.empty((1, 1, 1, 64), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        values = ttnn.empty((1, 1, 1, 64), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        words = ttnn.empty((1, 1, 1, 64), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ids, values, words

    def served(self, logits, rows):
        return self.t1.served_shards(self.ttnn, logits, rows)

    def s1(self, logits, rows, outputs, fold, scan_only=False):
        ids, values, words = outputs
        return self.sarg.launch(self.ttnn, logits, rows, self.partials, self.groups, ids, values, words, self.grid, fold == 'tree', scan_only)

    def read(self, tensor):
        return self.ttnn.to_torch(self.ttnn.get_device_tensors(tensor)[0])


def release_all(rig, *groups):
    """Deallocate every device tensor in the (nested) groups once, so a long run does not grow the card's DRAM case over case."""
    seen = set()

    def walk(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif value is not None and id(value) not in seen:
            seen.add(id(value))
            try:
                rig.ttnn.deallocate(value)
            except BaseException:  # noqa: BLE001
                pass

    walk(list(groups))


def compare_case(rig, host, rows, regime, folds=FOLDS):
    """One compare section per fold: the S1 answer against torch on the read-back logits, against the served composition and its words against
    its own ids and value bits. Returns the list of sections (one per fold)."""
    torch = rig.torch
    logits = rig.upload(host)
    served = ()
    sections = []
    try:
        readback = rig.read(logits).reshape(-1, SHARD)[:rows]
        uncertain = not torch.equal(bits16(torch, readback), bits16(torch, host.reshape(rows, SHARD)))
        want_ids, want_bits = reference_of(torch, readback)
        served = rig.served(logits, rows)
        served_ids = rig.read(served[0]).reshape(-1)[:rows].to(torch.int64)
        served_bits = bits16(torch, rig.read(served[1]).reshape(-1)[:rows]).to(torch.int64)
        served_ok = bool(((served_ids == want_ids) & numbers_equal(torch, served_bits, want_bits)).all())
        for fold in folds:
            outputs = rig.outputs()
            try:
                rig.s1(logits, rows, outputs, fold)
                ids = rig.read(outputs[0]).reshape(-1)
                values = rig.read(outputs[1]).reshape(-1)
                words = rig.read(outputs[2]).reshape(-1)
                got_ids = ids[:rows].to(torch.int64) & 0xFFFFFFFF
                got_bits = bits16(torch, values[:rows]).to(torch.int64)
                wrong = (got_ids != want_ids) | (got_bits != want_bits)
                packed = rig.sarg.pack_words(got_ids, values[:rows])
                words_wrong = (words[:rows].to(torch.int64) & 0xFFFFFFFF) != packed
                tail = int((ids[rows:] != 0).sum()) + int((bits16(torch, values[rows:]) != 0).sum()) + int((words[rows:] != 0).sum())
                sections.append(dict(
                    rows=rows, regime=regime, fold=fold, differing=int(wrong.sum()), words_differing=int(words_wrong.sum()) + tail,
                    differing_rows=[int(row) for row in torch.nonzero(wrong).reshape(-1)[:8]],
                    served_differing=0 if served_ok else int((~((served_ids == want_ids) & numbers_equal(torch, served_bits, want_bits))).sum()),
                    informational=regime in INFORMATIONAL, uncertain=bool(uncertain), fell_back=False))
            finally:
                release_all(rig, outputs)
    finally:
        release_all(rig, logits, served)
    return sections


def capture(rig, run, launches):
    """Capture `launches` back-to-back launches of one arm in one trace (the programs are compiled by a warm call beforehand, as the served stack
    does before its capture); returns (trace handle, the outputs the capture allocated, which stay alive until the trace is released)."""
    ttnn, mesh = rig.ttnn, rig.mesh
    handle = ttnn.begin_trace_capture(mesh, cq_id=0)
    kept = []
    try:
        try:
            for _ in range(launches):
                kept.append(run())
        finally:
            ttnn.end_trace_capture(mesh, handle, cq_id=0)
    except BaseException:
        ttnn.release_trace(mesh, handle)
        release_all(rig, kept)
        raise
    return handle, kept


def time_case(rig, host, rows, launches=None, rounds=None, clock=None):
    """Per-launch microseconds of the served composition, the scan alone, scan + one-core fold and scan + tree fold, each arm under a captured
    trace of `launches` launches, over serpentine rounds of replays."""
    ttnn, mesh = rig.ttnn, rig.mesh
    launches = LAUNCHES if launches is None else launches
    rounds = ROUNDS if rounds is None else rounds
    clock = time.perf_counter if clock is None else clock
    logits = rig.upload(host)
    outputs = rig.outputs()
    runs = dict(served=lambda: rig.served(logits, rows),
                scan=lambda: rig.s1(logits, rows, outputs, 'single', scan_only=True),
                fold1=lambda: rig.s1(logits, rows, outputs, 'single'),
                fold2=lambda: rig.s1(logits, rows, outputs, 'tree'))
    names = list(runs)
    samples = {name: [] for name in names}
    traces, kept = {}, []
    try:
        for name in names:
            warm = runs[name]()                                                  # the warm call: compile outside the capture
            release_all(rig, warm if name == 'served' else None)                 # (the S1 arms return a launch count, not tensors)
            ttnn.synchronize_device(mesh)
            traces[name], produced = capture(rig, runs[name], launches)
            kept.append(produced if name == 'served' else None)
            ttnn.synchronize_device(mesh)
        for name in names:                                                       # one untimed replay each
            ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        for index in range(rounds):
            order = names if index % 2 == 0 else list(reversed(names))
            for name in order:
                started = clock()
                ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                samples[name].append((clock() - started) / launches * 1e6)
    finally:
        for handle in traces.values():
            try:
                ttnn.release_trace(mesh, handle)
            except BaseException:  # noqa: BLE001
                pass
        release_all(rig, logits, outputs, kept)
    summary = {name: summarize(samples[name]) for name in names}
    return dict(rows=rows, mode='trace', launches=launches, **summary,
                fold1_cost_us=round(summary['fold1']['median_us'] - summary['scan']['median_us'], 2),
                fold2_cost_us=round(summary['fold2']['median_us'] - summary['scan']['median_us'], 2),
                s1_saves_us=round(summary['served']['median_us'] - min(summary['fold1']['median_us'], summary['fold2']['median_us']), 2),
                fold2_gain_us=round(summary['fold1']['median_us'] - summary['fold2']['median_us'], 2))


def main(argv=None, torch=None, ttnn=None, sarg=None, t1=None, tp_shapes=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--rows', default=','.join(str(value) for value in ROWS))
    parser.add_argument('--regimes', default=','.join(REGIMES))
    parser.add_argument('--seeds', default='17,23')
    parser.add_argument('--folds', default=','.join(FOLDS))
    parser.add_argument('--timing', choices=('on', 'always', 'off'), default='on')
    options = parser.parse_args(argv)
    row_cases = [int(value) for value in options.rows.split(',')]
    regimes = options.regimes.split(',')
    seeds = [int(value) for value in options.seeds.split(',')]
    folds = options.folds.split(',')
    if any(value not in (1, 4, 16, 32, 33, 48, 64) for value in row_cases) or any(value not in REGIMES for value in regimes) \
            or any(value not in FOLDS for value in folds):
        print('refusing: rows are 1 4 16 32 33 48 64, regimes %s, folds single tree' % ' '.join(REGIMES), file=sys.stderr)
        return 2
    timer = threading.Timer(WATCHDOG_S, lambda: (print('SHARD_ARGMAX watchdog', flush=True), os._exit(3)))
    timer.daemon = True
    timer.start()
    if torch is None:
        import torch
        import ttnn
        import tp4_shard_argmax as sarg
        import tp_shapes
        import verify_trace_t1 as t1
    report = dict(kind=KIND, rows=row_cases, regimes=regimes, seeds=seeds, folds=folds,
                  environment=dict(QWEN_FAST_TP=os.environ.get('QWEN_FAST_TP'), QWEN_FAST_TP4_SHARD_VALUES=os.environ.get('QWEN_FAST_TP4_SHARD_VALUES')))
    mesh, status, sections, rig = None, 4, [], None
    try:
        if os.environ.get('QWEN_FAST_TP') != '4':
            raise RuntimeError('QWEN_FAST_TP=4 required (the geometry is the four-card shard\'s)')
        if tp_shapes.vocab_shard() != SHARD:
            raise RuntimeError('vocab shard %r is not %d' % (tp_shapes.vocab_shard(), SHARD))
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=TRACE_REGION)
        grid = mesh.compute_with_storage_grid_size()
        report['grid'] = [grid.x, grid.y]
        workers = grid.x * grid.y
        report['workers'] = workers
        report['kernels_sha256'] = kernel_sha256(os.path.dirname(os.path.abspath(sarg.__file__)))
        pairs = tie_pairs(sarg, workers)
        report['tie_pairs'] = len(pairs)
        rig = Rig(ttnn, mesh, torch, sarg, t1, (grid.x, grid.y))
        rig.reserve()
        for rows in row_cases:
            for regime in regimes:
                for seed in seeds:
                    host = host_logits(torch, rows, regime, seed, pairs)
                    try:
                        found = compare_case(rig, host, rows, regime, folds)
                    except BaseException as error:  # noqa: BLE001
                        found = [dict(rows=rows, regime=regime, fold='-', error='%s: %s' % (error.__class__.__name__, str(error)[:400]))]
                    for section in found:
                        section.update(seed=seed)
                        sections.append(section)
                        print('SHARD_ARGMAX compare rows=%d regime=%s seed=%d fold=%s differing=%s words_differing=%s served_differing=%s uncertain=%s' % (
                            rows, regime, seed, section.get('fold'), section.get('differing', section.get('error')), section.get('words_differing'),
                            section.get('served_differing'), section.get('uncertain')), flush=True)
        report['compare'] = sections
        text, status = verdict(sections)
        if options.timing == 'always' or (options.timing == 'on' and text == 'PASS'):
            report['timing'] = []
            for rows in row_cases:
                host = host_logits(torch, rows, 'random', seeds[0], pairs)
                result = time_case(rig, host, rows)
                report['timing'].append(result)
                print('SHARD_ARGMAX timing rows=%d served_us=%s scan_us=%s fold1_us=%s fold2_us=%s fold1_cost_us=%s fold2_cost_us=%s s1_saves_us=%s' % (
                    rows, result['served']['median_us'], result['scan']['median_us'], result['fold1']['median_us'], result['fold2']['median_us'],
                    result['fold1_cost_us'], result['fold2_cost_us'], result['s1_saves_us']), flush=True)
            report['timing_verdict'] = dict(zip(('s1', 'fold2'), timing_verdict(report['timing'])))
            print('SHARD_ARGMAX timing_verdict s1=%s fold2=%s' % (report['timing_verdict']['s1'], report['timing_verdict']['fold2']), flush=True)
        report['verdict'] = text
        print('SHARD_ARGMAX verdict=%s sections=%d differing=%d' % (
            text, len(sections), sum(max(section.get('differing', 0), 0) + max(section.get('words_differing', 0), 0) for section in sections)), flush=True)
    except BaseException as error:  # noqa: BLE001
        report['verdict'] = 'NOT-RUN'
        report['error'] = '%s: %s' % (error.__class__.__name__, str(error)[:500])
        print('SHARD_ARGMAX verdict=NOT-RUN error=%s' % report['error'], flush=True)
        status = 4
    finally:
        if rig is not None:
            release_all(rig, rig.partials, rig.groups)
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
            except BaseException:  # noqa: BLE001
                pass
    with open(options.out, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True), flush=True)
    timer.cancel()
    return status


if __name__ == '__main__':
    sys.exit(main())
