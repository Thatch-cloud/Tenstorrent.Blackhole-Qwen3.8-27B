#!/usr/bin/env python3
"""E1: the four-card ordered K/V writers at page-table width 4,096 (a 262,144-token window), on ONE card.

Both writers a four-card process builds write one KV head per chip into a (N, 1, 64, 256) BF8 paged cache through the hash-pinned
ordered kernels (ordered_cache.load_kernels; imported, never copied):

  chained64  packed_ordered_cache.update_chained: the packed block's 64 rows, one launch, one semaphore chain per user.
  tiles32    ordered_cache_tp.update: the engines' 32-row tile, one chain over the tile (run under chip_view.ChipView, which shows
             the one chip as the four the builder loops over and launches chip 0's program).

Widths: 2,052 (the control: a 131,328-token window, admitted today by the pinned ordered_cache) and 4,096. The page table is
read by one noc.async_read into a CB sized from the tensor and never bounds-checked, so a misread of bytes 8,192-16,383 of a row is
the failure this run exists to catch; every earlier check at 2,052 shared that read with its oracle.

Per (writer, width, seed) the cases are (--sections): eager steps; replay_changed (trace captured once, the page table REWRITTEN IN
PLACE between replays - a replay that reused a stale table mispredicts); replay_unchanged (trace replayed with the table
untouched, positions and payloads rewritten). After every step the COMPLETE cache is read back and compared with the cache
predicted on the host from the host page table alone (ordered_cache_hw_plan.ExpectedCache / compare_cache: predicted blocks exact,
every other block still zero), so an extra write fails the case as surely as a missing one. Page-table entries cycled by every row:
0, 1, 1,023, 1,024, 2,047, 2,048, 2,051, 2,052, 3,071, 4,094 and 4,095; the last step puts rows 0-31 on positions 262,080-262,111
(the last 32 positions of a window under the 32-position drafter clamp), 131,296-131,327 at the control width. Every (row, entry)
the plan hits is a physical block no other row hits, so a chained launch never has two users on one tile row (kv_conflict).

The width 4,096 is not admitted by the pinned ordered_cache, and page_width_tp4 refuses it until THIS run's record exists. For the
run only, page_width_tp4.admitted is patched to admit exactly 4,096 (and nothing else); the patch is scoped (restored in a finally)
and written into the report (admission_patch). Nothing else about the writers is changed.

Run with QWEN_FAST_TP=4 (tp_shapes reads it at import). Verdict line:
  ORDERED_WRITER verdict=PASS|FAIL|NO-DECISION scope=full|reduced width=4096 chips=1of4 checks=N exact=N ...
scope=full needs both writers, both widths, seeds 0,1,2 and all three cases, each decisive; a section that raised is NO-DECISION.

  QWEN_FAST_TP=4 python3 ordered_writer_tp4_card_test.py --out results.json
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

VERDICT = 'ORDERED_WRITER'
BLOCKS = 4112
BLOCK_SIZE = 64
HEAD_DIM = 256
PADDED_HEADS = 32
WRITERS = ('chained64', 'tiles32')
LAUNCH_ROWS = {'chained64': 64, 'tiles32': 32}
WIDTHS = (2052, 4096)
CONTROL_WIDTH, WIDE_WIDTH = 2052, 4096
MODES = ('eager', 'replay_changed', 'replay_unchanged')
CASE_SECTIONS = MODES + ('complete_cache',)
SEEDS = (0, 1, 2)
# Page-table entries the plan hits, by width (the entries below the width).
WIDE_ENTRIES = (0, 1, 1023, 1024, 2047, 2048, 2051, 2052, 3071, 4094, 4095)
CONTROL_ENTRIES = (0, 1, 511, 512, 1023, 1024, 1025, 2047, 2048, 2049, 2050, 2051)
ANCHORS = {WIDE_WIDTH: 262080, CONTROL_WIDTH: 131296}
ANCHOR_ROWS = 32
KV_HEADS = 1
CHIPS = 'tp4-one-chip'


def entries_for(width):
    if width == WIDE_WIDTH:
        return WIDE_ENTRIES
    if width == CONTROL_WIDTH:
        return CONTROL_ENTRIES
    raise ValueError('no plan for width %r' % (width,))


def anchor_positions(width):
    first = ANCHORS[width]
    return [first + row for row in range(ANCHOR_ROWS)]


# ---------------------------------------------------------------------------------------------
# The plan (pure integer arithmetic: identical on every Python, no torch).
# ---------------------------------------------------------------------------------------------

MASK64 = (1 << 64) - 1


def _splitmix(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, value ^ (value >> 31)


def permutation(state, count):
    """A seeded Fisher-Yates permutation of range(count) and the advanced state."""
    ids = list(range(count))
    for index in range(count - 1, 0, -1):
        state, value = _splitmix(state)
        other = value % (index + 1)
        ids[index], ids[other] = ids[other], ids[index]
    return state, ids


def page_table(seed, rows, width, blocks, hits):
    """(rows, width) page table. Each (row, hit entry) gets a physical block no other (row, hit entry) has: a globally unique
    assignment from one permutation, so no two rows ever write one block. Every other entry is a distinct id of the row's own
    (none of the row's hit blocks), never targeted by the plan. Distinct within a row, as a served row is."""
    hits = sorted(set(hits))
    if rows * len(hits) > blocks or width > blocks or any(not 0 <= hit < width for hit in hits):
        raise ValueError('the plan does not fit %d blocks' % blocks)
    state, order = permutation(seed * 7919 + width, blocks)
    table = []
    for row in range(rows):
        mine = order[row * len(hits):(row + 1) * len(hits)]
        taken = set(mine)
        state, others = permutation(state, blocks)
        fill = [block for block in others if block not in taken]
        entry_blocks = {}
        for entry, block in zip(hits, mine):
            entry_blocks[entry] = block
        cursor = 0
        line = []
        for entry in range(width):
            if entry in entry_blocks:
                line.append(entry_blocks[entry])
            else:
                line.append(fill[cursor])
                cursor += 1
        table.append(line)
    return table


def schedule(entries, anchors, rows):
    """Positions per step: step s, row r targets entry entries[(r + s) % n] at offset (r * 4 + s * 5) % 64; the last step puts
    rows 0..len(anchors)-1 on the anchors and the rest on cycled entries."""
    count = len(entries)
    steps = [[entries[(row + step) % count] * BLOCK_SIZE + (row * 4 + step * 5) % BLOCK_SIZE for row in range(rows)]
             for step in range(count)]
    steps.append([anchors[row] if row < len(anchors)
                  else entries[(row + 3) % count] * BLOCK_SIZE + (row * 4 + 1) % BLOCK_SIZE for row in range(rows)])
    return steps


def payload_seed(case_index, step, row):
    return (case_index + 1) * 1000000 + step * 1000 + row + 1


def build_case(spec, blocks=BLOCKS):
    """Materialise one case spec: tables (two for replay_changed, else one), per-step positions, target blocks and payload
    seeds. Kept lazy (a spec is four words; a case holds up to two 64 x 4,096 tables)."""
    index, writer, width, mode, seed = (spec[key] for key in ('index', 'writer', 'width', 'mode', 'seed'))
    rows = LAUNCH_ROWS[writer]
    entries = entries_for(width)
    hits = set(entries) | {position // BLOCK_SIZE for position in anchor_positions(width)}
    tables = [page_table(seed * 31 + table_index + 1, rows, width, blocks, hits)
              for table_index in range(2 if mode == 'replay_changed' else 1)]
    steps = []
    for number, positions in enumerate(schedule(entries, anchor_positions(width), rows)):
        table_index = number % len(tables)
        table = tables[table_index]
        steps.append(dict(step=number, table=table_index, positions=positions,
                          blocks=[table[row][position // BLOCK_SIZE] for row, position in enumerate(positions)],
                          payload_seeds=[payload_seed(index, number, row) for row in range(rows)]))
    return dict(index=index, name=case_name(writer, width, mode, seed), writer=writer, width=width, mode=mode, seed=seed,
                rows=rows, tables=tables, steps=steps)


def case_name(writer, width, mode, seed):
    return '%s-%d-%s-s%d' % (writer, width, mode, seed)


def build_cases(widths=WIDTHS, writers=WRITERS, seeds=SEEDS, modes=MODES):
    """The case specs in run order (seed, width, writer, mode)."""
    cases = []
    for seed in seeds:
        for width in widths:
            for writer in writers:
                for mode in modes:
                    cases.append(dict(index=len(cases), name=case_name(writer, width, mode, seed), writer=writer, width=width,
                                      mode=mode, seed=seed))
    return cases


def conflicts(case):
    """(step, row, row) pairs that write one block in one step: a chained launch must have none (verify_trace_t2.kv_conflict
    is about (page, tile row); one block each is stricter)."""
    found = []
    for step in case['steps']:
        seen = {}
        for row, block in enumerate(step['blocks']):
            if block in seen:
                found.append((step['step'], seen[block], row))
            seen[block] = row
    return found


def table_digest(table):
    return hashlib.sha256(json.dumps(table, separators=(',', ':')).encode()).hexdigest()


def step_count(width):
    """Steps per case: one per cycled entry, then the anchor step."""
    return len(entries_for(width)) + 1


def required_checks(spec):
    """The (name, step) checks a case must record exact."""
    required = [('zero_baseline', None), ('pages_uploaded', None), ('input_unchanged', None), ('pages_unchanged', None)]
    required.extend(('complete_cache', step) for step in range(step_count(spec['width'])))
    return required


# ---------------------------------------------------------------------------------------------
# Verdict (pure).
# ---------------------------------------------------------------------------------------------

def decide(report):
    """PASS / FAIL / NO-DECISION: every requested case ran, recorded every required check, and every check is exact."""
    problems = []
    if report.get('error'):
        return dict(verdict='NO-DECISION', problems=[str(report['error'])[:200]])
    by_case = {}
    for check in report.get('checks', []):
        by_case.setdefault(check['case'], {})[(check['name'], check.get('step'))] = check
    for case in report.get('plan', []):
        recorded = by_case.get(case['name'])
        state = (report.get('cases') or {}).get(case['name']) or {}
        if state.get('error'):
            problems.append('%s raised' % case['name'])
            continue
        if state.get('skipped'):
            problems.append('%s was cut by the deadline' % case['name'])
            continue
        missing = [required for required in case['required'] if not recorded or tuple(required) not in recorded]
        if missing:
            problems.append('%s lacks %d checks (%s)' % (case['name'], len(missing), missing[0]))
    if problems:
        return dict(verdict='NO-DECISION', problems=problems[:20])
    inexact = [check for check in report.get('checks', []) if not check.get('exact')]
    if inexact:
        first = inexact[0]
        return dict(verdict='FAIL', problems=['%d checks differ (first: %s %s step %s)' % (
            len(inexact), first['case'], first['name'], first.get('step'))])
    return dict(verdict='PASS', problems=[])


def scope_of(report):
    requested = report.get('requested') or {}
    full = (set(requested.get('widths', ())) >= set(WIDTHS) and set(requested.get('writers', ())) >= set(WRITERS)
            and set(requested.get('seeds', ())) >= set(SEEDS) and set(requested.get('modes', ())) >= set(MODES))
    return 'full' if full else 'reduced'


def tally(report):
    checks = report.get('checks', [])
    return dict(checks=len(checks), exact=sum(1 for check in checks if check.get('exact')))


def verdict_line(report):
    counts = tally(report)
    requested = report.get('requested') or {}
    return '%s verdict=%s scope=%s width=%s chips=1of%s checks=%d exact=%d writers=%s widths=%s seeds=%s' % (
        VERDICT, report['decision']['verdict'], scope_of(report), max(requested.get('widths') or [0]), report.get('tp'),
        counts['checks'], counts['exact'], ','.join(requested.get('writers', ())),
        ','.join(str(width) for width in requested.get('widths', ())), ','.join(str(seed) for seed in requested.get('seeds', ())))


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--widths', default=','.join(str(width) for width in WIDTHS))
    parser.add_argument('--writers', default=','.join(WRITERS))
    parser.add_argument('--seeds', default=','.join(str(seed) for seed in SEEDS))
    parser.add_argument('--modes', default=','.join(MODES))
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('TT_METAL_HOME', '/opt/tt-metal')))
    parser.add_argument('--trace-region-bytes', type=int, default=4194304)
    # accepted for run_card_b.sh, which passes them to every harness
    parser.add_argument('--watchdog', type=float, default=0.0)
    parser.add_argument('--deadline-s', type=float, default=0.0)
    parser.add_argument('--expect-binary-sha256', default='')
    arguments = parser.parse_args(argv)
    arguments.widths = [int(item) for item in arguments.widths.split(',') if item]
    arguments.writers = [item for item in arguments.writers.split(',') if item]
    arguments.seeds = [int(item) for item in arguments.seeds.split(',') if item]
    arguments.modes = [item for item in arguments.modes.split(',') if item]
    if set(arguments.widths) - set(WIDTHS):
        parser.error('--widths within %s' % ','.join(map(str, WIDTHS)))
    if set(arguments.writers) - set(WRITERS):
        parser.error('--writers within %s' % ','.join(WRITERS))
    if set(arguments.modes) - set(MODES):
        parser.error('--modes within %s' % ','.join(MODES))
    return arguments


# ---------------------------------------------------------------------------------------------
# The device part.
# ---------------------------------------------------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def bitwise_equal(torch, actual, reference):
    """Same shape, both bf16, same 16-bit patterns. Anything else (another dtype, a -0 against a +0) is left to compare_cache."""
    if tuple(actual.shape) != tuple(reference.shape) or actual.dtype != torch.bfloat16 or reference.dtype != torch.bfloat16:
        return False
    return torch.equal(actual.contiguous().view(torch.int16), reference.contiguous().view(torch.int16))


class scoped_admission:
    """page_width_tp4.admitted patched to admit exactly 4,096 (and defer to the real answer for every other width), restored on
    exit; the patch is what the report records."""

    def __init__(self, module, width=WIDE_WIDTH):
        self.module, self.width, self.original = module, width, None

    def __enter__(self):
        self.original = self.module.admitted
        original, width = self.original, self.width

        def admitted(value, *args, **kwargs):
            if value == width:
                return True
            return original(value, *args, **kwargs)

        self.module.admitted = admitted
        return self

    def __exit__(self, *unused):
        self.module.admitted = self.original
        return False


class Rig:
    def __init__(self, ttnn, torch, mesh, report):
        self.ttnn, self.torch, self.mesh, self.report = ttnn, torch, mesh, report
        self.owned = []

    def upload(self, value, dtype):
        ttnn = self.ttnn
        result = ttnn.from_torch(value, device=self.mesh, dtype=dtype,
                                 layout=ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
        self.owned.append(result)
        return result

    def replace(self, value, destination, dtype):
        ttnn = self.ttnn
        host = ttnn.from_torch(value, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
        ttnn.copy_host_to_device_tensor(host, destination)

    def host(self, value):
        shards = self.ttnn.get_device_tensors(value)
        if len(shards) != 1:
            raise AssertionError('one chip expected, got %d shards' % len(shards))
        return self.ttnn.to_torch(shards[0])

    def release(self):
        for value in reversed(self.owned):
            self.ttnn.deallocate(value)
        self.owned.clear()


def run_case(rig, hw, case, kernels, report, writer_op):
    """One case on the open card; appends its checks to report['checks']."""
    ttnn, torch = rig.ttnn, rig.torch
    name = case['name']
    rows, width = case['rows'], case['width']

    def record(check_name, step, exact, **extra):
        report['checks'].append(dict(case=name, writer=case['writer'], width=width, mode=case['mode'], seed=case['seed'],
                                     name=check_name, step=step, exact=bool(exact), **extra))

    def check_cache(step, expected, **extra):
        actual = rig.host(cache)
        # The complete cache, every block: one bitwise pass first (a 71 MB cache compares in ~0.2 s instead of ~2 s, which is the
        # difference between a 25 and a 3 minute run over 450 readbacks); a difference falls through to the block-by-block
        # comparison that says which blocks, and which kind (an unpredicted write or a wrong predicted one).
        if bitwise_equal(torch, actual, expected.values):
            result = dict(exact=True, predicted_blocks=len(expected.predicted), mismatched_blocks=0,
                          unpredicted_nonzero_blocks=0, predicted_mismatch_blocks=0, samples=[])
        else:
            result = hw.compare_cache(actual, expected)
        record('complete_cache', step, result['exact'], **dict(extra, **{key: result[key] for key in (
            'predicted_blocks', 'mismatched_blocks', 'unpredicted_nonzero_blocks', 'predicted_mismatch_blocks')}))
        if not result['exact']:
            report['samples'].setdefault(name, result['samples'])
        return result['exact']

    def check_equal(check_name, value, host):
        actual = rig.host(value)
        record(check_name, None, tuple(actual.shape) == tuple(host.shape) and torch.equal(actual.to(host.dtype), host))

    expected = hw.ExpectedCache(BLOCKS, heads=KV_HEADS)
    cache = rig.upload(expected.values.clone(), ttnn.bfloat8_b)
    check_cache(None, expected)
    record_zero = report['checks'][-1]
    record_zero['name'] = 'zero_baseline'
    tables = [torch.tensor(table, dtype=torch.int32) for table in case['tables']]
    pages = rig.upload(tables[0], ttnn.int32)
    check_equal('pages_uploaded', pages, tables[0])
    first = case['steps'][0]
    positions = rig.upload(torch.tensor(first['positions'], dtype=torch.int32), ttnn.int32)
    first_payloads = hw.step_payloads(dict(payload_seeds=first['payload_seeds']))
    packed = rig.upload(first_payloads.unsqueeze(0).contiguous(), ttnn.bfloat16)
    current, trace, payloads = 0, None, first_payloads

    def operation():
        writer_op(cache, packed, positions, pages)

    try:
        for step in case['steps']:
            payloads = hw.step_payloads(dict(payload_seeds=step['payload_seeds']))
            if step['table'] != current:
                rig.replace(tables[step['table']], pages, ttnn.int32)
                current = step['table']
            rig.replace(torch.tensor(step['positions'], dtype=torch.int32), positions, ttnn.int32)
            rig.replace(payloads.unsqueeze(0).contiguous(), packed, ttnn.bfloat16)
            if case['mode'] == 'eager':
                operation()
            else:
                if trace is None:
                    # The warm-up compiles the program on these exact tensors (the step-0 write is idempotent); every later step
                    # is written by replay alone.
                    operation()
                    ttnn.synchronize_device(rig.mesh)
                    trace = ttnn.begin_trace_capture(rig.mesh, cq_id=0)
                    try:
                        operation()
                    finally:
                        ttnn.end_trace_capture(rig.mesh, trace, cq_id=0)
                ttnn.execute_trace(rig.mesh, trace, cq_id=0, blocking=True)
            ttnn.synchronize_device(rig.mesh)
            expected.apply(case['tables'][step['table']], step['positions'], payloads)
            check_cache(step['step'], expected, table=step['table'])
        check_equal('input_unchanged', packed, payloads.unsqueeze(0).contiguous())
        check_equal('pages_unchanged', pages, tables[current])
    finally:
        if trace is not None:
            ttnn.release_trace(rig.mesh, trace)
        ttnn.synchronize_device(rig.mesh)
        rig.release()


def write_report(arguments, report):
    report['tally'] = tally(report)
    arguments.out.write_text(json.dumps(report, indent=1, default=str) + '\n')


def run(arguments, report, cases):
    import torch
    import ttnn

    import chip_view
    import ordered_cache
    import ordered_cache_hw_plan as hw
    import ordered_cache_tp
    import packed_ordered_cache
    import page_width_tp4
    import tp_shapes

    tp = tp_shapes.chip_count()
    report['tp'] = tp
    if tp != 4:
        raise SystemExit('E1 qualifies the four-card writers: needs QWEN_FAST_TP=4, got a width of %d' % tp)
    for module in (ordered_cache, ordered_cache_tp, packed_ordered_cache, page_width_tp4):
        report['sources'][Path(module.__file__).name] = sha256_file(module.__file__)
    kernels = ordered_cache.load_kernels(arguments.root)
    report['generated_sha256'] = {role: hashlib.sha256(source.encode()).hexdigest() for role, source in kernels.items()}
    report['native_hashes'] = dict(ordered_cache.HASHES)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=arguments.trace_region_bytes)
    started = time.monotonic()
    try:
        mesh.enable_program_cache()
        grid = mesh.compute_with_storage_grid_size()
        report['grid'] = [grid.x, grid.y]
        rig = Rig(ttnn, torch, mesh, report)
        view = chip_view.ChipView(ttnn, chips=4)

        def chained(cache, packed, positions, pages):
            spans = tuple((row, row + 1) for row in range(LAUNCH_ROWS['chained64']))
            packed_ordered_cache.update_chained(mesh, cache, packed, positions, pages, kernels, spans, operations=ttnn)

        def tiles(cache, packed, positions, pages):
            with view.installed():
                ordered_cache_tp.update(mesh, cache, packed, positions, pages, kernels)

        with scoped_admission(page_width_tp4) as patch:
            report['admission_patch'] = dict(module='page_width_tp4.admitted', width=patch.width,
                                             scope='the run only; restored on exit')
            for spec in cases:
                state = report['cases'].setdefault(spec['name'], {})
                if arguments.deadline_s and time.monotonic() - started > arguments.deadline_s:
                    state['skipped'] = True
                    continue
                print('--- case %s' % spec['name'], flush=True)
                case = build_case(spec)
                state['table_sha256'] = [table_digest(table) for table in case['tables']]
                try:
                    run_case(rig, hw, case, kernels, report, chained if case['writer'] == 'chained64' else tiles)
                    mine = [check for check in report['checks'] if check['case'] == case['name']]
                    state['exact'] = all(check['exact'] for check in mine)
                    print(json.dumps(dict(case=case['name'], exact=state['exact'], checks=len(mine))), flush=True)
                except SystemExit:
                    raise
                except BaseException as error:  # noqa: BLE001 - recorded; the run continues
                    state['error'] = repr(error)
                    state['traceback'] = traceback.format_exc()
                    print(state['traceback'], flush=True)
                    rig.owned.clear()
                report['in_progress'] = case['name']
                write_report(arguments, report)
        report.pop('in_progress', None)
        report['view'] = dict(realised=view.realised, phantom=view.phantom, launches=view.launches)
    finally:
        ttnn.close_mesh_device(mesh)


def main(argv=None):
    arguments = parse(argv)
    cases = build_cases(arguments.widths, arguments.writers, arguments.seeds, arguments.modes)
    problems = []
    for spec in cases:
        if spec['writer'] == 'chained64' and spec['mode'] != 'replay_unchanged':
            found = conflicts(build_case(spec))
            if found:
                problems.append((spec['name'], found[0]))
    report = dict(scope='single-card four-card-geometry ordered K/V writers at page-table widths 2,052 and 4,096 (E1)',
                  argv=list(sys.argv[1:]), kv_heads=KV_HEADS, blocks=BLOCKS, entries=sorted(set(WIDE_ENTRIES) | set(CONTROL_ENTRIES)),
                  anchor_positions=[ANCHORS[WIDE_WIDTH], ANCHORS[WIDE_WIDTH] + ANCHOR_ROWS - 1],
                  requested=dict(widths=arguments.widths, writers=arguments.writers, seeds=arguments.seeds, modes=arguments.modes),
                  checks=[], cases={}, samples={}, sources={}, tp=None,
                  env={name: os.environ.get(name) for name in ('QWEN_FAST_TP',)})
    report['plan'] = [dict(name=spec['name'], required=[list(item) for item in required_checks(spec)]) for spec in cases]
    try:
        if problems:
            raise SystemExit('the plan has two rows on one block: %r' % (problems[0],))
        run(arguments, report, cases)
    except SystemExit as stop:
        report['error'] = str(stop)
    except BaseException as error:  # noqa: BLE001
        report['error'] = repr(error)
        report['traceback'] = traceback.format_exc()
    report['decision'] = decide(report)
    report['verdict_line'] = verdict_line(report)
    write_report(arguments, report)
    print(report['verdict_line'], flush=True)
    return 0 if report['decision']['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
