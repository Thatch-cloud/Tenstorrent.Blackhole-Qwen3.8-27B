"""Card-M qualification of the [QWEN-SDPA-PF] K-split (flag 0x10; apply_factory_pk.py; graft K64j-OQ-KS from build_k64j_oq_ks.sh). A TEMPLATE: nothing here has run on a card.

The served causal chunked paged SDPA at the TP4 shape (6 local Q heads, 1 KV head, head dim 256, q/k chunk 128, bf8 Q/K/V, HiFi2 with fp32 dest, the flexible path with a device
chunk_start tensor, the 4,096-block page table of the 8 x 262k pool) on ONE card of the mesh, in the programs oneq_card_m.py already knows plus the new one:

    stock    no program word                           the stock factory path
    oneq     0x5EFA000B                                one q chunk per core (the baseline the K-split is judged against; proven byte-equal to stock on card M)
    ksplit   0x5EFA001B                                the K-split: S = min(4, cores // q chunks) slices of every q chunk's K stream, merged by the pinned ring merge

It is NOT byte-equal to oneq (the association of the softmax differs: that is the numerics waiver), so the proof is a STATISTIC (ksplit_numerics.py, shared with the model's audit): per case
the distribution of |ksplit - oneq| in bf16 ulps of the row's scale, against the PROVISIONAL bound; and, up to --reference-max-start, both programs against a float32 reference computed
on the host from the very K/V the card read (the noise floor: the K-split must be no further from the truth than oneq, within the NOISE ratios). Rows 1024 and 512 only (2048 rows have one
slice; 256 and 128 are not chain-qualified).

THE CONFIGS (rows, grid -> slices; the planner reads the device grid): s2 = 1,024 rows on the native grid (48 q chunks, 2 slices), s4 = 512 rows native (24 chunks, 4 slices), s2g = 512 rows on an
8 x 8 grid (64 cores // 24 = 2 slices). The matrix: configs x Q memory (dram, l1) x seed x Q variant (normal, peaky) x 15 chunk starts 0 .. 251,904 (contexts 2k .. 254k).

ALSO: the refusals (TT_FATAL at program build, never a hang): a bare 0x10, 0x10 without the oneq bit, the hang flag without its test variable or without 0x10, the K-split at 2,048 rows (one
slice), on a grid with no room for a second slice, at an odd chunk count, with bf16 K/V, with a legacy int chunk_start, at TP2 heads; the CAP (256 rows on the native grid could take 10 slices
and must log exactly 4); the program cache (the word keys its own program, a chunk_start change adds none); an 11 x 10 grid pass; the factory log checked against the planner (the
'ksplit=1' line per program); the TIMING at 8 context points (interleaved rounds of oneq / ksplit): the median ksplit / oneq ratio from 32k, and the loss at 4k and below.

    python3 -B ksplit_card_m.py --out /results/ksplit.json [--expect-binary-sha256 <sha>]        the numerics, refusals, cache, grid and timing passes
    python3 -B ksplit_card_m.py --role hang --out /results/ksplit-hang.json                      the PLANTED HANG (needs QWEN_SDPA_PF_TEST=1): PASS = the watchdog fires; LAST in a session

Everything the C++ side prints goes to <out>.native.log. The helpers above run() import no ttnn and are unit-tested on CPU (test_ksplit_card.py, with a fake ttnn).
"""

import argparse
import collections
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT_CANDIDATES = (HERE.parent / 'sdpa_prefill_oneq', Path('/ops/sdpa_prefill_oneq'))
CI_CANDIDATES = (HERE.parents[2] / 'scripts' / 'ci', Path('/ops/ci'))
for _candidate in tuple(ROOT_CANDIDATES) + tuple(CI_CANDIDATES) + (HERE,):
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

import apply_factory_pk as pk  # noqa: E402
import ksplit_numerics as numerics  # noqa: E402
import ksplit_planner as planner  # noqa: E402
import oneq_card_m as card  # noqa: E402

PF_TAG = card.PF_TAG
ARMS = collections.OrderedDict([('stock', None), ('oneq', 0xB), ('ksplit', 0x1B)])
HANG_FLAGS = 0x1B | pk.FLAG_KS_HANG
HANG_CALL_LABEL = 'planted ksplit hang 0x41b'
HANG_READBACK_LABEL = 'planted ksplit hang read back'
HANG_LABELS = (HANG_CALL_LABEL, HANG_READBACK_LABEL)
HANG_ARMED = 'planted ksplit hang: armed'
CONFIGS = (('s2', 1024, None), ('s4', 512, None), ('s2g', 512, (8, 8)))
STARTS = card.STARTS
TIME_STARTS = card.TIME_STARTS
SEEDS = card.SEEDS
VARIANTS = card.VARIANTS
Q_MEMORY = card.Q_MEMORY
REFERENCE_MAX_START = 32768
REFERENCE_SEED = 0
GRID_STARTS = (0, 2048, 65536, 251904)
GRID_EXTRA = (11, 10)
KS_MARKER = b'[QWEN-SDPA-PF] ksplit needs at least 2 slices'
KS_LINE = re.compile(r'\[QWEN-SDPA-PF\] ksplit=1 slices=([0-9]+) q_chunks=([0-9]+) cores=([0-9]+) busy=([0-9]+) chains=([0-9]+) members=([0-9]+)')
KS_WIN_RATIO, KS_PARTIAL_RATIO, SMALL_CONTEXT_LOSS = 0.75, 0.95, 1.05
SMALL_CONTEXT = 4096

Q1 = card.Q1
WATCHDOG = card.WATCHDOG
COMPILE_GRACE_S = card.COMPILE_GRACE_S


# ---------------------------------------------------------------------------------------------
# Host helpers (no ttnn): configs, words, expected lines, refusals, verdicts.
# ---------------------------------------------------------------------------------------------

def word(flags):
    return card.word(flags)


def config_plan(rows, grid, native_grid, geometry='tp4'):
    """The planner's reading of a config on the device: dict from ksplit_planner.plan (refusal, slices, busy, chains)."""
    spec = card.GEOMETRIES[geometry]
    return planner.plan(spec['nqh'], spec['nkh'], rows, card.Q_CHUNK, grid or native_grid)


def case_label(config, qmem, seed, variant, start):
    return '%s-%s-s%d-%s@%d' % (config, qmem, seed, variant, start)


def expected_programs(requested):
    """Counters of the log lines a set of built programs must have produced. requested: the harness' {(geometry, rows, width, qmem, flags, grid)}.
    -> (Counter of (flags, chains, members, order) for the F4 line, Counter of (q_chunks, cores, chains, members) for the oneq line, Counter of
    (slices, q_chunks, cores, busy, chains, members) for the ksplit line). One program, one factory run, one line each; a ksplit program's F4 line counts ITS chains (one per slice)."""
    flags_lines, oneq_lines, ks_lines = collections.Counter(), collections.Counter(), collections.Counter()
    for geometry, rows, _width, _qmem, flags, grid in requested:
        spec = card.GEOMETRIES[geometry]
        base = card.planner.plan(spec['nqh'], spec['nkh'], rows, card.Q_CHUNK, grid, oneq=bool(flags & card.planner.FLAG_ONEQ), noc_order=bool(flags & card.planner.FLAG_NOC_ORDER))
        mine = planner.plan(spec['nqh'], spec['nkh'], rows, card.Q_CHUNK, grid) if flags & pk.FLAG_KSPLIT else None
        counted = mine or base
        flags_lines[(flags, counted['chain_count'], counted['member_count'], 'noc' if flags & card.planner.FLAG_NOC_ORDER else 'raster')] += 1
        if flags & card.planner.FLAG_ONEQ:
            oneq_lines[(base['total'], base['num_cores'], base['chain_count'], base['member_count'])] += 1
        if mine:
            ks_lines[(mine['slices'], mine['total'], mine['num_cores'], mine['busy'], mine['chain_count'], mine['member_count'])] += 1
    return flags_lines, oneq_lines, ks_lines


def expected_ks_lines(requested):
    return expected_programs(requested)[2]


def check_factory_lines(text, requested):
    """[problem] comparing the native log's factory lines (the F4 line, the oneq line, the ksplit line) with the planner's reading of every program the run built."""
    flags, oneq = card.factory_lines(text)
    got_ks = collections.Counter(tuple(int(value) for value in match.groups()) for match in KS_LINE.finditer(text))
    want_flags, want_oneq, want_ks = expected_programs(requested)
    problems = []
    for name, got, want in (('flags line', collections.Counter(flags), want_flags), ('oneq line', collections.Counter(oneq), want_oneq), ('ksplit line', got_ks, want_ks)):
        if got != want:
            problems.append('%s: unexpected %s, missing %s' % (name, dict(got - want) or 'none', dict(want - got) or 'none'))
    return problems


def loaded_binary(maps='/proc/self/maps'):
    """card.loaded_binary plus whether the mapped _ttnncpp.so has the ksplit edits."""
    info = card.loaded_binary(maps)
    with open(info['path'], 'rb') as handle:
        info['ksplit'] = KS_MARKER in handle.read()
    return info


def timing_verdict(timing, config):
    """Per Q memory: the median ksplit / oneq ratio over the starts of 32k and beyond (KS-WIN <= 0.75, KS-PARTIAL <= 0.95, else KS-NO-WIN) and the worst ratio at 4k and below
    (a loss above 5 percent there is KS-SMALL-LOSS: the lever must not cost the short contexts, where the device keeps slices small)."""
    out = {}
    for qmem, arms in (timing.get(config) or {}).items():
        oneq = {int(s): v['median_ms'] for s, v in (arms.get('oneq') or {}).items() if v.get('median_ms') is not None}
        mine = {int(s): v['median_ms'] for s, v in (arms.get('ksplit') or {}).items() if v.get('median_ms') is not None}
        common = sorted(set(oneq) & set(mine))
        long_ratios = [mine[s] / oneq[s] for s in common if s >= 32768]
        small_ratios = [mine[s] / oneq[s] for s in common if s <= SMALL_CONTEXT]
        entry = dict(ratio=None, verdict='KS-NO-WIN', small_worst=max(small_ratios) if small_ratios else None, ratios={s: mine[s] / oneq[s] for s in common})
        if long_ratios:
            entry['ratio'] = statistics.median(long_ratios)
            entry['verdict'] = ('KS-WIN' if entry['ratio'] <= KS_WIN_RATIO else 'KS-PARTIAL' if entry['ratio'] <= KS_PARTIAL_RATIO else 'KS-NO-WIN')
        entry['small_loss'] = bool(small_ratios) and max(small_ratios) > SMALL_CONTEXT_LOSS
        out[qmem] = entry
    return out


def numerics_summary(cases):
    """Worst statistics over the cases of a report (the numbers the waiver decision reads)."""
    rows = [case['stats'] for case in cases if case.get('stats')]
    if not rows:
        return {}
    return dict(cases=len(rows), worst_max_ulps=max(row['max_ulps'] for row in rows), worst_rms_ulps=max(row['rms_ulps'] for row in rows),
                worst_p999_ulps=max(row['p999_ulps'] for row in rows), worst_frac_gt2=max(row['frac_gt2'] for row in rows),
                worst_row_max_p99_ulps=max(row['row_max_p99_ulps'] for row in rows), worst_rel_l2=max(row['rel_l2'] for row in rows),
                nonfinite=sum(row['nonfinite'] for row in rows), in_bound=sum(1 for case in cases if case.get('stats') and not case['problems']))


# ---------------------------------------------------------------------------------------------
# Device part: the qualification card only.
# ---------------------------------------------------------------------------------------------

REFUSALS = (
    # (name, geometry, call changes, word flags, rows, needle)
    ('bare 0x10 (no chain bit)', 'tp4', {}, 0x10, 1024, 'unknown or incomplete flags'),
    ('0x10 with the chain bit but not the oneq bit (0x11)', 'tp4', {}, 0x11, 1024, 'unknown or incomplete flags'),
    ('the hang flag without 0x10 (0x40B)', 'tp4', {}, 0x40B, 1024, 'unknown or incomplete flags'),
    ('the hang flag without QWEN_SDPA_PF_TEST (0x41B)', 'tp4', {}, 0x41B, 1024, 'test-only flags'),
    ('ksplit at 2,048 rows (96 q chunks: one slice)', 'tp4', {}, 0x1B, 2048, 'ksplit needs at least 2 slices'),
    ('ksplit at an odd chunk count (1,920 rows)', 'tp4', {}, 0x1B, 1920, 'kv_chain outside its qualified envelope'),
    ('ksplit with bf16 K/V', 'tp4', dict(kv_bf16=True), 0x1B, 1024, 'kv_chain outside its qualified envelope'),
    ('ksplit with a legacy int chunk_start', 'tp4', dict(legacy_start=True), 0x1B, 1024, 'kv_chain outside its qualified envelope'),
    ('ksplit at TP2 heads, 1,024 rows (96 q chunks: one slice)', 'tp2', {}, 0x1B, 1024, 'ksplit needs at least 2 slices'),
    ('ksplit at TP2 heads, 2,048 rows (192 q chunks: more than the grid)', 'tp2', {}, 0x1B, 2048, 'oneq needs one q chunk per core'),
    ('ksplit on a grid with no room for a second slice (1,024 rows, 6 x 8 cores)', 'tp4', dict(grid=(6, 8)), 0x1B, 1024, 'ksplit needs at least 2 slices'),
)
CAP_ROWS = 256        # 12 q chunks on 130 cores: ten slices would fit, the factory takes four


def ks_refusal(geometry, rows, grid, native_grid, flags=0x1B):
    """The planner's reading of the factory: the TT_FATAL text a ksplit call must raise, or None when it runs."""
    spec = card.GEOMETRIES[geometry]
    return planner.plan(spec['nqh'], spec['nkh'], rows, card.Q_CHUNK, grid or native_grid)['refusal']


def reference_for(bench, rows, seed, variant, qmem, start, torch):
    """The float32 reference of one case from the tensors the card holds: Q, and the logical K / V gathered through the page table, read back."""
    ttnn, torch = bench.ttnn, bench.torch
    q = bench.query(rows, seed, variant, qmem)
    k, v = bench.pool(seed)
    table = bench.table(seed)
    with WATCHDOG.op('reference read back'):
        q_host = ttnn.to_torch(q).to(torch.float32)[0]
        k_host, v_host = ttnn.to_torch(k).to(torch.float32), ttnn.to_torch(v).to(torch.float32)
        table_host = ttnn.to_torch(table).reshape(-1)
    length = start + rows
    scale = 1.0 / (card.HD ** 0.5)
    return numerics.reference_attention(q_host, numerics.gather_pages(k_host, table_host, length), numerics.gather_pages(v_host, table_host, length), start, scale, torch)[None]


def sweep(bench, args, report):
    """Every config x Q memory x seed x variant x start: ksplit against oneq by the statistic, and the oneq arm against stock (the card-M proof it stands on, first seed only)."""
    failures = report['failures']
    torch = bench.torch
    for name, rows, grid in CONFIGS:
        if name not in args.configs:
            continue
        shape_plan = config_plan(rows, grid, bench.native_grid)
        if shape_plan['refusal']:
            failures.append('config %s: the planner says the factory refuses it on this grid (%s)' % (name, shape_plan['refusal']))
            continue
        card.validate_starts(args.starts, rows)
        for qmem in args.q_memory:
            for seed in args.seeds:
                for variant in args.variants:
                    for start in args.starts:
                        oneq = bench.host(bench.call(rows, qmem, seed, variant, start, word(ARMS['oneq']), grid=grid), 'read back oneq')
                        ks = bench.host(bench.call(rows, qmem, seed, variant, start, word(ARMS['ksplit']), grid=grid), 'read back ksplit')
                        stats = numerics.error_stats(ks, oneq, torch)
                        problems = numerics.within_bound(stats, numerics.bound_for(start + rows))
                        entry = dict(case=case_label(name, qmem, seed, variant, start), config=name, rows=rows, context=start + rows, slices=shape_plan['slices'],
                                     stats=stats, problems=problems)
                        if args.anchor and seed == args.seeds[0] and variant == args.variants[0]:
                            stock = bench.host(bench.call(rows, qmem, seed, variant, start, None, grid=grid), 'read back stock')
                            entry['oneq_equals_stock'] = bool(torch.equal(stock.contiguous().view(torch.int16), oneq.contiguous().view(torch.int16)))
                            if not entry['oneq_equals_stock']:
                                failures.append('%s: the oneq baseline differs from the stock path (card M proved them byte-equal)' % entry['case'])
                        if (start + rows <= args.reference_max_start and seed == REFERENCE_SEED and variant == args.variants[0] and qmem == 'dram'):
                            ref = reference_for(bench, rows, seed, variant, qmem, start, torch)
                            entry['oneq_vs_reference'] = numerics.error_stats(oneq, ref, torch)
                            entry['ksplit_vs_reference'] = numerics.error_stats(ks, ref, torch)
                            entry['noise_problems'] = numerics.noise_problems(entry['ksplit_vs_reference'], entry['oneq_vs_reference'])
                            problems = problems + ['noise floor: ' + text for text in entry['noise_problems']]
                            entry['problems'] = problems
                        report['cases'].append(entry)
                        if problems:
                            failures.append('%s: %s' % (entry['case'], '; '.join(problems)))
        print('config %s: %d cases so far, %d failures' % (name, len(report['cases']), len(failures)), flush=True)


def refusals(bench, tp2_bench, args, report):
    """Every refusal is a TT_FATAL at program build (a Python exception), never a hang. Run BEFORE the sweep builds these programs: the factory checks only on a program-cache miss."""
    results = {}
    for name, geometry, change, flags, rows, needle in REFUSALS:
        target = tp2_bench if geometry == 'tp2' else bench
        change = dict(change)
        env = os.environ.get('QWEN_SDPA_PF_TEST')
        os.environ.pop('QWEN_SDPA_PF_TEST', None)
        try:
            out = target.call(rows, 'dram', 0, 'normal', 2048, PF_TAG | flags, record=False, label='refusal %s' % name, **change)
        except Exception as error:  # noqa: BLE001 - the TT_FATAL surfaces as a RuntimeError
            text = str(error)
            results[name] = dict(refused=True, matched=needle in text, message=text[:400])
        else:
            target.ttnn.deallocate(out)
            results[name] = dict(refused=False, matched=False, message='returned an output')
        finally:
            if env is not None:
                os.environ['QWEN_SDPA_PF_TEST'] = env
        print('refusal %-72s %s' % (name, results[name]), flush=True)
        if not (results[name]['refused'] and results[name]['matched']):
            report['failures'].append('refusal: %s was not refused by its [QWEN-SDPA-PF] TT_FATAL (%s): %s' % (name, needle, results[name]['message']))
    report['refusals'] = results


def cap_check(bench, args, report):
    """256 rows (12 q chunks) on the native grid could take ten slices: the factory must take exactly four (busy = 48). The result is read from the factory log by check_factory_lines."""
    out = bench.call(CAP_ROWS, 'dram', 0, 'normal', 2048, word(ARMS['ksplit']), label='cap check 256 rows')
    bench.ttnn.deallocate(out)
    wanted = planner.plan(6, 1, CAP_ROWS, card.Q_CHUNK, bench.native_grid)
    report['cap'] = dict(rows=CAP_ROWS, slices=wanted['slices'], busy=wanted['busy'])
    if wanted['slices'] != planner.SLICE_CAP:
        report['failures'].append('cap: the planner reads %d slices at %d rows, not the cap %d' % (wanted['slices'], CAP_ROWS, planner.SLICE_CAP))


def cache_checks(bench, args, report):
    """oneq then ksplit at one fresh shape adds exactly one program (the word keys its own program); a chunk_start change adds none."""
    shape = (1024, 'dram', 0, 'normal')
    bench.pool(0)
    bench.table(0)
    bench.query(shape[0], shape[2], shape[3], shape[1])
    starts = [s for s in (0, 2048, 65536, 128) if card.blocks_needed(s, 1024) <= bench.pool_blocks]
    for start in starts:
        bench.start(start)
    before = bench.cache_entries()
    bench.host(bench.call(*shape, starts[0], word(ARMS['oneq'])))
    after_oneq = bench.cache_entries()
    bench.host(bench.call(*shape, starts[0], word(ARMS['ksplit'])))
    after_ks = bench.cache_entries()
    for start in starts[1:]:
        bench.host(bench.call(*shape, start, word(ARMS['ksplit'])))
    after_starts = bench.cache_entries()
    result = dict(before=before, after_oneq=after_oneq, after_ksplit=after_ks, after_starts=after_starts)
    report['cache'] = result
    print('program cache %r' % (result,), flush=True)
    if None in result.values():
        report['failures'].append('the device has no num_program_cache_entries(): the ksplit word keying its own program is unproven')
        return
    if after_ks != after_oneq + 1:
        report['failures'].append('cache: the ksplit word did not key its own program (cache %d -> %d)' % (after_oneq, after_ks))
    if after_starts != after_ks:
        report['failures'].append('cache: a chunk_start change rebuilt the ksplit program (cache %d -> %d)' % (after_ks, after_starts))


def grid_checks(bench, args, report):
    """The same statistic on an 11 x 10 grid (the grid the stack ran before the unlock) on a 13 x 10 device: s2 and s4 at a few starts."""
    failures = report['failures']
    torch = bench.torch
    grid = GRID_EXTRA
    if grid[0] > bench.native_grid[0] or grid[1] > bench.native_grid[1] or grid == bench.native_grid:
        return
    for name, rows in (('s2', 1024), ('s4', 512)):
        the_plan = config_plan(rows, grid, bench.native_grid)
        if the_plan['refusal']:
            failures.append('grid %dx%d %s: planner refusal %s' % (grid + (name, the_plan['refusal'])))
            continue
        for start in GRID_STARTS:
            oneq = bench.host(bench.call(rows, 'dram', 0, 'normal', start, word(ARMS['oneq']), grid=grid), 'read back oneq')
            ks = bench.host(bench.call(rows, 'dram', 0, 'normal', start, word(ARMS['ksplit']), grid=grid), 'read back ksplit')
            stats = numerics.error_stats(ks, oneq, torch)
            problems = numerics.within_bound(stats, numerics.bound_for(start + rows))
            label = '%s-g%dx%d@%d' % (name, grid[0], grid[1], start)
            report['grid_cases'].append(dict(case=label, rows=rows, slices=the_plan['slices'], stats=stats, problems=problems))
            if problems:
                failures.append('%s: %s' % (label, '; '.join(problems)))
    print('grid passes: %d cases, %d failures so far' % (len(report['grid_cases']), len(failures)), flush=True)


def time_arms(bench, args, report):
    """Interleaved single-call timings, one attention layer per sample: rounds of every (arm, start), the arm order rotated. arms: oneq and ksplit (stock with --time-stock)."""
    ttnn, device = bench.ttnn, bench.device
    timing = report['timing']
    arms = ['oneq', 'ksplit'] + (['stock'] if args.time_stock else [])
    for name, rows, grid in CONFIGS:
        if name not in args.time_configs:
            continue
        starts = list(args.time_starts)
        card.validate_starts(starts, rows)
        for qmem in args.time_q_memory:
            samples = {(arm, start): [] for arm in arms for start in starts}
            for arm in arms:
                for start in starts:
                    for _ in range(args.warmup):                         # the first call of a program JIT-compiles it
                        bench.ttnn.deallocate(bench.call(rows, qmem, 0, 'normal', start, word(ARMS[arm]), grid=grid, record=arm != 'stock'))
                    with WATCHDOG.op('timing warm sync'):
                        ttnn.synchronize_device(device)
            for arm, start in card.schedule(arms, starts, args.rounds):
                with WATCHDOG.op('%s@%d timed' % (arm, start)):
                    ttnn.synchronize_device(device)
                    began = time.perf_counter()
                    out = bench.call(rows, qmem, 0, 'normal', start, word(ARMS[arm]), grid=grid, record=False, grace=0.0)
                    ttnn.synchronize_device(device)
                    samples[(arm, start)].append((time.perf_counter() - began) * 1e3)
                ttnn.deallocate(out)
            block = timing.setdefault(name, {}).setdefault(qmem, {})
            for (arm, start), values in samples.items():
                block.setdefault(arm, {})[str(start)] = dict(median_ms=statistics.median(values), min_ms=min(values), samples=values)
    report['timing_verdict'] = {name: timing_verdict(timing, name) for name in timing}
    return report['timing_verdict']


def print_timing(report):
    for name, verdicts in sorted((report.get('timing_verdict') or {}).items()):
        for qmem, entry in sorted(verdicts.items()):
            print('KSPLIT TIME_VERDICT config=%s q=%s %s ksplit/oneq=%s small_worst=%s small_loss=%s' % (
                name, qmem, entry['verdict'], '%.3f' % entry['ratio'] if entry['ratio'] is not None else '-',
                '%.3f' % entry['small_worst'] if entry['small_worst'] is not None else '-', entry['small_loss']))
            for start, ratio in sorted(entry['ratios'].items()):
                print('KSPLIT TIME config=%s q=%s start=%d context=%d ksplit/oneq=%.3f' % (name, qmem, start, start + dict((c[0], c[1]) for c in CONFIGS)[name], ratio))


def planted_hang(bench, args, report):
    """The planted hang (flag 0x400, QWEN_SDPA_PF_TEST=1): every receiving slice withholds its hand-off credit, so the sending slice waits in the bounded credit wait (waypoint QKDC) and the
    call never returns. Returning is the failure; the watchdog's exit 3 is the pass."""
    if os.environ.get('QWEN_SDPA_PF_TEST') != '1':
        report['failures'].append('the planted hang needs QWEN_SDPA_PF_TEST=1 in the container')
        return
    shape = (1024, 'dram', 0, 'normal', 65536)
    # The pre-check: the card runs, the inputs are uploaded and the ksplit program (without the hang bit) is JIT-warm: the planted call then compiles nothing new but the
    # program word's own cache entry, so a watchdog on 'open device' or an upload is never mistaken for the planted hang.
    warm = bench.sha(*shape[:5], word(ARMS['ksplit']))
    report['hang'] = dict(precheck_ksplit_sha256=warm, armed=True, returned=False)
    print('%s: ksplit pre-check returned (%s); flags %#x, rows 1024, start 65536; PASS = the watchdog fires on %r or %r (%ss)'
          % (HANG_ARMED, warm[:16], HANG_FLAGS, HANG_CALL_LABEL, HANG_READBACK_LABEL, args.watchdog), flush=True)
    began = time.monotonic()
    out = bench.call(*shape[:5], word(HANG_FLAGS), label=HANG_CALL_LABEL, grace=card.COMPILE_GRACE_S)
    print('planted ksplit hang: enqueued; reading back (must never return)', flush=True)
    bench.host(out, label=HANG_READBACK_LABEL)
    report['hang'].update(returned=True, seconds=time.monotonic() - began)
    report['failures'].append('the planted ksplit hang RETURNED a result after %.1f s (a bounded wait fell through): the hand-off may not go near serving cards' % report['hang']['seconds'])


def run(args, report):
    import torch
    import ttnn

    with WATCHDOG.op('open device', extra=COMPILE_GRACE_S):
        device = ttnn.open_device(device_id=args.device_id, l1_small_size=24576)
    try:
        if args.maps:
            binary = loaded_binary()
            report['binary'] = binary
            print('binary %s sha256 %s chain_factory=%s oneq_edits=%s ksplit_edits=%s' % (binary['path'], binary['sha256'][:16], binary['chain'], binary['oneq'], binary['ksplit']),
                  flush=True)
            if not (binary['chain'] and binary['oneq'] and binary['ksplit']):
                report['failures'].append('the loaded _ttnncpp.so lacks the %s: the K64j-OQ-KS graft is not mounted' % (
                    '[QWEN-SDPA-PF] factory' if not binary['chain'] else 'oneq edits' if not binary['oneq'] else 'ksplit edits'))
                return
            if args.expect_binary_sha256 and binary['sha256'] != args.expect_binary_sha256:
                report['failures'].append('the loaded _ttnncpp.so is %s, not the expected %s' % (binary['sha256'][:16], args.expect_binary_sha256[:16]))
                return
        requested = set()
        report['_requested'] = requested
        bench = card.Bench(ttnn, torch, device, 'tp4', card.POOL_BLOCKS, requested)
        report['grid'] = list(bench.native_grid)
        print('device grid %dx%d (%d cores)' % (bench.native_grid + (bench.native_grid[0] * bench.native_grid[1],)), flush=True)
        if args.role == 'hang':
            planted_hang(bench, args, report)
            return
        tp2_bench = card.Bench(ttnn, torch, device, 'tp2', card.TP2_POOL_BLOCKS, requested)
        if not args.no_controls:
            cache_checks(bench, args, report)
            refusals(bench, tp2_bench, args, report)
            cap_check(bench, args, report)
        sweep(bench, args, report)
        if args.grids:
            grid_checks(bench, args, report)
        if not args.no_timing:
            time_arms(bench, args, report)
    finally:
        with WATCHDOG.op('close device'):
            ttnn.close_device(device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--role', choices=('numerics', 'hang'), default='numerics')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--expect-binary-sha256', default='', help='the K64j-OQ-KS _ttnncpp.so the run must have mapped (the runner passes it)')
    parser.add_argument('--configs', default=','.join(name for name, _r, _g in CONFIGS))
    parser.add_argument('--starts', default=','.join(map(str, STARTS)))
    parser.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--q-memory', default=','.join(Q_MEMORY))
    parser.add_argument('--reference-max-start', type=int, default=REFERENCE_MAX_START, help='the float reference is computed only up to this chunk start (host CPU time)')
    parser.add_argument('--no-anchor', dest='anchor', action='store_false', help='skip the oneq-equals-stock anchor of the first seed')
    parser.add_argument('--grids', action='store_true', default=True)
    parser.add_argument('--no-grids', dest='grids', action='store_false')
    parser.add_argument('--no-timing', action='store_true')
    parser.add_argument('--time-configs', default='s2,s4')
    parser.add_argument('--time-starts', default=','.join(map(str, TIME_STARTS)))
    parser.add_argument('--time-q-memory', default='dram')
    parser.add_argument('--time-stock', action='store_true')
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--no-controls', action='store_true', help='skip the cache, refusal and cap checks')
    parser.add_argument('--watchdog', type=float, default=120.0, help='seconds per device call; 0 off')
    parser.add_argument('--no-maps', dest='maps', action='store_false', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.configs = card.parse_list(args.configs, str)
    args.starts = card.parse_list(args.starts)
    args.seeds = card.parse_list(args.seeds)
    args.variants = card.parse_list(args.variants, str)
    args.q_memory = card.parse_list(args.q_memory, str)
    args.time_configs = card.parse_list(args.time_configs, str)
    args.time_starts = card.parse_list(args.time_starts)
    args.time_q_memory = card.parse_list(args.time_q_memory, str)
    known = [name for name, _r, _g in CONFIGS]
    for name in args.configs + args.time_configs:
        if name not in known:
            parser.error('unknown config %r (known: %s)' % (name, ', '.join(known)))
    for values, allowed, label in ((args.variants, VARIANTS, 'variant'), (args.q_memory, Q_MEMORY, 'Q memory'), (args.time_q_memory, Q_MEMORY, 'Q memory')):
        if any(value not in allowed for value in values):
            parser.error('unknown %s in %r' % (label, values))
    try:
        for _name, rows, _grid in CONFIGS:
            card.validate_starts(args.starts, rows)
            card.validate_starts(args.time_starts, rows)
    except ValueError as error:
        parser.error(str(error))
    if args.role == 'hang':
        args.no_timing = True
    return args


def main(argv=None):
    args = parse_args(argv)
    report = dict(passed=False, role=args.role, configs=args.configs, starts=args.starts, seeds=args.seeds, variants=args.variants, q_memory=args.q_memory,
                  bound=numerics.BOUND, bound_growth=dict(from_blocks=numerics.GROWTH_BLOCKS, frac_cap=numerics.FRAC_CAP), noise=numerics.NOISE, failures=[], cases=[], grid_cases=[], timing={}, expect_binary_sha256=args.expect_binary_sha256 or None)
    native = Q1.NativeLog(args.out.with_name(args.out.name + '.native.log'))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write_report(extra=None):
        payload = {key: value for key, value in report.items() if not key.startswith('_')}
        payload['requested_programs'] = sorted('%s rows=%d width=%d q=%s flags=%#x grid=%dx%d' % (key[:5] + key[5]) for key in report.get('_requested', ()))
        payload['numerics'] = numerics_summary(report['cases'])
        if extra:
            payload.update(extra)
        args.out.write_text(json.dumps(payload, indent=2, default=str))

    def on_fire(label):
        extra = dict(error='watchdog: %r exceeded %ss' % (label, args.watchdog), passed=False, watchdog=label)
        if args.role == 'hang':
            extra.update(passed=label in HANG_LABELS, hang=dict(report.get('hang') or {}, returned=False, watchdog=label))
        try:
            write_report(extra)
        except Exception:  # noqa: BLE001 - the WATCHDOG line stands
            pass

    globals()['WATCHDOG'] = card.WATCHDOG = Q1.Watchdog(args.watchdog, on_fire=on_fire).start()
    try:
        with native:
            run(args, report)
        if args.role == 'numerics' and (report.get('binary') or not args.maps):
            for problem in check_factory_lines(native.text(), report.get('_requested', set())):
                report['failures'].append('factory log: ' + problem)
        report['passed'] = not report['failures'] and (bool(report['cases']) if args.role == 'numerics' else False)
    except Exception as error:  # noqa: BLE001
        report['error'] = '%s: %s' % (type(error).__name__, error)
        report['traceback'] = traceback.format_exc()[-4000:]
    finally:
        write_report()
    for failure in report['failures']:
        print('FAIL', failure)
    if report.get('error'):
        print('ERROR', report['error'])
        print(report.get('traceback', ''))
    summary = numerics_summary(report['cases'])
    print_timing(report)
    print('KSPLIT_CARD_M role=%s verdict=%s cases=%d in_bound=%s worst_max_ulps=%s worst_rms_ulps=%s grid=%d failures=%d report=%s native_log=%s' % (
        args.role, 'PASS' if report['passed'] else 'FAIL', len(report['cases']), summary.get('in_bound', 0),
        '%.3f' % summary['worst_max_ulps'] if summary else '-', '%.3f' % summary['worst_rms_ulps'] if summary else '-', len(report['grid_cases']),
        len(report['failures']), args.out, native.path))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
