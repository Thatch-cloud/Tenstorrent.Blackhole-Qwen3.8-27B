"""CPU tests of the sub-device overlap harnesses' pure half and of H1 against a fake ttnn that models the command-queue and sub-device semantics.

Run: python -B -m unittest discover -s optimisation/ttnn-op/subdev_h -p 'test_*.py'
"""

import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fake_ttnn  # noqa: E402
import subdev_h1  # noqa: E402
import subdev_plan as plan  # noqa: E402

ROOT = HERE.parent.parent.parent
BASH = shutil.which('bash')
SMALL = ['--target-ops', '40', '--drafter-ops', '60', '--rounds', '6', '--warmup', '1', '--stress', '3', '--manager-cycles', '2']


def arms(**medians):
    return {name: dict(n=10, median_ms=value, min_ms=value) for name, value in medians.items()}


class SplitTests(unittest.TestCase):
    def test_the_default_is_80_and_30_of_the_110_core_grid_in_whole_columns(self):
        found = plan.plan_split(11, 10)
        self.assertEqual((found['target'], found['drafter']), ([0, 0, 7, 9], [8, 0, 10, 9]))
        self.assertEqual((found['target_cores'], found['drafter_cores'], found['idle_cores']), (80, 30, 0))
        self.assertEqual((found['target_grid'], found['drafter_grid']), ([8, 10], [3, 10]))
        self.assertTrue(found['exact'])

    def test_a_narrower_grid_scales_both_and_says_it_is_not_exact(self):
        found = plan.plan_split(10, 10)
        self.assertEqual((found['target_cores'], found['drafter_cores']), (70, 30))
        self.assertFalse(found['exact'])
        found = plan.plan_split(4, 10)
        self.assertEqual((found['target_grid'][0], found['drafter_grid'][0]), (3, 1))

    def test_the_rectangles_are_disjoint_and_inside_the_grid_for_every_grid(self):
        for grid_x in range(2, 15):
            for grid_y in (1, 8, 10, 12):
                found = plan.plan_split(grid_x, grid_y)
                self.assertTrue(plan.rectangles_disjoint(found['target'], found['drafter']), (grid_x, grid_y))
                for rect in (found['target'], found['drafter']):
                    self.assertTrue(0 <= rect[0] <= rect[2] < grid_x and 0 <= rect[1] <= rect[3] < grid_y)
                self.assertGreaterEqual(found['idle_cores'], 0)
                self.assertEqual(found['target_cores'] + found['drafter_cores'] + found['idle_cores'], grid_x * grid_y)

    def test_a_grid_that_cannot_hold_two_sub_devices_is_refused(self):
        with self.assertRaises(plan.PlanError):
            plan.plan_split(1, 10)
        with self.assertRaises(plan.PlanError):
            plan.plan_split(11, 10, 0, 30)

    def test_the_disjoint_test_sees_overlap(self):
        self.assertFalse(plan.rectangles_disjoint([0, 0, 7, 9], [7, 0, 10, 9]))
        self.assertTrue(plan.rectangles_disjoint([0, 0, 7, 9], [8, 0, 10, 9]))


class RecipeTests(unittest.TestCase):
    def test_the_target_recipe_is_about_four_hundred_ops_and_the_drafter_about_nine_hundred(self):
        target, drafter = plan.build_recipe('target', plan.TARGET_OPS), plan.build_recipe('drafter', plan.DRAFTER_OPS)
        self.assertEqual((target['count'], drafter['count']), (400, 900))
        self.assertEqual(target['matmuls'], 200)
        self.assertGreater(drafter['count'] - drafter['matmuls'], drafter['matmuls'])   # the drafter is eltwise-heavy: launch-bound

    def test_every_source_is_defined_before_it_is_used_and_every_output_is_distinct(self):
        for profile, ops in (('target', 400), ('drafter', 900), ('target', 7), ('drafter', 13)):
            recipe = plan.build_recipe(profile, ops)
            defined = set(recipe['inputs'])
            outputs = [op[1] for op in recipe['ops']]
            self.assertEqual(len(set(outputs)), len(outputs))
            for kind, dest, first, second in recipe['ops']:
                self.assertIn(first, defined)
                self.assertTrue(second is None or second in defined)
                self.assertEqual(second is None, kind == 'silu')
                defined.add(dest)
            self.assertEqual(recipe['output'], outputs[-1])

    def test_the_weights_a_recipe_reads_are_the_profiles_and_their_shapes_chain(self):
        for profile in plan.PROFILES:
            shapes, spec = plan.weight_shapes(profile), plan.PROFILES[profile]
            recipe = plan.build_recipe(profile, 120)
            used = {name for op in recipe['ops'] for name in op[2:] if name and name.startswith('w_')}
            self.assertEqual(used, set(shapes))
            self.assertEqual(shapes['w_gate'], (spec['hidden'], spec['inter']))
            self.assertEqual(shapes['w_down'], (spec['inter'], spec['hidden']))
            self.assertEqual(shapes['w_o'], (spec['attn'], spec['hidden']))
            for k, n in shapes.values():
                self.assertEqual((k % 32, n % 32), (0, 0))

    def test_a_recipe_is_deterministic_and_prefix_stable(self):
        self.assertEqual(plan.build_recipe('target', 50)['ops'], plan.build_recipe('target', 50)['ops'])
        self.assertEqual(plan.build_recipe('target', 50)['ops'], plan.build_recipe('target', 400)['ops'][:50])

    def test_bad_recipes_are_refused(self):
        with self.assertRaises(plan.PlanError):
            plan.build_recipe('nothing', 10)
        with self.assertRaises(plan.PlanError):
            plan.build_recipe('target', 0)

    def test_checkpoints_are_every_nth_output_and_always_the_last(self):
        recipe = plan.build_recipe('target', 20)
        names = plan.checkpoint_names(recipe, 8)
        self.assertEqual(names, ['o0', 'o8', 'o16', 'o19'])
        self.assertEqual(plan.checkpoint_names(plan.build_recipe('target', 17), 8)[-1], 'o16')
        self.assertEqual(len(plan.checkpoint_names(recipe, 1)), 20)


class StatisticsAndVerdictTests(unittest.TestCase):
    def test_summary(self):
        found = plan.summarize([5000000, 1000000, 3000000, 2000000, 4000000])
        self.assertEqual((found['n'], found['median_ms'], found['min_ms'], found['mean_ms']), (5, 3.0, 1.0, 3.0))
        self.assertEqual(plan.summarize([]), {'n': 0})

    def test_the_percentiles_bracket_the_median(self):
        found = plan.summarize(list(range(1000000, 101000000, 1000000)))
        self.assertLessEqual(found['p10_ms'], found['median_ms'])
        self.assertLessEqual(found['median_ms'], found['p90_ms'])

    def exact(self, compared=10, mismatched=0):
        return dict(compared=compared, mismatched=mismatched)

    def test_pass_is_concurrent_within_a_tenth_of_the_longer_solo(self):
        text, evidence = plan.verdict(arms(t_solo=10.0, d_solo=8.0, concurrent=11.0), self.exact())
        self.assertEqual(text, 'PASS')
        self.assertEqual(evidence['ratio'], 1.1)
        self.assertEqual(evidence['overlap'], 0.875)
        self.assertEqual(plan.verdict(arms(t_solo=10.0, d_solo=8.0, concurrent=11.01), self.exact())[0], 'FAIL-PARTIAL')

    def test_a_concurrent_wall_near_the_sum_is_serialised(self):
        text, evidence = plan.verdict(arms(t_solo=10.0, d_solo=8.0, concurrent=17.0), self.exact())
        self.assertEqual(text, 'FAIL-SERIALISED')
        self.assertEqual(evidence['concurrent_over_sum'], 0.9444)
        self.assertEqual(plan.verdict(arms(t_solo=10.0, d_solo=8.0, concurrent=16.2), self.exact())[0], 'FAIL-SERIALISED', 'exactly 0.9 of the sum')
        self.assertEqual(plan.verdict(arms(t_solo=10.0, d_solo=8.0, concurrent=16.1), self.exact())[0], 'FAIL-PARTIAL')

    def test_bytes_decide_before_timing(self):
        good = arms(t_solo=10.0, d_solo=8.0, concurrent=10.0)
        self.assertEqual(plan.verdict(good, self.exact(mismatched=1))[0], 'FAIL-BYTES')
        self.assertEqual(plan.verdict({}, self.exact(mismatched=1))[0], 'FAIL-BYTES')
        self.assertEqual(plan.verdict(good, self.exact(compared=0))[0], 'INCOMPLETE')

    def test_an_error_is_not_a_verdict_and_missing_arms_are_incomplete(self):
        self.assertEqual(plan.verdict({}, self.exact(0), error='TT_FATAL')[0], 'NOT-MEASURED')
        self.assertEqual(plan.verdict(arms(t_solo=10.0, d_solo=8.0), self.exact())[0], 'INCOMPLETE')
        self.assertEqual(plan.verdict(arms(t_solo=10.0, concurrent=9.0), self.exact())[0], 'INCOMPLETE')

    def test_without_timing_the_watcher_pass_judges_bytes_only(self):
        slow = arms(t_solo=10.0, d_solo=8.0, concurrent=40.0)
        self.assertEqual(plan.verdict(slow, self.exact(), timing=False)[0], 'UNTIMED-PASS')
        self.assertEqual(plan.verdict(slow, self.exact(mismatched=2), timing=False)[0], 'FAIL-BYTES')

    def test_a_custom_pass_ratio(self):
        self.assertEqual(plan.verdict(arms(t_solo=10.0, d_solo=8.0, concurrent=10.4), self.exact(), pass_ratio=1.03)[0], 'FAIL-PARTIAL')

    def test_exit_status(self):
        self.assertEqual([plan.exit_status(text) for text in ('PASS', 'UNTIMED-PASS', 'FAIL-BYTES', 'FAIL-SERIALISED', 'FAIL-PARTIAL', 'INCOMPLETE',
                                                              'NOT-MEASURED', 'HANG', 'PASS-SEPARATE-LINKS')], [0, 0, 1, 1, 1, 1, 2, 3, 0])

    def test_the_verdict_line_carries_the_numbers_and_prints_dashes_for_missing_ones(self):
        measured = arms(t_solo=10.0, d_solo=8.0, concurrent=10.5, chained=18.4, one_queue=17.0)
        text, evidence = plan.verdict(measured, self.exact())
        line = plan.verdict_line(text, plan.flatten_evidence(measured, evidence), plan=plan.plan_split(11, 10), manager=dict(load_ms_median=1.5, clear_ms_median=0.9))
        self.assertTrue(line.startswith('SUBDEV_H1 verdict=PASS ratio=1.05 conc_ms=10.5 t_ms=10.0 d_ms=8.0 sum_ms=18.0 chained_ms=18.4 one_queue_ms=17.0 '), line)
        self.assertIn('bytes=identical', line)
        self.assertIn('split=80/30 grid=11x10 load_ms=1.5 clear_ms=0.9', line)
        bare = plan.verdict_line('NOT-MEASURED', dict(timing=True, compared=0, mismatched=0), extra=dict(error='"boom"'))
        self.assertIn('ratio=- conc_ms=-', bare)
        self.assertIn('error="boom"', bare)
        self.assertEqual(len([line for line in (line, bare) if '\n' in line]), 0)

    def test_the_chained_and_one_queue_arms_are_reported_against_the_sum(self):
        found = plan.ratios(arms(t_solo=10.0, d_solo=8.0, concurrent=10.0, chained=19.0, one_queue=17.1))
        self.assertEqual((found['chained_over_sum'], found['one_queue_over_sum']), (1.0556, 0.95))

    def test_h2_verdict(self):
        base = dict(t_solo=10.0, d_solo=8.0)
        exact = self.exact()
        self.assertEqual(plan.verdict_h2(arms(shared=10.5, **base), exact)[0], 'PASS')
        self.assertEqual(plan.verdict_h2(arms(shared=17.5, **base), exact)[0], 'FAIL-SERIALISED')
        self.assertEqual(plan.verdict_h2(arms(shared=17.5, separate=10.2, **base), exact, separate_run=True)[0], 'PASS-SEPARATE-LINKS')
        self.assertEqual(plan.verdict_h2(arms(shared=17.5, separate=17.0, **base), exact, separate_run=True)[0], 'FAIL-SERIALISED')
        self.assertEqual(plan.verdict_h2(arms(shared=14.0, separate=13.0, **base), exact, separate_run=True)[0], 'FAIL-PARTIAL')
        self.assertEqual(plan.verdict_h2(arms(shared=17.5, separate=10.2, **base), exact)[0], 'FAIL-SERIALISED', 'a separate arm that did not run decides nothing')
        self.assertEqual(plan.verdict_h2(arms(shared=10.5, **base), self.exact(mismatched=1))[0], 'FAIL-BYTES')
        self.assertEqual(plan.verdict_h2(arms(shared=10.5, **base), exact, timing=False)[0], 'UNTIMED-PASS')
        self.assertEqual(plan.verdict_h2(arms(**base), exact)[0], 'INCOMPLETE')
        self.assertEqual(plan.verdict_h2({}, self.exact(0), error='x')[0], 'NOT-MEASURED')
        self.assertEqual(plan.verdict_h2(arms(shared=10.5, shared2=9.0, **base), exact)[1]['shared2_ratio'], 0.9)


class ArgumentTests(unittest.TestCase):
    def test_defaults_are_the_design(self):
        options, problems = plan.parse_args(['--out', 'x.json'])
        self.assertEqual(problems, [])
        self.assertEqual((options.target_cores, options.drafter_cores, options.target_ops, options.drafter_ops), (80, 30, 400, 900))
        self.assertEqual((options.rows, options.drafter_rows, options.pass_ratio, options.no_timing), (64, 32, 1.10, False))
        self.assertGreater(options.compile_watchdog_s, options.watchdog_s)

    def test_the_watcher_pass_shortens_the_loops(self):
        options, problems = plan.parse_args(['--out', 'x.json', '--no-timing'])
        self.assertEqual((problems, options.rounds <= 4, options.stress <= 6), ([], True, True))

    def test_timing_after_no_timing_turns_the_watcher_pass_back_into_a_timed_one(self):
        options, problems = plan.parse_args(['--out', 'x.json', '--no-timing', '--timing'])
        self.assertEqual((problems, options.no_timing, options.rounds), ([], False, 30))
        options, problems = plan.parse_args(['--out', 'x.json', '--timing', '--no-timing'])
        self.assertEqual((options.no_timing, options.rounds <= 4), (True, True))

    def test_bad_arguments_are_named(self):
        for bad, needle in ((['--rows', '48'], '--rows'), (['--drafter-rows', '0'], '--drafter-rows'), (['--target-ops', '2'], '--target-ops'),
                            (['--rounds', '2'], '--rounds'), (['--pass-ratio', '1.0'], '--pass-ratio'), (['--check-every', '0'], '--check-every'),
                            (['--target-cores', '0'], 'at least one core'), (['--watchdog-s', '-1'], 'watchdog')):
            options, problems = plan.parse_args(['--out', 'x.json'] + bad)
            self.assertTrue(any(needle in problem for problem in problems), (bad, problems))

    def test_the_out_path_is_required(self):
        with self.assertRaises(SystemExit):
            with unittest.mock.patch('sys.stderr', new=io.StringIO()):
                plan.parse_args([])


class WatchdogTests(unittest.TestCase):
    def make(self, **kwargs):
        clock = [0.0]
        stream, exits, fired = io.StringIO(), [], []
        dog = plan.Watchdog(backstop=False, exit_fn=exits.append, clock=lambda: clock[0], stream=stream, on_fire=fired.append, **kwargs)
        return dog, clock, stream, exits, fired

    def test_a_call_past_its_deadline_prints_the_hang_line_writes_the_partial_report_and_exits_three(self):
        dog, clock, stream, exits, fired = self.make()
        with dog.span('capture-t', 100):
            self.assertFalse(dog.check())
            clock[0] = 99.9
            self.assertFalse(dog.check())
            clock[0] = 100.0
            self.assertTrue(dog.check())
        self.assertEqual(exits, [3])
        self.assertEqual(fired, ['capture-t'])
        self.assertEqual(stream.getvalue(), 'SUBDEV_H1 verdict=HANG stage=capture-t after_s=100\n')
        self.assertEqual(dog.fired, 'capture-t')

    def test_nothing_fires_outside_a_span_or_after_it_returned(self):
        dog, clock, stream, exits, fired = self.make()
        clock[0] = 1000.0
        self.assertFalse(dog.check())
        with dog.span('quick', 5):
            clock[0] = 1001.0
        clock[0] = 5000.0
        self.assertFalse(dog.check())
        self.assertEqual((exits, stream.getvalue()), ([], ''))

    def test_spans_nest_and_the_inner_deadline_governs_until_it_returns(self):
        dog, clock, stream, exits, fired = self.make()
        with dog.span('outer', 100):
            clock[0] = 10.0
            with dog.span('inner', 20):
                self.assertEqual(dog.current(), 'inner')
                clock[0] = 31.0
                self.assertTrue(dog.check())
        self.assertEqual(fired, ['inner'])
        dog, clock, stream, exits, fired = self.make()
        with dog.span('outer', 100):
            with dog.span('inner', 20):
                clock[0] = 15.0
            self.assertEqual(dog.current(), 'outer')
            clock[0] = 99.0
            self.assertFalse(dog.check())
            clock[0] = 101.0
            self.assertTrue(dog.check())
        self.assertEqual(fired, ['outer'])

    def test_a_zero_budget_is_no_watchdog(self):
        dog, clock, stream, exits, fired = self.make()
        with dog.span('free', 0):
            clock[0] = 1e9
            self.assertFalse(dog.check())

    def test_the_tag_names_the_probe(self):
        stream = io.StringIO()
        dog = plan.Watchdog(tag='SUBDEV_H2', backstop=False, exit_fn=lambda code: None, clock=lambda: 50.0, stream=stream)
        dog.stack.append(('gather', 0.0, 10))
        dog.check()
        self.assertTrue(stream.getvalue().startswith('SUBDEV_H2 verdict=HANG stage=gather'))

    def test_the_faulthandler_backstop_is_armed_per_span_and_cancelled_after_the_outermost(self):
        calls = []
        with unittest.mock.patch('faulthandler.dump_traceback_later', lambda *a, **k: calls.append(('arm', a, k))), \
                unittest.mock.patch('faulthandler.cancel_dump_traceback_later', lambda: calls.append(('cancel',))):
            dog = plan.Watchdog(grace=60.0, exit_fn=lambda code: None, stream=io.StringIO())
            with dog.span('a', 10):
                with dog.span('b', 5):
                    pass
        self.assertEqual([c[0] for c in calls], ['arm', 'arm', 'arm', 'cancel'])
        self.assertEqual(calls[0][1], (70.0,))
        self.assertTrue(calls[0][2]['exit'])


class HeartbeatTests(unittest.TestCase):
    def test_the_line_names_the_stage_and_how_long_it_has_run(self):
        clock = [100.0]
        stream = io.StringIO()
        beat = plan.Heartbeat('SUBDEV_H2', clock=lambda: clock[0], stream=stream)
        beat.set('eager-concurrent')
        clock[0] = 147.9
        beat.beat()
        self.assertEqual(stream.getvalue(), 'SUBDEV_H2 alive stage=eager-concurrent in_stage_s=47\n')


def harness_run(extra=None, tmp=None, **fake):
    """main() of H1 against a FakeTTNN; returns (status, printed lines, the JSON report, the fake)."""
    ttnn, torch = fake_ttnn.FakeTTNN(**fake), fake_ttnn.FakeTorch()
    lines, out = [], os.path.join(tmp, 'report.json')
    dog = plan.Watchdog(backstop=False, exit_fn=lambda code: lines.append('EXIT %d' % code))
    status = subdev_h1.main(['--out', out] + SMALL + list(extra or ()), ttnn=ttnn, torch=torch, log=lines.append, clock=ttnn.clock, watchdog=dog)
    with open(out) as handle:
        report = json.load(handle)
    return status, lines, report, ttnn


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def run_fake(self, extra=None, **fake):
        return harness_run(extra, self.directory.name, **fake)

    def verdict_line(self, lines):
        found = [line for line in lines if isinstance(line, str) and line.startswith('SUBDEV_H1 verdict=')]
        self.assertEqual(len(found), 1, lines[-6:])
        return found[0]

    def test_queues_that_overlap_pass_and_the_protocol_never_trips_the_ownership_rules(self):
        status, lines, report, ttnn = self.run_fake()
        line = self.verdict_line(lines)
        self.assertEqual(status, 0, line)
        self.assertEqual(report['verdict'], 'PASS')
        self.assertNotIn('error', report, report.get('traceback'))
        self.assertLessEqual(report['evidence']['ratio'], 1.1)
        self.assertGreater(report['evidence']['overlap'], 0.8)
        self.assertIn('bytes=identical', line)
        self.assertEqual(json.loads(lines[-1])['kind'], 'subdev-h1')
        self.assertEqual(lines.index(line), len(lines) - 2, 'the verdict line then the JSON object are the last two lines')

    def test_the_queues_traces_and_sub_devices_are_the_designs(self):
        status, lines, report, ttnn = self.run_fake()
        self.assertEqual(ttnn.opened['queues'], 2)
        self.assertEqual(ttnn.opened['shape'], (1, 1))
        self.assertEqual(report['grid'], [11, 10])
        self.assertEqual(report['plan']['target_cores'], 80)
        self.assertEqual(report['plan']['drafter_cores'], 30)
        self.assertEqual(sorted(report['traces']), ['d0', 'd1', 't'])
        captured = {trace.ident: trace for trace in ttnn.traces.values()}
        target, drafter_q1, drafter_q0 = captured[1], captured[2], captured[3]
        self.assertEqual((target.cq, drafter_q1.cq, drafter_q0.cq), (0, 1, 0))
        self.assertEqual((target.sds, drafter_q1.sds, drafter_q0.sds), ({0}, {1}, {1}))
        self.assertEqual((len(target.ops), len(drafter_q1.ops)), (40, 60))
        self.assertEqual(report['recipes'], dict(target=dict(ops=40, matmuls=20), drafter=dict(ops=60, matmuls=ttnn_matmuls(60, 'drafter'))))

    def test_the_program_cache_is_on_before_any_op_runs(self):
        status, lines, report, ttnn = self.run_fake()
        self.assertEqual(ttnn.calls[:2], ['open', 'program_cache'])
        self.assertTrue(ttnn.program_cache)

    def test_without_the_program_cache_no_trace_can_be_captured(self):
        ttnn = fake_ttnn.FakeTTNN()
        fake_ttnn.FakeMesh.enable_program_cache, saved = lambda self: None, fake_ttnn.FakeMesh.enable_program_cache
        try:
            lines, out = [], os.path.join(self.directory.name, 'c.json')
            status = subdev_h1.main(['--out', out] + SMALL, ttnn=ttnn, torch=fake_ttnn.FakeTorch(), log=lines.append, clock=ttnn.clock,
                                    watchdog=plan.Watchdog(backstop=False, exit_fn=lambda code: None))
        finally:
            fake_ttnn.FakeMesh.enable_program_cache = saved
        self.assertEqual(status, 2)
        self.assertIn('Expected program binaries to be written', self.verdict_line(lines))

    def test_the_manager_is_timed_and_cleared_and_the_traces_are_released(self):
        status, lines, report, ttnn = self.run_fake()
        manager = report['manager']
        self.assertEqual((len(manager['load_ms']), len(manager['clear_ms'])), (3, 2))     # two cycles, then the load that stays
        self.assertIn('create_ms', manager)
        self.assertIn('load_ms_median', manager)
        self.assertIn('final_clear_ms', manager)
        calls = ttnn.calls
        self.assertLess(max(i for i, c in enumerate(calls) if c == 'release'), len(calls) - 1 - calls[::-1].index('clear'))
        self.assertEqual(calls[-3:], ['clear', 'remove', 'close'], 'the traces are released, then the manager cleared and removed, then the mesh closed')
        self.assertTrue(all(trace.released for trace in ttnn.traces.values()))
        self.assertIsNone(ttnn.loaded)
        self.assertTrue(report['closed'])

    def test_serialised_queues_fail_serialised(self):
        status, lines, report, ttnn = self.run_fake(overlap='serial')
        self.assertEqual((status, report['verdict']), (1, 'FAIL-SERIALISED'))
        self.assertGreaterEqual(report['evidence']['concurrent_over_sum'], 0.9)
        self.assertIn('verdict=FAIL-SERIALISED', self.verdict_line(lines))

    def test_partly_overlapping_queues_fail_partial(self):
        status, lines, report, ttnn = self.run_fake(overlap=0.2)
        self.assertEqual((status, report['verdict']), (1, 'FAIL-PARTIAL'))
        self.assertGreater(report['evidence']['ratio'], 1.1)

    def test_bytes_that_differ_under_overlap_fail_whatever_the_timing(self):
        status, lines, report, ttnn = self.run_fake(corrupt_on_overlap=True)
        self.assertEqual((status, report['verdict']), (1, 'FAIL-BYTES'))
        self.assertGreater(report['exactness']['mismatched'], 0)
        self.assertTrue(any(line.startswith('SUBDEV_H1 bytes DIFFER case=') for line in lines if isinstance(line, str)))
        cases = {case['case'] for case in report['exactness']['cases'] if case['mismatched']}
        self.assertTrue(any(case.startswith('concurrent') for case in cases), cases)
        self.assertFalse(any(case.startswith('one_queue') for case in cases) and not any(case.startswith('concurrent') for case in cases))

    def test_a_trace_that_wrote_only_zeros_is_not_measured_not_a_vacuous_pass(self):
        ttnn = fake_ttnn.FakeTTNN()
        original = ttnn.to_torch
        ttnn.to_torch = lambda tensor, **kwargs: fake_ttnn.FakeArray(tensor.shape, ('zeros',), 'bfloat16')
        lines, out = [], os.path.join(self.directory.name, 'z.json')
        status = subdev_h1.main(['--out', out] + SMALL, ttnn=ttnn, torch=fake_ttnn.FakeTorch(), log=lines.append, clock=ttnn.clock,
                                watchdog=plan.Watchdog(backstop=False, exit_fn=lambda code: None))
        self.assertEqual(status, 2)
        self.assertIn('a trace wrote only zeros', self.verdict_line(lines))
        del original

    def test_a_queue_that_silently_runs_nothing_cannot_pass_on_stale_or_poisoned_bytes(self):
        status, lines, report, ttnn = self.run_fake(drop_replays_on=(1,))
        self.assertEqual((status, report['verdict']), (2, 'NOT-MEASURED'))
        self.assertIn('NaN or inf', report['error'])
        status, lines, report, ttnn = self.run_fake(drop_replays_on=(0,))
        self.assertEqual((status, report['verdict']), (2, 'NOT-MEASURED'))

    def test_the_references_are_replays_of_poisoned_outputs(self):
        status, lines, report, ttnn = self.run_fake()
        self.assertEqual(report['verdict'], 'PASS')
        self.assertTrue(all(not tensor.poisoned for trace in ttnn.traces.values() for tensor in trace.outputs),
                        'every replay rewrote the poison it was given')

    def test_the_weight_gains_are_the_simulated_stable_ones(self):
        self.assertEqual((plan.WEIGHT_GAIN, plan.OUT_GAIN, plan.SCALE_AMPLITUDE), (0.5, 0.02, 0.02))
        self.assertLessEqual(plan.OUT_GAIN, 0.05, '0.1 overflows bfloat16 within 900 drafter ops')

    def test_the_outputs_are_reported_finite_and_live(self):
        status, lines, report, ttnn = self.run_fake()
        self.assertEqual(report['live'], dict(target=True, drafter=True))
        self.assertEqual(report['finite'], dict(target=True, drafter=True))

    def test_the_watcher_pass_is_untimed(self):
        status, lines, report, ttnn = self.run_fake(extra=['--no-timing'], overlap='serial')
        self.assertEqual((status, report['verdict']), (0, 'UNTIMED-PASS'))
        self.assertIn('timing=0', self.verdict_line(lines))

    def test_every_compared_case_is_listed_and_the_eager_comparison_is_information_only(self):
        status, lines, report, ttnn = self.run_fake()
        cases = report['exactness']['cases']
        names = [case['case'] for case in cases]
        for expected in ('drafter queue 0 vs queue 1', 'concurrent #0 target', 'chained #1 drafter', 'one_queue #0 drafter(q0)', 'stress x3 target',
                         'after timing drafter'):
            self.assertIn(expected, names)
        informational = [case for case in cases if case.get('informational')]
        self.assertEqual(sorted(case['case'] for case in informational), ['drafter trace vs eager', 'target trace vs eager'])
        counted = sum(case['compared'] for case in cases if not case.get('informational'))
        self.assertEqual(counted, report['exactness']['compared'])

    def test_each_arm_is_timed_the_requested_number_of_rounds(self):
        status, lines, report, ttnn = self.run_fake()
        for name in plan.ARMS:
            self.assertEqual(report['arms'][name]['n'], 6, name)
            self.assertEqual(len(report['raw_ms'][name]), 6)
        self.assertGreater(report['arms']['chained']['median_ms'], report['arms']['concurrent']['median_ms'])
        self.assertGreater(report['arms']['one_queue']['median_ms'], report['arms']['concurrent']['median_ms'])

    def test_a_chip_that_reports_a_narrower_grid_gets_a_scaled_split(self):
        status, lines, report, ttnn = self.run_fake(grid=(10, 10))
        self.assertEqual((report['plan']['target_cores'], report['plan']['drafter_cores'], report['plan']['exact']), (70, 30, False))
        self.assertEqual(report['verdict'], 'PASS')

    def test_a_device_error_is_not_measured_with_the_error_in_the_line(self):
        ttnn = fake_ttnn.FakeTTNN()
        original = ttnn.matmul

        def broken(*args, **kwargs):
            raise RuntimeError('TT_FATAL: Programs must be executed on a single sub-device')
        ttnn.matmul = broken
        lines, out = [], os.path.join(self.directory.name, 'r.json')
        status = subdev_h1.main(['--out', out] + SMALL, ttnn=ttnn, torch=fake_ttnn.FakeTorch(), log=lines.append, clock=ttnn.clock,
                                watchdog=plan.Watchdog(backstop=False, exit_fn=lambda code: None))
        self.assertEqual(status, 2)
        line = self.verdict_line(lines)
        self.assertIn('verdict=NOT-MEASURED', line)
        self.assertIn('error="RuntimeError: TT_FATAL: Programs must be executed on a single sub-device"', line)
        with open(out) as handle:
            report = json.load(handle)
        self.assertTrue(ttnn.traces == {} or all(trace.released for trace in ttnn.traces.values()))
        self.assertEqual(report['verdict'], 'NOT-MEASURED')
        self.assertTrue(report['closed'])
        self.assertIn('traceback', report)
        del original

    def test_a_ttnn_without_the_sub_device_api_is_not_measured_before_it_opens_anything(self):
        ttnn = fake_ttnn.FakeTTNN()
        del ttnn.SubDevice
        lines, out = [], os.path.join(self.directory.name, 'r.json')
        status = subdev_h1.main(['--out', out] + SMALL, ttnn=ttnn, torch=fake_ttnn.FakeTorch(), log=lines.append, clock=ttnn.clock,
                                watchdog=plan.Watchdog(backstop=False, exit_fn=lambda code: None))
        self.assertEqual(status, 2)
        self.assertIsNone(ttnn.opened)
        self.assertIn('this ttnn has no SubDevice', self.verdict_line(lines))

    def test_bad_arguments_never_reach_the_device(self):
        ttnn = fake_ttnn.FakeTTNN()
        with unittest.mock.patch('sys.stderr', new=io.StringIO()) as stderr:
            status = subdev_h1.main(['--out', os.path.join(self.directory.name, 'r.json'), '--rows', '50'], ttnn=ttnn, torch=fake_ttnn.FakeTorch(),
                                    log=lambda line: None)
        self.assertEqual(status, 2)
        self.assertIn('refusing: --rows', stderr.getvalue())
        self.assertIsNone(ttnn.opened)

    def test_a_hang_leaves_a_partial_report_and_the_hang_verdict(self):
        ttnn = fake_ttnn.FakeTTNN()
        exits = []
        lines, out = [], os.path.join(self.directory.name, 'r.json')

        class Stuck(plan.Watchdog):
            def span(self, label, seconds):
                if label.startswith('arm-concurrent'):
                    self.stack.append((label, 0.0, seconds))
                    self.check()           # the deadline has passed
                    self.stack.pop()
                return super().span(label, seconds)

        dog = Stuck(backstop=False, exit_fn=exits.append, clock=lambda: 1e9, stream=io.StringIO(), on_fire=lambda label: lines.append('fired ' + label))
        subdev_h1.main(['--out', out] + SMALL, ttnn=ttnn, torch=fake_ttnn.FakeTorch(), log=lines.append, clock=ttnn.clock, watchdog=dog)
        self.assertEqual(exits[0], 3)
        self.assertEqual(lines[0 if lines[0].startswith('fired') else [i for i, x in enumerate(lines) if str(x).startswith('fired')][0]], 'fired arm-concurrent')


def ttnn_matmuls(ops, profile):
    return plan.build_recipe(profile, ops)['matmuls']


class ScriptTests(unittest.TestCase):
    SCRIPT = HERE / 'run_card_m.sh'

    def text(self):
        return self.SCRIPT.read_text()

    def test_the_script_names_its_harness_files_image_and_contract_off_hook(self):
        text = self.text()
        for needle in ('subdev_plan.py', 'subdev_h1.py', 'QWEN_C2_SERVING=0', '-u TT_MESH_GRAPH_DESC_PATH', 'TT_METAL_WATCHER=5',
                       '--no-timing', 'SUBDEV_DRY_RUN', 'IMAGE_TAG', 'qual_card_recheck', 'qual_reset_hint', "grep -qF 'Timeout ('"):
            self.assertIn(needle, text)
        self.assertNotIn(chr(13), text)

    def test_the_tracked_files_name_no_host_address_registry_digest_or_home(self):
        banned = re.compile(r'\d{1,3}(\.\d{1,3}){3}|thatch\.local|zot\.|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/home/|ssh |@[A-Za-z0-9.-]+\.(?:local|com)|ghp_|token', re.I)
        for path in [p for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')] + [self.SCRIPT] + sorted(HERE.glob('*.patch')):
            text = path.read_text()
            if path == self.SCRIPT:
                start = text.index('# >>> qual_card.sh')
                end = text.index('# <<< qual_card.sh')
                text = text[:start] + text[end:]
            for number, line in enumerate(text.splitlines(), 1):
                self.assertIsNone(banned.search(line), '%s:%d %s' % (path.name, number, line))

    @unittest.skipUnless(BASH, 'bash not found')
    def test_the_script_parses(self):
        result = subprocess.run([BASH, '-n', str(self.SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(BASH, 'bash not found')
    def test_the_dry_run_launches_the_harness_on_one_board_with_two_mounts_and_the_extra_arguments_last(self):
        board = 'blackhole-0000000000000001'
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, HOME=directory, RESULTS=os.path.join(directory, 'results'), QUAL_CARD=board, SUBDEV_DRY_RUN='1',
                       CARD_B_ARGS='--rounds 60 --pass-ratio 1.2', IMAGE='an-image-ref')
            for key in ('WATCHER', 'WATCHDOG_S', 'ALLOW_SERVING_CARD'):
                env.pop(key, None)
            result = subprocess.run([BASH, str(self.SCRIPT)], capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            import shlex
            argv = shlex.split([line for line in result.stdout.splitlines() if line.startswith('### argv: ')][0][len('### argv: '):])
            self.assertEqual(argv[argv.index('--device') + 1], '/dev/tenstorrent/by-id/' + board)
            self.assertEqual(argv.count('--device'), 1)
            self.assertEqual(argv[argv.index('--name') + 1], 'qwen-subdev-' + board)
            self.assertIn('--network', argv)
            self.assertEqual(argv[argv.index('--network') + 1], 'none')
            mounts = [word for word in argv if word.startswith('type=bind,src=') and word.endswith(',readonly')]
            self.assertEqual(sorted(os.path.basename(m.split(',dst=')[1].split(',')[0]) for m in mounts), ['subdev_h1.py', 'subdev_plan.py'])
            self.assertNotIn('TT_METAL_WATCHER=5', argv)
            self.assertNotIn('--no-timing', argv)
            tail = argv[argv.index('/bench/subdev_h1.py') + 1:]
            self.assertEqual(tail[-4:], ['--rounds', '60', '--pass-ratio', '1.2'])
            self.assertEqual(tail[tail.index('--watchdog-s') + 1], '120')
            self.assertIn('an-image-ref', argv)
            self.assertFalse(os.path.exists(os.path.join(directory, 'results')), 'a dry run creates nothing')

    @unittest.skipUnless(BASH, 'bash not found')
    def test_the_watcher_dry_run_sets_the_watcher_mounts_its_log_and_runs_untimed(self):
        board = 'blackhole-0000000000000001'
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, HOME=directory, RESULTS=os.path.join(directory, 'results'), QUAL_CARD=board, SUBDEV_DRY_RUN='1', WATCHER='1',
                       WATCHDOG_S='45', IMAGE='an-image-ref')
            result = subprocess.run([BASH, str(self.SCRIPT)], capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            import shlex
            argv = shlex.split([line for line in result.stdout.splitlines() if line.startswith('### argv: ')][0][len('### argv: '):])
            self.assertIn('TT_METAL_WATCHER=5', argv)
            self.assertIn('--no-timing', argv)
            self.assertTrue(any(word.endswith(',dst=/opt/tt-metal/generated/watcher') for word in argv))
            self.assertEqual(argv[argv.index('--watchdog-s') + 1], '45')
            self.assertIn('container timeout 2700 s', result.stdout)

    @unittest.skipUnless(BASH, 'bash not found')
    def test_a_real_run_needs_an_image(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, HOME=directory, QUAL_CARD='blackhole-0000000000000001')
            for key in ('IMAGE', 'IMAGE_TAG', 'SUBDEV_DRY_RUN', 'ALLOW_SERVING_CARD'):
                env.pop(key, None)
            result = subprocess.run([BASH, str(self.SCRIPT)], capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 1)
            self.assertNotIn('### argv', result.stdout)


class PatchTests(unittest.TestCase):
    PATCH = HERE / 'ag_link_offset.patch'

    def test_the_prepared_graft_source_changes_only_the_link_the_fabric_connection_uses(self):
        text = self.PATCH.read_text()
        self.assertEqual(text.count('\n--- a/'), 0)
        self.assertTrue(text.startswith('--- a/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp\n'))
        removed = [line[1:].strip() for line in text.splitlines() if line.startswith('-') and not line.startswith('---')]
        added = [line[1:].strip() for line in text.splitlines() if line.startswith('+') and not line.startswith('+++')]
        self.assertEqual(len(removed), 4)
        self.assertTrue(all(', link, ' in line for line in removed), removed)
        self.assertEqual(sum(', fabric_link, ' in line for line in added), 4)
        self.assertIn('QWEN_AG_LINK_OFFSET_SD1', text)
        self.assertIn('*(sub_device_id.value()) == 1', text)
        self.assertIn('Unset or 0: byte-identical', text)

    def test_the_harness_names_the_same_environment_variable(self):
        import subdev_h2
        self.assertEqual(subdev_h2.LINK_ENV, 'QWEN_AG_LINK_OFFSET_SD1')


if __name__ == '__main__':
    unittest.main()
