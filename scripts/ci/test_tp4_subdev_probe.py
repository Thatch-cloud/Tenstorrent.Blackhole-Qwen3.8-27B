"""CPU tests of the four-card sub-device collective probe H2 (tp4_subdev_probe.py + optimisation/ttnn-op/subdev_h/subdev_h2.py) against a fake ttnn
that models the command-queue and sub-device ownership rules, the gather's semaphores and, in 'link' mode, the one-connection-per-link fabric.

Run: python -B -m unittest test_tp4_subdev_probe   (from scripts/ci)
"""

import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import unittest.mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tp4_subdev_probe as probe  # noqa: E402  (puts optimisation/ttnn-op/subdev_h on the path)
import fake_ttnn  # noqa: E402
import subdev_h2  # noqa: E402
import subdev_plan as plan  # noqa: E402
import tp4_mesh  # noqa: E402

SMALL = ['--gathers', '4', '--rounds', '5', '--warmup', '1', '--eager-rounds', '2', '--heartbeat-s', '0']


def run_probe(directory, extra=(), **fake):
    environ = {}
    ttnn = fake_ttnn.FakeTTNN(chips=4, environ=environ, **fake)
    lines = []
    out = os.path.join(directory, 'subdev-probe.json')
    dog = plan.Watchdog(tag=subdev_h2.TAG, backstop=False, exit_fn=lambda code: lines.append('EXIT %d' % code))
    status = probe.main(['--output', out] + SMALL + list(extra), ttnn=ttnn, torch=fake_ttnn.FakeTorch(), environ=environ, log=lines.append,
                        watchdog=dog, clock=ttnn.clock)
    with open(out) as handle:
        report = json.load(handle)
    return status, lines, report, ttnn


class H2Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def run_fake(self, extra=(), **fake):
        return run_probe(self.directory.name, extra, **fake)

    def verdict_line(self, lines):
        found = [line for line in lines if isinstance(line, str) and line.startswith('SUBDEV_H2 verdict=')]
        self.assertEqual(len(found), 1, lines[-5:])
        return found[0]

    def test_gathers_that_overlap_pass_on_a_shared_link(self):
        status, lines, report, ttnn = self.run_fake()
        line = self.verdict_line(lines)
        self.assertEqual((status, report['verdict']), (0, 'PASS'), line)
        self.assertNotIn('error', report, report.get('traceback'))
        self.assertLessEqual(report['evidence']['shared_ratio'], 1.1)
        self.assertIn('bytes=identical', line)
        self.assertIn('separate=NOT-RUN', line)
        self.assertIn('NOT-RUN', report['separate'])
        self.assertEqual(json.loads(lines[-1])['kind'], 'subdev-h2')

    def test_the_mesh_is_the_served_ring_with_two_queues_and_the_two_sub_devices(self):
        status, lines, report, ttnn = self.run_fake()
        self.assertEqual(ttnn.opened['shape'], tp4_mesh.MESH_SHAPE)
        self.assertEqual(ttnn.opened['queues'], 2)
        self.assertIn(('fabric', 'FABRIC_1D'), ttnn.calls)
        self.assertEqual((report['plan']['target_cores'], report['plan']['drafter_cores']), (80, 30))
        self.assertEqual(report['chips'], 4)
        self.assertEqual(sorted(report['traces']), ['d', 'd2', 't', 't2'])
        by_ident = ttnn.traces
        self.assertEqual({(t.cq, tuple(sorted(t.sds))) for t in by_ident.values()}, {(0, (0,)), (1, (1,))})
        self.assertEqual(report['manager_ms'] > 0, True)

    def test_each_sub_device_has_its_own_semaphores(self):
        status, lines, report, ttnn = self.run_fake()
        # five pools were built per side at most (eager, trace1, trace2): none is shared between the two sub-devices
        self.assertIsNone(report.get('error'), report.get('traceback'))

    def test_the_semaphore_pools_are_double_buffered_and_distinct_per_sub_device(self):
        ttnn = fake_ttnn.FakeTTNN(chips=4)
        pool_a, pool_b = subdev_h2.Pool(ttnn, None, None), subdev_h2.Pool(ttnn, None, None)
        first, second, third = pool_a.next(), pool_a.next(), pool_a.next()
        self.assertIsNot(first[0][0], second[0][0])
        self.assertIs(first[0][0], third[0][0])
        self.assertIs(first[1], third[1])
        self.assertTrue(all(a is not b for a in first[0] + [first[1]] for b in pool_b.next()[0] + [pool_b.ag[0][0]]))

    def test_one_dispatcher_for_both_queues_fails_serialised_and_says_so(self):
        status, lines, report, ttnn = self.run_fake(overlap='serial')
        self.assertEqual((status, report['verdict']), (1, 'FAIL-SERIALISED'))
        self.assertIn('verdict=FAIL-SERIALISED', self.verdict_line(lines))

    def test_a_shared_link_that_serialises_fails_and_the_link_offset_graft_would_pass_it(self):
        status, lines, report, ttnn = self.run_fake(overlap='link')
        self.assertEqual((status, report['verdict']), (1, 'FAIL-SERIALISED'), report['evidence'])
        status, lines, report, ttnn = self.run_fake(extra=['--link-offset', '1'], overlap='link')
        self.assertEqual((status, report['verdict']), (0, 'PASS-SEPARATE-LINKS'), report['evidence'])
        self.assertIn('separate=RUN', self.verdict_line(lines))
        self.assertLessEqual(report['evidence']['separate_ratio'], 1.1)
        self.assertGreater(report['evidence']['shared_ratio'], 1.1)

    def test_the_offset_is_in_the_environment_for_the_separate_capture_only(self):
        environ = {'QWEN_AG_LINK_OFFSET_SD1': '1'}         # an exported offset must not leak into the shared-link arm
        ttnn = fake_ttnn.FakeTTNN(chips=4, environ=environ, overlap='link')
        lines = []
        out = os.path.join(self.directory.name, 'r.json')
        probe.main(['--output', out, '--link-offset', '1'] + SMALL, ttnn=ttnn, torch=fake_ttnn.FakeTorch(), environ=environ, log=lines.append,
                   watchdog=plan.Watchdog(tag='SUBDEV_H2', backstop=False, exit_fn=lambda code: None), clock=ttnn.clock)
        offsets_sd1 = [offset for sd, offset in ttnn.gather_env if sd == 1]
        self.assertEqual(sorted(set(offsets_sd1)), [0, 1])
        self.assertEqual(offsets_sd1.count(1), 4, 'only the four gathers of the separate trace were built under the offset')
        self.assertNotIn('QWEN_AG_LINK_OFFSET_SD1', environ, 'removed again after the capture')
        with open(out) as handle:
            self.assertEqual(json.load(handle)['link_env_was_set'], '1')

    def test_bytes_that_differ_under_concurrency_fail(self):
        status, lines, report, ttnn = self.run_fake(corrupt_on_overlap=True)
        self.assertEqual((status, report['verdict']), (1, 'FAIL-BYTES'))
        self.assertGreater(report['exactness']['mismatched'], 0)

    def test_the_watcher_pass_is_untimed_sets_the_watcher_level_and_counts_its_log(self):
        with tempfile.TemporaryDirectory() as home:
            log_dir = Path(home) / 'generated' / 'watcher'
            log_dir.mkdir(parents=True)
            (log_dir / 'watcher.log').write_text('ok line\nNOC sanitization error at core (1,1)\nok line\nassert tripped\n')
            environ = {'TT_METAL_HOME': home}
            ttnn = fake_ttnn.FakeTTNN(chips=4, overlap='serial', environ=environ)
            lines = []
            out = os.path.join(self.directory.name, 'w.json')
            status = probe.main(['--output', out, '--watcher'] + SMALL, ttnn=ttnn, torch=fake_ttnn.FakeTorch(), environ=environ, log=lines.append,
                                watchdog=plan.Watchdog(tag='SUBDEV_H2', backstop=False, exit_fn=lambda code: None), clock=ttnn.clock)
            with open(out) as handle:
                report = json.load(handle)
            self.assertEqual((status, report['verdict']), (0, 'UNTIMED-PASS'))
            self.assertEqual(report['watcher_log']['errors'], 2)
            self.assertEqual(report['watcher_log']['log'], 'subdev-watcher.log')
            self.assertTrue((Path(self.directory.name) / 'subdev-watcher.log').is_file())
            self.assertIn('watcher_errors=2', self.verdict_line(lines))

    def test_a_missing_watcher_log_is_reported_as_none(self):
        self.assertEqual(probe.watcher_summary(os.path.join(self.directory.name, 'x.json'), {'TT_METAL_HOME': self.directory.name}), {'log': None})

    def test_a_mesh_that_is_not_four_cards_is_not_measured(self):
        ttnn = fake_ttnn.FakeTTNN(chips=1)
        lines = []
        out = os.path.join(self.directory.name, 'r.json')
        status = probe.main(['--output', out] + SMALL, ttnn=ttnn, torch=fake_ttnn.FakeTorch(), environ={}, log=lines.append,
                            watchdog=plan.Watchdog(tag='SUBDEV_H2', backstop=False, exit_fn=lambda code: None))
        self.assertEqual(status, 2)
        self.assertIn('H2 needs the four-card mesh', self.verdict_line(lines))
        with open(out) as handle:
            self.assertTrue(json.load(handle)['closed'])

    def test_the_mesh_is_not_opened_under_the_wrong_descriptor(self):
        lines = []
        out = os.path.join(self.directory.name, 'r.json')
        status = probe.main(['--output', out] + SMALL, environ={'TT_MESH_GRAPH_DESC_PATH': '/somewhere/other.textproto'}, log=lines.append,
                            watchdog=plan.Watchdog(tag='SUBDEV_H2', backstop=False, exit_fn=lambda code: None))
        self.assertEqual(status, 2)
        self.assertIn('verdict=NOT-MEASURED error="refused to open: TT_MESH_GRAPH_DESC_PATH', lines[0])

    def test_with_the_ring_descriptor_and_no_ttnn_installed_the_probe_is_not_measured_not_a_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            descriptor = Path(directory) / tp4_mesh.DESCRIPTOR_NAME
            descriptor.write_text(tp4_mesh.descriptor_text())
            environ = {'TT_MESH_GRAPH_DESC_PATH': str(descriptor)}
            lines = []
            out = os.path.join(directory, 'r.json')
            with unittest.mock.patch.dict(sys.modules, {'ttnn': None}):
                status = probe.main(['--output', out, '--watcher'] + SMALL, environ=environ, log=lines.append,
                                    watchdog=plan.Watchdog(tag='SUBDEV_H2', backstop=False, exit_fn=lambda code: None))
            self.assertEqual(status, 2)
            self.assertEqual(environ['TT_METAL_WATCHER'], '5')
            self.assertTrue(any('verdict=NOT-MEASURED' in str(line) and 'ModuleNotFoundError' in str(line) for line in lines), lines)

    def test_bad_arguments_never_reach_the_device(self):
        ttnn = fake_ttnn.FakeTTNN(chips=4)
        with unittest.mock.patch('sys.stderr', new=io.StringIO()) as stderr:
            status = probe.main(['--output', os.path.join(self.directory.name, 'r.json'), '--gathers', '1', '--link-offset', '2'], ttnn=ttnn,
                                torch=fake_ttnn.FakeTorch(), log=lambda line: None)
        self.assertEqual(status, 2)
        self.assertIn('--gathers', stderr.getvalue())
        self.assertIn('--link-offset', stderr.getvalue())
        self.assertIsNone(ttnn.opened)

    def test_a_hang_is_the_watchdogs_not_a_verdict_and_the_partial_report_names_the_stage(self):
        report_path = os.path.join(self.directory.name, 'r.json')
        exits = []
        stream = io.StringIO()
        ttnn = fake_ttnn.FakeTTNN(chips=4)
        clock = [0.0]
        dog = plan.Watchdog(tag='SUBDEV_H2', backstop=False, exit_fn=exits.append, clock=lambda: clock[0], stream=stream)
        original = ttnn.execute_trace

        def stuck(mesh, trace_id, **kwargs):
            clock[0] = 10000.0
            dog.check()
            return original(mesh, trace_id, **kwargs)
        ttnn.execute_trace = stuck
        probe.main(['--output', report_path] + SMALL, ttnn=ttnn, torch=fake_ttnn.FakeTorch(), environ={}, log=lambda line: None, watchdog=dog,
                   clock=ttnn.clock)
        self.assertEqual(exits[:1], [3])
        self.assertRegex(stream.getvalue(), r'^SUBDEV_H2 verdict=HANG stage=arm-')

    def test_the_heartbeat_follows_the_stages(self):
        clock = [0.0]
        beat = plan.Heartbeat('SUBDEV_H2', clock=lambda: clock[0], stream=io.StringIO())
        ttnn = fake_ttnn.FakeTTNN(chips=4)
        out = os.path.join(self.directory.name, 'r.json')
        seen = []
        original = beat.set
        beat.set = lambda stage: (seen.append(stage), original(stage))[1]
        probe.main(['--output', out] + SMALL, ttnn=ttnn, torch=fake_ttnn.FakeTorch(), environ={}, log=lambda line: None,
                   watchdog=plan.Watchdog(tag='SUBDEV_H2', backstop=False, exit_fn=lambda code: None), heartbeat=beat, clock=ttnn.clock)
        self.assertEqual(seen[0], 'manager')
        for stage in ('eager-solo', 'eager-concurrent', 'capture', 'exactness', 'timing', 'timing-done'):
            self.assertIn(stage, seen)

    def test_the_prepared_graft_and_the_harness_agree_on_the_variable(self):
        patch = (HERE.parent.parent / 'optimisation' / 'ttnn-op' / 'subdev_h' / 'ag_link_offset.patch').read_text()
        self.assertIn(subdev_h2.LINK_ENV, patch)


class ParserTests(unittest.TestCase):
    def test_defaults(self):
        options = probe.build_parser().parse_args(['--output', 'x.json'])
        self.assertEqual((options.fabric, options.topology, options.link_offset, options.watcher, options.no_timing, options.two_links),
                         ('FABRIC_1D', 'Ring', 0, False, False, True))
        self.assertEqual((options.target_cores, options.drafter_cores), (80, 30))
        self.assertEqual(subdev_h2.problems_of(options), [])

    def test_the_fabric_config_is_one_the_mesh_module_knows(self):
        with self.assertRaises(SystemExit):
            with unittest.mock.patch('sys.stderr', new=io.StringIO()):
                probe.build_parser().parse_args(['--output', 'x.json', '--fabric', 'FABRIC_2D'])


if __name__ == '__main__':
    unittest.main()
