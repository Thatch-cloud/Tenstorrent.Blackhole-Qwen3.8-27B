"""CPU tests of the four-card sub-device collective probe H2 (tp4_subdev_probe.py + optimisation/ttnn-op/subdev_h/subdev_h2.py) against a fake ttnn
that models the command-queue and sub-device ownership rules, the gather's semaphores and, in 'link' mode, the one-connection-per-link fabric.

What the first card runs (2026-10-10) put into these tests:
  - H2a: the watcher run died in open_mesh_device, in the fabric router's own program (ACTIVE_ETH kernel config buffer overflow), before any sub-device
    existed: the watcher variables must switch the watcher off on the ethernet cores (test_the_watcher_pass_...).
  - H2b: everything up to the first CONCURRENT replay of two gather traces on one fabric link worked, and that replay hung the mesh: the arms that cannot
    hang run first and keep their numbers when a later one hangs (test_a_hang_in_a_shared_link_arm_...).

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
SHARED = ['--arms', 'solo,chained,one_queue,shared']
ALL = ['--arms', 'solo,chained,one_queue,shared,shared2']


def make_dog(exits=None, **kwargs):
    return plan.Watchdog(tag=subdev_h2.TAG, backstop=False, exit_fn=(exits.append if exits is not None else (lambda code: None)), **kwargs)


def make_graft(directory, environ, marker=True, sha_ok=True, count=2):
    """Stand-ins for the graft's two mounted copies, and the sha256 the workflow exports for them."""
    import hashlib
    body = b'\x7fELF\x00fake graft\x00' + (probe.GRAFT_MARKER + b'\x00' if marker else b'')
    paths = []
    for index in range(count):
        path = os.path.join(directory, 'copy%d_ttnncpp.so' % index)
        with open(path, 'wb') as handle:
            handle.write(body)
        paths.append(path)
    environ[probe.GRAFT_ENV] = hashlib.sha256(body).hexdigest() if sha_ok else '0' * 64
    return tuple(paths)


def run_probe(directory, extra=(), environ=None, graft=None, **fake):
    environ = {} if environ is None else environ
    ttnn = fake_ttnn.FakeTTNN(chips=4, environ=environ, **fake)
    lines = []
    out = os.path.join(directory, 'subdev-probe.json')
    if graft is None and '--link-offset' in extra and extra[list(extra).index('--link-offset') + 1] != '0':
        graft = {}
    files = make_graft(directory, environ, **graft) if graft is not None else probe.GRAFT_FILES
    status = probe.main(['--output', out] + SMALL + list(extra), ttnn=ttnn, torch=fake_ttnn.FakeTorch(), environ=environ, log=lines.append,
                        watchdog=make_dog(lines), clock=ttnn.clock, graft_files=files)
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

    # ------------------------------------------------------------------ the arms that cannot hang

    def test_the_default_arms_cannot_hang_and_end_in_safe_pass(self):
        status, lines, report, ttnn = self.run_fake()
        line = self.verdict_line(lines)
        self.assertEqual((status, report['verdict']), (0, 'SAFE-PASS'), line)
        self.assertNotIn('error', report, report.get('traceback'))
        self.assertEqual(sorted(report['arms']), ['chained', 'd_solo', 'one_queue', 't_solo'])
        for name in ('t_solo', 'd_solo', 'chained', 'one_queue'):
            self.assertEqual(report['arms'][name]['n'], 5, name)
        self.assertEqual(report['arms_requested'], ['solo', 'chained', 'one_queue'])
        self.assertEqual(sorted(report['traces']), ['d', 'd0', 't'])
        self.assertIn('arms=solo+chained+one_queue', line)
        self.assertIn('bytes=identical', line)
        self.assertAlmostEqual(report['evidence']['chained_over_sum'], 1.0, delta=0.05)      # chained = one after the other
        self.assertEqual(json.loads(lines[-1])['kind'], 'subdev-h2')
        self.assertFalse(any('hazard' in str(line) for line in lines), 'no shared-link arm, no hazard announcement')

    def test_no_eager_concurrency_and_no_shared_semaphore_traffic_without_a_shared_link_arm(self):
        status, lines, report, ttnn = self.run_fake()
        self.assertFalse(any('eager concurrent' in case['case'] for case in report['exactness']['cases']))
        self.assertIn('eager target solo', [case['case'] for case in report['exactness']['cases']])

    def test_chained_and_one_queue_are_exact_and_cost_about_the_sum_of_the_solo_walls(self):
        status, lines, report, ttnn = self.run_fake(overlap='serial')
        names = [case['case'] for case in report['exactness']['cases']]
        for expected in ('chained #0 target', 'chained #1 drafter', 'one_queue #0 drafter', 'one_queue #1 target', 't_solo #0 target', 'd_solo #0 drafter'):
            self.assertIn(expected, names)
        self.assertAlmostEqual(report['evidence']['one_queue_over_sum'], 1.0, delta=0.05)
        self.assertEqual(report['verdict'], 'SAFE-PASS', 'with only the safe arms nothing is judged serial or concurrent')

    # ------------------------------------------------------------------ the arms that overlap the two streams

    def test_gathers_that_overlap_pass_on_a_shared_link(self):
        status, lines, report, ttnn = self.run_fake(ALL)
        line = self.verdict_line(lines)
        self.assertEqual((status, report['verdict']), (0, 'PASS'), line)
        self.assertLessEqual(report['evidence']['shared_ratio'], 1.1)
        self.assertLessEqual(report['evidence']['shared2_ratio'], 1.1)
        self.assertEqual(sorted(report['traces']), ['d', 'd0', 'd2', 't', 't2'])
        self.assertIn('separate=NOT-RUN', line)
        self.assertIn('NOT-RUN', report['separate'])
        self.assertTrue(any('next arm=shared hazard=' in str(line) for line in lines))

    def test_the_hazard_arms_run_after_every_arm_that_cannot_hang(self):
        status, lines, report, ttnn = self.run_fake(ALL)
        stages = [line.split('stage=')[1] for line in lines if isinstance(line, str) and line.startswith('SUBDEV_H2 stage=arm-')]
        self.assertEqual(stages, ['arm-t_solo', 'arm-d_solo', 'arm-chained', 'arm-one_queue', 'arm-shared', 'arm-shared2'])
        self.assertEqual([arm for arm in subdev_h2.ARMS], ['t_solo', 'd_solo', 't_solo2', 'chained', 'one_queue', 'separate', 'shared', 'shared2'])
        self.assertEqual(subdev_h2.parse_groups('shared,solo,separate,chained'), ['solo', 'chained', 'separate', 'shared'])

    def test_one_dispatcher_for_both_queues_fails_serialised_and_says_so(self):
        status, lines, report, ttnn = self.run_fake(SHARED, overlap='serial')
        self.assertEqual((status, report['verdict']), (1, 'FAIL-SERIALISED'))
        self.assertIn('verdict=FAIL-SERIALISED', self.verdict_line(lines))

    def test_a_shared_link_that_serialises_fails_and_the_link_offset_graft_would_pass_it(self):
        status, lines, report, ttnn = self.run_fake(SHARED, overlap='link')
        self.assertEqual((status, report['verdict']), (1, 'FAIL-SERIALISED'), report['evidence'])
        status, lines, report, ttnn = self.run_fake(['--arms', 'solo,chained,one_queue,separate,shared', '--link-offset', '1'], overlap='link')
        self.assertEqual((status, report['verdict']), (0, 'PASS-SEPARATE-LINKS'), report['evidence'])
        self.assertIn('separate=RUN', self.verdict_line(lines))
        self.assertLessEqual(report['evidence']['separate_ratio'], 1.1)
        self.assertGreater(report['evidence']['shared_ratio'], 1.1)

    def test_the_separate_arm_alone_can_pass(self):
        status, lines, report, ttnn = self.run_fake(['--arms', 'solo,chained,one_queue,separate', '--link-offset', '1'], overlap='link')
        self.assertEqual((status, report['verdict']), (0, 'PASS-SEPARATE-LINKS'))
        self.assertNotIn('shared', report['arms'])

    def test_the_offset_is_in_the_environment_for_the_separate_capture_only(self):
        environ = {'QWEN_AG_LINK_OFFSET_SD1': '1'}         # an exported offset must not leak into the shared-link arm
        status, lines, report, ttnn = self.run_fake(['--arms', 'solo,separate,shared', '--link-offset', '1'], environ=environ, overlap='link')
        offsets_sd1 = [offset for sd, offset in ttnn.gather_env if sd == 1]
        self.assertEqual(sorted(set(offsets_sd1)), [0, 1])
        self.assertEqual(offsets_sd1.count(1), 1, 'one program was built under the offset: the separate arm\'s (the capture reuses it from the program cache)')
        self.assertEqual(sorted({tuple(sorted(t.links)) for t in ttnn.traces.values() if t.sds == {1} and t.cq == 1}), [(0,), (1,)])
        self.assertNotIn('QWEN_AG_LINK_OFFSET_SD1', environ, 'removed again after the capture')
        self.assertEqual(report['link_env_was_set'], '1')

    def test_bytes_that_differ_under_concurrency_fail(self):
        status, lines, report, ttnn = self.run_fake(SHARED, corrupt_on_overlap=True)
        self.assertEqual((status, report['verdict']), (1, 'FAIL-BYTES'))
        cases = {case['case'] for case in report['exactness']['cases'] if case['mismatched']}
        self.assertTrue(cases and all(case.startswith('shared') for case in cases), cases)     # chained and one-queue are not overlap: still exact

    def test_a_queue_that_silently_runs_nothing_fails_on_the_poison_it_left(self):
        status, lines, report, ttnn = self.run_fake(drop_replays_on=(1,))
        self.assertEqual((status, report['verdict']), (1, 'FAIL-BYTES'))
        cases = {case['case'] for case in report['exactness']['cases'] if case['mismatched']}
        self.assertTrue(any('drafter' in case for case in cases), cases)

    # ------------------------------------------------------------------ the separate-links run: the graft, the link cost, the pre-registered rule

    LINKS = ['--arms', 'solo,link_cost,chained,one_queue,separate', '--link-offset', '1']

    def test_the_separate_links_run_measures_the_link_cost_and_applies_the_port_rule(self):
        status, lines, report, ttnn = self.run_fake(self.LINKS, overlap='link')
        line = self.verdict_line(lines)
        self.assertEqual((status, report['verdict']), (0, 'PASS-SEPARATE-LINKS'), report.get('error'))
        self.assertEqual(report['evidence']['link_cost'], 1.0)             # the fake's gather costs the same on one link and on two
        self.assertEqual(report['port_rule']['decision'], 'GO')
        self.assertIn('port=GO', line)
        self.assertIn('link_cost=1.0', line)
        self.assertEqual(sorted(report['traces']), ['d', 'd0', 'ds', 't', 't2'])
        self.assertNotIn('d2', report['traces'], 'the drafter at two links is not needed for the link cost')
        self.assertEqual(report['arms_requested'], ['solo', 'link_cost', 'chained', 'one_queue', 'separate'])
        self.assertTrue(report['graft']['ok'])
        stages = [line.split('stage=')[1] for line in lines if isinstance(line, str) and line.startswith('SUBDEV_H2 stage=arm-')]
        self.assertEqual(stages, ['arm-t_solo', 'arm-d_solo', 'arm-t_solo2', 'arm-chained', 'arm-one_queue', 'arm-separate'])

    def test_the_port_rule_follows_the_measured_link_cost(self):
        for costs, decision in (({'ag1': 22000, 'ag2': 20000}, 'GO'), ({'ag1': 30000, 'ag2': 20000}, 'CONDITIONAL'), ({'ag1': 50000, 'ag2': 20000}, 'NO-GO')):
            status, lines, report, ttnn = self.run_fake(self.LINKS, overlap='link', op_costs=costs)
            self.assertEqual(report['verdict'], 'PASS-SEPARATE-LINKS', costs)
            self.assertEqual(report['port_rule']['decision'], decision, (costs, report['evidence']['link_cost']))
            self.assertIn('port=%s' % decision, self.verdict_line(lines))
            self.assertEqual(status, 0)

    def test_without_an_overlap_on_separate_links_the_rule_says_no_go(self):
        status, lines, report, ttnn = self.run_fake(self.LINKS, overlap='serial')
        self.assertEqual((status, report['verdict']), (1, 'FAIL-SERIALISED'))
        self.assertEqual(report['port_rule']['decision'], 'NO-GO')
        status, lines, report, ttnn = self.run_fake(self.LINKS, corrupt_on_overlap=True)
        self.assertEqual((status, report['verdict']), (1, 'FAIL-BYTES'))
        self.assertEqual(report['port_rule']['decision'], 'NO-GO')

    def test_the_rule_is_not_applied_to_a_run_without_the_separate_arm(self):
        status, lines, report, ttnn = self.run_fake(['--arms', 'solo,link_cost,chained,one_queue'])
        self.assertEqual(report['verdict'], 'SAFE-PASS')
        self.assertNotIn('port_rule', report)
        self.assertEqual(report['evidence']['link_cost'], 1.0)

    def test_a_graft_that_is_not_the_mounted_binary_stops_before_anything_is_opened(self):
        for graft, needle in ((dict(marker=False), 'does not contain QWEN_AG_LINK_OFFSET_SD1'), (dict(sha_ok=False), 'not the graft 0000000000000000')):
            status, lines, report, ttnn = self.run_fake(self.LINKS, overlap='link', graft=graft)
            self.assertEqual((status, report['verdict']), (2, 'NOT-MEASURED'), graft)
            self.assertIn('the link-offset graft is not what is mounted', report['error'])
            self.assertIn(needle, report['error'])
            self.assertIsNone(ttnn.opened, 'nothing was opened')
            self.assertFalse(report['graft']['ok'])

    def test_no_sha_in_the_environment_is_no_graft(self):
        environ = {}
        ttnn = fake_ttnn.FakeTTNN(chips=4, environ=environ)
        out = os.path.join(self.directory.name, 'g.json')
        lines = []
        status = probe.main(['--output', out] + SMALL + self.LINKS, ttnn=ttnn, torch=fake_ttnn.FakeTorch(), environ=environ, log=lines.append,
                            watchdog=make_dog(), clock=ttnn.clock, graft_files=())
        self.assertEqual(status, 2)
        self.assertIsNone(ttnn.opened)
        self.assertIn('QWEN_AG_LINK_GRAFT_SHA256 is not set', self.verdict_line(lines))

    def test_the_loaded_libraries_are_checked_too(self):
        environ = {}
        files = make_graft(self.directory.name, environ)
        found = probe.verify_graft(environ, files)
        self.assertTrue(found['ok'], found['problems'])
        maps = '\n'.join('7f00-7f01 r-xp 0000 00:00 0 %s' % path for path in files)
        self.assertTrue(probe.loaded_graft(dict(found, binaries=dict(found['binaries']), problems=[]), maps)['ok'])
        other = os.path.join(self.directory.name, 'other_ttnncpp.so')
        with open(other, 'wb') as handle:
            handle.write(b'the image\'s own binary\x00')
        bad = probe.loaded_graft(dict(found, binaries=dict(found['binaries']), problems=[]), maps + '\n7f02-7f03 r-xp 0 00:00 0 ' + other)
        self.assertFalse(bad['ok'])
        self.assertIn('the process mapped %s' % other, bad['problems'][0])
        none = probe.loaded_graft(dict(found, binaries=dict(found['binaries']), problems=[]), '7f00-7f01 r-xp 0 00:00 0 /usr/lib/libc.so.6')
        self.assertFalse(none['ok'])
        self.assertIn('no _ttnncpp.so is mapped', none['problems'][0])

    def test_the_marker_search_finds_a_literal_across_a_block_boundary(self):
        path = os.path.join(self.directory.name, 'x.so')
        with open(path, 'wb') as handle:
            handle.write(b'a' * 100 + probe.GRAFT_MARKER + b'b' * 100)
        self.assertTrue(probe.file_contains(path, probe.GRAFT_MARKER, chunk=107))
        self.assertTrue(probe.file_contains(path, probe.GRAFT_MARKER, chunk=7))
        self.assertFalse(probe.file_contains(path, b'NOT_THERE', chunk=7))

    # ------------------------------------------------------------------ a hang keeps what was measured before it

    def test_a_deadlock_in_a_shared_link_arm_leaves_every_safe_arm_in_the_report_file(self):
        status, lines, report, ttnn = self.run_fake(SHARED, overlap='link', deadlock_on_shared_link=True)
        self.assertEqual((status, report['verdict']), (2, 'NOT-MEASURED'))
        self.assertIn('DEADLOCK', report['error'])
        for name in ('t_solo', 'd_solo', 'chained', 'one_queue'):
            self.assertEqual(report['arms'][name]['n'], 5, name)           # timed before the shared arm ran
        self.assertNotIn('shared', report['arms'])
        self.assertEqual(report['exactness']['mismatched'], 0)
        self.assertEqual(report['last_stage'], 'arm-shared')

    def test_the_watchdog_firing_in_the_shared_arm_writes_the_partial_arms_before_it_exits(self):
        report_path = os.path.join(self.directory.name, 'r.json')
        exits, stream = [], io.StringIO()
        clock = [0.0]
        dog = plan.Watchdog(tag=subdev_h2.TAG, backstop=False, exit_fn=exits.append, clock=lambda: clock[0], stream=stream)
        environ = {}
        ttnn = fake_ttnn.FakeTTNN(chips=4, environ=environ)
        original = ttnn.execute_trace

        def stuck(mesh, trace_id, **kwargs):
            if dog.current() == 'arm-shared':
                clock[0] = 10000.0
                dog.check()
            return original(mesh, trace_id, **kwargs)
        ttnn.execute_trace = stuck
        probe.main(['--output', report_path] + SMALL + SHARED, ttnn=ttnn, torch=fake_ttnn.FakeTorch(), environ=environ, log=lambda line: None,
                   watchdog=dog, clock=ttnn.clock)
        self.assertEqual(exits[:1], [3])
        self.assertRegex(stream.getvalue(), r'^SUBDEV_H2 verdict=HANG stage=arm-shared')
        # the report on disk when the watchdog fired (the process would exit here): the hang verdict and the safe arms' timings
        with open(report_path) as handle:
            report = json.load(handle)
        self.assertEqual(report['arms']['one_queue']['n'], 5)

    # ------------------------------------------------------------------ the arrangement the first card run settled

    def test_the_watcher_pass_is_untimed_keeps_the_watcher_off_the_ethernet_cores_and_counts_its_log(self):
        with tempfile.TemporaryDirectory() as home:
            log_dir = Path(home) / 'generated' / 'watcher'
            log_dir.mkdir(parents=True)
            (log_dir / 'watcher.log').write_text('ok line\nNOC sanitization error at core (1,1)\nok line\nassert tripped\n')
            environ = {'TT_METAL_HOME': home}
            status, lines, report, ttnn = self.run_fake(['--watcher'], environ=environ, overlap='serial')
            self.assertEqual((status, report['verdict']), (0, 'UNTIMED-PASS'))
            self.assertEqual(environ['TT_METAL_WATCHER'], '5')
            self.assertEqual(environ['TT_METAL_WATCHER_DISABLE_ETH'], '1')
            self.assertEqual(report['watcher_env'], {'TT_METAL_WATCHER': '5', 'TT_METAL_WATCHER_DISABLE_ETH': '1'})
            self.assertEqual(report['watcher_log']['errors'], 2)
            self.assertEqual(report['watcher_log']['log'], 'subdev-watcher.log')
            self.assertTrue((Path(self.directory.name) / 'subdev-watcher.log').is_file())
            self.assertIn('watcher_errors=2', self.verdict_line(lines))
            self.assertEqual(report['arms'], {}, 'untimed: no arm is timed under the watcher')

    def test_the_run_that_failed_on_the_first_card_fails_the_same_way_without_the_ethernet_switch(self):
        environ = {}
        with unittest.mock.patch.object(probe, 'apply_watcher_env', lambda options, env: env.update({'TT_METAL_WATCHER': '5'}) or dict(env)):
            status, lines, report, ttnn = self.run_fake(['--watcher'], environ=environ)
        self.assertEqual((status, report['verdict']), (2, 'NOT-MEASURED'))
        self.assertFalse(report['opened'], 'the failure is in the open, before the sub-device manager exists')
        self.assertIn('Program size (28256) too large for kernel config buffer (25600) on ACTIVE_ETH', report['error'])
        self.assertIn('TT_METAL_WATCHER_DISABLE_ETH=1', report['known_failure'])
        self.assertIn('known_failure="the fabric router', self.verdict_line(lines))
        self.assertNotIn('load', ttnn.calls)

    def test_the_timed_pass_sets_no_watcher_variable_at_all(self):
        environ = {}
        status, lines, report, ttnn = self.run_fake(environ=environ)
        self.assertFalse([name for name in environ if name.startswith('TT_METAL_WATCHER')])
        self.assertEqual(report['watcher_env'], {'TT_METAL_WATCHER': None, 'TT_METAL_WATCHER_DISABLE_ETH': None})

    def test_the_fabric_is_up_before_the_manager_and_no_sub_device_owns_an_ethernet_core(self):
        status, lines, report, ttnn = self.run_fake(SHARED)
        self.assertLess(ttnn.calls.index(('fabric', 'FABRIC_1D')), ttnn.calls.index('open'))
        self.assertLess(ttnn.calls.index('open'), ttnn.calls.index('load'))      # the fabric routers are launched by the open, outside every sub-device
        self.assertEqual(len(ttnn.managers), 1)
        rects = []
        for sub_device in ttnn.managers[0]:
            self.assertEqual(len(sub_device.sets), 1, 'one CoreRangeSet: Tensix only, no ACTIVE_ETH or IDLE_ETH set')
            self.assertEqual(len(sub_device.rects), 1)
            rects.append(sub_device.rects[0])
        self.assertTrue(plan.rectangles_disjoint(rects[0], rects[1]))
        self.assertEqual((rects[0], rects[1]), tuple(tuple(report['plan'][key]) for key in ('target', 'drafter')))

    def test_each_sub_device_has_its_own_semaphores_on_its_own_cores(self):
        status, lines, report, ttnn = self.run_fake(SHARED)
        target, drafter = (tuple(report['plan'][key]) for key in ('target', 'drafter'))
        inside = lambda rect, outer: outer[0] <= rect[0] and outer[1] <= rect[1] and rect[2] <= outer[2] and rect[3] <= outer[3]  # noqa: E731
        owners = {'target': [], 'drafter': []}
        for semaphore in ttnn.semaphores:
            rect = semaphore.rects[0]
            owner = 'target' if inside(rect, target) else 'drafter' if inside(rect, drafter) else None
            self.assertIsNotNone(owner, 'a semaphore outside both sub-devices: %s' % (rect,))
            owners[owner].append(semaphore)
        self.assertTrue(owners['target'] and owners['drafter'])
        self.assertFalse({id(s) for s in owners['target']} & {id(s) for s in owners['drafter']})

    def test_the_semaphore_pools_are_double_buffered_and_distinct_per_sub_device(self):
        ttnn = fake_ttnn.FakeTTNN(chips=4)
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        pool_a, pool_b = subdev_h2.Pool(ttnn, None, cores), subdev_h2.Pool(ttnn, None, cores)
        first, second, third = pool_a.next(), pool_a.next(), pool_a.next()
        self.assertIsNot(first[0][0], second[0][0])
        self.assertIs(first[0][0], third[0][0])
        self.assertIs(first[1], third[1])
        self.assertTrue(all(a is not b for a in first[0] + [first[1]] for b in pool_b.next()[0] + [pool_b.ag[0][0]]))

    def test_the_mesh_is_the_served_ring_with_two_queues_and_the_two_sub_devices(self):
        status, lines, report, ttnn = self.run_fake()
        self.assertEqual(ttnn.opened['shape'], tp4_mesh.MESH_SHAPE)
        self.assertEqual(ttnn.opened['queues'], 2)
        self.assertEqual((report['plan']['target_cores'], report['plan']['drafter_cores']), (80, 30))
        self.assertEqual(report['chips'], 4)
        self.assertEqual({(t.cq, tuple(sorted(t.sds))) for t in ttnn.traces.values()}, {(0, (0,)), (1, (1,))} | {(0, (1,))},
                         'target on queue 0, drafter on queue 1, and the one-queue control: the drafter again on queue 0')

    def test_a_grid_wider_than_the_planned_split_leaves_idle_cores_not_a_third_sub_device(self):
        status, lines, report, ttnn = self.run_fake(grid=(13, 10))
        self.assertEqual((report['plan']['target_cores'], report['plan']['drafter_cores'], report['plan']['idle_cores']), (80, 30, 20))
        self.assertEqual(len(ttnn.managers[0]), 2)

    # ------------------------------------------------------------------ refusals and edges

    def test_every_program_a_trace_captures_has_run_eagerly_first(self):
        status, lines, report, ttnn = self.run_fake(['--arms', 'solo,separate', '--link-offset', '1'], overlap='link')
        self.assertNotEqual(report['verdict'], 'NOT-MEASURED', report.get('error'))
        with unittest.mock.patch.object(subdev_h2.Harness2, 'warm', lambda *args, **kwargs: None):
            status, lines, report, ttnn = self.run_fake(ALL)
        self.assertEqual((status, report['verdict']), (2, 'NOT-MEASURED'))
        self.assertIn('Expected program binaries to be written', report['error'])

    def test_a_mesh_that_is_not_four_cards_is_not_measured(self):
        ttnn = fake_ttnn.FakeTTNN(chips=1)
        lines = []
        out = os.path.join(self.directory.name, 'r.json')
        status = probe.main(['--output', out] + SMALL, ttnn=ttnn, torch=fake_ttnn.FakeTorch(), environ={}, log=lines.append, watchdog=make_dog(),
                            clock=ttnn.clock)
        self.assertEqual(status, 2)
        self.assertIn('H2 needs the four-card mesh', self.verdict_line(lines))
        with open(out) as handle:
            self.assertTrue(json.load(handle)['closed'])

    def test_the_mesh_is_not_opened_under_the_wrong_descriptor(self):
        lines = []
        out = os.path.join(self.directory.name, 'r.json')
        status = probe.main(['--output', out] + SMALL, environ={'TT_MESH_GRAPH_DESC_PATH': '/somewhere/other.textproto'}, log=lines.append,
                            watchdog=make_dog())
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
                status = probe.main(['--output', out, '--watcher'] + SMALL, environ=environ, log=lines.append, watchdog=make_dog())
            self.assertEqual(status, 2)
            self.assertEqual(environ['TT_METAL_WATCHER'], '5')
            self.assertEqual(environ['TT_METAL_WATCHER_DISABLE_ETH'], '1')
            self.assertTrue(any('verdict=NOT-MEASURED' in str(line) and 'ModuleNotFoundError' in str(line) for line in lines), lines)

    def test_bad_arguments_never_reach_the_device(self):
        ttnn = fake_ttnn.FakeTTNN(chips=4)
        for bad, needle in ((['--gathers', '1'], '--gathers'), (['--link-offset', '2'], '--link-offset'), (['--arms', 'solo,bogus'], 'bogus'),
                            (['--arms', 'solo,separate'], 'needs --link-offset 1'), (['--link-offset', '1'], 'add it to --arms')):
            with unittest.mock.patch('sys.stderr', new=io.StringIO()) as stderr:
                status = probe.main(['--output', os.path.join(self.directory.name, 'r.json')] + bad, ttnn=ttnn, torch=fake_ttnn.FakeTorch(),
                                    log=lambda line: None)
            self.assertEqual(status, 2, bad)
            self.assertIn(needle, stderr.getvalue())
        self.assertIsNone(ttnn.opened)

    def test_a_hang_is_the_watchdogs_not_a_verdict_and_the_partial_report_names_the_stage(self):
        report_path = os.path.join(self.directory.name, 'r.json')
        exits, stream = [], io.StringIO()
        clock = [0.0]
        dog = plan.Watchdog(tag=subdev_h2.TAG, backstop=False, exit_fn=exits.append, clock=lambda: clock[0], stream=stream)
        ttnn = fake_ttnn.FakeTTNN(chips=4)
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
        probe.main(['--output', out] + SMALL, ttnn=ttnn, torch=fake_ttnn.FakeTorch(), environ={}, log=lambda line: None, watchdog=make_dog(),
                   heartbeat=beat, clock=ttnn.clock)
        self.assertEqual(seen[0], 'manager')
        for stage in ('eager-solo', 'capture', 'measure', 'arm-t_solo', 'arm-chained', 'measure-done'):
            self.assertIn(stage, seen)

    def test_the_prepared_graft_and_the_harness_agree_on_the_variable(self):
        patch = (HERE.parent.parent / 'optimisation' / 'ttnn-op' / 'subdev_h' / 'ag_link_offset.patch').read_text()
        self.assertIn(subdev_h2.LINK_ENV, patch)


class ParserTests(unittest.TestCase):
    def test_defaults(self):
        options = probe.build_parser().parse_args(['--output', 'x.json'])
        self.assertEqual((options.fabric, options.topology, options.link_offset, options.watcher, options.no_timing),
                         ('FABRIC_1D', 'Ring', 0, False, False))
        self.assertEqual(options.arms, 'solo,chained,one_queue')
        self.assertEqual((options.target_cores, options.drafter_cores), (80, 30))
        self.assertEqual(subdev_h2.problems_of(options), [])

    def test_the_fabric_config_is_one_the_mesh_module_knows(self):
        with self.assertRaises(SystemExit):
            with unittest.mock.patch('sys.stderr', new=io.StringIO()):
                probe.build_parser().parse_args(['--output', 'x.json', '--fabric', 'FABRIC_2D'])

    def test_known_failure_names_the_ethernet_overflow_and_leaves_other_signatures_alone(self):
        self.assertIn('TT_METAL_WATCHER_DISABLE_ETH=1', probe.known_failure('Program size (1) too large for kernel config buffer (2) on ACTIVE_ETH'))
        self.assertIsNone(probe.known_failure('too large for kernel config buffer (2) on TENSIX'))
        self.assertIn('wedge', probe.known_failure('Timed out while waiting for active ethernet core'))
        self.assertIsNone(probe.known_failure(None))


if __name__ == '__main__':
    unittest.main()
