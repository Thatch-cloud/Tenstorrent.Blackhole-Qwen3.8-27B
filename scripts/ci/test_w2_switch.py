"""The W2 runtime kill switch (w2_switch.py, docs/tp4-w2-kill-switch.md): the file w2.off beside levern.off, prefix-reuse.off and parked.off.

W2 (QWEN_FAST_TP4_SDPA=multi, QWEN_FAST_TP4_CONV_GATES_SPREAD=1) is baked into the packed blocks' captured traces, so the switch works between traces: live, the
next round decision sends every round on a W2 block to the exact sequential step; at the next attach, W2 is not attached at all. The flag-off identity is the contract:
a process whose environment names no W2 lever never makes a system call for it, and with W2 on and the file absent every round is the packed round it was.

Run at py 3.11: `py -3.11 -B -m unittest test_w2_switch` from scripts/ci."""

import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import c2_serving_job as job
import c2_smoke_check as check
import make_w2_kill_profiles as generator
import serving_c2_contract as contract
import serving_packed_step
import verifier_engine
import w2_switch
from serving_packed_step import PackedStep, packed_device_step, proposal_rows
from test_serving_packed_step import FakeBlock, FakeRequest, entry

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
FOLDER = HERE / 'references' / 'tp4-w2-kill-jobs'
PATH = '/models/.qwen-c2/w2.off'
W2 = {'QWEN_FAST_TP4_SDPA': 'multi', 'QWEN_FAST_TP4_CONV_GATES_SPREAD': '1'}
IMAGE = 'tp4-prodfix-1'


class Clock(object):
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class World(object):
    """A fake file system (a set of paths), a clock and a log, for one Switch."""

    def __init__(self, environ=None, files=()):
        self.files, self.clock, self.logs, self.checks, self.touched = set(files), Clock(), [], [], []
        self.switch = w2_switch.Switch(dict(W2 if environ is None else environ), clock=self.clock, exists=self.exists, log=self.logs.append,
                                       touch=lambda path: (self.touched.append(path), self.files.add(path))[0])

    def exists(self, path):
        self.checks.append(path)
        return path in self.files


def installed(world):
    w2_switch.reset(world.switch)
    return world


class SwitchTests(unittest.TestCase):
    def tearDown(self):
        w2_switch.reset()

    def test_a_process_without_w2_never_touches_the_file_system(self):
        for environ in ({}, {'QWEN_FAST_TP4_SDPA': 'served'}, {'QWEN_FAST_TP4_SDPA': 'grid8x4'}, {'QWEN_FAST_TP4_SDPA': 'off'},
                        {'QWEN_FAST_TP4_CONV_GATES_SPREAD': '0'}, {'QWEN_FAST_TP4_SDPA_AUDIT': '1'}):
            with self.subTest(environ=environ):
                world = installed(World(environ, files=[PATH]))
                block = SimpleNamespace(shape=SimpleNamespace(rows_per_user=16, users=4))
                self.assertFalse(world.switch.configured)
                self.assertFalse(w2_switch.attach_off())
                self.assertFalse(w2_switch.routes_sequential(block))
                self.assertFalse(world.switch.killed())
                w2_switch.note_round(block)
                self.assertEqual(world.checks, [], 'no stat() without W2')
                self.assertEqual(world.logs, [])

    def test_either_lever_configures_it(self):
        self.assertTrue(w2_switch.Switch({'QWEN_FAST_TP4_SDPA': 'multi'}).configured)
        self.assertTrue(w2_switch.Switch({'QWEN_FAST_TP4_CONV_GATES_SPREAD': '1'}).configured)
        self.assertTrue(w2_switch.Switch(W2).configured)

    def test_with_w2_on_and_the_file_absent_nothing_is_routed_and_nothing_is_logged(self):
        world = installed(World())
        block = SimpleNamespace(shape=SimpleNamespace(rows_per_user=16, users=4))
        for _ in range(10):
            world.clock.now += 1.0
            self.assertFalse(w2_switch.routes_sequential(block))
        self.assertFalse(w2_switch.attach_off())
        self.assertEqual(world.logs, [])

    def test_the_file_appearing_latches_within_one_poll_and_logs_once(self):
        world = installed(World())
        block = SimpleNamespace(shape=SimpleNamespace(rows_per_user=16, users=4))
        self.assertFalse(w2_switch.routes_sequential(block))
        world.files.add(PATH)
        self.assertFalse(w2_switch.routes_sequential(block), 'inside the poll interval the answer is the last one (one stat at most per interval)')
        world.clock.now += w2_switch.POLL_S
        self.assertTrue(w2_switch.routes_sequential(block))
        self.assertTrue(w2_switch.routes_sequential(block))
        kill = [line for line in world.logs if line.startswith(w2_switch.KILL_MARKER) and 'present:' in line]
        self.assertEqual(kill, [w2_switch.KILL_LINE.format(PATH)])
        self.assertTrue(kill[0].startswith('[PINDIAG] w2 kill switch %s present' % PATH))
        self.assertEqual(len([line for line in world.logs if line.startswith(w2_switch.ROUTED_LINE.split('{')[0])]), 1, 'one routed line per block')

    def test_polling_is_bounded_by_the_interval(self):
        world = installed(World())
        block = SimpleNamespace(shape=SimpleNamespace(rows_per_user=16, users=4))
        world.clock.now += 1.0
        for _ in range(100):
            w2_switch.routes_sequential(block)
        self.assertLessEqual(len(world.checks), 2, 'the attach question and one poll: no stat per call')
        world.clock.now += 0.3
        w2_switch.routes_sequential(block)
        self.assertLessEqual(len(world.checks), 3)

    def test_the_latch_outlives_the_file(self):
        world = installed(World(files=[]))
        block = SimpleNamespace(shape=SimpleNamespace(rows_per_user=16, users=4))
        w2_switch.routes_sequential(block)
        world.files.add(PATH)
        world.clock.now += 1.0
        self.assertTrue(w2_switch.routes_sequential(block))
        world.files.discard(PATH)
        world.clock.now += 5.0
        self.assertTrue(w2_switch.routes_sequential(block), 'removing the file does not bring the packed block back; a restart does')

    def test_only_sixteen_row_blocks_are_routed(self):
        world = installed(World())
        self.assertFalse(w2_switch.attach_off(), 'the attach asked with the file absent')
        world.files.add(PATH)
        world.clock.now += 1.0
        self.assertFalse(w2_switch.routes_sequential(SimpleNamespace(shape=SimpleNamespace(rows_per_user=8, users=8))), 'the octo block runs no W2')
        self.assertTrue(w2_switch.routes_sequential(SimpleNamespace(shape=SimpleNamespace(rows_per_user=16, users=4))))
        self.assertFalse(w2_switch.applies(SimpleNamespace()))

    def test_the_path_comes_from_the_environment_and_empty_disables_the_file(self):
        world = installed(World(dict(W2, QWEN_FAST_W2_OFF_PATH='/tmp/other.off'), files=['/tmp/other.off']))
        self.assertEqual(world.switch.path, '/tmp/other.off')
        self.assertTrue(w2_switch.attach_off())
        self.assertEqual(world.checks, ['/tmp/other.off'])
        off = installed(World(dict(W2, QWEN_FAST_W2_OFF_PATH=''), files=[PATH]))
        self.assertIsNone(off.switch.path)
        self.assertFalse(w2_switch.attach_off())
        self.assertEqual(off.checks, [])
        self.assertEqual(w2_switch.Switch(dict(W2)).path, PATH)
        self.assertEqual(w2_switch.OFF_PATH, '/models/.qwen-c2/w2.off')
        self.assertEqual(w2_switch.OFF_FILE, 'w2.off')

    def test_an_unreadable_path_is_an_absent_file(self):
        def broken(path):
            raise OSError('denied')

        state = w2_switch.Switch(dict(W2), exists=broken, log=lambda line: None)
        self.assertFalse(state.attach_off())
        self.assertFalse(state.killed())


class AttachTests(unittest.TestCase):
    def tearDown(self):
        w2_switch.reset()

    def test_the_file_at_the_attach_keeps_w2_out_and_is_decided_once(self):
        world = installed(World(files=[PATH]))
        self.assertTrue(w2_switch.attach_off())
        world.files.discard(PATH)
        self.assertTrue(w2_switch.attach_off(), 'every block of the process agrees')
        self.assertEqual([line for line in world.logs if 'at attach' in line], [w2_switch.ATTACH_LINE.format(PATH)])
        block = SimpleNamespace(shape=SimpleNamespace(rows_per_user=16, users=4))
        world.files.add(PATH)
        world.clock.now += 5.0
        self.assertFalse(w2_switch.routes_sequential(block), 'W2 was never attached: the packed blocks serve at full speed and there is nothing to route away from')
        self.assertFalse(world.switch.running())

    def test_the_file_absent_at_the_attach_leaves_w2_on_and_a_later_file_latches_the_live_switch(self):
        world = installed(World())
        self.assertFalse(w2_switch.attach_off())
        world.files.add(PATH)
        self.assertFalse(w2_switch.attach_off(), 'the attach decision is not revisited')
        world.clock.now += 1.0
        self.assertTrue(w2_switch.routes_sequential(SimpleNamespace(shape=SimpleNamespace(rows_per_user=16, users=4))))

    def test_an_audit_flag_cannot_be_skipped_so_the_attach_ignores_the_file(self):
        for audit in ('QWEN_FAST_TP4_SDPA_AUDIT', 'QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT'):
            with self.subTest(audit=audit):
                world = installed(World(dict(W2, **{audit: '1'}), files=[PATH]))
                self.assertFalse(w2_switch.attach_off())
                self.assertEqual(world.logs, [w2_switch.ATTACH_IGNORED_LINE.format(PATH)])

    def test_sdpa_apply_leaves_the_served_launches_when_the_file_is_at_the_attach(self):
        import sdpa_long_tp
        import sdpa_multi_tp

        reader = SimpleNamespace(multi=None, readers=[SimpleNamespace(sdpa_modes_applied=(0x23, 0x23), metadata=[1, 2])],
                                 mesh=SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10)), operations=None)
        sentinel = object()
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4', 'QWEN_FAST_TP4_SDPA': 'multi'}), \
                patch.object(sdpa_multi_tp, 'attach', return_value=sentinel) as attach, patch('sdpa_long_tp._log'):
            installed(World(files=[PATH]))
            self.assertIsNone(sdpa_long_tp.apply(reader))
            self.assertIsNone(reader.multi)
            attach.assert_not_called()
            installed(World())
            sdpa_long_tp.apply(reader)
            attach.assert_called_once()
            self.assertIs(reader.multi, sentinel)

    def test_the_conv_gates_stage_runs_the_served_call_when_the_file_is_at_the_attach(self):
        import gdn_conv_gates_spread as spread
        import test_gdn_conv_gates_spread as base

        case = base.StageTests('test_on_but_declined_it_is_the_served_call')
        case.setUp()
        self.addCleanup(case.doCleanups)
        installed(World(files=[PATH]))
        with patch.object(spread, 'launch', side_effect=AssertionError('the spread launch ran')) as launch:
            found, fallbacks = case.case.run_stage(**base.LEVER)
        launch.assert_not_called()
        self.assertEqual(case.case.conv_calls, [('empty0', ['empty1', 'empty2', 'empty3', 'empty4'], 64)])
        # ...and with the file absent the launch is the one the stage asks for
        installed(World())
        with patch.object(spread, 'launch', return_value=None) as launch:
            case.case.conv_calls.clear()
            case.case.run_stage(**base.LEVER)
        launch.assert_called_once()


class DrillTests(unittest.TestCase):
    def tearDown(self):
        w2_switch.reset()

    def test_the_server_writes_the_file_after_n_packed_rounds_and_the_poll_finds_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'w2.off')
            clock = Clock()
            logs = []
            state = w2_switch.Switch(dict(W2, QWEN_FAST_W2_OFF_PATH=path, QWEN_FAST_W2_OFF_AFTER='3'), clock=clock, log=logs.append)
            w2_switch.reset(state)
            block = SimpleNamespace(shape=SimpleNamespace(rows_per_user=16, users=4))
            for _ in range(2):
                w2_switch.note_round(block)
            self.assertFalse(os.path.exists(path))
            clock.now += 1.0
            self.assertFalse(w2_switch.routes_sequential(block))
            w2_switch.note_round(block)
            self.assertTrue(os.path.exists(path))
            w2_switch.note_round(block)
            self.assertEqual([line for line in logs if line.startswith('[PINDIAG] w2 kill switch drill')], [w2_switch.DRILL_LINE.format(path, 3)])
            clock.now += 1.0
            self.assertTrue(w2_switch.routes_sequential(block))

    def test_off_after_is_a_positive_whole_number(self):
        self.assertIsNone(w2_switch.off_after({}))
        self.assertIsNone(w2_switch.off_after({'QWEN_FAST_W2_OFF_AFTER': ''}))
        self.assertEqual(w2_switch.off_after({'QWEN_FAST_W2_OFF_AFTER': '480'}), 480)
        for bad in ('0', '-1', '3.5', 'x', '03', ' 3'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                w2_switch.off_after({'QWEN_FAST_W2_OFF_AFTER': bad})

    def test_only_a_running_w2_counts_rounds(self):
        world = installed(World(dict(W2, QWEN_FAST_W2_OFF_AFTER='1'), files=[PATH]))
        w2_switch.attach_off()
        w2_switch.note_round(SimpleNamespace(shape=SimpleNamespace(rows_per_user=16, users=4)))
        self.assertEqual(world.touched, [], 'W2 was not attached: there is nothing to kill')
        octo = installed(World(dict(W2, QWEN_FAST_W2_OFF_AFTER='1')))
        w2_switch.note_round(SimpleNamespace(shape=SimpleNamespace(rows_per_user=8, users=8)))
        self.assertEqual(octo.touched, [])


class PackedStepTests(unittest.TestCase):
    """The switch at the round decisions of serving_packed_step, over the fakes of test_serving_packed_step."""

    def setUp(self):
        verifier_engine.note_prefill()
        self.stepped = []
        self.addCleanup(w2_switch.reset)

    def pair(self, block, names=('A', 'B'), accept=(15, 9)):
        made = []
        for segment, (name, taken) in enumerate(zip(names, accept)):
            request = FakeRequest(name, 100 + 2900 * segment, self.stepped)
            block.bind(request.engine, segment)
            request.propose(block.predictions_for(segment), taken, 16)
            made.append(request)
        return made

    def round(self, block):
        requests = self.pair(block)
        return packed_device_step([entry(request) for request in reversed(requests)], cancelled=lambda: False, block=block)

    def test_the_file_absent_is_byte_for_byte_the_round_without_the_switch(self):
        plain, armed = FakeBlock(), FakeBlock()
        installed(World({}))
        without = self.round(plain)
        installed(World())
        with_switch = self.round(armed)
        self.assertEqual(plain.calls, armed.calls)
        self.assertEqual(without, with_switch)
        self.assertEqual(self.stepped, [], 'both rounds were packed')
        self.assertEqual(armed.rounds, 1)

    def test_the_file_present_sends_the_round_to_the_sequential_step_whole(self):
        block = FakeBlock()
        world = installed(World())
        self.round(block)
        self.assertEqual(block.rounds, 1)
        before = list(block.calls)
        world.files.add(PATH)
        world.clock.now += 1.0
        outputs = self.round(block)
        self.assertEqual(block.rounds, 1, 'the block replayed nothing after the latch')
        self.assertEqual(block.calls, before, 'no verify, no commit on the block')
        self.assertEqual(sorted(request_id for request_id, flag in self.stepped), ['A', 'B'], 'each request took its own sequential step')
        self.assertEqual([output.request_id for output in outputs], ['B', 'A'])
        self.assertEqual(len([line for line in world.logs if 'present:' in line]), 1)

    def test_ineligible_names_the_kill_switch_as_the_reason(self):
        block = FakeBlock()
        world = installed(World())
        self.assertFalse(w2_switch.attach_off(), 'the attach asked with the file absent')
        world.files.add(PATH)
        world.clock.now += 1.0
        requests = self.pair(block)
        self.assertEqual(serving_packed_step.ineligible([entry(request) for request in requests], block), w2_switch.REASON)
        installed(World())
        self.assertIsNone(serving_packed_step.ineligible([entry(request) for request in requests], block))

    def test_the_ticket_width_of_the_coming_round_is_the_engines_own_once_latched(self):
        block = FakeBlock()
        requests = self.pair(block)
        world = installed(World())
        self.assertEqual(proposal_rows(block, requests), 16)
        self.assertEqual(serving_packed_step.proposal_groups([block], requests)[0][1], 16)
        world.files.add(PATH)
        world.clock.now += 1.0
        self.assertIsNone(proposal_rows(block, requests), 'drafted at the engines widths for the sequential step')
        self.assertIsNone(serving_packed_step.proposal_groups([block], requests)[0][1])

    def test_a_block_that_does_not_carry_w2_is_untouched(self):
        eight = FakeBlock(users=8, rows=8)
        world = installed(World(files=[PATH]))
        world.clock.now += 1.0
        requests = []
        for segment in range(8):
            request = FakeRequest('U%d' % segment, 4100 + segment, self.stepped)
            eight.bind(request.engine, segment)
            request.propose(eight.predictions_for(segment), 3, 8)
            requests.append(request)
        self.assertIsNone(serving_packed_step.ineligible([entry(request) for request in requests], eight))
        self.assertEqual(proposal_rows(eight, requests), 8)
        self.assertEqual([line for line in world.logs if 'present:' in line], [])

    def test_two_blocks_both_go_sequential_after_the_latch(self):
        block_a, block_b = FakeBlock(), FakeBlock()
        world = installed(World())
        step = PackedStep([block_a, block_b])
        requests = self.pair(block_a) + self.pair(block_b, names=('C', 'D'), accept=(0, 12))
        step([entry(request) for request in requests], cancelled=lambda: False)
        self.assertEqual((block_a.rounds, block_b.rounds), (1, 1))
        world.files.add(PATH)
        world.clock.now += 1.0
        requests = self.pair(block_a) + self.pair(block_b, names=('C', 'D'), accept=(0, 12))
        del self.stepped[:]
        step([entry(request) for request in requests], cancelled=lambda: False)
        self.assertEqual((block_a.rounds, block_b.rounds), (1, 1))
        self.assertEqual(sorted(request_id for request_id, flag in self.stepped), ['A', 'B', 'C', 'D'])
        routed = [line for line in world.logs if line.startswith('[PINDIAG] w2 kill switch routed')]
        self.assertEqual(len(routed), 2, 'one line for each block')

    def test_the_drill_counts_only_packed_rounds_that_ran(self):
        block = FakeBlock()
        world = installed(World(dict(W2, QWEN_FAST_W2_OFF_AFTER='2')))
        self.round(block)
        self.assertEqual(world.touched, [])
        self.round(block)
        self.assertEqual(world.touched, [world.switch.path])


class ContractTests(unittest.TestCase):
    def profile(self, env, gate=True):
        return {'name': 'p', 'env': dict(env), 'gate_only': gate}

    def test_the_names_are_the_switchs_own(self):
        self.assertEqual(contract.W2_NAMES, w2_switch.NAMES)
        self.assertEqual(contract.W2_GATE_ONLY, w2_switch.GATE_ONLY)
        self.assertEqual(contract.W2_PREFIX, w2_switch.PREFIX)
        self.assertEqual(w2_switch.NAMES, (w2_switch.OFF_PATH_ENV, w2_switch.OFF_AFTER_FLAG))

    def test_no_names_no_problems(self):
        self.assertEqual(contract.w2_problems(self.profile(W2, gate=False)), [])
        self.assertEqual(contract.w2_problems(self.profile({}, gate=False)), [])

    def test_the_override_path_is_an_absolute_file_path_on_any_profile(self):
        self.assertEqual(contract.w2_problems(self.profile(dict(W2, QWEN_FAST_W2_OFF_PATH='/data/w2.off'), gate=False)), [])
        for bad in ('w2.off', '', '/data/'):
            with self.subTest(bad=bad):
                self.assertTrue(contract.w2_problems(self.profile(dict(W2, QWEN_FAST_W2_OFF_PATH=bad), gate=False)))

    def test_the_drill_is_a_gate_instrument(self):
        drill = dict(W2, QWEN_FAST_W2_OFF_PATH='/tmp/qwen-w2.off', QWEN_FAST_W2_OFF_AFTER='480')
        self.assertEqual(contract.w2_problems(self.profile(drill, gate=True)), [])
        problems = contract.w2_problems(self.profile(drill, gate=False))
        self.assertTrue(any('gate instrument' in problem for problem in problems), problems)
        self.assertTrue(any('QWEN_FAST_W2_OFF_AFTER' in problem for problem in contract.w2_problems(self.profile({}, gate=False), {'QWEN_FAST_W2_OFF_AFTER': '3'})))
        self.assertEqual(contract.w2_problems(self.profile({}, gate=True), {'QWEN_FAST_W2_OFF_AFTER': '3'}), [])

    def test_the_drill_needs_w2_a_count_and_a_scratch_path(self):
        base = {'QWEN_FAST_W2_OFF_PATH': '/tmp/qwen-w2.off', 'QWEN_FAST_W2_OFF_AFTER': '480'}
        self.assertTrue(any('needs W2' in problem for problem in contract.w2_problems(self.profile(base))))
        for bad in ('0', '-3', 'x', '1.5'):
            with self.subTest(bad=bad):
                self.assertTrue(contract.w2_problems(self.profile(dict(W2, **dict(base, QWEN_FAST_W2_OFF_AFTER=bad)))))
        self.assertTrue(any('scratch path' in problem for problem in contract.w2_problems(self.profile(dict(W2, QWEN_FAST_W2_OFF_AFTER='480')))))
        self.assertTrue(any('scratch path' in problem for problem in contract.w2_problems(
            self.profile(dict(W2, QWEN_FAST_W2_OFF_AFTER='480', QWEN_FAST_W2_OFF_PATH='/models/.qwen-c2/w2.off')))))

    def test_an_unknown_w2_name_is_refused(self):
        problems = contract.w2_problems(self.profile(dict(W2, QWEN_FAST_W2_OFF='1'), gate=False))
        self.assertTrue(any('QWEN_FAST_W2_OFF is not a W2 kill switch setting' in problem for problem in problems), problems)

    def test_an_inherited_name_does_not_reach_a_profile_that_never_named_it(self):
        profile = {'env': dict(W2), 'mesh_graph_descriptor': '/x'}
        environ = {'QWEN_FAST_W2_OFF_PATH': '/tmp/x', 'QWEN_FAST_W2_OFF_AFTER': '3'}
        contract.apply_environment(profile, environ)
        self.assertNotIn('QWEN_FAST_W2_OFF_PATH', environ)
        self.assertNotIn('QWEN_FAST_W2_OFF_AFTER', environ)
        named = {'env': dict(W2, QWEN_FAST_W2_OFF_PATH='/tmp/y'), 'mesh_graph_descriptor': '/x'}
        environ = {'QWEN_FAST_W2_OFF_AFTER': '3'}
        contract.apply_environment(named, environ)
        self.assertEqual(environ['QWEN_FAST_W2_OFF_PATH'], '/tmp/y')
        self.assertNotIn('QWEN_FAST_W2_OFF_AFTER', environ)

    def test_boot_refuses_a_misused_name(self):
        source = (HERE / 'serving_c2_contract.py').read_text(encoding='utf-8')
        self.assertIn('problems = w2_problems(profile, environ)', source)
        self.assertLess(source.index('problems = w2_problems(profile, environ)'), source.index('problems = parked_problems(profile, environ)'))


LOG_HEAD = ['[PINDIAG] tp4 sdpa engaged config=multi grid=11x10 entries=8 flags=0x21']
ROUND = '[PINDIAG] packed extent round round=%d live=4 families=[0:4096] idle=[] capped=[]'
DRILL = {'QWEN_FAST_W2_OFF_AFTER': '480', 'QWEN_FAST_W2_OFF_PATH': '/tmp/qwen-w2.off'}


def log(*parts):
    return '\n'.join(parts)


class SmokeRuleTests(unittest.TestCase):
    LATCH = w2_switch.KILL_LINE.format('/tmp/qwen-w2.off')
    DRILL_LINE = w2_switch.DRILL_LINE.format('/tmp/qwen-w2.off', 480)
    ROUTED = w2_switch.ROUTED_LINE.format(16, 4)

    def good(self):
        return log(ROUND % 1, ROUND % 2, self.DRILL_LINE, ROUND % 3, self.LATCH, self.ROUTED, 'sequential traffic continues')

    def test_the_constants_are_the_switchs_own(self):
        self.assertEqual(check.W2_KILL_PREFIX, w2_switch.KILL_MARKER)
        self.assertTrue(check.W2_KILL_LATCH.search(w2_switch.KILL_LINE.format('/x/w2.off')))
        self.assertFalse(check.W2_KILL_LATCH.search(w2_switch.ATTACH_LINE.format('/x/w2.off')))
        self.assertTrue(check.W2_KILL_ATTACH.search(w2_switch.ATTACH_LINE.format('/x/w2.off')))
        self.assertTrue(check.W2_KILL_ATTACH.search(w2_switch.ATTACH_IGNORED_LINE.format('/x/w2.off')))
        self.assertTrue(w2_switch.ROUTED_LINE.format(16, 4).startswith(check.W2_KILL_ROUTED))
        self.assertTrue(w2_switch.DRILL_LINE.format('/p', 3).startswith(check.W2_KILL_DRILL))
        self.assertEqual(check.W2_OFF_AFTER_FLAG, w2_switch.OFF_AFTER_FLAG)
        source = (HERE / 'packed_verifier.py').read_text(encoding='utf-8')
        self.assertIn("EXTENT_ROUND_MARKER = '%s'" % check.PACKED_ROUND_LINE, source)

    def test_a_log_with_no_kill_line_is_clean_without_the_drill_and_a_problem_with_it(self):
        text = log(*LOG_HEAD, ROUND % 1)
        self.assertEqual(check.w2_kill_problems({}, text), [])
        self.assertEqual(check.w2_kill_problems(None, text), [])
        self.assertEqual(check.w2_kill_problems(dict(W2), text), [])
        problems = check.w2_kill_problems(dict(W2, **DRILL), text)
        self.assertEqual(len(problems), 1)
        self.assertIn('the drill never latched', problems[0])

    def test_a_kill_line_outside_the_drill_arm_is_a_problem_even_at_the_attach(self):
        for line in (self.LATCH, w2_switch.ATTACH_LINE.format(PATH), w2_switch.ROUTED_LINE.format(16, 4)):
            with self.subTest(line=line[:60]):
                problems = check.w2_kill_problems(dict(W2), log(ROUND % 1, line))
                self.assertEqual(len(problems), 1)
                self.assertIn('outside the drill arm', problems[0])

    def test_the_drill_arm_passes_when_the_markers_stop_at_the_latch(self):
        self.assertEqual(check.w2_kill_problems(dict(W2, **DRILL), self.good()), [])

    def test_a_packed_round_after_the_latch_fails(self):
        text = log(self.good(), ROUND % 4)
        problems = check.w2_kill_problems(dict(W2, **DRILL), text)
        self.assertEqual(len(problems), 1)
        self.assertIn('after the W2 kill latch', problems[0])

    def test_no_packed_round_before_the_latch_fails(self):
        text = log(self.DRILL_LINE, self.LATCH, self.ROUTED)
        problems = check.w2_kill_problems(dict(W2, **DRILL), text)
        self.assertTrue(any('before the W2 kill latch' in problem for problem in problems), problems)

    def test_a_missing_routed_line_a_second_latch_and_a_latch_before_the_drill_fail(self):
        env = dict(W2, **DRILL)
        self.assertTrue(any('routed to the sequential step' in problem for problem in check.w2_kill_problems(
            env, log(ROUND % 1, self.DRILL_LINE, self.LATCH))))
        self.assertTrue(any('2 W2 kill switch latch lines' in problem for problem in check.w2_kill_problems(
            env, log(ROUND % 1, self.DRILL_LINE, self.LATCH, self.LATCH, self.ROUTED))))
        self.assertTrue(any('precedes the drill line' in problem for problem in check.w2_kill_problems(
            env, log(ROUND % 1, self.LATCH, self.DRILL_LINE, self.ROUTED))))
        self.assertTrue(any('0 W2 kill drill lines' in problem for problem in check.w2_kill_problems(
            env, log(ROUND % 1, self.LATCH, self.ROUTED))))

    def test_an_attach_line_in_the_drill_arm_fails(self):
        problems = check.w2_kill_problems(dict(W2, **DRILL), log(self.good(), w2_switch.ATTACH_LINE.format(PATH)))
        self.assertTrue(any('present at the attach' in problem for problem in problems), problems)

    def test_the_rule_runs_in_the_smoke_check_and_the_gates_engagement_rules(self):
        text = log(ROUND % 1, self.LATCH)
        self.assertTrue(any('outside the drill arm' in problem for problem in check.lever_engagement_problems(dict(W2), text)))
        smoke = 'SMOKE_JSON ' + json.dumps({'warmup': {'value': 200}})
        problems, facts = check.check(smoke, text, False, env=dict(W2))
        self.assertTrue(any('outside the drill arm' in problem for problem in problems), problems)


def profiles():
    with open(str(PROFILES_PATH), encoding='utf-8') as handle:
        return json.load(handle)['profiles']


class ProfileTests(unittest.TestCase):
    def test_the_checked_in_twin_is_what_the_parent_generates(self):
        self.assertEqual(generator.main(['--check']), 0)

    def test_the_twin_is_the_parent_plus_the_drill_and_nothing_else(self):
        found = profiles()
        parent, twin = found[generator.PARENT], found[generator.TWIN]
        self.assertEqual(sorted(set(twin['env']) - set(parent['env'])), ['QWEN_FAST_W2_OFF_AFTER', 'QWEN_FAST_W2_OFF_PATH'])
        self.assertEqual({key: value for key, value in twin['env'].items() if key in parent['env']}, parent['env'])
        for key in parent:
            if key not in ('env', 'description', 'owner_traffic_waiver', 'gate_only'):
                self.assertEqual(twin[key], parent[key], key)
        self.assertTrue(twin['gate_only'])
        self.assertNotIn('owner_traffic_waiver', twin)
        self.assertNotIn('gate_only', parent)
        self.assertEqual(twin['env']['QWEN_FAST_W2_OFF_PATH'], '/tmp/qwen-w2.off')
        self.assertEqual(twin['env']['QWEN_FAST_W2_OFF_AFTER'], generator.OFF_AFTER)
        self.assertEqual(contract.w2_problems(dict(twin, name=generator.TWIN)), [])
        self.assertEqual(list(found).index(generator.TWIN), list(found).index(generator.PARENT) + 1)

    def test_the_twin_is_exempted_from_the_profile_enumerating_tests(self):
        import profile_twins

        self.assertIn(generator.TWIN, profile_twins.twin_names())

    def test_no_profile_but_the_twin_carries_a_kill_switch_name_and_the_production_profile_names_the_file(self):
        found = profiles()
        for name, profile in found.items():
            if name != generator.TWIN:
                with self.subTest(name=name):
                    self.assertFalse([key for key in profile.get('env', {}) if key.startswith(w2_switch.PREFIX)], name)
        description = found[generator.PARENT]['description']
        self.assertIn('w2.off', description)
        self.assertNotIn('W2 has none', description)

    def test_the_control_is_the_production_stack_without_w2(self):
        found = profiles()
        control, twin = found[generator.PRODUCTION_CONTROL], found[generator.TWIN]
        self.assertTrue(control['gate_only'])
        self.assertNotIn('QWEN_FAST_TP4_SDPA', control['env'])
        self.assertNotIn('QWEN_FAST_TP4_CONV_GATES_SPREAD', control['env'])
        for key, value in control['env'].items():
            self.assertEqual(twin['env'].get(key), value, key)
        self.assertEqual(sorted(set(twin['env']) - set(control['env'])), sorted(['QWEN_FAST_TP4_SDPA', 'QWEN_FAST_TP4_CONV_GATES_SPREAD',
                                                                              'QWEN_FAST_W2_OFF_AFTER', 'QWEN_FAST_W2_OFF_PATH']))
        self.assertEqual(twin['engine'], control['engine'])


class ShippingTests(unittest.TestCase):
    def test_the_module_is_in_both_image_copy_lists_and_the_overlay(self):
        from test_serving_image_copy_closure import context_modules, dockerfile_modules, dockerfile_text

        self.assertIn('w2_switch.py', dockerfile_modules(dockerfile_text()))
        self.assertIn('w2_switch.py', context_modules())
        listed = {line.split()[0] for line in (ROOT / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8').splitlines()
                  if line.strip() and not line.startswith('#')}
        self.assertIn('scripts/ci/w2_switch.py', listed)
        for host_only in ('make_w2_kill_profiles.py', 'test_w2_switch.py'):
            self.assertNotIn(host_only, dockerfile_modules(dockerfile_text()))

    def test_the_importers_reach_the_image_beside_it(self):
        from test_serving_image_copy_closure import context_modules, dockerfile_modules, dockerfile_text

        for name in ('serving_packed_step.py', 'sdpa_long_tp.py', 'gdn_block_conv_tp.py'):
            with self.subTest(name=name):
                text = (HERE / name).read_text(encoding='utf-8')
                self.assertIn('import w2_switch', text)
        self.assertIn('serving_packed_step.py', dockerfile_modules(dockerfile_text()))
        self.assertIn('serving_packed_step.py', context_modules())

    def test_the_suite_runs_in_the_any_ref_regression_step(self):
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        self.assertRegex(workflow, r'python -B -m unittest [^\n]*\btest_w2_switch\b')
        step = workflow.split('T32 and unchanged T16 regression gates', 1)[1].split('lever-n:', 1)[0]
        self.assertIn('test_w2_switch', step)

    def test_the_stale_flag_check_of_the_gate_job_knows_the_file(self):
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
        self.assertIn('for name in prefix-reuse.off levern.off parked.off w2.off; do', workflow)

    def test_the_module_imports_only_the_standard_library_and_the_files_are_lf(self):
        source = (HERE / 'w2_switch.py').read_text(encoding='utf-8')
        top = [line for line in source.splitlines() if re.match(r'^(import|from) ', line)]
        self.assertEqual(top, ['import os', 'import time'])
        for name in ('w2_switch.py', 'make_w2_kill_profiles.py', 'test_w2_switch.py', 'c2_smoke_check.py', 'serving_c2_contract.py',
                     'serving_packed_step.py', 'qwen_c2_profiles.json'):
            with self.subTest(name=name):
                self.assertNotIn(b'\r\n', (HERE / name).read_bytes())

    def test_the_flag_off_path_adds_no_call_when_w2_is_not_configured(self):
        state = w2_switch.Switch({}, exists=lambda path: self.fail('stat'), clock=lambda: self.fail('clock'))
        w2_switch.reset(state)
        self.addCleanup(w2_switch.reset)
        block = SimpleNamespace(shape=SimpleNamespace(rows_per_user=16, users=4))
        self.assertFalse(w2_switch.routes_sequential(block))
        self.assertFalse(w2_switch.attach_off())
        w2_switch.note_round(block)


BANNED = re.compile(r'\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|[A-Za-z]:[/\\]Users[/\\]')
AGENT_ACTIONS = {'agentstop', 'agentstart', 'unserve', 'platform', 'replay', 'priority', 'cardm'}


def text_of(name):
    return (FOLDER / (name + '.env')).read_text(encoding='utf-8')


def parsed(name):
    return job.read_job(job.parse_env(text_of(name)), sorted(profiles()), root=str(ROOT))


def order_lines():
    return [line.split() for line in (FOLDER / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]


class JobPackTests(unittest.TestCase):
    def names(self):
        return [line[0] for line in order_lines()]

    def test_order_lists_exactly_the_templates_and_the_dependencies_are_greppable(self):
        files = sorted(path.name[:-4] for path in FOLDER.iterdir() if path.name.endswith('.env'))
        self.assertEqual(files, sorted(self.names()))
        self.assertEqual(self.names(), ['B0-build', 'X0-status-rescan-reset', 'K-W2a-control', 'K-W2b-kill-switch', 'Z-reset'])
        for name, mode, image, minutes in order_lines():
            self.assertIn(mode, ('stop', 'soft'), name)
            self.assertEqual(image, IMAGE, name)
            self.assertTrue(minutes.isdigit() and int(minutes) > 0, name)
        needs = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        for line in ('# NEEDS X0 <- B0', '# NEEDS K-W2a <- X0', '# NEEDS K-W2b <- K-W2a'):
            self.assertIn(line, needs)

    def test_every_template_parses_on_the_quad_with_the_one_image_and_never_touches_the_agent(self):
        for name in self.names():
            with self.subTest(name):
                result = parsed(name)
                self.assertEqual(result['cards'], 'quad')
                self.assertEqual(result['tag'], IMAGE)
                self.assertFalse(AGENT_ACTIONS & set(result['actions'].split()))
        self.assertEqual(parsed('B0-build')['bake_default_profile'], 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic')
        self.assertEqual(parsed('Z-reset')['actions'], 'status reset')

    def test_the_pair_runs_the_same_tests_on_the_control_and_the_drill_profile(self):
        control, kill = parsed('K-W2a-control'), parsed('K-W2b-kill-switch')
        self.assertEqual(control['tests'], kill['tests'])
        self.assertEqual(control['actions'], 'reset smoke')
        self.assertEqual(kill['actions'], 'reset smoke')
        self.assertEqual(control['profile'], generator.PRODUCTION_CONTROL)
        self.assertEqual(kill['profile'], generator.TWIN)
        self.assertEqual(kill['tests'].split(','), ['warmup', 'concurrent8_steady', 'concurrent8_code_equal', 'coding'])
        found = profiles()
        self.assertTrue(found[control['profile']]['gate_only'])
        self.assertTrue(found[kill['profile']]['gate_only'])

    def test_every_test_a_template_names_is_defined_by_the_smoke(self):
        import ast

        tree = ast.parse((HERE / 'c2_serving_smoke.py').read_text(encoding='utf-8'))
        defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        for name in ('K-W2a-control', 'K-W2b-kill-switch'):
            for test in parsed(name)['tests'].split(','):
                with self.subTest(name=name, test=test):
                    self.assertTrue(test in ('warmup', 'coding', 'concurrent8_steady') or test in defined, test)

    def test_the_drill_template_states_the_read_rules_and_the_comparison(self):
        text = text_of('K-W2b-kill-switch')
        for needle in ('levern_compare.py', 'w2_kill_problems', 'drill (gate only): wrote', 'packed extent round', 'routed the round to the sequential step',
                       'QWEN_FAST_W2_OFF_AFTER=480', 'NO-GO'):
            self.assertIn(needle, text)
        self.assertIn(str(generator.OFF_AFTER), text)
        self.assertIn('K-W2a', text)

    def test_the_templates_are_public_safe(self):
        for path in sorted(FOLDER.iterdir()):
            with self.subTest(path=path.name):
                self.assertIsNone(BANNED.search(path.read_text(encoding='utf-8')))
                self.assertNotIn(b'\r\n', path.read_bytes())


if __name__ == '__main__':
    unittest.main()
