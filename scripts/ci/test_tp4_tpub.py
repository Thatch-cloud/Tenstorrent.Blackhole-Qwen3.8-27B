"""tp4/tpub: the sequential step's GDN carry copies as captured traces (verifier_engine_tp, QWEN_FAST_TP4_TRACED_PUBLISH).

Everything runs on fakes: a device whose helpers and trace calls record their order, so the capture order (every copy kernel run
eagerly before its capture, nothing allocated after the first capture), the replay path, the decline and audit paths and the flag-off
byte-identity are held without a card.
"""

import os
from itertools import count
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding' / 'harness'))
import c2_smoke_check
import verifier_engine
import verifier_engine_tp
from verifier_engine_tp import VerifierEngine

LAYERS = 48
_addresses = count(1000)


def tensor():
    return SimpleNamespace(buffer_address=Mock(return_value=next(_addresses)))


class Device:
    """Records every call in order; 'save' and 'restore' carry whether a capture was open."""

    def __init__(self):
        self.events, self.capturing, self.released = [], False, []
        self.traces = count(7)
        self.executed = []
        self.operations = SimpleNamespace(
            begin_trace_capture=self.begin, end_trace_capture=self.end, release_trace=self.release,
            execute_trace=self.execute, synchronize_device=lambda mesh: self.events.append('sync'),
            get_device_tensors=lambda value: [value, value], to_torch=lambda part: part.value, deallocate=lambda value: None)

    def begin(self, mesh, cq_id):
        self.capturing = True
        self.events.append('begin')
        return next(self.traces)

    def end(self, mesh, trace, cq_id):
        self.capturing = False
        self.events.append('end')

    def release(self, mesh, trace):
        self.released.append(trace)

    def execute(self, mesh, trace, cq_id, blocking):
        self.executed.append((trace, blocking))
        self.events.append('execute')

    def helpers(self, direct=True):
        def make(name):
            return Mock(side_effect=lambda slot: self.events.append((name, self.capturing)))

        return [SimpleNamespace(direct=direct, live=[tensor() for index in range(5)], allocate=Mock(side_effect=lambda: self.alloc()),
                                save=make('save'), restore=make('restore')) for layer in range(LAYERS)]

    def alloc(self):
        self.events.append(('allocate', self.capturing))
        return [tensor() for index in range(5)]


def engine(device, *, arm, direct=True, audit=False, phase='preparing'):
    live = VerifierEngine.__new__(VerifierEngine)
    live.session = SimpleNamespace(request_id='request-T')
    live.phase, live.pending = phase, None
    live.mesh, live.operations = object(), device.operations
    live.helpers = device.helpers(direct)
    live.carry_trace_arm = arm
    live.carry_traces = None if arm else False
    live.carry_trace_audit = audit
    live.carry_audit_checked = live.carry_audit_mismatches = 0
    live.buckets, live.initial, live.mtp_row_reader = {}, [], None
    live.allocate_carry()
    return live


class FlagTests(unittest.TestCase):
    def test_the_flags_are_off_unless_exactly_one(self):
        for value in (None, '0', '', 'true', '2'):
            environ = {} if value is None else {verifier_engine_tp.TRACED_FLAG: value}
            self.assertFalse(verifier_engine_tp.traced_publish_enabled(environ), value)
        self.assertTrue(verifier_engine_tp.traced_publish_enabled({verifier_engine_tp.TRACED_FLAG: '1'}))

    def test_the_audit_needs_the_arm(self):
        audit = verifier_engine_tp.TRACED_AUDIT_FLAG
        self.assertFalse(verifier_engine_tp.traced_audit_enabled({audit: '1'}))
        self.assertTrue(verifier_engine_tp.traced_audit_enabled({verifier_engine_tp.TRACED_FLAG: '1', audit: '1'}))
        self.assertFalse(verifier_engine_tp.traced_audit_enabled({verifier_engine_tp.TRACED_FLAG: '1'}))

    def test_the_constructor_reads_the_flags_before_the_base_constructor_captures(self):
        seen = {}

        def base(self, *args, **options):
            seen.update(arm=self.carry_trace_arm, traces=self.carry_traces, audit=self.carry_trace_audit)

        for environ, expected in (({}, (False, False, False)),
                                  ({verifier_engine_tp.TRACED_FLAG: '1'}, (True, None, False)),
                                  ({verifier_engine_tp.TRACED_FLAG: '1', verifier_engine_tp.TRACED_AUDIT_FLAG: '1'}, (True, None, True))):
            with self.subTest(environ=environ), patch.dict(os.environ, environ), \
                    patch.object(verifier_engine_tp.PairVerifierEngine, '__init__', base):
                VerifierEngine(sampler=None)
                self.assertEqual((seen['arm'], seen['traces'], seen['audit']), expected)

    def test_the_pair_engine_the_pinned_file_defines_is_not_touched(self):
        source = Path(verifier_engine.__file__).read_text()
        self.assertNotIn('TRACED_PUBLISH', source)
        self.assertIn('Capturing them as one trace is the follow-up.', source)


class CaptureOrderTests(unittest.TestCase):
    def build(self, **options):
        device = Device()
        live = engine(device, **options)
        live.save_carry()
        return device, live

    def test_every_copy_kernel_runs_eagerly_before_its_capture_and_nothing_is_allocated_after_one(self):
        device, live = self.build(arm=True)
        events = device.events
        first_capture = events.index('begin')
        before = events[:first_capture]
        # allocate (48 layers, at attach) -> eager save -> identity restore warm -> sync -> captures
        self.assertEqual([e for e in before if isinstance(e, tuple)].count(('save', False)), LAYERS)
        self.assertEqual([e for e in before if isinstance(e, tuple)].count(('restore', False)), LAYERS)
        self.assertLess(max(i for i, e in enumerate(before) if e == ('save', False)),
                        min(i for i, e in enumerate(before) if e == ('restore', False)), 'the save seeds the carry, then the warm restores it')
        self.assertEqual(before[-1], 'sync')
        captured = events[first_capture:]
        self.assertEqual(captured.count('begin'), 2)
        self.assertEqual(captured.count('end'), 2)
        self.assertEqual(captured.count(('save', True)), LAYERS)
        self.assertEqual(captured.count(('restore', True)), LAYERS)
        self.assertFalse([e for e in captured if isinstance(e, tuple) and e[0] == 'allocate'], 'no persistent buffer after a capture')
        self.assertEqual(set(live.carry_traces), {'save', 'restore'})
        self.assertEqual(captured[-1], 'sync')

    def test_the_captured_programs_run_in_the_eager_loops_layer_order(self):
        device, live = self.build(arm=True)
        order = [call for helper in live.helpers for call in helper.save.call_args_list]
        self.assertEqual(len(live.helpers[0].save.call_args_list), 2, 'the eager seed and the capture')
        for helper, slot in zip(live.helpers, live.carry):
            self.assertIs(helper.save.call_args_list[-1].args[0], slot)
            self.assertIs(helper.restore.call_args_list[-1].args[0], slot)
        self.assertEqual(len(order), 2 * LAYERS)

    def test_flag_off_the_build_is_the_eager_seed_alone(self):
        device, live = self.build(arm=False)
        self.assertNotIn('begin', device.events)
        self.assertIs(live.carry_traces, False)
        self.assertEqual([len(helper.save.call_args_list) for helper in live.helpers], [1] * LAYERS)
        self.assertEqual([len(helper.restore.call_args_list) for helper in live.helpers], [0] * LAYERS)

    def test_a_save_after_the_build_never_captures(self):
        device = Device()
        live = engine(device, arm=True, phase='idle')
        live.save_carry()
        self.assertNotIn('begin', device.events)
        self.assertIsNone(live.carry_traces, 'still waiting for a build-time save')

    def test_a_restore_first_never_captures(self):
        device = Device()
        live = engine(device, arm=True)
        live.copy_carry('restore', 'other')
        self.assertNotIn('begin', device.events)

    def test_a_helper_that_is_not_the_direct_copy_declines_with_a_logged_reason_and_stays_eager(self):
        device = Device()
        lines = []
        live = engine(device, arm=True, direct=False)
        with patch.object(verifier_engine_tp.verify_trace_t1, 'log_line', lines.append):
            live.save_carry()
        self.assertNotIn('begin', device.events)
        self.assertIs(live.carry_traces, False)
        self.assertEqual(lines, ['%s request=request-T: layer 0 helper is not the direct copy' % verifier_engine_tp.TRACED_DECLINED])
        with patch.object(verifier_engine, '_resident', None):
            live.restore_carry()
        self.assertEqual(len(live.helpers[3].restore.call_args_list), 1, 'eager from then on')

    def test_an_incomplete_carry_declines(self):
        device = Device()
        live = engine(device, arm=True)
        live.carry[5] = live.carry[5][:4]
        self.assertIn('layer 5 carry holds 4 tensors for 5 live', verifier_engine_tp.carry_trace_problem(live.helpers, live.carry))
        self.assertIn('carry holds 47 layers', verifier_engine_tp.carry_trace_problem(live.helpers, live.carry[:-1]))

    def test_the_engaged_line_is_logged_once_per_engine(self):
        device = Device()
        lines = []
        live = engine(device, arm=True)
        with patch.object(verifier_engine_tp.verify_trace_t1, 'log_line', lines.append):
            live.save_carry()
        self.assertEqual(lines, ['%s request=request-T layers=48 audit=0' % verifier_engine_tp.TRACED_ENGAGED])

    def test_a_failed_capture_releases_what_it_took_and_leaves_the_engine_eager(self):
        device = Device()
        live = engine(device, arm=True)
        original = device.end
        calls = []

        def end(mesh, trace, cq_id):
            calls.append(trace)
            original(mesh, trace, cq_id)
            if len(calls) == 2:
                raise RuntimeError('capture failed')

        device.operations.end_trace_capture = end
        with self.assertRaises(RuntimeError):
            live.save_carry()
        self.assertIs(live.carry_traces, False)
        self.assertEqual(device.released, [calls[0], calls[1]][:len(device.released)])
        self.assertIn(calls[0], device.released, 'the first trace is released')


class ReplayTests(unittest.TestCase):
    def built(self, **options):
        device = Device()
        live = engine(device, arm=True, **options)
        live.save_carry()
        for helper in live.helpers:
            helper.save.reset_mock()
            helper.restore.reset_mock()
        device.events.clear()
        return device, live

    def test_save_and_restore_replay_their_trace_and_launch_no_eager_copy(self):
        device, live = self.built()
        saves, restores = live.carry_traces['save'], live.carry_traces['restore']
        live.save_carry()
        self.assertEqual(device.executed, [(saves, False)])
        with patch.object(verifier_engine, '_resident', None):
            self.assertTrue(live.restore_carry())
        self.assertEqual(device.executed, [(saves, False), (restores, False)])
        self.assertEqual(sum(len(helper.save.call_args_list) + len(helper.restore.call_args_list) for helper in live.helpers), 0)

    def test_replay_still_refuses_a_carry_that_moved(self):
        device, live = self.built()
        live.carry[3][0].buffer_address = Mock(return_value=-1)
        with self.assertRaisesRegex(ValueError, 'Carried GDN state moved under the engine'):
            live.save_carry()
        self.assertEqual(device.executed, [])

    def test_close_releases_each_trace_once_after_a_sync_and_the_base_close_still_runs(self):
        device, live = self.built()
        traces = dict(live.carry_traces)
        live.phase = 'idle'
        live.close()
        self.assertEqual(sorted(device.released), sorted(traces.values()))
        self.assertEqual(live.phase, 'closed')
        live.close()
        self.assertEqual(len(device.released), 2)
        self.assertIs(live.carry_traces, False)

    def test_close_of_an_engine_that_never_captured_releases_nothing(self):
        device = Device()
        live = engine(device, arm=False)
        live.close()
        self.assertEqual(device.released, [])

    def test_the_carry_log_says_traced(self):
        device, live = self.built()
        lines = []
        with patch.dict(os.environ, {'QWEN_FAST_CARRY_LOG': '1'}), patch.object(verifier_engine_tp, 'carry_log_line',
                lambda message, **values: lines.append(message.format(**values))):
            live.save_carry()
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(line.endswith('traced=1') for line in lines), lines)
        self.assertIn('begin', lines[0])


class AuditTests(unittest.TestCase):
    """Audited, the trace and then the eager loop write the same destination; the bytes between them are compared on every chip."""

    def audited(self, *, corrupt):
        device = Device()
        live = engine(device, arm=True, audit=True)
        live.save_carry()
        lines = []
        for slot in live.carry:
            for index, value in enumerate(slot):
                value.value = torch.full((4,), float(index), dtype=torch.bfloat16)
        if corrupt:
            # the traced copy leaves one tensor different from what the eager loop then writes
            def execute(mesh, trace, cq_id, blocking):
                live.carry[2][1].value = torch.full((4,), 99.0, dtype=torch.bfloat16)

            device.operations.execute_trace = execute
        for helper, slot in zip(live.helpers, live.carry):
            helper.save.side_effect = lambda destination, slot=slot: [setattr(value, 'value', torch.full((4,), float(i), dtype=torch.bfloat16))
                                                                      for i, value in enumerate(destination)]
        with patch.object(verifier_engine_tp.verify_trace_t1, 'log_line', lines.append):
            live.save_carry()
        return live, lines

    def test_equal_bytes_log_zero_mismatches(self):
        live, lines = self.audited(corrupt=False)
        self.assertEqual(lines, ['[TPUB-AUDIT] op=save checked=%d mismatches=0' % (LAYERS * 5 * 2)])
        self.assertEqual((live.carry_audit_checked, live.carry_audit_mismatches), (LAYERS * 5 * 2, 0))

    def test_a_differing_tensor_is_counted_on_every_chip_and_the_eager_bytes_stand(self):
        live, lines = self.audited(corrupt=True)
        self.assertEqual(lines, ['[TPUB-AUDIT] op=save checked=%d mismatches=2' % (LAYERS * 5 * 2)])
        self.assertEqual(live.carry_audit_mismatches, 2)
        self.assertTrue(torch.equal(live.carry[2][1].value, torch.full((4,), 1.0, dtype=torch.bfloat16)))

    def restore_audit(self, *, corrupt):
        device = Device()
        live = engine(device, arm=True, audit=True)
        live.carry_traces = {'save': 1, 'restore': 2}
        for helper in live.helpers:
            for value in helper.live:
                value.value = torch.zeros(4, dtype=torch.bfloat16)
        for helper in live.helpers:
            helper.restore.side_effect = lambda slot, helper=helper: [setattr(v, 'value', torch.zeros(4, dtype=torch.bfloat16)) for v in helper.live]
        if corrupt:
            def execute(mesh, trace, cq_id, blocking):
                live.helpers[3].live[1].value = torch.full((4,), 5.0, dtype=torch.bfloat16)

            device.operations.execute_trace = execute
        lines = []
        with patch.object(verifier_engine_tp.verify_trace_t1, 'log_line', lines.append):
            live.audited_carry_copy('restore')
        return live, lines

    def test_the_restore_audit_reads_the_live_state_not_the_carry(self):
        live = engine(Device(), arm=True, audit=True)
        live_tensors = [value for helper in live.helpers for value in helper.live]
        self.assertEqual([id(v) for v in live.carry_destination('restore')], [id(v) for v in live_tensors])
        carry = [value for slot in live.carry for value in slot]
        self.assertFalse({id(v) for v in carry} & {id(v) for v in live.carry_destination('restore')})

    def test_a_restore_that_differs_from_the_eager_loop_is_counted(self):
        live, lines = self.restore_audit(corrupt=True)
        self.assertEqual(lines, ['[TPUB-AUDIT] op=restore checked=%d mismatches=2' % (LAYERS * 5 * 2)])
        live, lines = self.restore_audit(corrupt=False)
        self.assertEqual(lines, ['[TPUB-AUDIT] op=restore checked=%d mismatches=0' % (LAYERS * 5 * 2)])

    def test_bit_equality_distinguishes_signed_zero_and_shape(self):
        zero, negative = torch.zeros(2, dtype=torch.bfloat16), -torch.zeros(2, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(zero, negative))
        self.assertFalse(verifier_engine_tp._bits_equal(zero, negative))
        self.assertFalse(verifier_engine_tp._bits_equal(zero, torch.zeros(3, dtype=torch.bfloat16)))
        self.assertTrue(verifier_engine_tp._bits_equal(zero, torch.zeros(2, dtype=torch.bfloat16)))


class SmokeCheckTests(unittest.TestCase):
    """c2_smoke_check.tpub_problems: the container log must show what the profile asked for."""
    ENGAGED = '[TPUB] carry traces engaged request=r layers=48 audit=1'
    LINE = '[TPUB-AUDIT] op=%s checked=%d mismatches=%d'

    def problems(self, text, **env):
        return c2_smoke_check.tpub_problems(dict({'QWEN_FAST_TP': '4'}, **env), text)

    def test_a_profile_without_the_flag_must_log_no_traced_carry_line(self):
        self.assertEqual(self.problems('nothing'), [])
        self.assertEqual(len(self.problems(self.ENGAGED)), 1)
        self.assertEqual(len(self.problems(self.LINE % ('save', 960, 0))), 1)

    def test_the_timed_arm_needs_an_engaged_line_and_no_declined_one(self):
        flag = {c2_smoke_check.TPUB_FLAG: '1'}
        self.assertEqual(self.problems(self.ENGAGED, **flag), [])
        self.assertIn('never traced', self.problems('nothing', **flag)[0])
        self.assertIn('declined 1', self.problems(self.ENGAGED + '\n' + c2_smoke_check.TPUB_DECLINED + ' request=r: x', **flag)[0])

    def test_the_audited_arm_needs_equal_lines_of_the_width_implied_count(self):
        env = {c2_smoke_check.TPUB_FLAG: '1', c2_smoke_check.TPUB_AUDIT_FLAG: '1'}
        good = self.ENGAGED + '\n' + self.LINE % ('save', 960, 0) + '\n' + self.LINE % ('restore', 960, 0)
        self.assertEqual(self.problems(good, **env), [])
        self.assertIn('no [TPUB-AUDIT] line', self.problems(self.ENGAGED, **env)[0])
        self.assertIn('mismatches>0', ' '.join(self.problems(good + '\n' + self.LINE % ('save', 960, 3), **env)))
        self.assertIn('960', ' '.join(self.problems(self.ENGAGED + '\n' + self.LINE % ('save', 480, 0), **env)))

    def test_the_audited_arm_needs_a_restore_line_the_restore_writes_the_live_slot(self):
        env = {c2_smoke_check.TPUB_FLAG: '1', c2_smoke_check.TPUB_AUDIT_FLAG: '1'}
        saves_only = self.ENGAGED + '\n' + self.LINE % ('save', 960, 0)
        self.assertIn('op=restore', ' '.join(self.problems(saves_only, **env)))

    def test_check_runs_the_tpub_rules(self):
        env = {'QWEN_FAST_TP': '4', c2_smoke_check.TPUB_FLAG: '1'}
        problems, _facts = c2_smoke_check.check('', 'nothing', False, env=env)
        self.assertTrue([p for p in problems if c2_smoke_check.TPUB_FLAG in p and 'engaged' in p], problems)

    def test_audit_lines_without_the_audit_flag_fail(self):
        self.assertEqual(len(self.problems(self.ENGAGED + '\n' + self.LINE % ('save', 960, 0), **{c2_smoke_check.TPUB_FLAG: '1'})), 1)


if __name__ == '__main__':
    unittest.main()
