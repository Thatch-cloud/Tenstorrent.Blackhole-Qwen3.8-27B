"""Each engine carries its own GDN slot-zero state across the other engine's blocks."""

import io
from itertools import count
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding' / 'harness'))
from greedy_session import GreedySession
import verifier_engine
from verifier_engine import VerifierEngine, carry_log, note_prefill
from verifier_pack import GDN_LAYERS

_addresses = count(1)


def tensor():
    """A device tensor stand-in with one stable address on both chips."""
    return SimpleNamespace(buffer_address=Mock(return_value=next(_addresses)))


def helpers():
    """The 48 shared layer helpers; allocate hands out a fresh snapshot each call."""
    return [SimpleNamespace(live=[tensor() for index in range(5)],
                            allocate=Mock(side_effect=lambda: [tensor() for index in range(5)]),
                            save=Mock(), restore=Mock()) for index in range(GDN_LAYERS)]


def engine(request_id, shared, rows=4):
    """A prepared engine over the shared helpers, ending the way __init__ ends."""
    session = GreedySession(request_id, [0, 1, 2] * 12, 0, vocab_size=100, max_new_tokens=64)
    engine = VerifierEngine.__new__(VerifierEngine)
    engine.session, engine.position = session, session.position
    engine.phase, engine.pending = 'idle', None
    engine.mesh = object()
    engine.model = SimpleNamespace(args=SimpleNamespace(vocab_size=100))
    engine.operations = SimpleNamespace(execute_trace=Mock(return_value=None), synchronize_device=Mock(),
        release_trace=Mock(), get_device_tensors=lambda value: [value, value], to_torch=lambda value: value)
    engine.validate_bindings = Mock()
    engine.helpers, engine.initial = shared, []
    retained = Mock() if rows > 1 else None
    if retained is not None:
        retained.replay.side_effect = lambda operation: operation()
        retained.commit.side_effect = lambda prefix, **kwargs: kwargs['publication'](prefix)
    bucket = dict(first=True, fixture=SimpleNamespace(retained=retained, close=Mock()), trace=7,
        output=(None, torch.tensor([1, 2, 0, 1])), commits={prefix: 20 + prefix for prefix in range(5)},
        checkpoints=[[tensor()] for index in range(GDN_LAYERS)])
    engine.buckets = {width: bucket for width in (1, 2, 4) if width <= rows}
    engine.allocate_carry()
    engine.save_carry()
    return engine, session


def cycle(engine, session, rows=4, abort=False):
    """One verify and its publication, the way FastRequest.step drives them."""
    with patch('verifier_engine.stage_inputs'):
        ticket = session.propose(session.request_id, max_rows=rows)
        predictions, metrics = engine.verify(ticket)
    if abort:
        session.abort(session.request_id, ticket, engine.publish)
    else:
        session.commit(session.request_id, ticket, predictions, engine.publish)
    return metrics


def reset(shared):
    for helper in shared:
        helper.restore.reset_mock()
        helper.save.reset_mock()


class CarryTests(unittest.TestCase):
    def setUp(self):
        note_prefill()

    def test_a_lone_engine_never_restores_and_saves_after_each_publication(self):
        shared = helpers()
        alone, session = engine('A', shared)
        reset(shared)
        for index in range(2):
            self.assertFalse(cycle(alone, session)['carry_restored'])
        for helper, slot in zip(shared, alone.carry, strict=True):
            helper.restore.assert_not_called()
            self.assertEqual(helper.save.call_args_list, [call(slot), call(slot)])

    def test_the_single_row_branch_saves_on_commit_and_on_abort(self):
        shared = helpers()
        alone, session = engine('A', shared, rows=1)
        reset(shared)
        cycle(alone, session, rows=1)
        cycle(alone, session, rows=1, abort=True)
        for helper, slot, checkpoint in zip(shared, alone.carry, alone.buckets[1]['checkpoints'], strict=True):
            # the abort put the pre-block checkpoint back, and the carry followed it
            helper.restore.assert_called_once_with(checkpoint)
            self.assertEqual(helper.save.call_args_list, [call(slot), call(slot)])

    def test_alternating_engines_each_restore_their_own_carry(self):
        shared = helpers()
        first, first_session = engine('A', shared)
        cycle(first, first_session)
        # B's prefill overwrites the slot; B is seeded from it and is the resident.
        note_prefill()
        second, second_session = engine('B', shared)
        reset(shared)
        self.assertFalse(cycle(second, second_session)['carry_restored'])
        self.assertTrue(cycle(first, first_session)['carry_restored'])
        self.assertTrue(cycle(second, second_session)['carry_restored'])
        for helper, own_first, own_second in zip(shared, first.carry, second.carry, strict=True):
            self.assertIsNot(own_first, own_second)
            self.assertEqual(helper.restore.call_args_list, [call(own_first), call(own_second)])
            self.assertEqual(helper.save.call_args_list, [call(own_second), call(own_first), call(own_second)])

    def test_a_prefill_displaces_the_resident_engine(self):
        shared = helpers()
        alone, session = engine('A', shared)
        cycle(alone, session)
        note_prefill()
        reset(shared)
        self.assertTrue(cycle(alone, session)['carry_restored'])
        self.assertFalse(cycle(alone, session)['carry_restored'])
        for helper, slot in zip(shared, alone.carry, strict=True):
            helper.restore.assert_called_once_with(slot)

    def test_a_failed_verify_after_restoring_still_displaces_the_other_engine(self):
        """A restored itself into the slot and then died; B must not trust its old residency."""
        shared = helpers()
        first, first_session = engine('A', shared)
        cycle(first, first_session)
        note_prefill()
        second, second_session = engine('B', shared)
        cycle(second, second_session)
        first.operations.execute_trace.side_effect = RuntimeError('device failure')
        with patch('verifier_engine.stage_inputs'), self.assertRaises(RuntimeError):
            first.verify(first_session.propose('A', max_rows=4))
        reset(shared)
        self.assertTrue(cycle(second, second_session)['carry_restored'])
        for helper, slot in zip(shared, second.carry, strict=True):
            helper.restore.assert_called_once_with(slot)

    def test_a_moved_carry_is_refused_before_any_copy(self):
        shared = helpers()
        first, first_session = engine('A', shared)
        note_prefill()
        second, second_session = engine('B', shared)
        first.carry[3][0].buffer_address.return_value = -1
        reset(shared)
        with patch('verifier_engine.stage_inputs'), self.assertRaisesRegex(ValueError, 'moved'):
            first.verify(first_session.propose('A', max_rows=4))
        self.assertEqual((first.phase, first_session.phase), ('failed', 'failed'))
        for helper in shared:
            helper.restore.assert_not_called()
        # the same check guards the save that follows publication
        with patch('verifier_engine.stage_inputs'):
            ticket = second_session.propose('B', max_rows=4)
            predictions, unused = second.verify(ticket)
        second.carry[0][4].buffer_address.return_value = -2
        with self.assertRaisesRegex(ValueError, 'moved'):
            second_session.commit('B', ticket, predictions, second.publish)
        self.assertEqual(second.phase, 'failed')
        for helper in shared:
            helper.save.assert_not_called()

    def test_log_lines_are_fenced_bounded_and_only_emitted_when_asked(self):
        """Ids are cut to 48 characters so two of them and the fields stay under 180."""
        first_id, second_id = 'A' * 100, 'B' * 100

        def round_trip(enabled):
            note_prefill()
            shared = helpers()
            environment = {'QWEN_FAST_CARRY_LOG': '1' if enabled else '0'}
            with patch.dict('os.environ', environment), patch('verifier_engine.carry_log') as log:
                first, first_session = engine(first_id, shared)
                cycle(first, first_session)
                note_prefill()
                second, second_session = engine(second_id, shared)
                cycle(second, second_session)
                fences = first.operations.synchronize_device.call_count
                cycle(first, first_session)
                fences = first.operations.synchronize_device.call_count - fences
            return [entry.args[0].format(**entry.kwargs) for entry in log.call_args_list], fences

        lines, fences = round_trip(False)
        self.assertEqual((lines, fences), ([], 0))
        lines, fences = round_trip(True)
        # A's last cycle: one restore and one save, each fenced
        self.assertEqual(fences, 2)
        self.assertTrue(all(len(line) < 180 for line in lines), max(map(len, lines)))
        request, source = 'A' * 48, 'B' * 48
        self.assertEqual(lines[-4], f'[CARRY] op=restore request={request} from={source} layers=48 begin')
        self.assertRegex(lines[-3], rf'^\[CARRY\] op=restore request={request} layers=48 enqueue_ms=\d+\.\d fence_ms=\d+\.\d$')
        self.assertEqual(lines[-2], f'[CARRY] op=save request={request} layers=48 begin')
        self.assertRegex(lines[-1], rf'^\[CARRY\] op=save request={request} layers=48 enqueue_ms=\d+\.\d fence_ms=\d+\.\d$')
        # seeds and lone-engine saves log too, and a resident engine restores nothing
        self.assertEqual([line.split()[1] for line in lines],
                         ['op=save'] * 8 + ['op=restore'] * 2 + ['op=save'] * 2)

    def test_carry_log_prints_when_loguru_is_absent(self):
        with patch.dict(sys.modules, loguru=None), patch('sys.stdout', new_callable=io.StringIO) as output:
            carry_log('[CARRY] op={op} request={request}{origin} layers={layers} begin',
                      op='save', request='A', origin='', layers=48)
        self.assertEqual(output.getvalue().strip(), '[CARRY] op=save request=A layers=48 begin')

    def test_the_carry_is_allocated_beside_initial_before_any_fixture_exists(self):
        """Allocate before capture: the slots must exist before the first fixture, let alone its trace."""
        session = GreedySession('request', [0, 1, 2], 0, vocab_size=100, max_new_tokens=33)
        shared = helpers()
        seen = []

        def fail(engine, *args, **kwargs):
            seen.append((list(engine.carry), list(engine.carry_addresses), [helper.save.call_count for helper in shared]))
            raise RuntimeError('fixture failure')

        operations = SimpleNamespace(synchronize_device=Mock(), get_device_tensors=lambda value: [value, value])
        with patch.dict(sys.modules, ttnn=operations), patch('verifier_engine.release_owned') as release, \
             patch.object(VerifierEngine, 'fixture', autospec=True, side_effect=fail):
            with self.assertRaises(RuntimeError):
                VerifierEngine(SimpleNamespace(mesh_device=object()), session, SimpleNamespace(shape=(1, 1024)), shared)
        (carry, recorded, saves), = seen
        self.assertEqual(len(carry), GDN_LAYERS)
        self.assertEqual(recorded, [[(value.buffer_address(),) * 2 for value in slot] for slot in carry])
        # only the initial snapshot had been saved: the carry is seeded after the last restore
        self.assertEqual(saves, [1] * GDN_LAYERS)
        self.assertIn(call(operations, [value for slot in carry for value in slot]), release.call_args_list)
        self.assertIsNone(verifier_engine._resident)

    def test_close_releases_the_carry_and_gives_up_residency(self):
        shared = helpers()
        alone, session = engine('A', shared)
        carried = [value for slot in alone.carry for value in slot]
        self.assertEqual(len(carried), 5 * GDN_LAYERS)
        self.assertIs(verifier_engine._resident, alone)
        with patch('verifier_engine.release_owned') as release:
            alone.close()
        self.assertIn(call(alone.operations, carried), release.call_args_list)
        self.assertEqual((alone.carry, alone.carry_addresses), ([], []))
        self.assertIsNone(verifier_engine._resident)
        self.assertEqual(alone.phase, 'closed')


if __name__ == '__main__':
    unittest.main()
