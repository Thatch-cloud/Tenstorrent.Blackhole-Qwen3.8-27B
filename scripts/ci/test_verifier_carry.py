"""Each engine carries its own GDN slot-zero state across the other engine's blocks."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding' / 'harness'))
from greedy_session import GreedySession
import verifier_engine
from verifier_engine import VerifierEngine, note_prefill
from verifier_pack import GDN_LAYERS


def helpers():
    """The 48 shared layer helpers; allocate hands out a fresh snapshot each call."""
    return [SimpleNamespace(live=[object()] * 5, allocate=Mock(side_effect=lambda: [object()] * 5),
                            save=Mock(), restore=Mock()) for index in range(GDN_LAYERS)]


def engine(request_id, shared, rows=4):
    """A prepared engine over the shared helpers, seeded the way __init__ ends."""
    session = GreedySession(request_id, [0, 1, 2] * 12, 0, vocab_size=100, max_new_tokens=64)
    engine = VerifierEngine.__new__(VerifierEngine)
    engine.session, engine.position = session, session.position
    engine.phase, engine.pending = 'idle', None
    engine.mesh = object()
    engine.model = SimpleNamespace(args=SimpleNamespace(vocab_size=100))
    engine.operations = SimpleNamespace(execute_trace=Mock(return_value=None), synchronize_device=Mock(),
        release_trace=Mock(), get_device_tensors=lambda tensor: [tensor, tensor], to_torch=lambda tensor: tensor)
    engine.validate_bindings = Mock()
    engine.helpers, engine.initial = shared, []
    retained = Mock() if rows > 1 else None
    if retained is not None:
        retained.replay.side_effect = lambda operation: operation()
        retained.commit.side_effect = lambda prefix, **kwargs: kwargs['publication'](prefix)
    bucket = dict(first=True, fixture=SimpleNamespace(retained=retained, close=Mock()), trace=7,
        output=(None, torch.tensor([1, 2, 0, 1])), commits={prefix: 20 + prefix for prefix in range(5)},
        checkpoints=[[object()] for index in range(GDN_LAYERS)])
    engine.buckets = {width: bucket for width in (1, 2, 4) if width <= rows}
    engine.seed_carry()
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

    def test_close_releases_the_carry_and_gives_up_residency(self):
        shared = helpers()
        alone, session = engine('A', shared)
        carry = alone.carry
        self.assertIs(verifier_engine._resident, alone)
        with patch('verifier_engine.release_owned'), patch('verifier_engine.release_slots') as release:
            alone.close()
        release.assert_called_once_with(alone.operations, carry)
        self.assertEqual(alone.carry, ())
        self.assertIsNone(verifier_engine._resident)
        self.assertEqual(alone.phase, 'closed')


if __name__ == '__main__':
    unittest.main()
