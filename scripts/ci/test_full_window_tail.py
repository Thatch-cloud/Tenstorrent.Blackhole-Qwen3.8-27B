from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

from attention_request_plan import Capture, ReplayPlan
from full_window_tail import runtime_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding' / 'harness'))
from greedy_session import GreedySession


class FullWindowTailTests(unittest.TestCase):
    def test_real_session_tail_does_not_call_neural_drafter(self):
        neural = Mock(side_effect=AssertionError('Tail must not execute fixed-width drafting'))
        session = GreedySession('tail', [1] * 262130, 2, vocab_size=8,
            max_new_tokens=14, neural={'dspark': neural}, lookup_enabled=False)
        plan = ReplayPlan(261888, 262144, tuple(Capture(rows, 261888, None)
            for rows in (1, 2, 4, 8, 16)))
        publication = Mock(return_value=None)
        with runtime_scope():
            while not session.finished:
                remaining = session.max_new_tokens - len(session.emitted)
                ticket = session.propose('tail', max_rows=plan.max_rows(session.position, remaining),
                    selected='dspark')
                self.assertEqual(ticket.source, 'target')
                self.assertEqual(ticket.tokens, (2,))
                session.commit('tail', ticket, (2,), publication)
        neural.assert_not_called()
        self.assertEqual(publication.call_count, 13)
        self.assertEqual(session.committed_decode_tokens, 13)
        self.assertEqual(session.position, 262143)
        self.assertEqual(session.emitted, [2] * 14)
        session.close('tail')

    def test_fixed_draft_boundary_and_singleton_tail(self):
        plan = ReplayPlan(261888, 262144, tuple(Capture(rows, 261888, None)
            for rows in (1, 2, 4, 8, 16)))
        original = ReplayPlan.available
        with runtime_scope() as evidence:
            self.assertEqual(plan.max_rows(262129, 15), 8)
            for position in range(262130, 262144):
                remaining = 262144 - position
                self.assertEqual(plan.max_rows(position, remaining), 1)
                self.assertEqual(plan.select(position, 1, remaining).rows, 1)
                with self.assertRaises(ValueError):
                    plan.select(position, 2, remaining)
            with self.assertRaises(ValueError):
                plan.max_rows(262144, 1)
        self.assertIs(ReplayPlan.available, original)
        self.assertTrue(evidence['restored'])
        self.assertGreater(evidence['target_only_calls'], 0)

    def test_wrong_request_and_missing_singleton_fail_closed(self):
        for plan in (ReplayPlan(131072, 131328, (Capture(1, 131072, None),)),
                ReplayPlan(261888, 262400, (Capture(1, 261888, None),)),
                ReplayPlan(261888, 262144, (Capture(2, 261888, None),))):
            original = ReplayPlan.available
            with self.assertRaises(ValueError), runtime_scope() as evidence:
                position = max(plan.start, 262130) if plan.start == 261888 else plan.start
                plan.max_rows(position, plan.stop - position)
            self.assertIs(ReplayPlan.available, original)
            self.assertTrue(evidence['restored'])


if __name__ == '__main__':
    unittest.main()
