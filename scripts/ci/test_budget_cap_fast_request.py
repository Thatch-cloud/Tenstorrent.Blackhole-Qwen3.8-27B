"""FastRequest.prepare and step under QWEN_FAST_BUDGET_CAP (tail caps).

test_serving_fast_request's fixture, in a module of its own: that file is imported by an in-image test
(test_serving_vllm_contract), so changing it would have to ship in the C2 overlay."""

import unittest
from unittest.mock import Mock, patch

import test_serving_fast_request as base


class BudgetCapTests(unittest.TestCase):
    """QWEN_FAST_BUDGET_CAP: a proposal keeps the engine's (or the block's) width past the remaining
    budget, and the session's commit cuts the emission there."""

    def setUp(self):
        stack = patch.dict('os.environ', {'QWEN_FAST_BUDGET_CAP': '1'})
        stack.start()
        self.addCleanup(stack.stop)

    def fixture(self, budget, rows):
        request, events = base.FastServingTests.fixture(self, budget)
        request.engine.proposal_rows = Mock(return_value=rows)
        return request, events

    def test_prepare_for_a_block_round_keeps_sixteen_rows_at_three_left(self):
        request, events = self.fixture(4, 16)
        ticket = request.prepare('request', packed_rows=16)
        request.engine.proposal_rows.assert_called_once_with(packed_rows=16)
        self.assertEqual(len(ticket.tokens), 16)

    def test_flag_off_the_ticket_is_narrowed_to_the_budget_as_before(self):
        with patch.dict('os.environ', {'QWEN_FAST_BUDGET_CAP': '0'}):
            request, events = self.fixture(4, 16)
            ticket = request.prepare('request', packed_rows=16)
            self.assertEqual(len(ticket.tokens), 2)
            request, events = self.fixture(4, 4)
            self.assertEqual(len(request.prepare('request').tokens), 2)

    def test_a_sequential_step_at_three_left_drafts_four_rows_emits_three_and_publishes_prefix_three(self):
        request, events = self.fixture(4, 4)
        output = request.step('request', cancelled=lambda: False)
        self.assertEqual(output.token_ids, (11, 12, 13))
        self.assertTrue(output.finished)
        self.assertEqual(events, [('verify', 4), ('target', 3), ('history', 3)])

    def test_the_capped_emission_is_the_uncapped_prefix_for_every_remaining(self):
        for left in (1, 2, 3, 4, 5):
            request, events = self.fixture(left + 1, 4)
            uncapped, _ = self.fixture(32, 4)
            expected = uncapped.step('request', cancelled=lambda: False).token_ids
            output = request.step('request', cancelled=lambda: False)
            with self.subTest(left=left):
                self.assertEqual(output.token_ids, expected[:left])
                self.assertEqual(request.session.finished, left <= 4)

    def test_a_cancelled_step_aborts_as_before(self):
        request, events = self.fixture(4, 4)
        output = request.step('request', cancelled=Mock(side_effect=(False, True)))
        self.assertTrue(output.cancelled)
        self.assertEqual(events, [('verify', 4), ('target', 0)])
        self.assertEqual(request.session.emitted, [10])


if __name__ == '__main__':
    unittest.main()
