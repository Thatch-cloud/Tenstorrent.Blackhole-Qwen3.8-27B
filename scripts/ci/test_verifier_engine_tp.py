"""verifier_engine_tp.VerifierEngine.proposal_rows under QWEN_FAST_BUDGET_CAP (tail caps, H3).

The engine is built without a device (object.__new__): proposal_rows reads only the session's budget and
position, the engine's captured widths, its page table and its replay plan.
"""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from verifier_engine_tp import VerifierEngine

PAGES = 2052  # the served geometry: 131328 positions


def engine(remaining, *, widths=(1, 2, 4), position=5000, verifier_rows=16, replay_plan=None):
    live = object.__new__(VerifierEngine)
    live.session = SimpleNamespace(max_new_tokens=remaining + 1, emitted=[0], position=position,
                                   verifier_rows=verifier_rows)
    live.position = position
    live.widths, live.pages, live.replay_plan = tuple(widths), SimpleNamespace(shape=(1, PAGES)), replay_plan
    return live


class BudgetCapProposalRowsTests(unittest.TestCase):
    def setUp(self):
        stack = patch.dict(os.environ, {'QWEN_FAST_BUDGET_CAP': '1'})
        stack.start()
        self.addCleanup(stack.stop)

    def test_a_request_at_its_last_tokens_drafts_the_engines_widest_width(self):
        for remaining in (1, 2, 3, 4, 40):
            with self.subTest(remaining=remaining):
                self.assertEqual(engine(remaining).proposal_rows(), 4)

    def test_without_the_flag_the_widest_width_that_fits_the_remaining_tokens(self):
        with patch.dict(os.environ, {'QWEN_FAST_BUDGET_CAP': '0'}):
            self.assertEqual([engine(remaining).proposal_rows() for remaining in (3, 2, 1)], [2, 2, 1])
        os.environ.pop('QWEN_FAST_BUDGET_CAP')
        self.assertEqual([engine(remaining).proposal_rows() for remaining in (3, 2, 1)], [2, 2, 1])

    def test_the_scheduler_cannot_offer_four_rows_at_the_end_of_the_context(self):
        # vLLM offers max_model_len - 1 - position rows: four fit through position 131323
        top = PAGES * 64 - 1
        self.assertEqual([engine(1, position=top - rows).proposal_rows() for rows in (4, 3, 2, 1)], [4, 2, 2, 1])

    def test_an_engine_built_without_a_four_row_width_still_narrows(self):
        for widths, expected in (((1, 2), 2), ((1,), 1)):
            self.assertEqual(engine(1, widths=widths).proposal_rows(), expected)

    def test_a_block_round_answers_the_blocks_rows_from_one_token_left(self):
        self.assertEqual(engine(1).proposal_rows(packed_rows=16), 16)
        self.assertEqual(engine(40).proposal_rows(packed_rows=16), 16)
        for rows in (3, 128, '16'):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                engine(40).proposal_rows(packed_rows=rows)
        with self.assertRaises(ValueError):
            engine(40, verifier_rows=16).proposal_rows(packed_rows=32)

    def test_a_replay_plan_keeps_its_own_width_selection(self):
        plan = SimpleNamespace(max_rows=lambda position, remaining: 2)
        self.assertEqual(engine(1, replay_plan=plan).proposal_rows(), 2)
        self.assertEqual(engine(1, replay_plan=plan).proposal_rows(packed_rows=16), 16, 'a block round, as without a plan')

    def test_the_inherited_answer_without_the_flag_for_a_block_round(self):
        with patch.dict(os.environ, {'QWEN_FAST_BUDGET_CAP': '0'}):
            self.assertEqual(engine(1).proposal_rows(packed_rows=16), 1)
            self.assertEqual(engine(16).proposal_rows(packed_rows=16), 16)


if __name__ == '__main__':
    unittest.main()
