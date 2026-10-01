"""verifier_engine_tp.VerifierEngine.proposal_rows under QWEN_FAST_BUDGET_CAP (tail caps).

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

    def test_a_block_round_answers_the_blocks_rows_from_one_token_left(self):
        self.assertEqual(engine(1).proposal_rows(packed_rows=16), 16)
        self.assertEqual(engine(40).proposal_rows(packed_rows=16), 16)
        for rows in (3, 128, '16'):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                engine(40).proposal_rows(packed_rows=rows)
        with self.assertRaises(ValueError):
            engine(40, verifier_rows=16).proposal_rows(packed_rows=32)

    def test_the_inherited_answer_without_the_flag_for_a_block_round(self):
        with patch.dict(os.environ, {'QWEN_FAST_BUDGET_CAP': '0'}):
            self.assertEqual(engine(1).proposal_rows(packed_rows=16), 1)
            self.assertEqual(engine(16).proposal_rows(packed_rows=16), 16)
        os.environ.pop('QWEN_FAST_BUDGET_CAP')
        self.assertEqual(engine(3).proposal_rows(packed_rows=16), 2)

    def test_without_the_blocks_hint_the_inherited_widest_width_that_fits_the_remaining_tokens(self):
        self.assertEqual([engine(remaining).proposal_rows() for remaining in (3, 2, 1)], [2, 2, 1])


if __name__ == '__main__':
    unittest.main()
