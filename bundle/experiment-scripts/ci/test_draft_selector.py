import unittest

import torch

from draft_selector import greedy_selector_reference


class DraftSelectorTests(unittest.TestCase):
    def operands(self):
        return [torch.ones(1, 2, 1), torch.tensor([[[1, 2], [1, 2]]]), torch.zeros(1, 2, 2),
            torch.tensor([[1.], [-1.], [1.]]), torch.tensor([[0.], [1.], [-1.]]), torch.tensor([0])]

    def test_previous_selected_token_changes_next_position(self):
        operands = self.operands()
        before = [value.clone() for value in operands]
        path, scores = greedy_selector_reference(*operands)
        self.assertEqual(path.tolist(), [[1, 2]])
        self.assertEqual(scores.tolist(), [[[1., -1.], [-1., 1.]]])
        self.assertTrue(all(torch.equal(value, original) for value, original in zip(operands, before, strict=True)))
        operands[-1] = torch.tensor([1])
        self.assertEqual(greedy_selector_reference(*operands)[0].tolist(), [[2, 1]])

    def test_ties_preserve_supplied_candidate_order(self):
        operands = self.operands()
        operands[0].zero_()
        operands[1] = torch.tensor([[[2, 1], [1, 2]]])
        self.assertEqual(greedy_selector_reference(*operands)[0].tolist(), [[2, 1]])

    def test_unary_scores_can_override_transition_scores(self):
        operands = self.operands()
        operands[2][0, 0, 1] = 3
        self.assertEqual(greedy_selector_reference(*operands)[0].tolist(), [[2, 1]])

    def test_batches_keep_independent_predecessors(self):
        operands = self.operands()
        for index in (0, 1, 2):
            operands[index] = operands[index].repeat(2, 1, 1)
        operands[-1] = torch.tensor([0, 1])
        self.assertEqual(greedy_selector_reference(*operands)[0].tolist(), [[1, 2], [2, 1]])

    def test_invalid_operands_are_rejected(self):
        for index, replacement in ((0, torch.ones(1, 9, 1)), (1, torch.tensor([[[1, 1], [1, 2]]])),
                (1, torch.tensor([[[1, 3], [1, 2]]])), (2, torch.full((1, 2, 2), float('nan'))),
                (3, torch.ones(3, 2)), (4, torch.ones(2, 1)), (5, torch.tensor([-1])),
                (5, torch.tensor([0.]))):
            with self.subTest(index=index, shape=replacement.shape):
                operands = self.operands()
                operands[index] = replacement
                with self.assertRaises(ValueError):
                    greedy_selector_reference(*operands)
