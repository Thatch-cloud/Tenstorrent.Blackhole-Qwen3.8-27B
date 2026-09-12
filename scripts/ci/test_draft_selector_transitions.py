import unittest

import torch

from draft_selector import greedy_selector_reference
from draft_selector_transitions import transition_scores_reference, select_transition_scores


class SelectorTransitionTests(unittest.TestCase):
    def test_parallel_scores_preserve_greedy_path_and_selected_scores(self):
        generator = torch.Generator().manual_seed(38127)
        for batch, positions, count, rank in ((1, 1, 1, 4), (2, 7, 16, 256), (1, 8, 4, 32)):
            vocabulary = 64
            candidates = torch.stack([torch.randperm(vocabulary, generator=generator)[:count]
                for _ in range(batch * positions)]).reshape(batch, positions, count)
            operands = (torch.randn(batch, positions, rank, generator=generator), candidates,
                torch.randn(batch, positions, count, generator=generator),
                torch.randn(vocabulary, rank, generator=generator),
                torch.randn(vocabulary, rank, generator=generator), torch.arange(batch))
            expected_path, expected_scores = greedy_selector_reference(*operands)
            transitions = transition_scores_reference(*operands)
            actual_path, actual_scores = select_transition_scores(transitions, candidates)
            self.assertTrue(torch.equal(actual_path, expected_path))
            torch.testing.assert_close(actual_scores, expected_scores, atol=1e-12, rtol=1e-12)

    def test_ties_keep_original_candidate_order(self):
        candidates = torch.tensor([[[5, 2], [8, 3]]])
        path, _ = select_transition_scores(torch.zeros(1, 2, 2, 2), candidates)
        self.assertEqual(path.tolist(), [[5, 8]])

    def test_next_row_uses_selected_candidate_index_not_token_id(self):
        scores = torch.tensor([[[[0., 1.], [0., 1.]], [[5., 0.], [0., 5.]]]])
        candidates = torch.tensor([[[100, 500], [900, 20]]])
        self.assertEqual(select_transition_scores(scores, candidates)[0].tolist(), [[500, 20]])

    def test_invalid_matrix_and_duplicate_candidates_fail(self):
        candidates = torch.tensor([[[1, 2]]])
        for scores in (torch.zeros(1, 1, 2), torch.full((1, 1, 2, 2), float('nan'))):
            with self.assertRaises(ValueError):
                select_transition_scores(scores, candidates)
        with self.assertRaisesRegex(ValueError, 'Unique'):
            select_transition_scores(torch.zeros(1, 1, 2, 2), torch.tensor([[[1, 1]]]))
