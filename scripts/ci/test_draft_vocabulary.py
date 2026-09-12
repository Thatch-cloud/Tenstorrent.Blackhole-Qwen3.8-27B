import unittest

import torch

from draft_vocabulary import select_vocabulary, materialize_head, global_argmax


class DraftVocabularyTests(unittest.TestCase):
    def test_frequency_selection_preserves_required_tokens_and_ties(self):
        self.assertEqual(select_vocabulary([1, 9, 9, 0, 5], 3, required=(3,)), (1, 2, 3))
        self.assertEqual(select_vocabulary([0] * 5, 3), (0, 1, 2))
        with self.assertRaises(ValueError):
            select_vocabulary([1, 2], 1, required=(0, 1))

    def test_materialization_matches_selected_full_head_columns(self):
        weight = torch.arange(24, dtype=torch.float32).reshape(6, 4).bfloat16()
        original = weight.clone()
        packed, mapping = materialize_head(weight, (0, 2, 5))
        hidden = torch.tensor([[1., -1., 2., 0.]])
        torch.testing.assert_close(hidden @ packed.float().T, (hidden @ weight.float().T)[:, [0, 2, 5]], rtol=0, atol=0)
        self.assertEqual(global_argmax(torch.tensor([[3., 3., 2.]]), mapping).item(), 0)
        packed.zero_()
        self.assertTrue(torch.equal(weight, original))

    def test_invalid_inputs_fail_closed(self):
        for counts, size in (([1, -1], 1), ([1.0, 2], 1), ([1, 2], True), ([], 1)):
            with self.assertRaises(ValueError):
                select_vocabulary(counts, size)
        weight = torch.ones(4, 2)
        for mapping in ((1, 1), (3, 1), (4,), (True,)):
            with self.assertRaises(ValueError):
                materialize_head(weight, mapping)
        with self.assertRaises(ValueError):
            global_argmax(torch.tensor([[float('nan')]]), torch.tensor([1]))


if __name__ == '__main__':
    unittest.main()
