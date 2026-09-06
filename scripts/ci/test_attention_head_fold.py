import unittest

import torch

from attention_head_fold import causal_mask, fold_query, unfold_output


class HeadFoldTests(unittest.TestCase):
    def test_roundtrip_and_gqa_mapping(self):
        for rows in (1, 2, 4, 8, 16, 32):
            query = torch.arange(rows * 12 * 256).reshape(1, rows, 12, 256)
            folded = fold_query(query)
            self.assertTrue(torch.equal(unfold_output(folded, rows), query))
            for head in range(12):
                for token in range(rows):
                    mapped = (head // 6) * rows * 6 + token * 6 + head % 6
                    self.assertTrue(torch.equal(folded[0, 0, mapped], query[0, token, head]))

    def test_each_virtual_head_masks_future_tokens(self):
        for rows in (1, 2, 4, 8, 16, 32):
            mask = causal_mask(rows, 63, 128)
            for head in range(rows * 12):
                position = 63 + (head % (rows * 6)) // 6
                self.assertTrue(torch.all(mask[0, 0, head, :position + 1] == 0))
                self.assertTrue(torch.all(torch.isneginf(mask[0, 0, head, position + 1:])))

    def test_invalid_geometry_is_rejected(self):
        for geometry in ((3, 0, 64), (2, -1, 64), (2, 63, 64), (True, 0, 64)):
            with self.assertRaises(ValueError):
                causal_mask(*geometry)
