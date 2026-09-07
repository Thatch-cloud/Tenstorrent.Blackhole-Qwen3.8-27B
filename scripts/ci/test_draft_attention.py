import unittest

import torch

from draft_attention import draft_attention_mask


class DraftAttentionTests(unittest.TestCase):
    def test_all_proposals_visible_but_padding_hidden(self):
        mask = draft_attention_mask(31)
        self.assertEqual(tuple(mask.shape), (1, 1, 32, 64))
        self.assertTrue((mask[..., :8, 31:39] == 0).all())
        self.assertTrue(torch.isneginf(mask[..., :8, 39:]).all())
        self.assertTrue((mask[..., 8:, :] == 0).sum(-1).eq(1).all())

    def test_sliding_boundary_advances_per_query(self):
        mask = draft_attention_mask(2048)[0, 0]
        for row in range(8):
            self.assertTrue(torch.isneginf(mask[row, :row + 1]).all())
            self.assertEqual(mask[row, row + 1].item(), 0)
            self.assertTrue((mask[row, 2048:2056] == 0).all())

    def test_empty_context_is_bidirectional_block(self):
        mask = draft_attention_mask(0)
        self.assertTrue((mask[..., :8, :8] == 0).all())
        self.assertTrue(torch.isneginf(mask[..., :8, 8:]).all())

    def test_invalid_geometry_is_rejected(self):
        for context in (-1, True, 262145):
            with self.assertRaises(ValueError):
                draft_attention_mask(context)
        with self.assertRaises(ValueError):
            draft_attention_mask(31, block_rows=2)
