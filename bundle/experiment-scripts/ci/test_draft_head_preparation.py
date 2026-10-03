import unittest

import torch

from draft_head_preparation import rope_tables, rope_reference, head_norm_reference


class DraftHeadPreparationTests(unittest.TestCase):
    def test_zero_position_is_identity_and_halves_share_angles(self):
        cosine, sine = rope_tables(0, 32)
        value = torch.randn(1, 16, 32, 128).bfloat16()
        self.assertTrue(torch.equal(rope_reference(value, cosine, sine)[..., 0, :], value[..., 0, :]))
        self.assertTrue(torch.equal(cosine[..., :64], cosine[..., 64:]))
        shifted, unused = rope_tables(4096, 32)
        self.assertFalse(torch.equal(shifted, cosine))

    def test_rotation_uses_half_split_not_adjacent_pairs(self):
        value = torch.arange(128).bfloat16().reshape(1, 1, 1, 128)
        actual = rope_reference(value, torch.zeros_like(value), torch.ones_like(value))
        self.assertTrue(torch.equal(actual[..., :64], -value[..., 64:]))
        self.assertTrue(torch.equal(actual[..., 64:], value[..., :64]))

    def test_head_norm_uses_direct_weight(self):
        value = torch.full((1, 4, 32, 128), 2., dtype=torch.bfloat16)
        weight = torch.full((128,), .5, dtype=torch.bfloat16)
        self.assertTrue(head_norm_reference(value, weight).eq(.5).all())
        with self.assertRaises(ValueError):
            rope_tables(-1, 32)
