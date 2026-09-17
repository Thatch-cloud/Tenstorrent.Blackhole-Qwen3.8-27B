import unittest

import torch

from draft_attention import draft_attention_mask
from draft_live_qk import live_qk_geometry, validate_live_qk_mask


class LiveQKTests(unittest.TestCase):
    def test_only_qk_shapes_are_accepted(self):
        self.assertEqual(live_qk_geometry((1, 16, 32, 128), (1, 16, 2080, 128)), (64, 65, 4))
        for left, right in (((1, 16, 32, 2080), (1, 16, 128, 2080)),
                ((1, 16, 8, 128), (1, 16, 32, 128)),
                ((1, 16, 32, 128), (1, 4, 32, 128))):
            with self.assertRaises(ValueError):
                live_qk_geometry(left, right)

    def test_padding_score_elision_preserves_every_softmax_row(self):
        generator = torch.Generator().manual_seed(3827)
        for context in (0, 31, 32, 2048):
            mask = draft_attention_mask(context)
            validate_live_qk_mask(mask)
            scores = torch.randn((1, 16, 32, mask.shape[-1]), generator=generator)
            candidate = scores.clone()
            candidate[..., 8:, :] = 0
            self.assertTrue(torch.equal((scores * (128 ** -.5) + mask).softmax(-1),
                (candidate * (128 ** -.5) + mask).softmax(-1)))

    def test_arbitrary_padding_and_nonfinite_bias_are_rejected(self):
        for bad_value in (float('nan'), float('inf'), 1.):
            mask = draft_attention_mask(31)
            mask[..., 0, 0] = bad_value
            with self.assertRaises(ValueError):
                validate_live_qk_mask(mask)
        for mode in ('two_keys', 'no_key', 'live_empty', 't32'):
            mask = draft_attention_mask(31, block_rows=32 if mode == 't32' else 8)
            if mode == 'two_keys':
                mask[..., 8, 0] = 0
            elif mode == 'no_key':
                mask[..., 8, :] = float('-inf')
            elif mode == 'live_empty':
                mask[..., 0, :] = float('-inf')
            with self.assertRaises(ValueError):
                validate_live_qk_mask(mask)
