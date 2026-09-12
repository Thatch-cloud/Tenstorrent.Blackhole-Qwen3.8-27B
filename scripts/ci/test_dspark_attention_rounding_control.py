"""CPU checks for the diagnostic, independent of device kernel qualification."""

import unittest

import torch

from dspark_attention_rounding_control import bf16_storage, online_attention


class OnlineAttentionTests(unittest.TestCase):
    def test_truncation_differs_from_round_to_nearest(self):
        values = torch.tensor([1.007, -1.007, float('-inf')])
        truncated = bf16_storage(values, truncate=True)
        self.assertEqual(truncated[:2].tolist(), [1., -1.])
        self.assertEqual(bf16_storage(values)[:2].tolist(), [1.0078125, -1.0078125])
        self.assertTrue(torch.isneginf(truncated[2]))

    def test_masked_chunks_and_changing_maximum(self):
        generator = torch.Generator().manual_seed(383928)
        query = torch.randn(1, 2, 3, 128, generator=generator)
        key = torch.randn(1, 2, 192, 128, generator=generator)
        value = torch.randn(1, 2, 192, 128, generator=generator)
        key[:, :, 128:] = query[:, :, :1] * 2
        mask = torch.zeros(1, 1, 3, 192)
        mask[:, :, :, 64:128] = float('-inf')
        expected = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask).bfloat16().float()
        actual = online_attention(query, key, value, mask)
        torch.testing.assert_close(actual, expected, rtol=.01, atol=.01)
        value[:, :, 64:128] = 8192
        torch.testing.assert_close(online_attention(query, key, value, mask), actual, rtol=0, atol=0)

    def test_constant_values_survive_all_rounding_variants(self):
        query = torch.ones(1, 1, 2, 128)
        key = torch.ones(1, 1, 128, 128)
        value = torch.full_like(key, 2)
        mask = torch.zeros(1, 1, 2, 128)
        for round_statistics in (False, True):
            for truncate_scale in (False, True):
                actual = online_attention(query, key, value, mask, round_output=True,
                    round_probability=True, round_statistics=round_statistics,
                    truncate_scale=truncate_scale)
                torch.testing.assert_close(actual, torch.full_like(query, 2), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
