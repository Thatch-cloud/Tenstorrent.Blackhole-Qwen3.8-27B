import unittest

import torch

from dspark_splitk_layout import fold_mask, fold_query, scheduling, unfold_output


class SplitKLayoutTests(unittest.TestCase):
    def test_every_query_row_head_and_mask_is_preserved(self):
        query = torch.arange(16 * 32 * 128).reshape(1, 16, 32, 128)
        self.assertTrue(torch.equal(unfold_output(fold_query(query)), query))
        mask = torch.arange(32 * 64).reshape(1, 1, 32, 64)
        actual = fold_mask(mask)
        for kv_head in range(4):
            for row in range(32):
                for grouped_head in range(4):
                    index = kv_head * 128 + row * 4 + grouped_head
                    self.assertTrue(torch.equal(actual[0, 0, index], mask[0, 0, row]))

    def test_folded_attention_retains_full_history_and_masked_gap(self):
        torch.manual_seed(0)
        query = torch.randn(1, 16, 32, 128)
        keys, values = torch.randn(1, 4, 64, 128), torch.randn(1, 4, 64, 128)
        mask = torch.zeros(1, 1, 32, 64)
        mask[..., 39:48] = float('-inf')
        mask[..., 63] = float('-inf')
        expected = torch.nn.functional.scaled_dot_product_attention(query,
            keys.repeat_interleave(4, dim=1), values.repeat_interleave(4, dim=1), attn_mask=mask)
        folded = fold_query(query).reshape(1, 4, 128, 128)
        actual = torch.nn.functional.scaled_dot_product_attention(folded, keys, values,
            attn_mask=fold_mask(mask).reshape(1, 4, 128, 64))
        self.assertTrue(torch.allclose(unfold_output(actual.reshape(1, 1, 512, 128)), expected,
            rtol=1e-5, atol=1e-6))

    def test_grid_size_alone_does_not_create_prefill_work(self):
        self.assertEqual(scheduling()['prefill_query_work_items'], 16)
        self.assertEqual(scheduling()['decode_active_cores'], 64)
        self.assertEqual(scheduling((11, 10), 24)['decode_active_cores'], 96)


if __name__ == '__main__':
    unittest.main()
