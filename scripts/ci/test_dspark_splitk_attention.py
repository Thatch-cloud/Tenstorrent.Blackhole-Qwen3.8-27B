from types import SimpleNamespace
import unittest

import torch

from dspark_splitk_attention import execute_folded


class SplitKAdapterTests(unittest.TestCase):
    def test_device_layout_matches_oracle_without_copying_kv(self):
        torch.manual_seed(3)
        query = torch.randn(1, 16, 32, 128)
        keys, values = torch.randn(1, 4, 256, 128), torch.randn(1, 4, 256, 128)
        mask = torch.zeros(1, 1, 32, 256)
        mask[..., 128:240] = float('-inf')
        calls = []

        def decode(folded, actual_keys, actual_values, **options):
            self.assertEqual(actual_keys.data_ptr(), keys.data_ptr())
            self.assertEqual(actual_values.data_ptr(), values.data_ptr())
            self.assertEqual(tuple(actual_keys.shape), (4, 1, 256, 128))
            self.assertEqual(tuple(folded.shape), (1, 4, 128, 128))
            self.assertFalse(options['is_causal'])
            self.assertEqual(options['program_config']['max_cores_per_head_batch'], 16)
            calls.append(options)
            return torch.nn.functional.scaled_dot_product_attention(folded.reshape(4, 1, 128, 128),
                actual_keys, actual_values, attn_mask=options['attn_mask'],
                scale=options['scale']).reshape(1, 4, 128, 128)

        operations = SimpleNamespace(DRAM_MEMORY_CONFIG='dram', ROW_MAJOR_LAYOUT='row', TILE_LAYOUT='tile',
            MathFidelity=SimpleNamespace(HiFi4='hifi4'), WormholeComputeKernelConfig=lambda **kwargs: kwargs,
            SDPAProgramConfig=lambda **kwargs: kwargs, transformer=SimpleNamespace(scaled_dot_product_attention_decode=decode),
            to_layout=lambda tensor, layout, **kwargs: tensor.clone(), reshape=lambda tensor, shape: tensor.reshape(shape),
            permute=lambda tensor, order, **kwargs: tensor.permute(order).contiguous(),
            repeat_interleave=lambda tensor, count, dim, **kwargs: tensor.repeat_interleave(count, dim),
            repeat=lambda tensor, shape, **kwargs: tensor.repeat(shape))
        actual = execute_folded(operations, query, keys, values, mask, [])
        expected = torch.nn.functional.scaled_dot_product_attention(query, keys.repeat_interleave(4, 1),
            values.repeat_interleave(4, 1), attn_mask=mask)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
