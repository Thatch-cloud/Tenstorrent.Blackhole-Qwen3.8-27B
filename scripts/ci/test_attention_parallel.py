import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from attention_head_fold import fold_query, parallel_groups, unfold_output
from attention_parallel import execute


class ParallelAdapterTests(unittest.TestCase):
    def test_device_operation_chain_preserves_all_rows_and_tail_order(self):
        def tensor(value):
            return SimpleNamespace(value=value, shape=value.shape)

        def layout(mesh, value, count, owned, *, inverse=False, offset=0):
            result = tensor(unfold_output(value.value, count) if inverse else
                fold_query(value.value[:, offset:offset + count].contiguous()))
            owned.append(result)
            return result

        for rows in (1, 2, 4, 8, 16, 32):
            for start in (4095, 4096):
                operations = SimpleNamespace(DRAM_MEMORY_CONFIG='dram',
                    concat=lambda values, dim, **kwargs: tensor(torch.cat([value.value for value in values], dim=dim)),
                    slice=lambda value, first, last, **kwargs: tensor(value.value[tuple(slice(begin, end)
                        for begin, end in zip(first, last, strict=True))]),
                    transformer=SimpleNamespace(paged_scaled_dot_product_attention_decode=Mock(
                        side_effect=lambda query, *args, **kwargs: tensor(query.value.clone()))))
                query = tensor(torch.arange(rows * 12 * 256).reshape(1, rows, 12, 256))
                bundles = parallel_groups(start, rows)
                metadata = [(bundle, None, None, None) for bundle in bundles]
                owned = []
                with patch('attention_parallel.device_layout_dma', side_effect=layout):
                    result = execute(None, operations, query, None, None, metadata, owned, scale=0.0625, memory_config='L1')
                self.assertTrue(torch.equal(result.value, query.value))
                self.assertIs(owned[-1], result)
                self.assertEqual(operations.transformer.paged_scaled_dot_product_attention_decode.call_count, len(bundles))

    def test_invalid_group_shape_rejected_before_dispatch(self):
        groups = [dict(offset=0, rows=4, signature=(256, 4352)), dict(offset=4, rows=3, signature=(256, 4352))]
        with self.assertRaises(ValueError):
            execute(None, None, None, None, None, [(groups, None, None, None)], [], scale=0.0625, memory_config=None)
