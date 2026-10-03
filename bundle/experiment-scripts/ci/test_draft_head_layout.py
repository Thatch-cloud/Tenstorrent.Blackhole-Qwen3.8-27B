from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from draft_head_layout import split_projected_heads, concatenate_query_heads


def tensor(shape, dtype='bf16'):
    return SimpleNamespace(shape=shape, dtype=dtype, layout='tile', memory_config=lambda: 'dram')


class DraftHeadLayoutTests(unittest.TestCase):
    def operations(self, rows):
        return SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
            pad=Mock(return_value=tensor((1, 1, rows, 2048))),
            concat=Mock(return_value=tensor((1, 1, rows, 1024))),
            slice=Mock(return_value=tensor((1, 16, 32, 128))),
            experimental=SimpleNamespace(nlp_create_qkv_heads=Mock(return_value=tuple(
                tensor((1, heads, rows, 128)) for heads in (16, 4, 4))),
                nlp_concat_heads=Mock(return_value=tensor((1, 1, 32, 2048)))))

    def test_native_split_keeps_query_rows_and_complete_kv_history(self):
        for rows in (32, 192, 2080):
            operations = self.operations(rows)
            query, key, value = tensor((1, 1, 32, 2048)), tensor((1, 1, rows, 512)), tensor((1, 1, rows, 512))
            retained = []
            def retain(result):
                retained.append(result)
                return result
            heads = split_projected_heads(operations, query, key, value, retain)
            self.assertEqual(heads['q'].shape, (1, 16, 32, 128))
            self.assertEqual(heads['k'].shape, (1, 4, rows, 128))
            self.assertEqual(heads['v'].shape, (1, 4, rows, 128))
            self.assertEqual(operations.pad.call_count, int(rows != 32))
            self.assertEqual(operations.slice.call_count, int(rows != 32))
            call = operations.experimental.nlp_create_qkv_heads.call_args
            self.assertEqual(call.kwargs, dict(num_heads=16, num_kv_heads=4, transpose_k_heads=False, memory_config='dram'))
            self.assertEqual(concatenate_query_heads(operations, heads['q'], retain).shape, (1, 1, 32, 2048))
            self.assertIn(heads['q'], retained)

    def test_invalid_projection_geometry_fails_before_native_dispatch(self):
        operations = self.operations(192)
        for query, key, value in (
                (tensor((1, 1, 8, 2048)), tensor((1, 1, 192, 512)), tensor((1, 1, 192, 512))),
                (tensor((1, 1, 32, 2048)), tensor((1, 1, 178, 512)), tensor((1, 1, 178, 512))),
                (tensor((1, 1, 32, 2048), 'fp32'), tensor((1, 1, 192, 512)), tensor((1, 1, 192, 512)))):
            with self.assertRaises(ValueError):
                split_projected_heads(operations, query, key, value, lambda result: result)
        operations.experimental.nlp_create_qkv_heads.assert_not_called()
        with self.assertRaises(ValueError):
            concatenate_query_heads(operations, tensor((1, 4, 32, 128)), lambda result: result)
