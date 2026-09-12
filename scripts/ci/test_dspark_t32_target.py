from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

import dspark_t32_target as wide
from test_dspark_projection import tensor


class DSparkWideTargetTests(unittest.TestCase):
    def operations(self):
        operations = MagicMock()
        operations.uint32, operations.bfloat16, operations.float32 = 'uint32', 'bf16', 'float32'
        operations.ROW_MAJOR_LAYOUT, operations.TILE_LAYOUT, operations.DRAM_MEMORY_CONFIG = 'row', 'tile', 'dram'
        return operations

    def test_anchor_row_and_thirty_masks_use_the_entire_requested_geometry(self):
        for proposals in (31,):
            value = wide.query_inputs(248319, 4096, proposals)
            self.assertEqual(tuple(value.shape), (1, proposals))
            self.assertEqual(value.dtype, torch.int64)
            self.assertEqual(value[0, 0].item(), 248319)
            self.assertTrue((value[:, 1:] == 248070).all())
        for anchor, position, proposals in ((True, 4096, 31), (-1, 4096, 31), (248320, 4096, 31), (1, 8193, 31), (1, 4096, 15)):
            with self.assertRaises(ValueError):
                wide.query_inputs(anchor, position, proposals)

    def test_thirty_one_embeddings_gather_both_hidden_shards_then_pad_one_row(self):
        operations = self.operations()
        mesh, collectives = SimpleNamespace(shape=(1, 2)), MagicMock()
        target = SimpleNamespace(mesh_device=mesh, num_devices=2, vocab_size=248320, embd=MagicMock())
        identifiers = SimpleNamespace(shape=(1, 31), dtype='uint32', layout='row', memory_config=lambda: 'dram')
        operations.reshape.return_value = tensor((1, 1, 31, 2560))
        operations.experimental.all_gather_async.return_value = tensor((1, 1, 31, 5120))
        operations.pad.return_value = tensor((1, 1, 32, 5120))
        with patch.object(wide, 'projection_links', return_value=4):
            output = wide.noise_embeddings(operations, target, mesh, collectives, identifiers, lambda value: value, proposals=31)
        self.assertIs(output, operations.pad.return_value)
        target.embd.assert_called_once_with(identifiers, memory_config='dram')
        self.assertEqual(operations.experimental.all_gather_async.call_args.kwargs['num_links'], 4)
        self.assertEqual(operations.experimental.all_gather_async.call_args.kwargs['dim'], 3)
        self.assertEqual(operations.pad.call_args.args[1], [(0, 0), (0, 0), (0, 1), (0, 0)])
        operations.to_torch.assert_not_called()

    def test_complete_vocab_is_gathered_before_slicing_all_thirty_one_rows_from_zero(self):
        operations = self.operations()
        mesh = SimpleNamespace(shape=(1, 2))
        local = tensor((1, 1, 32, 124160))
        operations.experimental.all_gather_async.return_value = tensor((1, 1, 32, 248320))
        operations.slice.return_value = tensor((1, 1, 31, 248320))
        operations.typecast.return_value = tensor((1, 1, 31, 248320), dtype='float32')
        with patch.object(wide, 'projection_links', return_value=4):
            output = wide.gather_logits(operations, mesh, MagicMock(), local, lambda value: value, proposals=31)
        self.assertIs(output, operations.typecast.return_value)
        operations.slice.assert_called_once_with(operations.experimental.all_gather_async.return_value,
            (0, 0, 0, 0), (1, 1, 31, 248320))
        operations.typecast.assert_called_once_with(operations.slice.return_value, 'float32')
        operations.to_torch.assert_not_called()

    def test_packed_tokens_preserve_row_zero_order_and_use_bounded_concat(self):
        operations = self.operations()
        values = [SimpleNamespace(shape=(1, 1, 1), dtype='uint32', layout='row',
            memory_config=lambda: 'dram', tokens=[index]) for index in range(31)]
        operations.reshape.side_effect = lambda value, shape: value

        def concat(parts, **kwargs):
            self.assertEqual(kwargs, dict(dim=2, memory_config='dram'))
            self.assertLessEqual(len(parts), 8)
            return SimpleNamespace(tokens=[token for part in parts for token in part.tokens])

        operations.concat.side_effect = concat
        output = wide.pack_tokens(operations, [dict(token=value) for value in values], lambda value: value)
        self.assertEqual(output.tokens, list(range(31)))
        self.assertEqual(operations.concat.call_count, 5)
        operations.to_torch.assert_not_called()
        for count in (0, 8, 14, 16):
            with self.assertRaises(ValueError):
                wide.pack_tokens(operations, [dict(token=values[0])] * count, lambda value: value)


if __name__ == '__main__':
    unittest.main()
