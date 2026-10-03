from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import torch

from draft_head_preparation import rope_reference
from dspark_rope_tables import DSparkRotary
from dspark_rotary_device import CASES, execute, fixtures
from test_dspark_intake import configuration


def tensor(shape):
    return SimpleNamespace(shape=shape, dtype='bf16', layout='tile', memory_config=lambda: 'dram')


def runtime():
    return SimpleNamespace(bfloat16='bf16', float32='fp32', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
        MathFidelity=SimpleNamespace(HiFi4='hifi4'), WormholeComputeKernelConfig=MagicMock(return_value='kernel'),
        typecast=MagicMock(side_effect=lambda *args: object()),
        slice=MagicMock(side_effect=lambda *args, **kwargs: object()),
        neg=MagicMock(return_value=object()), concat=MagicMock(return_value=object()),
        multiply=MagicMock(side_effect=lambda *args, **kwargs: object()), add=MagicMock(return_value=object()),
        experimental=SimpleNamespace(rotary_embedding_hf=MagicMock(return_value=object())))


class DSparkRotaryDeviceTests(unittest.TestCase):
    def test_composed_path_uses_half_split_and_separate_fp32_products(self):
        operations = runtime()
        inputs = [tensor((1, 16, 32, 128)), tensor((1, 1, 32, 128)), tensor((1, 1, 32, 128))]
        owned = []
        result = execute(operations, *inputs, owned, composed=True)
        self.assertEqual(len(owned), 11)
        self.assertIs(result, owned[-1])
        self.assertEqual([call.args for call in operations.slice.call_args_list],
            [(owned[0], (0, 0, 0, 0), (1, 16, 32, 64)),
             (owned[0], (0, 0, 0, 64), (1, 16, 32, 128))])
        operations.neg.assert_called_once_with(owned[4], memory_config='dram')
        operations.concat.assert_called_once_with([owned[5], owned[3]], dim=3, memory_config='dram')
        self.assertEqual([call.args for call in operations.multiply.call_args_list],
            [(owned[0], owned[1]), (owned[6], owned[2])])
        for call in operations.multiply.call_args_list:
            self.assertFalse(call.kwargs['fast_and_approximate_mode'])
        operations.add.assert_called_once_with(owned[7], owned[8], dtype='fp32', memory_config='dram')
        operations.typecast.assert_called_with(owned[9], 'bf16')
        operations.experimental.rotary_embedding_hf.assert_not_called()
        operations.WormholeComputeKernelConfig.assert_not_called()

    def test_variant_requires_explicit_boolean(self):
        operations = runtime()
        with self.assertRaises(ValueError):
            execute(operations, None, None, None, [], composed='false')
        operations.typecast.assert_not_called()

    def test_native_path_widens_all_operands_and_preserves_borrowed_inputs(self):
        operations = runtime()
        inputs = [tensor((1, 16, 32, 128)), tensor((1, 1, 32, 128)), tensor((1, 1, 32, 128))]
        owned = []
        result = execute(operations, *inputs, owned)
        self.assertEqual(len(owned), 5)
        self.assertIs(result, owned[-1])
        self.assertEqual([call.args for call in operations.typecast.call_args_list[:3]],
            [(value, 'fp32') for value in inputs])
        call = operations.experimental.rotary_embedding_hf.call_args
        self.assertEqual(call.args, tuple(owned[:3]))
        self.assertFalse(call.kwargs['is_decode_mode'])
        self.assertFalse(operations.WormholeComputeKernelConfig.call_args.kwargs['packer_l1_acc'])
        self.assertTrue(operations.WormholeComputeKernelConfig.call_args.kwargs['fp32_dest_acc_en'])

    def test_bad_head_layout_or_precision_fails_before_dispatch(self):
        for index, field, value in ((0, 'shape', (1, 8, 32, 128)), (0, 'shape', (1, 16, 7, 128)),
                (0, 'shape', (2, 16, 32, 128)), (1, 'shape', (1, 1, 64, 128)), (2, 'dtype', 'fp32'),
                (0, 'memory_config', lambda: 'l1'), (1, 'layout', 'row')):
            operations = runtime()
            inputs = [tensor((1, 16, 32, 128)), tensor((1, 1, 32, 128)), tensor((1, 1, 32, 128))]
            setattr(inputs[index], field, value)
            with self.assertRaises(ValueError):
                execute(operations, *inputs, [])
            operations.typecast.assert_not_called()

    def test_every_fixture_separates_absolute_positions_and_padding(self):
        config = configuration()
        config['max_position_embeddings'] = 262144
        rotary = DSparkRotary(config)
        for heads, live_rows, padded_rows in CASES:
            patterns = fixtures(rotary, heads, live_rows, padded_rows)
            self.assertTrue(torch.equal(patterns[0][0], patterns[1][0]))
            self.assertFalse(torch.equal(patterns[0][1], patterns[1][1]))
            for chip in range(2):
                outputs = [rope_reference(pattern[0][chip:chip + 1], *pattern[1:]) for pattern in patterns]
                valid = [value[..., :live_rows, :] for value in outputs]
                self.assertFalse(torch.equal(valid[0], valid[1]))
                self.assertFalse(torch.equal(valid[1], valid[2]))
                self.assertTrue(torch.equal(valid[2], valid[3]))
                self.assertFalse(torch.equal(outputs[2], outputs[3]))
            for original, changed in zip(patterns[2], patterns[3], strict=True):
                self.assertNotEqual(original.data_ptr(), changed.data_ptr())


if __name__ == '__main__':
    unittest.main()
