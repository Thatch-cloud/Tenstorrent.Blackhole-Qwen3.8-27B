from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from gdn_output_l1_scope import GDNOutputL1Arm


class OutputPlacementTests(unittest.TestCase):
    def fixture(self):
        layers = [SimpleNamespace(is_full_attention=False, attention=SimpleNamespace(
            args=SimpleNamespace(proj_1d_decode=True, gdn_out_decode_1d_progcfg=object()),
            cfg=object(), tw={'out': object()}, _row_proj=Mock())) for index in range(48)]
        operations = SimpleNamespace(L1_MEMORY_CONFIG=object())
        matmul = Mock(return_value=object())
        return GDNOutputL1Arm(operations, SimpleNamespace(layers=layers), matmul), matmul

    def test_same_native_operands_and_math_with_l1_output(self):
        arm, matmul = self.fixture()
        value = SimpleNamespace(shape=(1, 16, 3072))
        layer = arm.layers[0]
        with arm.install():
            layer._row_proj(value, layer.tw['out'])
        matmul.assert_called_once_with(value, layer.tw['out'], layer.args.gdn_out_decode_1d_progcfg,
            layer.cfg, out_memory_config=arm.operations.L1_MEMORY_CONFIG)
        self.assertEqual(arm.hits[0], 1)

    def test_other_shapes_and_weights_fall_back(self):
        arm, matmul = self.fixture()
        with arm.install():
            arm.layers[0]._row_proj(SimpleNamespace(shape=(1, 1, 3072)), arm.layers[0].tw['out'])
            arm.layers[0]._row_proj(SimpleNamespace(shape=(1, 16, 3072)), object())
        matmul.assert_not_called()
        self.assertEqual(arm.originals[0].call_count, 2)

    def test_failure_restores_every_layer(self):
        arm, matmul = self.fixture()
        matmul.side_effect = RuntimeError('allocation')
        with self.assertRaisesRegex(RuntimeError, 'allocation'):
            with arm.install():
                arm.layers[0]._row_proj(SimpleNamespace(shape=(1, 16, 3072)), arm.layers[0].tw['out'])
        self.assertFalse(arm.active)
        self.assertTrue(all(layer._row_proj == original for layer, original in zip(arm.layers, arm.originals)))
