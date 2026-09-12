from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gdn_output_grid_scope import GDNOutputGridArm


class GridScopeTests(unittest.TestCase):
    @patch('gdn_output_grid_scope.widen')
    def test_native_weights_math_and_placement_preserved(self, widen):
        programs = [object() for index in range(48)]
        widen.side_effect = programs
        layers = [SimpleNamespace(is_full_attention=False, attention=SimpleNamespace(
            args=SimpleNamespace(proj_1d_decode=True, gdn_out_decode_1d_progcfg=object()),
            cfg=object(), tw={'out': object()}, _row_proj=Mock())) for index in range(48)]
        operations = SimpleNamespace(DRAM_MEMORY_CONFIG=object())
        matmul = Mock()
        arm = GDNOutputGridArm(operations, SimpleNamespace(layers=layers), matmul)
        value = SimpleNamespace(shape=(1, 16, 3072))
        with arm.install():
            for index, layer in enumerate(arm.layers):
                layer._row_proj(value, layer.tw['out'])
                matmul.assert_called_with(value, layer.tw['out'], programs[index], layer.cfg,
                    out_memory_config=operations.DRAM_MEMORY_CONFIG)
            arm.layers[0]._row_proj(SimpleNamespace(shape=(1, 1, 3072)), arm.layers[0].tw['out'])
        self.assertEqual(arm.hits, [1] * 48)
        arm.originals[0].assert_called_once()
        self.assertFalse(arm.active)
