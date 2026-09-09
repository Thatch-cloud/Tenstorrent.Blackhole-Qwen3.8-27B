from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from tiny_tile_matmul import PROJECTIONS, project


class TinyTileMatmulTests(unittest.TestCase):
    def fixture(self):
        operations = SimpleNamespace(bfloat16='BF16', bfloat4_b='BF4', bfloat8_b='BF8',
            L1_MEMORY_CONFIG='L1', Tile=lambda shape: shape, tilize=Mock(side_effect=['small-input', 'restored-output']),
            linear=Mock(return_value='projected-output'))
        value = SimpleNamespace(shape=(1, 1, 8, 5120), dtype='BF16')
        weight = SimpleNamespace(dtype='BF4')
        return operations, value, weight

    def test_candidate_retiles_both_boundaries_and_retains_every_intermediate(self):
        operations, value, weight = self.fixture()
        retained = []
        def retain(tensor):
            retained.append(tensor)
            return tensor
        result = project(operations, value, weight, 'program', 'compute', retain, tiny=True)
        self.assertEqual(result, 'restored-output')
        self.assertEqual(retained, ['small-input', 'projected-output', 'restored-output'])
        self.assertEqual([call.kwargs['tile'] for call in operations.tilize.call_args_list], [(16, 32), (32, 32)])
        operations.linear.assert_called_once_with('small-input', weight, program_config='program',
            compute_kernel_config='compute', memory_config='L1')

    def test_control_preserves_native_matmul_and_never_retiles(self):
        operations, value, weight = self.fixture()
        self.assertEqual(project(operations, value, weight, 'program', 'compute', lambda tensor: tensor, tiny=False),
            'projected-output')
        operations.tilize.assert_not_called()
        self.assertIs(operations.linear.call_args.args[0], value)

    def test_changed_scope_or_precision_fails_before_device_operations(self):
        for mutation in ('width', 'activation', 'weight', 'flag'):
            operations, value, weight = self.fixture()
            if mutation == 'width':
                value.shape = (1, 1, 32, 5120)
            elif mutation == 'activation':
                value.dtype = 'BF8'
            elif mutation == 'weight':
                weight.dtype = 'BF16'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                project(operations, value, weight, 'program', 'compute', lambda tensor: tensor,
                    tiny=1 if mutation == 'flag' else True)
            operations.linear.assert_not_called()
            operations.tilize.assert_not_called()

    def test_local_shapes_and_weight_precision_match_the_native_mlp(self):
        self.assertEqual(PROJECTIONS['gate'], (5120, 8704, 44, 'bfloat4_b', True))
        self.assertEqual(PROJECTIONS['up'], (5120, 8704, 44, 'bfloat4_b', False))
        self.assertEqual(PROJECTIONS['down'], (8704, 5120, 33, 'bfloat8_b', False))
