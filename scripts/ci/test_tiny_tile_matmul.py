from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from tiny_tile_matmul import PROJECTIONS, project
from tiny_tile_dma import validate_layout


class TinyTileMatmulTests(unittest.TestCase):
    def fixture(self):
        operations = SimpleNamespace(bfloat16='BF16', bfloat4_b='BF4', bfloat8_b='BF8',
            L1_MEMORY_CONFIG='L1', Tile=lambda shape: shape, tilize=Mock(side_effect=['small-input', 'restored-output']),
            linear=Mock(return_value='projected-output'))
        value = SimpleNamespace(shape=(1, 1, 8, 5120), dtype='BF16')
        weight = SimpleNamespace(dtype='BF4')
        return operations, value, weight

    def test_candidate_borrows_prepared_boundaries_and_retains_its_matmul_output(self):
        operations, value, weight = self.fixture()
        retained = []
        def retain(tensor):
            retained.append(tensor)
            return tensor
        retile = Mock(side_effect=['small-input', 'restored-output'])
        result = project(operations, value, weight, 'program', 'compute', retain, tiny=True, retile=retile)
        self.assertEqual(result, 'restored-output')
        self.assertEqual(retained, ['projected-output'])
        self.assertEqual([call.args[1] for call in retile.call_args_list], [16, 32])
        operations.tilize.assert_not_called()
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

    def test_prepared_conversion_is_required_before_any_candidate_operation(self):
        operations, value, weight = self.fixture()
        with self.assertRaisesRegex(ValueError, 'Prepared bounded-memory'):
            project(operations, value, weight, 'program', 'compute', lambda tensor: tensor, tiny=True)
        operations.linear.assert_not_called()

    def test_dma_only_accepts_matching_native_t8_geometries(self):
        for width in (5120, 8704):
            for tiles in (((16, 32), (32, 32)), ((32, 32), (16, 32))):
                shape = (1, 1, 8, width)
                self.assertEqual(validate_layout(shape, shape, *tiles), width // 32)
        for shapes, tiles in (
                (((1, 1, 8, 5120), (1, 1, 8, 8704)), ((32, 32), (16, 32))),
                (((1, 1, 16, 5120),) * 2, ((32, 32), (16, 32))),
                (((1, 1, 8, 4096),) * 2, ((32, 32), (16, 32))),
                (((1, 1, 8, 5120),) * 2, ((32, 32), (32, 32))),
                (((1, 1, 8, 5120),) * 2, ((32, 32), (8, 32)))):
            with self.subTest(shapes=shapes, tiles=tiles), self.assertRaises(ValueError):
                validate_layout(*shapes, *tiles)
