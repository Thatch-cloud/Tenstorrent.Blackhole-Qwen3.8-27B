from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from tiny_mlp import execute
from tiny_tile_product import validate_layout


class TinyMlpTests(unittest.TestCase):
    def fixture(self):
        operations = SimpleNamespace(bfloat16='BF16', bfloat4_b='BF4', bfloat8_b='BF8', L1_MEMORY_CONFIG='L1',
            linear=Mock(side_effect=['gate', 'up', 'partial']), mul=Mock(return_value='hidden'))
        source = SimpleNamespace(shape=(1, 1, 8, 5120), dtype='BF16')
        weights = {name: SimpleNamespace(dtype=dtype) for name, dtype in (('gate', 'BF4'), ('up', 'BF4'), ('down', 'BF8'))}
        return operations, source, weights, {name: name + '-program' for name in weights}

    def test_candidate_shares_small_input_and_only_converts_outer_boundaries(self):
        operations, source, weights, programs = self.fixture()
        retile = Mock(side_effect=['small-input', 'restored'])
        multiply = Mock(return_value='hidden')
        retained = []
        def retain(value):
            retained.append(value)
            return value
        state = execute(operations, source, weights, programs, 'compute', retain, tiny=True, retile=retile, multiply=multiply)
        self.assertEqual(state, dict(gate='gate', up='up', hidden='hidden', partial='restored'))
        self.assertEqual(retained, ['gate', 'up', 'partial'])
        self.assertEqual([call.args for call in retile.call_args_list], [(source, 16), ('partial', 32)])
        self.assertEqual([call.args[0] for call in operations.linear.call_args_list], ['small-input', 'small-input', 'hidden'])
        operations.mul.assert_not_called()
        multiply.assert_called_once_with('gate', 'up')

    def test_control_does_not_convert_or_replace_native_weight_program_pairs(self):
        operations, source, weights, programs = self.fixture()
        retile = Mock()
        multiply = Mock()
        state = execute(operations, source, weights, programs, 'compute', lambda value: value, tiny=False,
            retile=retile, multiply=multiply)
        self.assertEqual(state['partial'], 'partial')
        retile.assert_not_called()
        multiply.assert_not_called()
        operations.mul.assert_called_once_with('gate', 'up', memory_config='L1')
        for name, call in zip(('gate', 'up', 'down'), operations.linear.call_args_list, strict=True):
            self.assertIs(call.args[1], weights[name])
            self.assertEqual(call.kwargs['program_config'], programs[name])
            self.assertEqual(call.kwargs['compute_kernel_config'], 'compute')

    def test_missing_boundaries_changed_precision_and_other_widths_fail_closed(self):
        for mutation in ('boundaries', 'precision', 'width', 'weights'):
            operations, source, weights, programs = self.fixture()
            if mutation == 'precision':
                weights['down'].dtype = 'BF4'
            elif mutation == 'width':
                source.shape = (1, 1, 32, 5120)
            elif mutation == 'weights':
                del weights['gate']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                execute(operations, source, weights, programs, 'compute', lambda value: value, tiny=True)
            operations.linear.assert_not_called()

    def test_candidate_requires_both_prepared_callbacks(self):
        for callbacks in ({'retile': Mock()}, {'multiply': Mock()}):
            operations, source, weights, programs = self.fixture()
            with self.subTest(callbacks=callbacks), self.assertRaisesRegex(ValueError, 'Prepared DMA'):
                execute(operations, source, weights, programs, 'compute', lambda value: value, tiny=True, **callbacks)
            operations.linear.assert_not_called()

    def test_product_requires_all_three_matching_small_tiles(self):
        shapes, tiles = [(1, 1, 8, 8704)] * 3, [(16, 32)] * 3
        self.assertEqual(validate_layout(shapes, tiles), 272)
        for index in range(3):
            for invalid in ((1, 1, 8, 5120), (1, 1, 16, 8704), (8, 8704)):
                changed = list(shapes)
                changed[index] = invalid
                with self.subTest(index=index, invalid=invalid), self.assertRaises(ValueError):
                    validate_layout(changed, tiles)
            changed = list(tiles)
            changed[index] = (32, 32)
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_layout(shapes, changed)
        with self.assertRaises(ValueError):
            validate_layout(shapes[:2], tiles)
