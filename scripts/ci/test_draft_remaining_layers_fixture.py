import math
from pathlib import Path
import unittest
from unittest.mock import patch

from draft_remaining_layers_fixture import specifications, fetch_layers


class RemainingLayerFixtureTests(unittest.TestCase):
    def test_each_layer_has_complete_unique_bounded_tensor_selection(self):
        for layer in range(1, 5):
            tensors = specifications(layer)
            self.assertEqual(len(tensors), 15)
            self.assertEqual(len({filename for shape, filename in tensors.values()}), 15)
            self.assertTrue(all(name.startswith(f'layers.{layer}.') for name in tensors))
            self.assertEqual(sum(2 * math.prod(shape) for shape, filename in tensors.values()), 665948672)

    def test_invalid_selection_fails_before_network_access(self):
        with patch('draft_remaining_layers_fixture.fetch_subset') as fetch:
            for layers in ((), (1, 1), (0,), (5,), (True,)):
                with self.assertRaises(ValueError):
                    fetch_layers(Path('unused'), layers)
            fetch.assert_not_called()

    def test_layers_are_kept_in_separate_fixture_directories(self):
        with patch('draft_remaining_layers_fixture.fetch_subset', return_value={}) as fetch:
            fetch_layers(Path('staged'), (1, 4))
            self.assertEqual([call.args[0] for call in fetch.call_args_list], [Path('staged/layer-1'), Path('staged/layer-4')])
            self.assertEqual(fetch.call_args_list[1].kwargs['specifications'], specifications(4))
