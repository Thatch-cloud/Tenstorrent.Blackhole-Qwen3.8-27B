from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from prepared_target_features import PreparedTargetFeatures


class Layer:
    def forward(self, value):
        return value + 1


class PreparedTargetFeatureTests(unittest.TestCase):
    def fixture(self):
        model = SimpleNamespace(layers=[Layer() for index in range(4)])
        destinations = [torch.zeros(1, 1, 4, 8) for index in range(2)]
        capture = PreparedTargetFeatures(model, (3, 1), destinations,
            copy=lambda source, destination: destination.copy_(source),
            storage_ids=lambda value: (value.untyped_storage().data_ptr(),))
        return model, destinations, capture

    def test_repeated_capture_updates_fixed_storage_without_changing_target(self):
        model, destinations, capture = self.fixture()
        identities = [value.data_ptr() for value in destinations]
        for seed in (0, 9, 0):
            hidden = torch.full((1, 1, 4, 8), float(seed))
            with capture.capture():
                for layer in model.layers:
                    hidden = layer.forward(hidden)
            self.assertTrue(torch.all(hidden == seed + 4))
            self.assertEqual([value[0, 0, 0, 0].item() for value in capture.outputs()], [seed + 4, seed + 2])
            self.assertEqual([value.data_ptr() for value in destinations], identities)
            self.assertTrue(all('forward' not in layer.__dict__ for layer in model.layers))
            self.assertFalse(hasattr(model, '_qwen_target_feature_capture'))

    def test_missing_duplicate_and_reentrant_forwards_fail_closed(self):
        for case in ('missing', 'duplicate', 'reentrant'):
            model, destinations, capture = self.fixture()
            with self.assertRaises(RuntimeError):
                with capture.capture():
                    if case == 'duplicate':
                        model.layers[1].forward(torch.zeros_like(destinations[0]))
                        model.layers[1].forward(torch.zeros_like(destinations[0]))
                    elif case == 'reentrant':
                        with capture.capture():
                            self.fail('Reentrant capture entered')
            with self.assertRaises(RuntimeError):
                capture.outputs()
            self.assertFalse(capture.active)
            self.assertFalse(hasattr(model, '_qwen_target_feature_capture'))

    def test_aliases_are_rejected_without_copy(self):
        model, destinations, capture = self.fixture()
        copy = Mock()
        with self.assertRaises(ValueError):
            PreparedTargetFeatures(model, (0, 1), [destinations[0], destinations[0]], copy=copy,
                storage_ids=lambda value: (value.data_ptr(),))
        model.layers[1].forward = lambda value: destinations[0]
        capture.copy = copy
        with self.assertRaises(ValueError):
            with capture.capture():
                model.layers[1].forward(torch.zeros_like(destinations[0]))
        copy.assert_not_called()
