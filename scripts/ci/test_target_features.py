from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from target_features import LayerOutputCapture


class Layer:
    def forward(self, value, *, mode):
        return value + 1


class FeatureCaptureTests(unittest.TestCase):
    def fixture(self, ids=(3, 1)):
        model = SimpleNamespace(layers=[Layer() for index in range(4)])
        released = []
        capture = LayerOutputCapture(model, ids, snapshot=lambda value: value.clone(), release=released.append,
            storage_ids=lambda value: (value.untyped_storage().data_ptr(),))
        return model, capture, released

    def test_post_layer_values_preserve_configured_order_and_model_output(self):
        model, capture, released = self.fixture()
        hidden = torch.zeros(1, 3, 8)
        with capture.capture():
            for layer in model.layers:
                hidden = layer.forward(hidden, mode='decode')
        self.assertTrue(torch.all(hidden == 4))
        self.assertEqual([value[0, 0, 0].item() for value in capture.outputs()], [4, 2])
        hidden.fill_(99)
        self.assertEqual([value[0, 0, 0].item() for value in capture.outputs()], [4, 2])
        self.assertFalse(hasattr(model, '_qwen_target_feature_capture'))
        self.assertTrue(all('forward' not in layer.__dict__ for layer in model.layers))
        capture.close()
        capture.close()
        self.assertEqual(len(released), 2)
        with self.assertRaises(RuntimeError):
            capture.outputs()

    def test_partial_forward_restores_overrides_and_releases_snapshots(self):
        model, capture, released = self.fixture()
        with self.assertRaisesRegex(RuntimeError, 'target failed'):
            with capture.capture():
                model.layers[1].forward(torch.zeros(1), mode='decode')
                raise RuntimeError('target failed')
        self.assertEqual(len(released), 1)
        self.assertTrue(capture.closed)
        self.assertFalse(hasattr(model, '_qwen_target_feature_capture'))
        self.assertNotIn('forward', model.layers[1].__dict__)

    def test_missing_duplicate_and_borrowed_taps_fail_closed(self):
        for failure in ('missing', 'duplicate', 'borrowed', 'view'):
            model, capture, released = self.fixture(ids=(1,))
            value = torch.zeros(1)
            if failure == 'borrowed':
                capture.snapshot = lambda output: output
            if failure == 'view':
                capture.snapshot = lambda output: output.view_as(output)
            with self.assertRaises((AssertionError, RuntimeError, ValueError)):
                with capture.capture():
                    if failure != 'missing':
                        model.layers[1].forward(value, mode='decode')
                    if failure == 'duplicate':
                        model.layers[1].forward(value, mode='decode')
            self.assertTrue(capture.closed)
            self.assertEqual(len(released), 1 if failure == 'duplicate' else 0)

    def test_nested_scope_cannot_replace_other_capture(self):
        model, capture, released = self.fixture(ids=(1,))
        other = LayerOutputCapture(model, (1,), snapshot=Mock(), release=Mock(), storage_ids=Mock())
        with capture.capture():
            with self.assertRaisesRegex(RuntimeError, 'already has'):
                with other.capture():
                    self.fail('Nested model capture must not enter')
            model.layers[1].forward(torch.zeros(1), mode='decode')
        self.assertFalse(other.started)
        capture.close()

    def test_tap_indices_are_unique_explicit_decoder_indices(self):
        for ids in ((), (1, 1), (True,), (-1,), (4,), (1.0,)):
            with self.assertRaises(ValueError):
                self.fixture(ids)

    def test_override_setup_failure_closes_capture(self):
        model, capture, released = self.fixture()
        model.layers[3] = object()
        with self.assertRaises(AttributeError):
            with capture.capture():
                self.fail('Invalid layer must not enter capture')
        self.assertTrue(capture.closed)

    def test_alias_on_only_one_chip_is_rejected_without_releasing_borrowed_storage(self):
        hidden, owned = object(), object()
        model = SimpleNamespace(layers=[SimpleNamespace(forward=lambda value: value)])
        release = Mock()
        capture = LayerOutputCapture(model, (0,), snapshot=lambda value: owned, release=release,
            storage_ids=lambda value: ((0, 11), (1, 12 if value is hidden else 13)))
        with self.assertRaisesRegex(ValueError, 'independent ownership'):
            with capture.capture():
                model.layers[0].forward(hidden)
        release.assert_not_called()
        self.assertTrue(capture.closed)
    def test_release_failure_attempts_all_owned_buffers(self):
        model, capture, released = self.fixture()
        with capture.capture():
            for layer in model.layers:
                layer.forward(torch.zeros(1), mode='decode')
        capture.release = Mock(side_effect=RuntimeError('release failed'))
        with self.assertRaisesRegex(RuntimeError, 'release failed'):
            capture.close()
        self.assertEqual(capture.release.call_count, 2)
        self.assertTrue(capture.closed)
