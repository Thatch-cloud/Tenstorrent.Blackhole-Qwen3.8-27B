from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from force_argmax import sample_rows


class ForceArgmaxTests(unittest.TestCase):
    def fixture(self):
        return SimpleNamespace(tt_sampling=SimpleNamespace(force_argmax_sampling=True, max_batch_size=32),
            seed_manager=SimpleNamespace(has_active_request_seed=Mock(return_value=False)),
            sample=Mock(return_value=('ids', None))), SimpleNamespace(pad=Mock(return_value='padded'), deallocate=Mock(),
                get_device_tensors=lambda value: [SimpleNamespace(buffer_address=lambda: id(value))] * 2)

    def test_all_widths_use_one_untraced_sampler_and_own_only_padding(self):
        for rows in (1, 2, 4, 8, 16, 32):
            sampler, operations = self.fixture()
            logits = SimpleNamespace(shape=(1, 1, rows, 124160))
            self.assertEqual(sample_rows(sampler, logits, rows, operations), 'ids')
            sampler.sample.assert_called_once_with(logits if rows == 32 else 'padded', enable_trace=False)
            if rows == 32:
                operations.pad.assert_not_called()
                operations.deallocate.assert_not_called()
            else:
                operations.pad.assert_called_once_with(logits, [(0, 0), (0, 0), (0, 32 - rows), (0, 0)], value=0.0)
                operations.deallocate.assert_called_once_with('padded')

    def test_rejects_sampling_mode_seed_or_geometry_before_device_ops(self):
        for invalid in ('mode', 'batch', 'seed', 'shape', 'width'):
            sampler, operations = self.fixture()
            logits = SimpleNamespace(shape=(1, 1, 16, 124160))
            if invalid == 'mode':
                sampler.tt_sampling.force_argmax_sampling = False
            elif invalid == 'batch':
                sampler.tt_sampling.max_batch_size = 64
            elif invalid == 'seed':
                sampler.seed_manager.has_active_request_seed.return_value = True
            elif invalid == 'shape':
                logits.shape = (16, 124160)
            with self.assertRaises(ValueError):
                sample_rows(sampler, logits, 3 if invalid == 'width' else 16, operations)
            sampler.sample.assert_not_called()
            operations.pad.assert_not_called()

    def test_sampler_failure_releases_owned_padding(self):
        sampler, operations = self.fixture()
        sampler.sample.side_effect = RuntimeError('device')
        with self.assertRaises(RuntimeError):
            sample_rows(sampler, SimpleNamespace(shape=(1, 1, 2, 124160)), 2, operations)
        operations.deallocate.assert_called_once_with('padded')

    def test_padding_view_never_deallocates_original_storage(self):
        sampler, operations = self.fixture()
        operations.get_device_tensors = lambda value: [SimpleNamespace(buffer_address=lambda: 123)] * 2
        sample_rows(sampler, SimpleNamespace(shape=(1, 1, 2, 124160)), 2, operations)
        operations.deallocate.assert_not_called()

    def test_native_shape_need_not_support_slicing(self):
        class Shape:
            def __iter__(self):
                return iter((1, 1, 2, 124160))

            def __getitem__(self, index):
                raise TypeError('Native shape does not support slices')

        sampler, operations = self.fixture()
        self.assertEqual(sample_rows(sampler, SimpleNamespace(shape=Shape()), 2, operations), 'ids')
