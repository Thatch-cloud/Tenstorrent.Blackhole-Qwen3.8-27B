from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from feature_prefix import allocate_prefixes, copy_prefix, publish_prefix


class FeaturePrefixTests(unittest.TestCase):
    def fixture(self, allocated_slice=False):
        features = [torch.arange(32).reshape(1, 1, 32, 1).expand(1, 1, 32, 2560).clone().bfloat16() + tap
                    for tap in range(5)]
        freed = []

        def shards(value):
            address = value.untyped_storage().data_ptr()
            return [SimpleNamespace(buffer_address=lambda address=address + chip: address) for chip in (0, 1)]

        def sliced(value, start, end, memory_config):
            self.assertEqual(start, (0, 0, 0, 0))
            result = value[:, :, :end[2], :end[3]]
            return result.clone() if allocated_slice else result

        operations = SimpleNamespace(DRAM_MEMORY_CONFIG=object(), get_device_tensors=shards,
            slice=Mock(side_effect=sliced), clone=Mock(side_effect=lambda value, **kwargs: value.clone()),
            copy=Mock(side_effect=lambda source, target: target.copy_(source)),
            deallocate=lambda value: freed.append(value.untyped_storage().data_ptr()))
        return features, operations, freed

    def test_publication_reuses_preallocated_addresses(self):
        features, operations, freed = self.fixture()
        groups = allocate_prefixes(operations, features, (0, 1, 17, 32))
        for feature in features:
            feature.add_(100)
        operations.clone.reset_mock()
        for prefix, group in zip((0, 1, 17, 32), groups):
            before = [value.data_ptr() for value in group]
            publish_prefix(operations, features, group, prefix)
            self.assertEqual([value.data_ptr() for value in group], before)
            for value, source in zip(group, features):
                self.assertTrue(torch.equal(value, source[:, :, :prefix]))
        operations.clone.assert_not_called()
        with self.assertRaises(ValueError):
            publish_prefix(operations, features, features, 32)

    def test_only_committed_rows_survive_source_overwrite(self):
        for prefix in (0, 1, 17, 32):
            for allocated in (False, True):
                features, operations, freed = self.fixture(allocated)
                source_addresses = [value.untyped_storage().data_ptr() for value in features]
                expected = [value[:, :, :prefix].clone() for value in features]
                result = copy_prefix(operations, features, prefix)
                for feature in features:
                    feature.fill_(-1)
                self.assertEqual(len(result), 5 if prefix else 0)
                for actual, gold in zip(result, expected):
                    self.assertTrue(torch.equal(actual, gold))
                    self.assertEqual(actual.shape[2], prefix)
                self.assertFalse(set(freed) & set(source_addresses))
                if not prefix:
                    operations.slice.assert_not_called()
                    operations.clone.assert_not_called()

    def test_failed_copy_releases_completed_outputs_not_sources(self):
        features, operations, freed = self.fixture()
        first = features[0][:, :, :1].clone()
        operations.clone.side_effect = [first, RuntimeError('copy failed')]
        with self.assertRaisesRegex(RuntimeError, 'copy failed'):
            copy_prefix(operations, features, 1)
        self.assertEqual(freed, [first.untyped_storage().data_ptr()])

    def test_borrowed_clone_and_invalid_prefix_fail(self):
        features, operations, freed = self.fixture()
        operations.clone.side_effect = lambda value, **kwargs: value
        with self.assertRaisesRegex(ValueError, 'independent chip storage'):
            copy_prefix(operations, features, 32)
        self.assertEqual(freed, [])
        for prefix in (-1, True, 33):
            with self.assertRaises(ValueError):
                copy_prefix(operations, features, prefix)
