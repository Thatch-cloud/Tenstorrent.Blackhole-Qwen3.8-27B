from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from feature_prefix import allocate_prefix_pool, allocate_prefixes, copy_prefix, publish_prefix


class FeaturePrefixTests(unittest.TestCase):
    def test_pool_allocates_both_epochs_upfront_and_publishes_without_clones(self):
        features, operations, freed = self.fixture()
        factory = Mock(side_effect=lambda prefix: torch.zeros((1, 1, prefix, 2560), dtype=torch.bfloat16))
        pool = allocate_prefix_pool(operations, factory)
        self.assertEqual(factory.call_count, 40)
        self.assertEqual(pool[0, 0], ())
        self.assertEqual(pool[1, 0], ())
        first = pool[0, 1]
        second = pool[1, 1]
        publish_prefix(operations, features, first, 1)
        for feature in features:
            feature.add_(100)
        publish_prefix(operations, features, second, 1)
        for before, after in zip(first, second):
            self.assertTrue(torch.equal(after, before + 100))
        self.assertEqual(factory.call_count, 40)
        operations.clone.assert_not_called()
        self.assertEqual(freed, [])

    def test_pool_rejects_aliases_geometry_and_releases_partial_allocation(self):
        features, operations, freed = self.fixture()
        allocated = torch.zeros((1, 1, 1, 2560), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, 'independent chip storage'):
            allocate_prefix_pool(operations, lambda prefix: allocated)
        self.assertEqual(freed, [allocated.data_ptr()])
        freed.clear()
        with self.assertRaisesRegex(ValueError, 'geometry'):
            allocate_prefix_pool(operations, lambda prefix: features[0])
        self.assertEqual(freed, [features[0].data_ptr()])
        for prefixes in ((0,), (True,), (1, 1), ()):
            with self.assertRaises(ValueError):
                allocate_prefix_pool(operations, Mock(), prefixes=prefixes)

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
