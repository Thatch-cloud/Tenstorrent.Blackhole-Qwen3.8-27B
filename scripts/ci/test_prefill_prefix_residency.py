from dataclasses import replace
from types import SimpleNamespace
import unittest

from prefill_prefix_lookup import PrefixIdentity
from prefill_prefix_residency import OfflinePrefixResidency
from test_prefill_prefix_integration import HostOperations


class ResidencyTests(unittest.TestCase):
    def fixture(self):
        operations = HostOperations()
        model = SimpleNamespace(_paged_kv_caches=[tuple(operations.tensor((104, 1, 64, 256))
            for unused in range(2)) for unused in range(16)])
        identity = PrefixIdentity('a' * 64, 'b' * 64, 'c' * 64, 'session', 0)
        guard = OfflinePrefixResidency(operations, model, identity, list(range(96)))
        return operations, model, identity, guard

    def test_valid_subset_and_inactive_pages(self):
        unused, model, identity, guard = self.fixture()
        guard(identity, list(range(65)), list(range(96, 104)))
        self.assertTrue(guard.active)

    def test_reserved_but_unused_page_cannot_belong_to_another_slot(self):
        unused, model, identity, guard = self.fixture()
        with self.assertRaises(ValueError):
            guard(identity, list(range(65)), [95])
        self.assertFalse(guard.active)

    def test_rebound_allocation_or_metadata_rejected(self):
        for changed in ('address', 'dtype'):
            operations, model, identity, guard = self.fixture()
            if changed == 'address':
                model._paged_kv_caches[0] = (operations.tensor((104, 1, 64, 256)), model._paged_kv_caches[0][1])
            else:
                model._paged_kv_caches[0][0].dtype = 'other'
            with self.assertRaises(ValueError):
                guard(identity, list(range(65)), [])
            self.assertFalse(guard.active)

    def test_identity_range_and_reuse_after_revocation_rejected(self):
        unused, model, identity, guard = self.fixture()
        with self.assertRaises(ValueError):
            guard(replace(identity, allocation_generation=1), [0], [])
        with self.assertRaises(ValueError):
            guard(identity, [0], [])
        unused, model, identity, guard = self.fixture()
        with self.assertRaises(ValueError):
            guard(identity, [104], [])


if __name__ == '__main__':
    unittest.main()
