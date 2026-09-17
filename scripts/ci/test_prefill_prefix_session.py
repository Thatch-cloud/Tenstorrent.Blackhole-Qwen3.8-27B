"""Offline lifetime integration; not device numerical acceptance."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from prefill_prefix_lookup import PrefixIdentity
from prefill_prefix_session import offline_prefix_session
from test_prefill_prefix_integration import HostModel, HostOperations


class PrefixSessionTests(unittest.TestCase):
    def fixture(self):
        operations = HostOperations()
        model = HostModel(operations)
        for index, layer in enumerate(model.layers):
            layer.is_full_attention = index >= 48
            if index < 48:
                layer.attention = model.gdn[index]
        model._bucket_trace_id = None
        model._bind_gdn_prefill_scratch = Mock(return_value='decode')
        model._unbind_gdn_prefill_scratch = Mock()
        generator = SimpleNamespace(model=[model], trace_ids_decode={False: {}, True: {}})
        identity = PrefixIdentity('a' * 64, 'b' * 64, 'c' * 64, 'offline', 0)
        pages = torch.arange(96, dtype=torch.int32).reshape(1, -1)
        options = dict(prefix_position=4096, reserved_pages=list(range(96)), inactive_pages=[100])
        allocated = []
        clone = operations.clone

        def record_clone(source, **kwargs):
            result = clone(source, **kwargs)
            allocated.append(result)
            return result

        operations.clone = record_clone
        return operations, generator, model, identity, pages, options, allocated

    def test_cold_then_cached_and_all_owned_storage_released(self):
        operations, generator, model, identity, pages, options, allocated = self.fixture()
        tokens = list(range(6144))

        def prefill(values):
            return model._prefill_chunked_eager_tp(torch.tensor([values]), pages, 6144, 3, 2048, 0)

        with offline_prefix_session(operations, generator, model, identity, pages, **options) as factory:
            self.assertEqual(len(allocated), 288)
            with factory(tokens, prefill, 0) as (cold, unused, record):
                self.assertFalse(record['cache_hit'])
            model.starts.clear()
            with factory(tokens, prefill, 1) as (cached, unused, record):
                self.assertTrue(record['cache_hit'])
                self.assertEqual(cold, cached)
                self.assertEqual(model.starts, [4096])
        self.assertTrue(all(value.freed for value in allocated))
        self.assertFalse(any(value.freed for value in model.states))
        self.assertFalse(hasattr(model, '_qwen_prefix_session'))
        with self.assertRaisesRegex(ValueError, 'reservation'):
            with factory(tokens, prefill, 0):
                self.fail('Closed session reused')

    def test_consumer_failure_releases_checkpoint_and_features(self):
        operations, generator, model, identity, pages, options, allocated = self.fixture()
        with self.assertRaisesRegex(RuntimeError, 'consumer'):
            with offline_prefix_session(operations, generator, model, identity, pages, **options) as factory:
                def prefill(values):
                    return model.cold(torch.tensor([values]))
                with factory(list(range(6144)), prefill, 0):
                    raise RuntimeError('consumer')
        self.assertTrue(all(value.freed for value in allocated))
        self.assertFalse(hasattr(model, '_qwen_prefix_session'))

    def test_nested_or_overlapping_reservation_fails_before_extra_allocation(self):
        operations, generator, model, identity, pages, options, allocated = self.fixture()
        with offline_prefix_session(operations, generator, model, identity, pages, **options):
            with self.assertRaisesRegex(ValueError, 'Exclusive'):
                with offline_prefix_session(operations, generator, model, identity, pages, **options):
                    self.fail('Nested cache session admitted')
            self.assertEqual(len(allocated), 288)
        allocated.clear()
        options['inactive_pages'] = [0]
        with self.assertRaisesRegex(ValueError, 'reservation'):
            with offline_prefix_session(operations, generator, model, identity, pages, **options):
                self.fail('Overlapping pages admitted')
        self.assertFalse(allocated)


if __name__ == '__main__':
    unittest.main()
