from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from draft_head_preparation import rope_reference, rope_tables
from draft_kv_history import DraftKVHistory


class DraftKVHistoryTests(unittest.TestCase):
    def operations(self):
        return SimpleNamespace(bfloat16=torch.bfloat16, TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
            from_torch=lambda value, **kwargs: value.clone(), ReplicateTensorToMesh=lambda mesh: mesh,
            slice=lambda value, start, end: value[tuple(slice(first, last) for first, last in zip(start, end, strict=True))],
            pad=lambda value, padding, fill: torch.nn.functional.pad(value, tuple(item for pair in reversed(padding) for item in pair), value=fill),
            concat=lambda values, dim, **kwargs: torch.cat(values, dim=dim), zeros_like=torch.zeros_like,
            copy=Mock(side_effect=lambda source, destination: destination.copy_(source)),
            synchronize_device=Mock(), deallocate=Mock())

    def features(self, rows, seed):
        return torch.randn((1, 1, rows, 5120), generator=torch.Generator().manual_seed(seed)).bfloat16()

    def reference(self, features, start, layer):
        rows = features.shape[2]
        value = (features[..., :512] * (layer + 1)).reshape(1, rows, 4, 128).transpose(1, 2).contiguous()
        return dict(k=rope_reference(value, *rope_tables(start, rows)), v=value)

    @contextmanager
    def fixture(self, features, position, *, layers=2):
        operations = self.operations()
        def project(operations, inputs, query, tables, retain, *, parameters):
            rows = inputs.shape[2]
            value = (inputs[..., :512] * (parameters['layer'] + 1)).reshape(1, rows, 4, 128).transpose(1, 2).contiguous()
            return dict(k=retain(rope_reference(value, *tables)), v=retain(value))
        def address(operations, value):
            pointer = value.untyped_storage().data_ptr()
            return pointer, pointer + 1
        with patch('draft_kv_history.project_key_value', side_effect=project), \
                patch('draft_kv_history.addresses', side_effect=address), \
                patch('draft_kv_history.release_owned', side_effect=lambda operations, owned: [operations.deallocate(value) for value in owned]):
            cache = DraftKVHistory(operations, object(), [dict(layer=layer) for layer in range(layers)],
                features, position=position, history_rows=min(position, 2048))
            try:
                yield cache, operations
            finally:
                cache.close()

    def assert_history(self, cache, features):
        for layer, pair in enumerate(cache.active):
            expected = self.reference(features, cache.position - cache.history_rows, layer)
            for name in ('k', 'v'):
                actual = pair[name][..., :cache.history_rows, :].contiguous()
                self.assertTrue(torch.equal(actual.view(torch.int16), expected[name].contiguous().view(torch.int16)))
                self.assertEqual(torch.count_nonzero(pair[name][..., cache.history_rows:, :]).item(), 0)

    def test_prepare_discard_and_commit_keep_all_layers_at_one_frontier(self):
        for position in (170, 254, 4093):
            features = self.features(min(position, 2048), position)
            with self.fixture(features, position) as (cache, operations):
                self.assert_history(cache, features)
                for prefix in (1, 7, 8, 32):
                    candidate = self.features(32, cache.position)
                    old_active = cache.active
                    saved = [{name: value.clone() for name, value in pair.items()} for pair in cache.active]
                    publication = cache.prepare(candidate, prefix, position=cache.position)
                    self.assertIs(cache.active, old_active)
                    for pair, before in zip(cache.active, saved, strict=True):
                        for name in ('k', 'v'):
                            self.assertTrue(torch.equal(pair[name].view(torch.int16), before[name].view(torch.int16)))
                    cache.discard(publication)
                    self.assertIs(cache.active, old_active)
                    with self.assertRaises(ValueError):
                        cache.commit(publication)
                    publication = cache.prepare(candidate, prefix, position=cache.position)
                    operations.copy.reset_mock()
                    cache.commit(publication)
                    operations.copy.assert_not_called()
                    self.assertIsNot(cache.active, old_active)
                    features = torch.cat((features, candidate[..., :prefix, :]), dim=2)[..., -2048:, :]
                    self.assert_history(cache, features)
                    with self.assertRaises(ValueError):
                        cache.commit(publication)

    def test_failed_preparation_cannot_publish_part_of_a_layer_or_prefix(self):
        features = self.features(2048, 41)
        with self.fixture(features, 4093) as (cache, operations):
            active = cache.active
            operations.copy.side_effect = [None, RuntimeError('injected spare-copy failure')]
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                cache.prepare(self.features(32, 91), 7, position=4093)
            self.assertIsNone(cache.pending)
            self.assertIs(cache.active, active)
            self.assertEqual(cache.position, 4093)
            self.assert_history(cache, features)
            operations.copy.side_effect = lambda source, destination: destination.copy_(source)
            publication = cache.prepare(self.features(32, 91), 1, position=4093)
            cache.discard(publication)

    def test_invalid_or_nested_publications_do_not_run_projection(self):
        with self.fixture(self.features(170, 9), 170) as (cache, operations):
            for prefix, position in ((0, 170), (33, 170), (True, 170), (1, 169), (1, True)):
                with self.assertRaises(ValueError):
                    cache.prepare(self.features(32, 4), prefix, position=position)
            pending = cache.prepare(self.features(32, 4), 7, position=170)
            with self.assertRaises(ValueError):
                cache.prepare(self.features(32, 5), 1, position=170)
            cache.close()
            self.assertEqual(pending.status, 'discarded')
            operations.deallocate.reset_mock()
            cache.close()
            operations.deallocate.assert_not_called()
            with self.assertRaises(ValueError):
                cache.prepare(self.features(32, 6), 1, position=170)

    def test_initial_geometry_and_feature_count_are_checked(self):
        for position, rows, layers in ((0, 1, 1), (True, 1, 1), (4093, 4093, 1), (170, 170, 0), (170, 170, 6)):
            with self.assertRaises(ValueError):
                DraftKVHistory(self.operations(), object(), [None] * layers, self.features(32, 4),
                    position=position, history_rows=rows)
        with self.assertRaisesRegex(ValueError, 'Complete replicated'):
            with self.fixture(self.features(32, 4), 170):
                pass


if __name__ == '__main__':
    unittest.main()
