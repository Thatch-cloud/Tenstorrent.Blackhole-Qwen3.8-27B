from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from draft_head_preparation import rope_reference, rope_tables
from draft_kv_history import KV_SHAPE, DraftKVHistory, bank_tensors


def pooled_storage(layers=2):
    """Zeroed banks as the serving pool hands them over: per layer, active and spare k and v."""
    return [{side: {head: torch.zeros(KV_SHAPE, dtype=torch.bfloat16) for head in ('k', 'v')}
             for side in ('active', 'spare')} for layer in range(layers)]


def freed(operations):
    return [call.args[0] for call in operations.deallocate.call_args_list]


class DraftKVHistoryTests(unittest.TestCase):
    def operations(self):
        return SimpleNamespace(bfloat16=torch.bfloat16, TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
            from_torch=lambda value, **kwargs: value.clone(), ReplicateTensorToMesh=lambda mesh: mesh,
            slice=lambda value, start, end: value[tuple(slice(first, last) for first, last in zip(start, end, strict=True))],
            pad=lambda value, padding, fill: torch.nn.functional.pad(value, tuple(item for pair in reversed(padding) for item in pair), value=fill),
            concat=lambda values, dim, **kwargs: torch.cat(values, dim=dim), zeros_like=torch.zeros_like,
            copy=Mock(side_effect=lambda source, destination: destination.copy_(source)),
            synchronize_device=Mock(), deallocate=Mock(), get_device_tensors=lambda value: [value, value], to_torch=lambda value: value)

    def features(self, rows, seed):
        return torch.randn((1, 1, rows, 5120), generator=torch.Generator().manual_seed(seed)).bfloat16()

    def reference(self, features, start, layer):
        rows = features.shape[2]
        value = (features[..., :512] * (layer + 1)).reshape(1, rows, 4, 128).transpose(1, 2).contiguous()
        return dict(k=rope_reference(value, *rope_tables(start, rows)), v=value)

    @staticmethod
    def address(operations, value):
        pointer = value.untyped_storage().data_ptr()
        return pointer, pointer + 1

    @contextmanager
    def fixture(self, features, position, *, layers=2, storage=None):
        operations = self.operations()
        def project(operations, inputs, query, tables, retain, *, parameters):
            rows = inputs.shape[2]
            value = (inputs[..., :512] * (parameters['layer'] + 1)).reshape(1, rows, 4, 128).transpose(1, 2).contiguous()
            return dict(k=retain(rope_reference(value, *tables)), v=retain(value))
        with patch('draft_kv_history.project_key_value', side_effect=project), \
                patch('draft_kv_history.addresses', side_effect=self.address), \
                patch('draft_kv_history.release_owned', side_effect=lambda operations, owned: [operations.deallocate(value) for value in owned]):
            cache = DraftKVHistory(operations, object(), [dict(layer=layer) for layer in range(layers)],
                features, position=position, history_rows=min(position, 2048),
                **(dict(storage=storage) if storage is not None else {}))
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
                for prefix in range(1, 33):
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

    def test_audit_recomputes_the_full_committed_history_and_detects_corruption(self):
        features = self.features(170, 31)
        with self.fixture(features, 170) as (cache, operations):
            cache.audit(features)
            self.assertEqual(len(cache.checks), 8)
            cache.active[1]['k'][..., 7, 0] = 300
            with self.assertRaisesRegex(AssertionError, 'historical K/V differs'):
                cache.audit(features)

    def test_without_storage_nothing_is_borrowed_and_the_banks_are_owned_and_freed(self):
        with self.fixture(self.features(170, 5), 170) as (cache, operations):
            self.assertEqual(cache.borrowed, [])
            banks = [pair[name] for pairs in (cache.active, cache.spare) for pair in pairs for name in ('k', 'v')]
            self.assertEqual(len(cache.owned), 1 + len(banks))
            self.assertTrue(all(any(value is owned for owned in cache.owned) for value in banks))
        self.assertTrue(all(any(value is released for released in freed(operations)) for value in banks))

    def test_lent_banks_are_adopted_in_place_swapped_by_commits_and_never_freed(self):
        for position in (170, 4093):
            features = self.features(min(position, 2048), position)
            storage = pooled_storage()
            banks = bank_tensors(storage)
            with self.fixture(features, position, storage=storage) as (cache, operations):
                self.assertEqual(cache.borrowed, banks)
                self.assertEqual(cache.owned, [cache.query])
                for layer, bank in enumerate(storage):
                    for name in ('k', 'v'):
                        self.assertIs(cache.active[layer][name], bank['active'][name])
                        self.assertIs(cache.spare[layer][name], bank['spare'][name])
                # Written in place: the lent active banks hold the projected history and a zero tail.
                self.assert_history(cache, features)
                self.assertTrue(all(torch.count_nonzero(bank['spare'][name]).item() == 0
                                    for bank in storage for name in ('k', 'v')))
                # The padded sources were temporaries, freed with the scope; no bank was.
                self.assertTrue(freed(operations))
                self.assertFalse(any(any(released is value for value in banks) for released in freed(operations)))
                for prefix in (5, 32):
                    candidate = self.features(32, cache.position)
                    cache.commit(cache.prepare(candidate, prefix, position=cache.position))
                    features = torch.cat((features, candidate[..., :prefix, :]), dim=2)[..., -2048:, :]
                    self.assert_history(cache, features)
                cache.audit(features)
                # Two commits: the banks are back the way round they were lent.
                for layer, bank in enumerate(storage):
                    for name in ('k', 'v'):
                        self.assertIs(cache.active[layer][name], bank['active'][name])
                        self.assertIs(cache.spare[layer][name], bank['spare'][name])
                self.assertEqual(cache.owned, [cache.query])
            self.assertEqual(cache.borrowed, [])
            self.assertFalse(any(any(released is value for value in banks) for released in freed(operations)))
            self.assertTrue(any(released is cache.query for released in freed(operations)))

    def test_lent_banks_are_protected_from_the_temporaries(self):
        storage = pooled_storage()
        with self.fixture(self.features(170, 3), 170, storage=storage) as (cache, operations):
            operations.deallocate.reset_mock()
            with cache.temporaries([]) as retain:
                self.assertIs(retain(storage[1]['spare']['v']), storage[1]['spare']['v'])
                self.assertIs(retain(cache.query), cache.query)
                scratch = retain(torch.zeros(4))
            self.assertEqual(freed(operations), [scratch])

    def test_storage_geometry_is_checked_before_anything_is_uploaded(self):
        bad_shape, bad_dtype, missing, aliased = (pooled_storage() for _ in range(4))
        bad_shape[0]['active']['k'] = torch.zeros((1, 4, 2048, 64), dtype=torch.bfloat16)
        bad_dtype[1]['spare']['v'] = torch.zeros(KV_SHAPE)
        del missing[1]['spare']['v']
        aliased[1]['spare']['v'] = aliased[0]['active']['k']
        for storage in (pooled_storage()[:1], pooled_storage(3), bad_shape, bad_dtype, missing, aliased,
                        [dict(active=bank['active']) for bank in pooled_storage()], [['k', 'v'], ['k', 'v']]):
            operations = self.operations()
            operations.from_torch = Mock(side_effect=AssertionError('uploaded before the banks were checked'))
            with self.subTest(storage=storage), patch('draft_kv_history.addresses', side_effect=self.address), \
                    self.assertRaises(ValueError):
                DraftKVHistory(operations, object(), [dict(layer=layer) for layer in range(2)], self.features(170, 3),
                               position=170, history_rows=170, storage=storage)
            operations.from_torch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
