from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import dspark_stable_history as stable
from dspark_history import TensorScope, leaves
import test_dspark_history as fixtures


class StableHistoryTests(unittest.TestCase):
    def test_captured_adapter_pads_features_and_restores_hook_on_failure(self):
        from dspark_publication_scope import CapturedPublicationArm

        cache = self.cache(position=32, capacity=128)
        cache.rotary = SimpleNamespace(tables=Mock(side_effect=lambda position, rows:
            (torch.full((1, 1, rows, 128), position, dtype=torch.bfloat16),
             torch.full((1, 1, rows, 128), -position, dtype=torch.bfloat16))))
        outputs = self.projected(start=32, rows=32)

        def project(features, tables):
            for value in features:
                self.assertEqual(tuple(value.shape), (1, 1, 32, 2560))
                self.assertEqual(torch.count_nonzero(value[..., 16:, :]), 0)
            self.assertTrue(torch.all(tables[0][..., :3, :] == 32))
            self.assertTrue(torch.all(tables[0][..., 3:, :] == 1))
            self.assertTrue(torch.all(tables[1][..., :3, :] == -32))
            self.assertEqual(torch.count_nonzero(tables[1][..., 3:, :]), 0)
            return outputs

        projection = SimpleNamespace(operations=self.operations, mesh=self.mesh, outputs=outputs,
            closed=False, project=Mock(side_effect=project), close=Mock())
        try:
            with patch('dspark_publication_scope.PreparedHistoryProjection', return_value=projection), \
                    patch('dspark_publication_scope.require_tensor', side_effect=fixtures.require_tensor), \
                    patch('dspark_captured_publication.require_tensor', side_effect=fixtures.require_tensor):
                arm = CapturedPublicationArm(cache, audit=True)
                with self.assertRaisesRegex(RuntimeError, 'request failed'):
                    with arm.install():
                        publication = cache.prepare_publication(fixtures.features(32, 3, 16), 3, position=32)
                        cache.discard_publication(publication)
                        raise RuntimeError('request failed')
                self.assertNotIn('prepare_publication', vars(cache))
                projection.close.assert_called_once()
                self.assertEqual(cache.position, 32)
                self.assertIsNone(cache.pending)
                self.assertEqual(cache.rotary.tables.call_args_list[-1].args, (32, 3))
        finally:
            cache.close()

    def test_captured_prefix_publication_preserves_transaction_and_borrowed_outputs(self):
        from dspark_captured_publication import prepare

        cache = self.cache(position=32, capacity=128)
        try:
            with patch('dspark_captured_publication.require_tensor', side_effect=fixtures.require_tensor):
                for prefix, commit in ((1, False), (15, True), (32, True)):
                    position = cache.position
                    outputs = self.projected(start=position, rows=32)
                    identities = {fixtures.identity(value) for value in leaves(outputs)}
                    projection = SimpleNamespace(operations=self.operations, mesh=self.mesh,
                        project=Mock(return_value=outputs))
                    previous = tuple(value.clone() for value in leaves(cache.layers))
                    publication = prepare(cache, projection, (), (), prefix, position=position)
                    self.assertEqual(cache.position, position)
                    for value, golden in zip(leaves(cache.layers), previous, strict=True):
                        self.assertTrue(torch.equal(value, golden))
                    with self.assertRaises(ValueError):
                        prepare(cache, projection, (), (), prefix, position=position)
                    self.assertEqual(projection.project.call_count, 1)
                    if commit:
                        cache.commit_publication(publication)
                    else:
                        cache.discard_publication(publication)
                    for value, golden in zip(leaves(cache.layers),
                            leaves(self.projected(start=0, rows=cache.position)), strict=True):
                        self.assertTrue(torch.equal(value[..., :cache.position, :], golden))
                        self.assertEqual(torch.count_nonzero(value[..., cache.position:, :]), 0)
                    self.assertTrue(identities.isdisjoint(self.operations.deallocated))
        finally:
            cache.close()

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.operations = fixtures.HostOperations()
        self.operations.copy = Mock(side_effect=lambda value, destination: destination.copy_(value))
        self.mesh = SimpleNamespace(shape=(1, 2))
        self.stack.enter_context(patch('dspark_history.addresses', side_effect=lambda operations, value: fixtures.identity(value)))
        self.stack.enter_context(patch.object(stable, 'require_tensor', side_effect=fixtures.require_tensor))
        self.project = self.stack.enter_context(patch('dspark_history.project_chunks', side_effect=self.projected))
        self.delta_project = self.stack.enter_context(patch.object(stable, 'project_chunks', side_effect=self.projected))

    def projected(self, *args, start, rows):
        return tuple(tuple((torch.arange(start, start + rows) % 193 + layer + operand * 20)
            .reshape(1, 1, rows, 1).expand(1, 4, rows, 128).bfloat16().clone()
            for operand in range(2)) for layer in range(5))

    def cache(self, position=4093, capacity=4384):
        return stable.StableHistoryKV(self.operations, self.mesh, None, {}, [], (), None,
            position=position, capacity=capacity)

    def test_both_banks_allocated_once_and_every_commit_keeps_the_same_addresses(self):
        cache = self.cache()
        bindings = {fixtures.identity(value) for value in leaves(cache.layers) + leaves(cache.spare_layers)}
        self.assertEqual(len(bindings), 20)
        for prefix in (3, 15, 1):
            old = tuple(value.clone() for value in leaves(cache.layers))
            position = cache.position
            delta = self.projected(start=position, rows=prefix)
            publication = cache.prepare_projected(delta, prefix, position=position)
            for previous, current in zip(old, leaves(cache.layers), strict=True):
                self.assertTrue(torch.equal(previous, current))
            cache.commit_publication(publication)
            for actual, expected in zip(leaves(cache.layers), leaves(self.projected(start=0, rows=cache.position)), strict=True):
                self.assertTrue(torch.equal(actual[..., :cache.position, :], expected))
                self.assertEqual(torch.count_nonzero(actual[..., cache.position:, :]), 0)
            self.assertEqual(bindings, {fixtures.identity(value) for value in leaves(cache.layers) + leaves(cache.spare_layers)})
            self.assertTrue(bindings.isdisjoint(self.operations.deallocated))
        self.assertEqual(self.project.call_count, 1)
        self.operations.to_torch.assert_not_called()
        cache.close()
        self.assertTrue(bindings.issubset(self.operations.deallocated))
        count = len(self.operations.deallocated)
        cache.close()
        self.assertEqual(len(self.operations.deallocated), count)

    def test_discard_reuses_spare_bank_without_advancing_or_freeing_committed_history(self):
        cache = self.cache()
        before = tuple(value.clone() for value in leaves(cache.layers))
        publication = cache.prepare_projected(self.projected(start=4093, rows=15), 15, position=4093)
        spare = publication.layers
        cache.discard_publication(publication)
        self.assertEqual(cache.position, 4093)
        self.assertIs(cache.spare_layers, spare)
        for value, expected in zip(leaves(cache.layers), before, strict=True):
            self.assertTrue(torch.equal(value, expected))
        next_publication = cache.prepare_projected(self.projected(start=4093, rows=3), 3, position=4093)
        self.assertIs(next_publication.layers, spare)
        cache.commit_publication(next_publication)
        cache.discard_publication(next_publication)
        self.assertEqual(cache.position, 4096)
        cache.close()

    def test_logical_proposal_views_exclude_all_fixed_capacity_padding(self):
        cache = self.cache()
        scope = TensorScope(self.operations, leaves(cache.layers))
        values = cache.logical_layers(scope.retain)
        self.assertTrue(all(tuple(value.shape) == (1, 4, 4093, 128) for value in leaves(values)))
        scope.release()
        self.assertTrue(all(fixtures.identity(value) not in self.operations.deallocated for value in leaves(cache.layers)))
        cache.close()

    def test_feature_publication_still_projects_only_the_verified_delta(self):
        cache = self.cache()
        features = fixtures.features(4093, 3, 16)
        publication = cache.prepare_publication(features, 3, position=4093)
        self.assertEqual(self.delta_project.call_args.kwargs, dict(start=4093, rows=3))
        self.assertEqual(publication.prefix, 3)
        self.assertTrue(all(tuple(value.shape) == (1, 4, 4384, 128) for value in leaves(publication.layers)))
        cache.close()

    def test_copy_failure_does_not_publish_a_partial_spare_bank(self):
        cache = self.cache()
        self.operations.copy.side_effect = RuntimeError('injected copy failure')
        with self.assertRaisesRegex(RuntimeError, 'copy failure'):
            cache.prepare_projected(self.projected(start=4093, rows=3), 3, position=4093)
        self.assertIsNone(cache.pending)
        self.assertEqual(cache.position, 4093)
        cache.close()

    def test_capacity_and_frontier_guards(self):
        for capacity in (True, 4092, 4385, 8224):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                self.cache(capacity=capacity)
        cache = self.cache(position=31, capacity=32)
        with self.assertRaises(ValueError):
            cache.prepare_projected(self.projected(start=31, rows=2), 2, position=31)
        publication = cache.prepare_projected(self.projected(start=31, rows=1), 1, position=31)
        cache.commit_publication(publication)
        self.assertIs(cache.logical_layers(lambda value: value)[0][0], cache.layers[0][0])
        cache.close()


if __name__ == '__main__':
    unittest.main()
