from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import dspark_history as history
from dspark_intake import TAPS
from dspark_prefill import FeatureChunk, FullHistoryCapture, validate_chunks


def identity(value):
    return (value.untyped_storage().data_ptr(),) * 2


def require_tensor(operations, value, shape, dtype):
    if tuple(value.shape) != shape or value.dtype != dtype:
        raise ValueError('Unexpected test tensor geometry or precision')


class HostOperations:
    bfloat16 = torch.bfloat16
    TILE_LAYOUT = 'tile'
    DRAM_MEMORY_CONFIG = 'dram'

    def __init__(self):
        self.deallocated = []
        self.concat_widths = []
        self.to_torch = Mock(side_effect=AssertionError('No hidden-state readback allowed'))

    def slice(self, value, start, end):
        return value[tuple(slice(first, last) for first, last in zip(start, end, strict=True))].clone()

    def clone(self, value, **kwargs):
        return value.clone()

    def concat(self, values, dim, **kwargs):
        self.concat_widths.append(len(values))
        if len(values) > 8:
            raise AssertionError('Native concat fan-in exceeded')
        return torch.cat(values, dim=dim)

    def pad(self, value, padding, fill):
        return torch.nn.functional.pad(value, tuple(amount for pair in reversed(padding) for amount in pair), value=fill)

    def from_torch(self, value, **kwargs):
        return value.clone()

    def ReplicateTensorToMesh(self, mesh):
        return mesh

    def synchronize_device(self, mesh):
        pass

    def deallocate(self, value):
        self.deallocated.append(identity(value))


def features(start, rows, bucket):
    result = []
    for index in range(len(TAPS)):
        value = torch.full((1, 1, bucket, 2560), -8192, dtype=torch.bfloat16)
        value[..., :rows, :] = ((torch.arange(start, start + rows) % 193) + index).reshape(1, 1, rows, 1)
        result.append(value)
    return tuple(result)


def chunks(position):
    return tuple(FeatureChunk(start, min(2048, position - start), features(start,
        min(2048, position - start), ((min(2048, position - start) + 31) // 32) * 32))
        for start in range(0, position, 2048))


class FullHistoryCaptureTests(unittest.TestCase):
    def model(self):
        model = SimpleNamespace(layers=[SimpleNamespace(forward=lambda value: value) for index in range(62)])

        def chunk(token_buf, valid_len, chunk_start, page_table, bucket):
            for layer in model.layers:
                token_buf = layer.forward(token_buf)
            return token_buf

        model._forward_prefill_chunk_masked_tp = chunk
        return model

    def test_all_4k_rows_survive_native_buffer_reuse_and_partial_chunk_padding(self):
        for position in (1, 2048, 2049, 4093, 4096, 8192):
            operations, model = HostOperations(), self.model()
            capture = FullHistoryCapture(operations, model, position)
            original = model._forward_prefill_chunk_masked_tp
            expected = []
            with patch('dspark_prefill.addresses', side_effect=lambda operations, value: identity(value)), \
                    patch('dspark_projection.require_tensor', side_effect=require_tensor):
                with capture.capture():
                    for piece in chunks(position):
                        source = piece.features[0]
                        expected.append(source.clone())
                        model._forward_prefill_chunk_masked_tp(source, piece.rows, piece.start, None, source.shape[2])
                        source.zero_()
                output = capture.outputs()
                validate_chunks(output, start=0, rows=position)
                self.assertEqual(sum(piece.rows for piece in output), position)
                self.assertEqual(output[0].start, 0)
                for piece, golden in zip(output, expected, strict=True):
                    self.assertEqual(len({identity(value) for value in piece.features}), 5)
                    for value in piece.features:
                        self.assertTrue(torch.equal(value, golden))
                with self.assertRaises(ValueError):
                    with capture.capture():
                        pass
                capture.close()
                self.assertEqual(len(operations.deallocated), 5 * len(expected))
                capture.close()
                self.assertEqual(len(operations.deallocated), 5 * len(expected))
            self.assertIs(model._forward_prefill_chunk_masked_tp, original)
            self.assertFalse(hasattr(model, '_qwen_dspark_prefill_capture'))
            operations.to_torch.assert_not_called()

    def test_incomplete_repeated_out_of_order_and_overrun_chunks_fail_and_restore_hooks(self):
        for starts in ((), (0,), (2048,), (0, 0), (0, 2048, 4096)):
            operations, model = HostOperations(), self.model()
            original = model._forward_prefill_chunk_masked_tp
            capture = FullHistoryCapture(operations, model, 4096)
            with patch('dspark_prefill.addresses', side_effect=lambda operations, value: identity(value)), \
                    patch('dspark_projection.require_tensor', side_effect=require_tensor), self.assertRaises(ValueError):
                with capture.capture():
                    for start in starts:
                        model._forward_prefill_chunk_masked_tp(features(start, 2048, 2048)[0], 2048, start, None, 2048)
            self.assertTrue(capture.closed)
            self.assertIs(model._forward_prefill_chunk_masked_tp, original)
            self.assertFalse(hasattr(model, '_qwen_dspark_prefill_capture'))
            with self.assertRaises(ValueError):
                capture.outputs()

    def test_other_capture_owners_and_invalid_frontiers_reject_before_mutation(self):
        for marker in ('_qwen_dspark_prefill_capture', '_qwen_dflash_prefill_capture', '_qwen_target_feature_capture'):
            model = self.model()
            setattr(model, marker, object())
            capture = FullHistoryCapture(HostOperations(), model, 4096)
            with self.assertRaises(ValueError):
                with capture.capture():
                    pass
            self.assertFalse(capture.started)
        for position in (True, 0, -1, 8193, 1.0):
            with self.assertRaises(ValueError):
                FullHistoryCapture(HostOperations(), self.model(), position)

    def test_history_contract_rejects_truncation_padding_as_history_and_missing_taps(self):
        complete = chunks(4093)
        invalid = ((complete[1],), (complete[0],),
            (complete[0], FeatureChunk(2048, 2048, complete[1].features)),
            (FeatureChunk(0, 2048, complete[0].features[:-1]), complete[1]))
        for candidate in invalid:
            with self.assertRaises(ValueError):
                validate_chunks(candidate, start=0, rows=4093)


class FullHistoryKVTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.operations = HostOperations()
        self.mesh = SimpleNamespace(shape=(1, 2))
        self.parameters = {name: torch.zeros(1) for name in ('fc.weight', 'hidden_norm.weight')}
        self.weights = [{name: torch.zeros(1) for name in history.SPECIFICATIONS} for layer in range(5)]
        self.calls, self.table_calls = [], []
        self.rotary = SimpleNamespace(tables=self.tables)
        self.stack.enter_context(patch.object(history, 'addresses', side_effect=lambda operations, value: identity(value)))
        self.stack.enter_context(patch.object(history, 'require_tensor', side_effect=require_tensor))
        self.stack.enter_context(patch.object(history, 'project_block', side_effect=self.project))

    def tables(self, start, rows):
        self.table_calls.append((start, rows))
        value = torch.arange(start, start + rows).reshape(1, 1, rows, 1).expand(1, 1, rows, 128).bfloat16()
        return value, -value

    def project(self, operations, mesh, collectives, values, parameters, weights, tables, retain):
        self.assertTrue(all(tuple(value.shape) == (1, 1, 32, 2560) for value in values))
        self.assertTrue(all(tuple(table.shape) == (1, 1, 32, 128) for table in tables))
        self.calls.append(tuple(value[..., :1].clone() for value in values))
        return tuple(tuple(retain((values[layer][..., :128] + offset).expand(1, 4, 32, 128).contiguous())
            for offset in (0, 20)) for layer in range(5))

    def cache(self, position):
        return history.FullHistoryKV(self.operations, self.mesh, None, self.parameters, self.weights,
            chunks(position), self.rotary, position=position)

    def test_prefill_projects_each_32_rows_once_and_retains_complete_history_without_padding(self):
        cache = self.cache(4093)
        self.assertEqual(len(self.calls), 128)
        self.assertEqual(self.table_calls, [(start, min(32, 4093 - start)) for start in range(0, 4093, 32)])
        self.assertLessEqual(max(self.operations.concat_widths), 8)
        expected = features(0, 4093, 4096)
        for layer, pair in enumerate(cache.layers):
            for index, value in enumerate(pair):
                self.assertEqual(tuple(value.shape), (1, 4, 4093, 128))
                golden = (expected[layer][..., :4093, :128] + index * 20).expand(1, 4, 4093, 128)
                self.assertTrue(torch.equal(value, golden))
                self.assertNotIn(identity(value), self.operations.deallocated)
        cache.close()
        count = len(self.operations.deallocated)
        cache.close()
        self.assertEqual(len(self.operations.deallocated), count)
        self.operations.to_torch.assert_not_called()

    def test_prepare_discard_and_commit_project_only_valid_committed_delta(self):
        cache = self.cache(4093)
        old = cache.layers
        before = tuple(value.clone() for value in history.leaves(old))
        proposal = features(4093, 15, 32)
        pending = cache.prepare_publication(proposal, 15, position=4093)
        self.assertEqual(len(self.calls), 129)
        self.assertEqual(self.table_calls[-1], (4093, 15))
        self.assertIs(cache.layers, old)
        self.assertEqual(cache.position, 4093)
        for original, saved in zip(history.leaves(old), before, strict=True):
            self.assertTrue(torch.equal(original, saved))
            self.assertNotIn(identity(original), self.operations.deallocated)
        for layer, pair in enumerate(pending.layers):
            for index, value in enumerate(pair):
                self.assertTrue(torch.equal(value[..., :4093, :], old[layer][index]))
                expected = (proposal[layer][..., :15, :128] + index * 20).expand(1, 4, 15, 128)
                self.assertTrue(torch.equal(value[..., 4093:, :], expected))
        with self.assertRaises(ValueError):
            cache.prepare_publication(proposal, 15, position=4093)
        cache.discard_publication(pending)
        self.assertIs(cache.layers, old)
        self.assertEqual(cache.position, 4093)
        with self.assertRaises(ValueError):
            cache.commit_publication(pending)
        pending = cache.prepare_publication(proposal, 3, position=4093)
        cache.commit_publication(pending)
        self.assertEqual(cache.position, 4096)
        self.assertEqual(len(self.calls), 130)
        self.assertTrue(all(value.shape[2] == 4096 for value in history.leaves(cache.layers)))
        for value in history.leaves(old):
            self.assertIn(identity(value), self.operations.deallocated)
        with self.assertRaises(ValueError):
            cache.commit_publication(pending)
        cache.close()

    def test_ragged_verifier_features_pad_without_publishing_unaccepted_suffix(self):
        cache = self.cache(32)
        for width, prefix in ((16, 1), (8, 7), (32, 16)):
            start = cache.position
            values = features(start, width, width)
            pending = cache.prepare_publication(values, prefix, position=start)
            self.assertEqual(self.table_calls[-1], (start, prefix))
            cache.commit_publication(pending)
            self.assertEqual(cache.position, start + prefix)
            for layer, pair in enumerate(cache.layers):
                for index, value in enumerate(pair):
                    expected = (values[layer][..., :prefix, :128] + index * 20).expand(1, 4, prefix, 128)
                    self.assertTrue(torch.equal(value[..., start:, :], expected))
        cache.close()

    def test_bad_publication_bounds_do_not_run_projection_or_change_history(self):
        cache = self.cache(32)
        original = cache.layers
        for prefix, position in ((0, 32), (33, 32), (True, 32), (1, 31), (1, True)):
            with self.assertRaises(ValueError):
                cache.prepare_publication(features(32, 16, 32), prefix, position=position)
        self.assertEqual(len(self.calls), 1)
        self.assertIs(cache.layers, original)
        cache.close()

    def test_projection_failure_leaves_current_cache_and_no_pending_publication(self):
        cache = self.cache(32)
        original = cache.layers
        with patch.object(history, 'project_block', side_effect=RuntimeError('injected failure')), self.assertRaises(RuntimeError):
            cache.prepare_publication(features(32, 15, 32), 15, position=32)
        self.assertIs(cache.layers, original)
        self.assertEqual(cache.position, 32)
        self.assertIsNone(cache.pending)
        for value in history.leaves(original):
            self.assertNotIn(identity(value), self.operations.deallocated)
        cache.close()

    def test_scope_protects_aliases_and_releases_remaining_tensors_after_one_release_error(self):
        borrowed, first, second = torch.zeros(1), torch.ones(1), torch.ones(2)
        scope = history.TensorScope(self.operations, (borrowed,))
        scope.retain(borrowed.view(1, 1))
        scope.retain(first)
        scope.retain(first.view(1, 1))
        scope.retain(second)
        with patch.object(self.operations, 'deallocate', side_effect=[RuntimeError('injected'), None]) as release:
            with self.assertRaises(RuntimeError):
                scope.release()
            self.assertEqual(release.call_count, 2)
            self.assertEqual({identity(call.args[0]) for call in release.call_args_list}, {identity(first), identity(second)})
        self.assertEqual(scope.owned, {})


if __name__ == '__main__':
    unittest.main()
