"""Host orchestration equivalence, not TT-NN numerical or hardware acceptance."""

from types import SimpleNamespace
import unittest

import torch

from dspark_prefill import FullHistoryCapture
from prefill_gdn_checkpoint import PrefillGDNCheckpoint
from prefill_prefix_boundary import checkpoint_boundary
from prefill_prefix_features import prefix_features
from prefill_prefix_lookup import PrefixIdentity, PrefixLookup
from prefill_prefix_resume import native_resume_scope
from prefill_prefix_controller import PrefixController
from prefill_prefix_residency import OfflinePrefixResidency


class HostOperations:
    bfloat16 = 'bf16'
    TILE_LAYOUT = 'tile'
    DRAM_MEMORY_CONFIG = 'dram'

    def __init__(self):
        self.next_address = 1

    def tensor(self, shape=(1, 1, 32, 32), values=(0, 0)):
        address = self.next_address
        self.next_address += 1
        return SimpleNamespace(shape=shape, dtype=self.bfloat16, layout=self.TILE_LAYOUT,
            values=values, freed=False, memory_config=lambda: self.DRAM_MEMORY_CONFIG,
            parts=tuple(SimpleNamespace(buffer_address=lambda chip=chip: address + chip * 100000)
                for chip in (0, 1)))

    def get_device_tensors(self, value):
        if value.freed:
            raise ValueError('Released host-fixture storage')
        return value.parts

    def copy(self, source, destination):
        self.get_device_tensors(source)
        self.get_device_tensors(destination)
        destination.values = source.values

    def clone(self, source, **kwargs):
        self.get_device_tensors(source)
        return self.tensor(source.shape, source.values)

    def deallocate(self, value):
        self.get_device_tensors(value)
        value.freed = True

    def synchronize_device(self, mesh):
        pass


class HostModel:
    def __init__(self, operations):
        self.operations, self.device = operations, 'mesh'
        self.num_devices, self._chunked_trace_id = 2, None
        self.states = [operations.tensor() for unused in range(288)]
        self.gdn = [SimpleNamespace(B=1, _stable_state=True, rec_state=self.states[index],
            conv_states=self.states[index + 1:index + 5], conv_carry=self.states[index + 5])
            for index in range(0, 288, 6)]
        self.layers = [SimpleNamespace(forward=self.layer_forward) for unused in range(64)]
        self._paged_kv_caches = [tuple(operations.tensor((104, 1, 64, 256))
            for unused in range(2)) for unused in range(16)]
        self.kv, self.starts = {}, []
        self.inactive = ('untouched', 42)

    def layer_forward(self, hidden):
        return hidden

    def _set_vision_merge(self, *args):
        pass

    def _prefill_chunked_eager_tp(self, *args, **kwargs):
        return self.cold(args[0])

    def _forward_prefill_chunk_masked_tp(self, tokens, valid_len, chunk_start, page_table, bucket, **kwargs):
        self.starts.append(chunk_start)
        total = int(tokens.sum())
        for index, value in enumerate(self.states):
            value.values = tuple(previous + total * (index + 1) for previous in value.values)
        self.kv[chunk_start] = tuple(tokens.flatten().tolist())
        hidden = self.operations.tensor((1, 1, bucket, 2560), self.states[0].values)
        for layer in self.layers:
            hidden = layer.forward(hidden)
        return hidden

    def _masked_bucket_logits_tp(self, hidden, valid_len, bucket):
        return hidden.values

    def cold(self, tokens):
        for value in self.states:
            value.values = (0, 0)
        self.kv.clear()
        for start in range(0, tokens.shape[1], 2048):
            hidden = self._forward_prefill_chunk_masked_tp(tokens[:, start:start + 2048],
                2048, start, None, 2048)
            logits = self._masked_bucket_logits_tp(hidden, 2048, 2048)
            self.operations.deallocate(hidden)
        return logits


def feature_values(capture):
    return tuple((chunk.start, chunk.rows, tuple(value.values for value in chunk.features))
        for chunk in capture.outputs())


class PrefixIntegrationTests(unittest.TestCase):
    def test_controller_cold_hit_miss_and_failed_residency(self):
        operations = HostOperations()
        model = HostModel(operations)
        checkpoint = PrefillGDNCheckpoint(operations, model.device, model.gdn,
            [operations.tensor() for unused in model.states])
        tokens = torch.arange(6144).reshape(1, -1)
        pages = torch.arange(96, dtype=torch.int32).reshape(1, -1)
        identity = PrefixIdentity('a' * 64, 'b' * 64, 'c' * 64, 'session', 0)
        residency = OfflinePrefixResidency(operations, model, identity, list(range(96)))
        controller = PrefixController(operations, model, checkpoint, residency)
        prefill = lambda values: model._prefill_chunked_eager_tp(values, pages, 6144, 3, 2048, 0)
        with controller.request(identity, tokens, pages, prefill,
                prefix_position=4096, inactive_pages=[100]) as (result, capture, evidence):
            self.assertFalse(evidence['cache_hit'])
            self.assertEqual(len(capture.outputs()), 3)
        self.assertEqual(len(controller.owner.outputs()), 2)
        self.assertEqual(controller.owner.position, 4096)
        changed = tokens.clone()
        changed[:, 4096:] += 7
        for value in model.states:
            value.values = (-333, -444)
        model.starts.clear()
        with controller.request(identity, changed, pages, prefill,
                prefix_position=4096, inactive_pages=[100]) as (actual, capture, evidence):
            self.assertTrue(evidence['cache_hit'])
            actual_features = feature_values(capture)
        self.assertEqual(model.starts, [4096])
        expected_model = HostModel(HostOperations())
        expected_capture = FullHistoryCapture(expected_model.operations, expected_model, 6144)
        with expected_capture.capture():
            expected = expected_model.cold(changed)
        self.assertEqual(actual, expected)
        self.assertEqual(actual_features, feature_values(expected_capture))
        self.assertEqual([value.values for value in model.states],
            [value.values for value in expected_model.states])
        self.assertEqual(model.kv, expected_model.kv)
        changed[0, 0] += 1
        model.starts.clear()
        with controller.request(identity, changed, pages, prefill,
                prefix_position=4096, inactive_pages=[100]) as (unused, capture, evidence):
            self.assertFalse(evidence['cache_hit'])
        self.assertEqual(model.starts, [0, 2048, 4096])
        residency.invalidate()
        with self.assertRaisesRegex(ValueError, 'reservation'):
            with controller.request(identity, changed, pages, prefill,
                    prefix_position=4096, inactive_pages=[100]):
                self.fail('Lost page lease admitted')
        self.assertIsNone(controller.owner)
        self.assertIsNone(controller.lookup.identity)
        expected_capture.close()

    def test_changed_suffix_matches_cold_state_features_and_output(self):
        operations = HostOperations()
        model = HostModel(operations)
        checkpoint = PrefillGDNCheckpoint(operations, model.device, model.gdn,
            [operations.tensor() for unused in model.states])
        tokens = torch.arange(6144).reshape(1, -1)
        owner = FullHistoryCapture(operations, model, 6144)
        with checkpoint_boundary(model, checkpoint, 4096) as boundary:
            with owner.capture():
                model.cold(tokens)
        self.assertTrue(boundary['complete'] and boundary['restored'])
        identity = PrefixIdentity('a' * 64, 'b' * 64, 'c' * 64, 'session', 0)
        pages = list(range(96))
        lookup = PrefixLookup()
        lookup.publish(identity, tokens.flatten().tolist(), 4096, pages, inactive_pages=[100])
        changed = tokens.clone()
        changed[:, 4096:] += 5
        for value in model.states:
            value.values = (-777, -888)
        position = lookup.match(identity, changed.flatten().tolist(), pages, inactive_pages=[100])
        self.assertEqual(position, 4096)
        model.starts.clear()
        with prefix_features(operations, model, 6144, owner, position) as capture:
            page_table = torch.tensor([pages], dtype=torch.int32)
            with native_resume_scope(operations, model, changed, page_table,
                    prefix_position=position, restore=checkpoint.restore) as native_route:
                with capture.capture():
                    actual = model._prefill_chunked_eager_tp(changed, page_table, 6144, 3, 2048, 0)
            self.assertTrue(native_route['completed'] and native_route['restored'])
            actual_features = feature_values(capture)
        self.assertEqual(model.starts, [4096])
        actual_state = [value.values for value in model.states]
        actual_kv = dict(model.kv)
        cold_capture = FullHistoryCapture(operations, model, 6144)
        with cold_capture.capture():
            expected = model.cold(changed)
        self.assertEqual(actual, expected)
        self.assertEqual(actual_state, [value.values for value in model.states])
        self.assertEqual(actual_kv, model.kv)
        self.assertEqual(actual_features, feature_values(cold_capture))
        self.assertEqual(model.inactive, ('untouched', 42))
        owner.close()
        cold_capture.close()


if __name__ == '__main__':
    unittest.main()
