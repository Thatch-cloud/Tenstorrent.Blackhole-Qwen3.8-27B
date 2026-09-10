from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

import dspark_device as device


class DSparkDeviceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.operations = MagicMock()
        self.operations.bfloat16, self.operations.float32, self.operations.uint32 = torch.bfloat16, torch.float32, torch.int64
        self.operations.from_torch.side_effect = lambda value, **kwargs: value.clone()
        self.operations.to_torch.side_effect = lambda value: value
        self.mesh = SimpleNamespace(shape=(1, 2))
        self.target = SimpleNamespace(num_devices=2, vocab_size=248320, _lmhead_vocab_sharded=True,
            mesh_device=self.mesh, lm_head_weight=object())
        self.parameters = {name: object() for name in ('fc.weight', 'hidden_norm.weight', 'norm.weight')}
        self.weights = [{str(index): object()} for index in range(5)]
        self.history = SimpleNamespace(position=4096, pending=None,
            layers=tuple((object(), object()) for index in range(5)), prepare_publication=Mock(),
            commit_publication=Mock(), discard_publication=Mock(), close=Mock())
        self.history_factory = self.stack.enter_context(patch.object(device, 'FullHistoryKV', return_value=self.history))
        self.scope = SimpleNamespace(retain=lambda value: value, release=Mock())
        self.scope_factory = self.stack.enter_context(patch.object(device, 'TensorScope', return_value=self.scope))
        self.hidden = [object() for index in range(6)]
        self.embed = self.stack.enter_context(patch.object(device, 'noise_embeddings', return_value=self.hidden[0]))
        self.layer = self.stack.enter_context(patch.object(device, 'layer', side_effect=[dict(finish={'output': value}) for value in self.hidden[1:]]))
        self.norm = self.stack.enter_context(patch.object(device, 'norm', return_value=object()))
        self.head = self.stack.enter_context(patch.object(device, 'shared_head_logits', return_value=object()))
        self.markov = self.stack.enter_context(patch.object(device, 'markov', return_value=[dict(token=object()) for index in range(15)]))
        self.tokens = torch.arange(100, 115).reshape(1, 1, 15, 1)
        self.pack = self.stack.enter_context(patch.object(device, 'pack_tokens', return_value=self.tokens))
        self.operations.get_device_tensors.return_value = [self.tokens, self.tokens.clone()]
        self.rotary = SimpleNamespace(tables=Mock(return_value=(torch.ones(1, 1, 15, 128), torch.zeros(1, 1, 15, 128))))
        self.drafter = device.DSparkDevice(self.operations, self.target, object(), self.parameters, self.weights,
            object(), object(), (object(),), self.rotary, position=4096)

    def test_fixed_history_uses_temporary_logical_views_not_capacity_padding_for_attention(self):
        views = tuple((object(), object()) for index in range(5))
        self.history.logical_layers = Mock(return_value=views)
        with patch('dspark_stable_history.StableHistoryKV', return_value=self.history) as factory:
            drafter = device.DSparkDevice(self.operations, self.target, object(), self.parameters, self.weights,
                object(), object(), (object(),), self.rotary, position=4096, history_capacity=4384)
        self.assertEqual(factory.call_args.kwargs, dict(position=4096, capacity=4384))
        self.assertEqual(drafter.propose(777, 15), tuple(range(100, 115)))
        self.history.logical_layers.assert_called_once_with(self.scope.retain)
        for index, call in enumerate(self.layer.call_args_list):
            self.assertIs(call.args[4], views[index])
        self.scope.release.assert_called_once()

    def test_actual_target_boundaries_and_five_cached_layers_feed_all_markov_rows(self):
        self.assertEqual(self.drafter.propose(777, 15), tuple(range(100, 115)))
        self.assertEqual(self.drafter.position, 4096)
        self.assertEqual(self.history_factory.call_count, 1)
        self.assertEqual(self.layer.call_count, 5)
        for index, call in enumerate(self.layer.call_args_list):
            self.assertIs(call.args[3], self.hidden[index])
            self.assertIs(call.args[4], self.history.layers[index])
            self.assertIs(call.args[5], self.weights[index])
            self.assertEqual(call.kwargs, dict(position=4096, proposals=15, mask_validated=True))
            self.assertEqual(call.args[8].dtype, torch.float32)
            self.assertEqual(int(call.args[8].sum()), 15)
        self.assertIs(self.norm.call_args.args[1], self.hidden[-1])
        self.assertIs(self.head.call_args.args[4], self.norm.return_value)
        self.assertIs(self.markov.call_args.args[2], self.head.return_value)
        self.assertIs(self.pack.call_args.args[1], self.markov.return_value)
        self.rotary.tables.assert_called_once_with(4096, 15)
        self.assertEqual(self.operations.to_torch.call_count, 2)
        self.assertTrue(all(torch.equal(call.args[0], self.tokens) for call in self.operations.to_torch.call_args_list))
        self.scope.release.assert_called_once()
        self.history.prepare_publication.assert_not_called()

    def test_request_tail_takes_first_proposals_without_dropping_query_row_zero(self):
        self.assertEqual(self.drafter.propose(777, 3), (100, 101, 102))
        self.assertEqual(self.layer.call_count, 5)
        self.assertEqual(self.embed.call_args.kwargs['proposals'], 15)

    def test_divergent_replicas_fail_without_advancing_history_and_release_temporaries(self):
        self.operations.get_device_tensors.return_value[1][..., 0, 0] = 0
        with self.assertRaisesRegex(AssertionError, 'divergent'):
            self.drafter.propose(777, 15)
        self.assertEqual(self.drafter.position, 4096)
        self.scope.release.assert_called_once()

    def test_pending_cache_or_invalid_count_rejects_before_any_proposal_operation(self):
        for count in (0, 16, True):
            with self.assertRaises(ValueError):
                self.drafter.propose(777, count)
        self.history.pending = object()
        with self.assertRaises(ValueError):
            self.drafter.propose(777, 15)
        self.embed.assert_not_called()
        self.scope_factory.assert_not_called()

    def test_publication_is_delegated_and_close_does_not_release_borrowed_target_weights(self):
        features, publication = object(), object()
        self.drafter.prepare_publication(features, 16, position=4096)
        self.history.prepare_publication.assert_called_once_with(features, 16, position=4096)
        self.drafter.commit_publication(publication)
        self.drafter.discard_publication(publication)
        self.history.commit_publication.assert_called_once_with(publication)
        self.history.discard_publication.assert_called_once_with(publication)
        self.drafter.close()
        self.drafter.close()
        self.history.close.assert_called_once()
        self.operations.deallocate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
