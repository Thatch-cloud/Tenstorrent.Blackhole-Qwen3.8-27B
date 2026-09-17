from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import test_draft_attention as attention_fixture
import test_draft_attention_branch as branch_fixture
from dflash_device import DFlashDevice
from dflash_proposal_inputs import proposal_inputs
from dflash_proposal_trace import PreparedDFlashProposal
from draft_attention_branch import prepare_attention_branch, execute_attention_branch
from draft_live_attention import live_attention


class LiveAttentionTests(unittest.TestCase):
    def test_trace_owns_every_intermediate_and_preserves_arithmetic(self):
        operations, query, key, value, mask = attention_fixture.DraftAttentionTests().operations_fixture()
        owned = []
        def kernel(mesh, *arguments, **options):
            output = object()
            arguments[-1].append(output)
            return output
        with patch('draft_live_attention.live_qk', side_effect=kernel) as qk, \
                patch('draft_live_attention.row_sum', side_effect=kernel) as row_sum, \
                patch('draft_live_attention.fused_dot', side_effect=kernel) as pv:
            output = live_attention(operations, object(), query, key, value, mask,
                trace_owned=owned, mask_validated=True)
        self.assertEqual(len(owned), 17)
        self.assertIs(owned[-1], output)
        self.assertEqual(qk.call_count, 1)
        self.assertEqual(row_sum.call_count, 1)
        self.assertEqual(pv.call_args.kwargs, dict(cache_tiles=True))
        self.assertEqual(operations.typecast.call_count, 4)
        self.assertTrue(all(call.args[1] == 'fp32' for call in operations.typecast.call_args_list))
        self.assertEqual(operations.multiply.call_args_list[0].args[1], 128 ** -.5)
        self.assertEqual(operations.exp.call_args.kwargs, dict(fast_and_approximate_mode=False))
        operations.softmax.assert_not_called()
        operations.synchronize_device.assert_not_called()
        operations.deallocate.assert_not_called()
        for borrowed in (query, key, value, mask):
            self.assertFalse(any(borrowed is tensor for tensor in owned))

    def test_failure_transfers_allocations_to_trace_owner(self):
        operations, query, key, value, mask = attention_fixture.DraftAttentionTests().operations_fixture()
        operations.exp.side_effect = RuntimeError('exp failed')
        owned = []
        def qk(mesh, query, keys, local):
            local.append(object())
            return local[-1]
        with patch('draft_live_attention.live_qk', side_effect=qk), self.assertRaisesRegex(RuntimeError, 'exp failed'):
            live_attention(operations, object(), query, key, value, mask, trace_owned=owned, mask_validated=True)
        self.assertEqual(len(owned), 11)
        operations.deallocate.assert_not_called()
        operations.synchronize_device.assert_not_called()

    def test_missing_validation_or_ownership_fails_before_dispatch(self):
        operations, query, key, value, mask = attention_fixture.DraftAttentionTests().operations_fixture()
        for validated, owned in ((False, []), (1, []), (True, None), (True, ())):
            with self.assertRaises(ValueError):
                live_attention(operations, object(), query, key, value, mask,
                    trace_owned=owned, mask_validated=validated)
        operations.repeat_interleave.assert_not_called()

    def test_branch_routes_only_explicit_validated_t8(self):
        operations, weights, convolution, hidden, history, mask, rope = branch_fixture.DraftAttentionBranchTests().fixture()
        mesh = object()
        for options in (dict(live_query_qk=1), dict(live_query_qk=True, block_rows=32),
                dict(live_query_qk=True, precise_native=True)):
            with self.assertRaises(ValueError):
                prepare_attention_branch(operations, mesh, weights, convolution, lambda value: value, **options)
        operations.from_torch.assert_not_called()
        parameters = prepare_attention_branch(operations, mesh, weights, convolution, lambda value: value, live_query_qk=True)
        for options in ({}, dict(live_query_mask_validated=1),
                dict(live_query_mask_validated=True, wide_dot_placement=True)):
            with self.assertRaisesRegex(ValueError, 'Live-query'):
                execute_attention_branch(operations, mesh, object(), hidden, history, mask, rope, lambda value: value,
                    parameters=parameters, context=31, **options)
        operations.matmul.assert_not_called()
        with patch('draft_attention_branch.grouped_causal_convolution', return_value=object()), \
                patch('draft_attention_branch.gather_add_projection', return_value=object()), \
                patch('draft_attention_branch.composed_draft_attention') as control, \
                patch('draft_live_attention.live_attention', return_value=object()) as candidate:
            execute_attention_branch(operations, mesh, object(), hidden, history, mask, rope, lambda value: value,
                parameters=parameters, context=31, live_query_mask_validated=True)
        control.assert_not_called()
        self.assertIs(candidate.call_args.kwargs['mask_validated'], True)

    def test_device_rejects_unregistered_mask_before_embedding(self):
        device = DFlashDevice.__new__(DFlashDevice)
        device.operations = SimpleNamespace(embedding=Mock())
        device.live_query_qk, device.validated_live_masks = True, set()
        with patch('dflash_device.addresses', return_value=(11, 12)), self.assertRaisesRegex(ValueError, 'mask'):
            device.execute_proposal(None, None, None, None, context=256, owned=[], retain=lambda value: value, stage=Mock())
        device.operations.embedding.assert_not_called()

    def test_update_revalidates_masks_and_revokes_failed_uploads(self):
        for failure in (None, 'mask', 'upload', 'sync'):
            with self.subTest(failure=failure):
                host = proposal_inputs(17, 170, 170, 8, 256)
                operations = SimpleNamespace(ReplicateTensorToMesh=lambda mesh: mesh,
                    from_torch=lambda value, **kwargs: value.clone(),
                    copy_host_to_device_tensor=Mock(side_effect=lambda source, destination: destination.copy_(source)),
                    slice=Mock(return_value=object()), pad=Mock(return_value=object()), copy=Mock(),
                    synchronize_device=Mock())
                device = SimpleNamespace(live_query_qk=True, validated_live_masks={(11, 12)},
                    position=170, history_rows=170, block_rows=8, progress=None, history=object(), spare_history=object(),
                    temporaries=lambda protected: ([], lambda value: value))
                bucket = SimpleNamespace(context=256, identifiers=host['identifiers'].clone(), mask=host['mask'].clone(),
                    history=object(), rope={name: tuple(value.clone() for value in host['rope'][name]) for name in ('q', 'k')},
                    inputs=[], addresses=[])
                prepared = PreparedDFlashProposal.__new__(PreparedDFlashProposal)
                prepared.operations, prepared.device, prepared.mesh = operations, device, object()
                prepared.kv_history, prepared.owned = None, []
                if failure == 'mask':
                    host['mask'][..., 8, :2] = 0
                if failure == 'upload':
                    operations.copy_host_to_device_tensor.side_effect = RuntimeError('upload failed')
                if failure == 'sync':
                    operations.synchronize_device.side_effect = RuntimeError('sync failed')
                with patch('dflash_proposal_trace.proposal_inputs', return_value=host), \
                        patch('dflash_proposal_trace.addresses', return_value=(11, 12)), \
                        patch('dflash_proposal_trace.release_owned'):
                    if failure:
                        with self.assertRaises((ValueError, RuntimeError)):
                            prepared.update(bucket, 17)
                        self.assertEqual(device.validated_live_masks, set())
                    else:
                        prepared.update(bucket, 17)
                        self.assertEqual(device.validated_live_masks, {(11, 12)})
                        self.assertTrue(torch.equal(bucket.mask, host['mask']))
                if failure == 'mask':
                    operations.copy_host_to_device_tensor.assert_not_called()
