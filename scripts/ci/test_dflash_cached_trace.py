from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import Mock, patch

import torch

from dflash_device import DFlashDevice
from dflash_proposal_inputs import proposal_inputs
from dflash_proposal_trace import PreparedDFlashProposal


class CachedProposalTests(unittest.TestCase):
    def test_update_follows_committed_bank_swaps_without_copying_feature_history(self):
        self.exercise_update(8, False)

    def test_t16_native_update_validates_all_live_rows_and_bank_swaps(self):
        self.exercise_update(16, True)

    def exercise_update(self, block_rows, native):
        operations = SimpleNamespace(ReplicateTensorToMesh=lambda mesh: mesh,
            from_torch=lambda value, **kwargs: value.clone(),
            copy_host_to_device_tensor=Mock(side_effect=lambda source, destination: destination.copy_(source)),
            slice=lambda value, start, end: value[tuple(slice(first, last) for first, last in zip(start, end, strict=True))],
            copy=Mock(side_effect=lambda source, destination: destination.copy_(source)), synchronize_device=Mock())
        active = [dict(k=torch.ones((1, 4, 2048, 128), dtype=torch.bfloat16),
            v=torch.full((1, 4, 2048, 128), 3, dtype=torch.bfloat16))]
        cache = SimpleNamespace(position=4093, history_rows=2048, active=active, pending=None,
            owned=list(active[0].values()), borrowed=[])
        device = SimpleNamespace(operations=operations, mesh=object(), position=4093, history_rows=2048, block_rows=block_rows,
            native_proposal_attention=native, validated_native_proposal_masks=set(),
            history=torch.zeros((1, 1, 2048, 5120), dtype=torch.bfloat16),
            spare_history=torch.zeros((1, 1, 2048, 5120), dtype=torch.bfloat16), progress=None)
        device.temporaries = MethodType(DFlashDevice.temporaries, device)
        host = proposal_inputs(17, 4093, 2048, block_rows, 2048)
        bucket = SimpleNamespace(context=2048, identifiers=host['identifiers'].clone(), mask=host['mask'].clone(),
            rope={name: tuple(value.clone() for value in host['rope'][name]) for name in ('q', 'k')},
            history=torch.full((1, 1, 2080, 5120), 9, dtype=torch.bfloat16),
            cached_history=[{name: torch.zeros_like(value) for name, value in active[0].items()}])
        bucket.inputs = [bucket.identifiers, bucket.history, bucket.mask, *bucket.rope['q'], *bucket.rope['k'],
            *bucket.cached_history[0].values()]
        address = lambda operations, value: (value.untyped_storage().data_ptr(), value.untyped_storage().data_ptr() + 1)
        bucket.addresses = [address(operations, value) for value in bucket.inputs]
        prepared = PreparedDFlashProposal.__new__(PreparedDFlashProposal)
        prepared.operations, prepared.device, prepared.mesh = operations, device, device.mesh
        prepared.kv_history, prepared.owned = cache, bucket.inputs
        with patch('dflash_proposal_trace.addresses', side_effect=address), \
                patch('dflash_t16_native_scope.require_active', return_value={}) as admission, \
                patch('dflash_device.addresses', side_effect=address), patch('dflash_proposal_trace.release_owned'):
            for position, scalar in ((4093, 5), (4100, 7)):
                device.position = cache.position = position
                cache.active = [{name: torch.full_like(value, scalar) for name, value in active[0].items()}]
                cache.owned = list(cache.active[0].values())
                prepared.update(bucket, position % 17)
                for name in ('k', 'v'):
                    self.assertTrue(torch.equal(bucket.cached_history[0][name], cache.active[0][name]))
                self.assertTrue(torch.all(bucket.history == 9))
                self.assertEqual(bucket.addresses, [address(operations, value) for value in bucket.inputs])
                if native:
                    self.assertIn(address(operations, bucket.mask), device.validated_native_proposal_masks)
            self.assertEqual(admission.call_count, 2 if native else 0)
            if native:
                poisoned = proposal_inputs(17, device.position, 2048, block_rows, 2048)
                poisoned['mask'][..., 15, :] = float('-inf')
                operations.copy_host_to_device_tensor.reset_mock()
                with patch('dflash_proposal_trace.proposal_inputs', return_value=poisoned):
                    with self.assertRaisesRegex(ValueError, 'live T16'):
                        prepared.update(bucket, 17)
                operations.copy_host_to_device_tensor.assert_not_called()
                self.assertNotIn(address(operations, bucket.mask), device.validated_native_proposal_masks)
            operations.copy_host_to_device_tensor.reset_mock()
            cache.pending = object()
            with self.assertRaisesRegex(ValueError, 'fully committed'):
                prepared.update(bucket, 31)
            operations.copy_host_to_device_tensor.assert_not_called()

    def test_audit_requires_uncached_eager_cached_eager_and_trace_to_match(self):
        for corruption in (None, 'uncached', 'trace'):
            prepared = PreparedDFlashProposal.__new__(PreparedDFlashProposal)
            operations = SimpleNamespace(synchronize_device=Mock(), execute_trace=Mock())
            bucket = SimpleNamespace(context=256, trace=17, outputs='trace')
            values = {name: (torch.tensor([2.0]),) for name in ('uncached', 'cached', 'trace')}
            if corruption:
                values[corruption] = (torch.tensor([3.0]),)
            device = SimpleNamespace(history_rows=170, progress=lambda **kwargs: None, position=170, owned=[],
                temporaries=lambda protected: ([], lambda value: value), proposal_snapshot=lambda output: values[output],
                select_proposal=lambda output, seed, count: tuple(range(count)))
            prepared.operations, prepared.device, prepared.mesh = operations, device, object()
            prepared.closed, prepared.owned, prepared.checks, prepared.cache_checks = False, [], [], []
            prepared.kv_history, prepared.buckets = object(), {256: bucket}
            prepared.update = Mock()
            prepared.execute = Mock(side_effect=lambda bucket, owned, retain, **kwargs:
                'uncached' if kwargs.get('use_cache') is False else 'cached')
            with patch('dflash_proposal_trace.release_owned'):
                if corruption:
                    with self.assertRaises(AssertionError):
                        prepared.propose(17, 7)
                else:
                    self.assertEqual(prepared.propose(17, 7), tuple(range(7)))
                    self.assertEqual(prepared.checks, prepared.cache_checks)
                    self.assertEqual(len(prepared.checks), 1)
            if corruption == 'uncached':
                operations.execute_trace.assert_not_called()

    def build_pooled_case(self, context, scalar=5, spare_scalar=11):
        """One request's DraftKVHistory-shaped fixture over a POOLED bank: `active` and
        `spare` are the lent tensors themselves (draft_kv_history.py, storage=), and
        `borrowed` lists every one of them regardless of which list currently calls it
        active - exactly as DraftKVHistory.__init__ builds it, and unlike `owned`,
        which is empty whenever the pool lends the banks."""
        operations = SimpleNamespace(ReplicateTensorToMesh=lambda mesh: mesh,
            from_torch=lambda value, **kwargs: value.clone(),
            copy_host_to_device_tensor=Mock(side_effect=lambda source, destination: destination.copy_(source)),
            slice=lambda value, start, end: value[tuple(slice(first, last) for first, last in zip(start, end, strict=True))],
            copy=Mock(side_effect=lambda source, destination: destination.copy_(source)),
            synchronize_device=Mock(), deallocate=Mock())
        active = [dict(k=torch.full((1, 4, 2048, 128), scalar, dtype=torch.bfloat16),
            v=torch.full((1, 4, 2048, 128), scalar + 1, dtype=torch.bfloat16))]
        spare = [dict(k=torch.full((1, 4, 2048, 128), spare_scalar, dtype=torch.bfloat16),
            v=torch.full((1, 4, 2048, 128), spare_scalar + 1, dtype=torch.bfloat16))]
        cache = SimpleNamespace(position=4093, history_rows=context, active=active, spare=spare, pending=None,
            owned=[], borrowed=[*active[0].values(), *spare[0].values()])
        device = SimpleNamespace(operations=operations, mesh=object(), position=4093, history_rows=context, block_rows=8,
            native_proposal_attention=False, validated_native_proposal_masks=set(),
            history=torch.zeros((1, 1, 2048, 5120), dtype=torch.bfloat16),
            spare_history=torch.zeros((1, 1, 2048, 5120), dtype=torch.bfloat16), progress=None)
        device.temporaries = MethodType(DFlashDevice.temporaries, device)
        host = proposal_inputs(17, 4093, context, 8, context)
        bucket = SimpleNamespace(context=context, identifiers=host['identifiers'].clone(), mask=host['mask'].clone(),
            rope={name: tuple(value.clone() for value in host['rope'][name]) for name in ('q', 'k')},
            history=torch.full((1, 1, context + 32, 5120), 9, dtype=torch.bfloat16),
            cached_history=[{name: torch.zeros((1, 4, context, 128), dtype=torch.bfloat16) for name in ('k', 'v')}])
        bucket.inputs = [bucket.identifiers, bucket.history, bucket.mask, *bucket.rope['q'], *bucket.rope['k'],
            *bucket.cached_history[0].values()]
        address = lambda operations, value: (value.untyped_storage().data_ptr(), value.untyped_storage().data_ptr() + 1)
        bucket.addresses = [address(operations, value) for value in bucket.inputs]
        prepared = PreparedDFlashProposal.__new__(PreparedDFlashProposal)
        prepared.operations, prepared.device, prepared.mesh = operations, device, device.mesh
        prepared.kv_history, prepared.owned = cache, bucket.inputs
        return SimpleNamespace(operations=operations, cache=cache, device=device, bucket=bucket,
            prepared=prepared, address=address)

    def released_addresses(self, operations, address):
        """A release_owned exactly as gdn_multitoken_conv.release_owned dedups and frees,
        but keyed by this test's own address() rather than a real get_device_tensors -
        the same substitution the neighbouring tests make for `addresses` itself."""
        def fake_release_owned(ops, tensors):
            unique = {address(ops, tensor): tensor for tensor in tensors}
            for tensor in unique.values():
                ops.deallocate(tensor)
        return fake_release_owned

    def test_update_does_not_release_the_pooled_kv_banks_it_reads(self):
        """Run 35561480877: the FIRST traced proposal after four-user pool admission died
        with TT_FATAL input_tensor.is_allocated() reading kv_history.active - the context
        2048 bucket (dflash_proposal_inputs.proposal_contexts's top rung, exactly the pool
        bank's own row extent) slices the bank's full extent, which shares the bank's own
        storage identity; unprotected in update()'s device.temporaries() (kv_history.owned
        is empty under the pool - only kv_history.borrowed lists the lent banks), retain()
        queued that reslice as a temporary and this call's own release_owned() freed the
        live pool bank. This must never happen, for the top bucket or any other."""
        for context in (256, 2048):
            case = self.build_pooled_case(context)
            release_owned = self.released_addresses(case.operations, case.address)
            with patch('dflash_proposal_trace.addresses', side_effect=case.address), \
                    patch('dflash_device.addresses', side_effect=case.address), \
                    patch('dflash_proposal_trace.release_owned', side_effect=release_owned):
                case.prepared.update(case.bucket, 5)
            bank_addresses = {case.address(case.operations, value) for value in case.cache.borrowed}
            freed_addresses = {case.address(case.operations, call.args[0])
                for call in case.operations.deallocate.call_args_list}
            self.assertFalse(bank_addresses & freed_addresses,
                'context=%d: update() released a pool-lent K/V bank as its own temporary' % context)
            for name in ('k', 'v'):
                self.assertTrue(torch.equal(case.bucket.cached_history[0][name],
                    case.cache.active[0][name][:, :, :context, :]))

    def test_update_protects_both_sides_of_a_committed_bank_swap(self):
        """The rebinding scenario: DraftKVHistory.commit() (draft_kv_history.py) swaps
        `self.active` and `self.spare` to the OTHER physical bank without reallocating
        either one - `borrowed` lists both sides permanently, from construction, so a
        proposal update() reading whichever bank is active now must never free either
        side, before or after the swap it did not itself request."""
        case = self.build_pooled_case(2048)
        release_owned = self.released_addresses(case.operations, case.address)
        with patch('dflash_proposal_trace.addresses', side_effect=case.address), \
                patch('dflash_device.addresses', side_effect=case.address), \
                patch('dflash_proposal_trace.release_owned', side_effect=release_owned):
            case.prepared.update(case.bucket, 5)
            # DraftKVHistory.commit(): self.active, self.spare = self.spare, self.active -
            # the same two physical banks, roles swapped; borrowed is untouched by design.
            case.cache.active, case.cache.spare = case.cache.spare, case.cache.active
            case.bucket.cached_history = [{name: torch.zeros_like(value)
                for name, value in case.cache.active[0].items()}]
            case.bucket.inputs = [case.bucket.identifiers, case.bucket.history, case.bucket.mask,
                *case.bucket.rope['q'], *case.bucket.rope['k'], *case.bucket.cached_history[0].values()]
            case.bucket.addresses = [case.address(case.operations, value) for value in case.bucket.inputs]
            case.prepared.owned = case.bucket.inputs
            case.prepared.update(case.bucket, 11)
        bank_addresses = {case.address(case.operations, value) for value in case.cache.borrowed}
        freed_addresses = {case.address(case.operations, call.args[0])
            for call in case.operations.deallocate.call_args_list}
        self.assertFalse(bank_addresses & freed_addresses,
            'a committed bank swap must not expose either physical bank to release')
        for name in ('k', 'v'):
            self.assertTrue(torch.equal(case.bucket.cached_history[0][name], case.cache.active[0][name]))


if __name__ == '__main__':
    unittest.main()
