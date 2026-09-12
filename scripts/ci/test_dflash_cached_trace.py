from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import Mock, patch

import torch

from dflash_device import DFlashDevice
from dflash_proposal_inputs import proposal_inputs
from dflash_proposal_trace import PreparedDFlashProposal


class CachedProposalTests(unittest.TestCase):
    def test_update_follows_committed_bank_swaps_without_copying_feature_history(self):
        operations = SimpleNamespace(ReplicateTensorToMesh=lambda mesh: mesh,
            from_torch=lambda value, **kwargs: value.clone(),
            copy_host_to_device_tensor=Mock(side_effect=lambda source, destination: destination.copy_(source)),
            slice=lambda value, start, end: value[tuple(slice(first, last) for first, last in zip(start, end, strict=True))],
            copy=Mock(side_effect=lambda source, destination: destination.copy_(source)), synchronize_device=Mock())
        active = [dict(k=torch.ones((1, 4, 2048, 128), dtype=torch.bfloat16),
            v=torch.full((1, 4, 2048, 128), 3, dtype=torch.bfloat16))]
        cache = SimpleNamespace(position=4093, history_rows=2048, active=active, pending=None,
            owned=list(active[0].values()))
        device = SimpleNamespace(operations=operations, mesh=object(), position=4093, history_rows=2048, block_rows=8,
            history=torch.zeros((1, 1, 2048, 5120), dtype=torch.bfloat16),
            spare_history=torch.zeros((1, 1, 2048, 5120), dtype=torch.bfloat16), progress=None)
        device.temporaries = MethodType(DFlashDevice.temporaries, device)
        host = proposal_inputs(17, 4093, 2048, 8, 2048)
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


if __name__ == '__main__':
    unittest.main()
