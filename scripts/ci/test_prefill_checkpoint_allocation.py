from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from prefill_checkpoint_allocation import prepare
from test_prefill_prefix_integration import HostOperations


class CheckpointAllocationTests(unittest.TestCase):
    def fixture(self):
        operations = HostOperations()
        scratch = [SimpleNamespace(B=1, _stable_state=True, rec_state=operations.tensor(),
            conv_states=[operations.tensor() for unused in range(4)], conv_carry=operations.tensor())
            for unused in range(48)]
        for layer in scratch:
            layer.rec_state.dtype = 'fp32'
        base_clone = operations.clone
        def clone(source, **kwargs):
            result = base_clone(source, **kwargs)
            result.dtype = source.dtype
            return result
        operations.clone = Mock(side_effect=clone)
        model = SimpleNamespace(_chunked_trace_id=None, _bucket_trace_id=None, device='mesh',
            layers=[SimpleNamespace(is_full_attention=False, attention=layer) for layer in scratch],
            _bind_gdn_prefill_scratch=Mock(return_value='decode-bindings'),
            _unbind_gdn_prefill_scratch=Mock())
        generator = SimpleNamespace(model=[model], trace_ids_decode={False: {}, True: {}})
        return operations, generator, model

    def test_allocate_preserves_types_and_restores_native_bindings(self):
        operations, generator, model = self.fixture()
        allocation = prepare(operations, generator, model)
        self.assertEqual(len(allocation.buffers), 288)
        self.assertEqual(allocation.buffers[0].dtype, 'fp32')
        self.assertEqual(allocation.buffers[1].dtype, 'bf16')
        model._unbind_gdn_prefill_scratch.assert_called_once_with('decode-bindings')
        allocation.close()
        allocation.close()
        self.assertTrue(all(value.freed for value in allocation.buffers))
        self.assertFalse(any(value.freed for value in allocation.checkpoint.live))

    def test_existing_trace_rejected_before_binding_or_allocation(self):
        for kind in ('decode', 'prefill'):
            operations, generator, model = self.fixture()
            if kind == 'decode':
                generator.trace_ids_decode[False][1] = 0
            else:
                model._bucket_trace_id = 0
            with self.assertRaises(ValueError):
                prepare(operations, generator, model)
            model._bind_gdn_prefill_scratch.assert_not_called()
            operations.clone.assert_not_called()

    def test_allocation_failure_releases_partial_storage_and_restores_bindings(self):
        operations, generator, model = self.fixture()
        allocated = operations.tensor()
        operations.clone.side_effect = [allocated, RuntimeError('allocation')]
        with self.assertRaisesRegex(RuntimeError, 'allocation'):
            prepare(operations, generator, model)
        self.assertTrue(allocated.freed)
        model._unbind_gdn_prefill_scratch.assert_called_once_with('decode-bindings')


if __name__ == '__main__':
    unittest.main()
