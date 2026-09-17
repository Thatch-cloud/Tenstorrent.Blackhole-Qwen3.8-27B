from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from dspark_cached_markov import BiasCache


class CachedMarkovTests(unittest.TestCase):
    def setUp(self):
        self.operations = SimpleNamespace(uint32='uint32', float32='float32', ROW_MAJOR_LAYOUT='row',
            DRAM_MEMORY_CONFIG='dram', ReplicateTensorToMesh=Mock(return_value='replicate'),
            from_torch=Mock(side_effect=lambda *arguments, **keywords: object()),
            deallocate=Mock(), synchronize_device=Mock(), copy_host_to_device_tensor=Mock())
        self.mesh = SimpleNamespace(shape=[1, 2])
        self.predecessor = SimpleNamespace(shape=(1, 1, 64, 256))
        self.successor = SimpleNamespace(shape=(1, 1, 256, 64))

    def create(self):
        return BiasCache(self.operations, self.mesh, self.predecessor, self.successor)

    def test_reset_changes_epoch_without_reallocating_device_buffers(self):
        cache = self.create()
        original = list(cache.owned)
        cache.reset()
        self.assertEqual(cache.epoch, 2)
        self.assertEqual(cache.owned, original)
        self.assertEqual(self.operations.copy_host_to_device_tensor.call_count, 2)
        state, request = [call.args[0] for call in self.operations.from_torch.call_args_list[-2:]]
        self.assertEqual(int(state[0, 0, 0, 0]), 2)
        self.assertEqual(int(state.to(dtype=torch.uint32)[0, 0, 1:, 0].long().count_nonzero()), 0)
        self.assertEqual(request.reshape(-1).tolist(), [0, 248320, 2, 0, 0, 0, 0, 0])
        self.assertEqual(self.operations.deallocate.call_count, 0)

    def test_close_requires_trace_release_and_is_idempotent(self):
        cache = self.create()
        with self.assertRaises(ValueError):
            cache.close()
        cache.close(traces_released=True)
        cache.close(traces_released=True)
        self.assertEqual(self.operations.deallocate.call_count, 4)
        with self.assertRaises(ValueError):
            cache.reset()

    def test_exhausted_epoch_rejected_before_mutation(self):
        cache = self.create()
        cache.epoch = 0xffffffff
        with self.assertRaises(ValueError):
            cache.reset()
        self.operations.copy_host_to_device_tensor.assert_not_called()

    def test_partial_allocation_failure_releases_owned_buffers(self):
        self.operations.from_torch.side_effect = [object(), RuntimeError('allocation failed')]
        with self.assertRaises(RuntimeError):
            self.create()
        self.assertEqual(self.operations.deallocate.call_count, 1)


if __name__ == '__main__':
    unittest.main()
