"""The pre-trace draft history pool: sized for the scheduler, lent once per device,
loud when exhausted, exact about its addresses, and invisible to a device without one."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from dflash_device import DFlashDevice
from serving_buffer_pool import HISTORY_SHAPE, ServingBufferPool


class FakeShard:
    def __init__(self, address):
        self.address = address

    def buffer_address(self):
        return self.address


class FakeTensor:
    def __init__(self, shape, shards):
        self.shape, self.shards = tuple(shape), shards


class FakeOperations:
    """Two independent bump allocators, one per chip, so chip addresses never coincide."""

    bfloat16, TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'tile', 'row_major', 'dram'
    MathFidelity = SimpleNamespace(HiFi4='HiFi4')

    def __init__(self):
        self.next = [0x1000, 0x900000]
        self.live, self.deallocated, self.copies, self.fills = [], [], [], []
        self.zeros_like_calls = self.synchronized = 0

    def allocate(self, shape):
        shards = []
        for chip in range(2):
            shards.append(FakeShard(self.next[chip]))
            self.next[chip] += 0x100
        tensor = FakeTensor(shape, shards)
        self.live.append(tensor)
        return tensor

    def from_torch(self, value, **options):
        return self.allocate(value.shape)

    def zeros_like(self, tensor):
        self.zeros_like_calls += 1
        return self.allocate(tensor.shape)

    def pad(self, tensor, padding, value):
        return self.allocate(tuple(size + before + after for size, (before, after) in zip(tensor.shape, padding)))

    def get_device_tensors(self, tensor):
        return tensor.shards

    def deallocate(self, tensor):
        if any(tensor is value for value in self.deallocated):
            raise AssertionError('Double free')
        self.deallocated.append(tensor)

    def copy(self, source, destination):
        self.copies.append((source, destination))

    def full_like(self, tensor, value, *, optional_tensor):
        if optional_tensor is not tensor:
            raise AssertionError('In-place fill expected')
        self.fills.append((tensor, value))

    def synchronize_device(self, mesh):
        self.synchronized += 1

    def ReplicateTensorToMesh(self, mesh):
        return ('replicate', mesh)

    def ShardTensorToMesh(self, mesh, dim):
        return ('shard', mesh, dim)

    def WormholeComputeKernelConfig(self, **options):
        return options


class PoolTests(unittest.TestCase):
    def test_one_zeroed_history_pair_per_scheduler_request_on_independent_storage(self):
        operations = FakeOperations()
        pool = ServingBufferPool(operations, 'mesh', users=3)
        self.assertEqual(len(pool.slots), 3)
        self.assertEqual([value.shape for value in operations.live], [HISTORY_SHAPE] * 6)
        for chip in range(2):
            self.assertEqual(len({slot.addresses[pair][chip] for slot in pool.slots for pair in range(2)}), 6)
        report = pool.describe()
        self.assertEqual((report['users'], report['bytes_per_slot']), (3, 2 * 2 * 2048 * 5120))
        self.assertEqual([slot['lent'] for slot in report['slots']], [False] * 3)
        self.assertEqual(operations.synchronized, 1)

    def test_users_must_be_an_explicit_count_within_the_native_slots(self):
        for users in (0, 9, '2', 2.0, None):
            with self.subTest(users=users), self.assertRaises(ValueError):
                ServingBufferPool(FakeOperations(), 'mesh', users=users)

    def test_each_slot_is_lent_once_and_the_next_request_is_refused_loudly(self):
        operations = FakeOperations()
        pool = ServingBufferPool(operations, 'mesh', users=2)
        first, second = pool.acquire(), pool.acquire()
        self.assertIsNot(first, second)
        self.assertTrue(first.lent and second.lent)
        with self.assertRaisesRegex(ValueError, 'already lent'):
            pool.acquire()
        first.release()
        self.assertFalse(first.lent)
        with self.assertRaisesRegex(ValueError, 'not lent'):
            first.release()
        self.assertIs(pool.acquire(), first)
        foreign = ServingBufferPool(FakeOperations(), 'mesh', users=1).acquire()
        with self.assertRaisesRegex(ValueError, 'Only a slot lent by this pool'):
            pool.release(foreign)

    def test_a_loan_hands_over_zeroed_storage(self):
        operations = FakeOperations()
        pool = ServingBufferPool(operations, 'mesh', users=1)
        slot = pool.acquire()
        self.assertEqual(operations.fills, [(slot.history, 0.0), (slot.spare_history, 0.0)])

    def test_moved_storage_is_refused_at_both_ends_of_the_loan(self):
        operations = FakeOperations()
        pool = ServingBufferPool(operations, 'mesh', users=1)
        slot = pool.slots[0]
        original = slot.spare_history.shards[1].address
        slot.spare_history.shards[1].address += 1
        with self.assertRaisesRegex(AssertionError, 'slot 0 moved'):
            pool.acquire()
        slot.spare_history.shards[1].address = original
        self.assertIs(pool.acquire(), slot)
        slot.history.shards[0].address += 1
        with self.assertRaisesRegex(AssertionError, 'slot 0 moved'):
            slot.release()
        self.assertTrue(slot.lent)

    def test_close_frees_everything_once_and_reports_slots_still_lent(self):
        operations = FakeOperations()
        pool = ServingBufferPool(operations, 'mesh', users=2)
        pool.acquire()
        with self.assertRaisesRegex(ValueError, r'slots \[0\] still lent'):
            pool.close()
        self.assertEqual(len(operations.deallocated), 4)
        pool.close()
        self.assertEqual(len(operations.deallocated), 4)
        with self.assertRaisesRegex(ValueError, 'Closed'):
            pool.acquire()
        operations = FakeOperations()
        pool = ServingBufferPool(operations, 'mesh', users=2)
        pool.close()
        self.assertEqual(len(operations.deallocated), 4)

    def test_partial_allocation_is_freed_when_construction_fails(self):
        operations = FakeOperations()
        uploads = operations.from_torch

        def failing(value, **options):
            if len(operations.live) == 3:
                raise RuntimeError('out of DRAM')
            return uploads(value, **options)

        operations.from_torch = failing
        with self.assertRaisesRegex(RuntimeError, 'out of DRAM'):
            ServingBufferPool(operations, 'mesh', users=2)
        self.assertEqual(operations.deallocated, operations.live)


class DeviceLoanTests(unittest.TestCase):
    def device(self, operations, pool=None):
        model = SimpleNamespace(num_devices=2, vocab_size=248320, _lmhead_vocab_sharded=True, mesh_device='mesh')
        projection = {'fc.weight': torch.zeros(4, 4), 'hidden_norm.weight': torch.zeros(5120)}
        selector = {'norm.weight': torch.zeros(5120), 'candidate_selector.hidden_projection.weight': torch.zeros(2, 2),
                    'candidate_selector.predecessor_codebook': torch.zeros(1),
                    'candidate_selector.successor_codebook': torch.zeros(1)}
        features = [SimpleNamespace(shape=(1, 1, 2048, 2560))] * 5
        return DFlashDevice(operations, model, Mock(), [('attention', 'convolution', 'mlp')] * 5, projection,
            selector, features, position=4096, feature_start=2048, block_rows=16,
            **(dict(buffer_pool=pool) if pool is not None else {}))

    def build(self, operations, pool=None, project=None):
        def project_features(device, features, count):
            return device.operations.allocate((1, 1, count, 5120))

        with patch('dflash_device.prepare_attention_branch', return_value='attention'), \
                patch('dflash_device.prepare_mlp_branch', return_value='mlp'), \
                patch('dflash_device.projection_shards', return_value=[torch.zeros(2, 4)]), \
                patch.object(DFlashDevice, 'project_features', project or project_features):
            return self.device(operations, pool)

    def test_device_borrows_the_slot_and_never_frees_the_pool(self):
        operations = FakeOperations()
        pool = ServingBufferPool(operations, 'mesh', users=1)
        device = self.build(operations, pool)
        slot = pool.slots[0]
        self.assertTrue(slot.lent)
        self.assertIs(device.history, slot.history)
        self.assertIs(device.spare_history, slot.spare_history)
        self.assertEqual(operations.zeros_like_calls, 0)
        padded = operations.live[-1]
        self.assertEqual(padded.shape, HISTORY_SHAPE)
        self.assertEqual(operations.copies, [(padded, slot.history)])
        self.assertTrue(any(padded is value for value in operations.deallocated))
        device.close()
        self.assertFalse(slot.lent)
        self.assertFalse(any(value is pooled for value in operations.deallocated for pooled in slot.tensors))
        self.assertEqual(len(operations.deallocated), len(operations.live) - 2)
        pool.close()
        self.assertEqual(len(operations.deallocated), len(operations.live))

    def test_committed_swap_still_returns_the_whole_pair(self):
        operations = FakeOperations()
        pool = ServingBufferPool(operations, 'mesh', users=1)
        device = self.build(operations, pool)
        device.history, device.spare_history = device.spare_history, device.history
        device.close()
        self.assertFalse(pool.slots[0].lent)
        self.assertFalse(any(value is pooled for value in operations.deallocated for pooled in pool.slots[0].tensors))

    def test_without_a_pool_the_device_allocates_and_frees_its_own_pair(self):
        operations = FakeOperations()
        device = self.build(operations)
        self.assertIsNone(device.pool_slot)
        self.assertEqual(operations.zeros_like_calls, 1)
        self.assertIs(device.history, operations.live[-2])
        self.assertIs(device.spare_history, operations.live[-1])
        self.assertEqual(operations.copies, [])
        device.close()
        self.assertEqual(len(operations.deallocated), len(operations.live))

    def test_an_exhausted_pool_refuses_the_device_before_it_uploads_anything(self):
        operations = FakeOperations()
        pool = ServingBufferPool(operations, 'mesh', users=1)
        self.build(operations, pool)
        allocated = len(operations.live)
        with self.assertRaisesRegex(ValueError, 'already lent'):
            self.build(operations, pool)
        self.assertEqual(len(operations.live), allocated)

    def test_a_device_that_fails_after_borrowing_returns_the_slot(self):
        operations = FakeOperations()
        pool = ServingBufferPool(operations, 'mesh', users=1)
        with self.assertRaisesRegex(RuntimeError, 'projection failed'):
            self.build(operations, pool, project=Mock(side_effect=RuntimeError('projection failed')))
        self.assertFalse(pool.slots[0].lent)
        self.assertFalse(any(value is pooled for value in operations.deallocated for pooled in pool.slots[0].tensors))
        self.assertEqual(len(operations.deallocated), len(operations.live) - 2)

    def test_a_pool_without_a_loan_method_is_refused(self):
        with self.assertRaises(ValueError):
            self.build(FakeOperations(), pool=object())


if __name__ == '__main__':
    unittest.main()
