"""The pre-trace draft history pool and the shared draft weights: sized for the
scheduler, lent once per device, loud when exhausted, exact about their addresses,
logged when lent, and invisible to a device that has neither."""

from contextlib import ExitStack
import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import dflash_device
from dflash_device import DFlashDevice, PreparedDraftWeights, pindiag
from serving_buffer_pool import DRAFT_LAYERS, HISTORY_SHAPE, KV_SHAPE, ServingBufferPool, bank_tensors

# A slot: the history pair plus, per draft layer, active and spare k and v.
SLOT_TENSORS = 2 + 4 * DRAFT_LAYERS


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
        self.assertEqual([value.shape for value in operations.live],
                         ([HISTORY_SHAPE] * 2 + [KV_SHAPE] * 4 * DRAFT_LAYERS) * 3)
        for slot in pool.slots:
            self.assertEqual(len(slot.tensors), SLOT_TENSORS)
            self.assertEqual(slot.tensors[:2], (slot.history, slot.spare_history))
            self.assertEqual(len(slot.kv), DRAFT_LAYERS)
            self.assertEqual(list(slot.tensors[2:]), bank_tensors(slot.kv))
            self.assertEqual(slot.tensors[2:6], (slot.kv[0]['active']['k'], slot.kv[0]['active']['v'],
                                                 slot.kv[0]['spare']['k'], slot.kv[0]['spare']['v']))
        for chip in range(2):
            self.assertEqual(len({address[chip] for slot in pool.slots for address in slot.addresses}), 3 * SLOT_TENSORS)
        report = pool.describe()
        self.assertEqual((report['users'], report['layers'], report['kv_shape']), (3, DRAFT_LAYERS, list(KV_SHAPE)))
        self.assertEqual(report['bytes_per_slot'], 2 * 2 * 2048 * 5120 + 4 * DRAFT_LAYERS * 2 * 4 * 2048 * 128)
        self.assertEqual([slot['lent'] for slot in report['slots']], [False] * 3)
        # Every K/V address in the stage line, placed by layer, side and head.
        for slot, described in zip(pool.slots, report['slots'], strict=True):
            self.assertEqual(described['addresses'], [list(value) for value in slot.addresses[:2]])
            self.assertEqual(described['kv'], [{side: {head: [shard.address for shard in bank[side][head].shards]
                                                        for head in ('k', 'v')} for side in ('active', 'spare')}
                                               for bank in slot.kv])
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
        self.assertEqual(operations.fills, [(value, 0.0) for value in slot.tensors])
        self.assertEqual(len(operations.fills), SLOT_TENSORS)

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
        slot.history.shards[0].address -= 1
        slot.kv[3]['spare']['v'].shards[1].address += 1
        with self.assertRaisesRegex(AssertionError, 'slot 0 moved'):
            slot.release()
        slot.kv[3]['spare']['v'].shards[1].address -= 1
        slot.release()
        self.assertFalse(slot.lent)

    def test_close_frees_everything_once_and_reports_slots_still_lent(self):
        operations = FakeOperations()
        pool = ServingBufferPool(operations, 'mesh', users=2)
        pool.acquire(owner='request-A')
        with self.assertRaisesRegex(ValueError, r"slots \[\(0, 'request-A'\)\] still lent"):
            pool.close()
        self.assertEqual(len(operations.deallocated), 2 * SLOT_TENSORS)
        pool.close()
        self.assertEqual(len(operations.deallocated), 2 * SLOT_TENSORS)
        with self.assertRaisesRegex(ValueError, 'Closed'):
            pool.acquire()
        operations = FakeOperations()
        pool = ServingBufferPool(operations, 'mesh', users=2)
        pool.close()
        self.assertEqual(len(operations.deallocated), 2 * SLOT_TENSORS)

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

    def test_every_loan_and_return_is_logged_with_its_owner(self):
        with patch('serving_buffer_pool.pindiag') as log:
            pool = ServingBufferPool(FakeOperations(), 'mesh', users=1)
            slot = pool.acquire(owner='request-A')
            self.assertEqual(slot.owner, 'request-A')
            self.assertEqual(pool.describe()['slots'][0]['owner'], 'request-A')
            slot.release()
        lines = [call.args[0].format(*call.args[1:]) for call in log.call_args_list]
        self.assertEqual(lines, ['[PINDIAG] pool slot 0 acquired for request-A: history at %s, %d K/V banks from %s'
                                 % (slot.addresses[:2], SLOT_TENSORS - 2, slot.addresses[2]),
                                 '[PINDIAG] pool slot 0 released by request-A'])
        self.assertIsNone(slot.owner)


class DiagnosticLineTests(unittest.TestCase):
    def test_loguru_carries_the_line_when_present_and_print_when_absent(self):
        logger = Mock()
        with patch.dict(sys.modules, {'loguru': SimpleNamespace(logger=logger)}):
            pindiag('[PINDIAG] pool slot {} acquired for {}', 0, 'A')
        logger.info.assert_called_once_with('[PINDIAG] pool slot {} acquired for {}', 0, 'A')
        with patch.dict(sys.modules, {'loguru': None}), patch('sys.stdout', new_callable=io.StringIO) as out:
            pindiag('[PINDIAG] pool slot {} acquired for {}', 0, 'A')
        self.assertEqual(out.getvalue(), '[PINDIAG] pool slot 0 acquired for A\n')


def fake_attention_branch(operations, mesh, attention, convolution, retain, **options):
    return dict(norm=retain(operations.allocate((1, 1, 160, 32))), block_rows=options['block_rows'])


def fake_mlp_branch(operations, mesh, mlp, convolution, retain):
    return dict(device_norm=retain(operations.allocate((1, 1, 160, 32))))


WEIGHT_UPLOADS = 5 * 2 + 4


class FakeKVHistory:
    """Records how the device constructs its draft cache; owns nothing."""

    def __init__(self, operations, mesh, parameters, features, **options):
        self.operations, self.mesh, self.parameters, self.features = operations, mesh, list(parameters), features
        self.options = options
        storage = options.get('storage')
        self.active = [bank['active'] for bank in storage] if storage is not None else []
        self.pending, self.closed = None, False

    def close(self):
        self.closed = True


class DeviceFixture(unittest.TestCase):
    def setUp(self):
        self.layers = [('attention', 'convolution', 'mlp')] * 5
        self.projection = {'fc.weight': torch.zeros(4, 4), 'hidden_norm.weight': torch.zeros(5120)}
        self.selector = {'norm.weight': torch.zeros(5120), 'candidate_selector.hidden_projection.weight': torch.zeros(2, 2),
                         'candidate_selector.predecessor_codebook': torch.zeros(1),
                         'candidate_selector.successor_codebook': torch.zeros(1)}

    def patches(self, stack):
        stack.enter_context(patch('dflash_device.prepare_attention_branch', side_effect=fake_attention_branch))
        stack.enter_context(patch('dflash_device.prepare_mlp_branch', side_effect=fake_mlp_branch))
        stack.enter_context(patch('dflash_device.projection_shards', return_value=[torch.zeros(2, 4)]))

    def device(self, operations, pool=None, weights=None, layers=None, projection=None, block_rows=16, **extra):
        model = SimpleNamespace(num_devices=2, vocab_size=248320, _lmhead_vocab_sharded=True, mesh_device='mesh')
        features = [SimpleNamespace(shape=(1, 1, 2048, 2560))] * 5
        return DFlashDevice(operations, model, Mock(), self.layers if layers is None else layers,
            self.projection if projection is None else projection, self.selector, features,
            position=4096, feature_start=2048, block_rows=block_rows,
            **(dict(buffer_pool=pool) if pool is not None else {}),
            **(dict(shared_weights=weights) if weights is not None else {}), **extra)

    def build(self, operations, pool=None, weights=None, project=None, **options):
        def project_features(device, features, count):
            return device.operations.allocate((1, 1, count, 5120))

        with ExitStack() as stack:
            self.patches(stack)
            stack.enter_context(patch.object(DFlashDevice, 'project_features', project or project_features))
            return self.device(operations, pool, weights, **options)

    def prepare(self, operations, **geometry):
        with ExitStack() as stack:
            self.patches(stack)
            return PreparedDraftWeights(operations, 'mesh', self.layers, self.projection, self.selector,
                                        **dict(dict(block_rows=16), **geometry))


class DeviceLoanTests(DeviceFixture):
    def test_device_borrows_the_slot_and_never_frees_the_pool(self):
        operations = FakeOperations()
        pool = ServingBufferPool(operations, 'mesh', users=1)
        device = self.build(operations, pool)
        slot = pool.slots[0]
        self.assertTrue(slot.lent)
        self.assertEqual(slot.owner, device.name)
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
        self.assertEqual(len(operations.deallocated), len(operations.live) - SLOT_TENSORS)
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
        self.assertEqual(len(operations.deallocated), len(operations.live) - SLOT_TENSORS)

    def test_a_pool_without_a_loan_method_is_refused(self):
        with self.assertRaises(ValueError):
            self.build(FakeOperations(), pool=object())

    def test_pooled_device_lends_its_kv_banks_to_the_draft_cache_and_an_unpooled_one_does_not(self):
        # The serving geometry: cached native T16 proposals with the capture deferred.
        serving = dict(cache_history=True, proposal_capture=True, defer_proposal_capture=True,
                       native_proposal_attention=True)
        operations = FakeOperations()
        pool = ServingBufferPool(operations, 'mesh', users=1)
        with patch('draft_kv_history.DraftKVHistory', FakeKVHistory):
            device = self.build(operations, pool, **serving)
            cache = device.kv_history
            self.assertIs(cache.options['storage'], pool.slots[0].kv)
            self.assertIs(cache.features, device.history)
            self.assertEqual(cache.parameters, [layer[0] for layer in device.layers])
            self.assertEqual((cache.options['position'], cache.options['history_rows'], cache.options['capture_projection']),
                             (4096, 2048, False))
            device.close()
            self.assertTrue(cache.closed)
            self.assertFalse(pool.slots[0].lent)
            plain = self.build(operations, **serving)
            self.assertNotIn('storage', plain.kv_history.options)
            plain.close()
            self.assertTrue(plain.kv_history.closed)


class SharedWeightTests(DeviceFixture):
    def test_private_weights_go_through_the_device_in_the_old_order_and_are_freed_by_it(self):
        operations = FakeOperations()
        device = self.build(operations)
        self.assertIsNone(device.shared_weights)
        self.assertEqual(device.borrowed, [])
        self.assertEqual(device.owned, operations.live[:WEIGHT_UPLOADS])
        self.assertEqual([layer[0]['norm'] for layer in device.layers], operations.live[0:WEIGHT_UPLOADS - 4:2])
        self.assertEqual([device.projection, device.feature_norm, device.final_norm, device.selector_projection],
                         operations.live[WEIGHT_UPLOADS - 4:WEIGHT_UPLOADS])
        self.assertEqual([layer[2:] for layer in device.layers], [('mlp', 'convolution')] * 5)
        device.close()
        self.assertEqual(len(operations.deallocated), len(operations.live))

    def test_shared_weights_are_uploaded_once_and_lent_to_every_device(self):
        operations = FakeOperations()
        weights = self.prepare(operations)
        self.assertEqual(len(weights.tensors), WEIGHT_UPLOADS)
        self.assertEqual(weights.owned, weights.tensors)
        uploaded = len(operations.live)
        first, second = self.build(operations, weights=weights), self.build(operations, weights=weights)
        # Only the history pair and its projection per device; not one weight.
        self.assertEqual(len(operations.live), uploaded + 2 * 3)
        for device in (first, second):
            self.assertIs(device.shared_weights, weights)
            self.assertIs(device.layers, weights.layers)
            for name in ('projection', 'feature_norm', 'final_norm', 'selector_projection', 'predecessors', 'successors'):
                self.assertIs(getattr(device, name), getattr(weights, name))
            self.assertEqual(device.borrowed, weights.tensors)
            self.assertFalse(any(value is weight for value in device.owned for weight in weights.tensors))
        self.assertEqual(weights.borrowers, [first, second])
        first.close()
        self.assertEqual(weights.borrowers, [second])
        self.assertFalse(any(value is weight for value in operations.deallocated for weight in weights.tensors))
        second.close()
        self.assertEqual(weights.borrowers, [])
        self.assertEqual(len(operations.deallocated), len(operations.live) - WEIGHT_UPLOADS)
        weights.close()
        self.assertEqual(len(operations.deallocated), len(operations.live))
        weights.close()
        self.assertEqual(len(operations.deallocated), len(operations.live))

    def test_borrowed_weights_are_protected_from_the_device_temporaries(self):
        operations = FakeOperations()
        weights = self.prepare(operations)
        device = self.build(operations, weights=weights)
        owned, retain = device.temporaries([])
        self.assertIs(retain(weights.selector_projection), weights.selector_projection)
        self.assertEqual(owned, [])
        aliased = FakeTensor((1, 1, 32, 32), [FakeShard(weights.projection.shards[0].address), FakeShard(0xdead)])
        with self.assertRaisesRegex(ValueError, 'partially alias'):
            retain(aliased)
        fresh = operations.allocate((1, 1, 32, 32))
        self.assertEqual([retain(fresh)], owned)

    def test_lend_refuses_another_geometry_or_other_learned_layers(self):
        operations = FakeOperations()
        weights = self.prepare(operations)
        with self.assertRaisesRegex(ValueError, 'another mesh or proposal geometry'):
            self.build(operations, weights=weights, block_rows=8)
        with self.assertRaisesRegex(ValueError, 'other learned layers'):
            self.build(operations, weights=weights, layers=[('attention', 'convolution', {'other': 1})] * 5)
        with self.assertRaisesRegex(ValueError, 'other learned layers'):
            self.build(operations, weights=weights, projection=dict(self.projection))
        with self.assertRaises(ValueError):
            self.build(operations, weights=object())
        self.assertEqual(weights.borrowers, [])
        weights.close()
        with self.assertRaisesRegex(ValueError, 'Closed'):
            self.build(operations, weights=weights)

    def test_geometry_and_layer_count_are_checked_before_any_upload(self):
        operations = FakeOperations()
        for options in (dict(block_rows=7), dict(live_query_qk=1), dict(native_proposal_attention='yes')):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.prepare(operations, **options)
        with self.assertRaises(ValueError), ExitStack() as stack:
            self.patches(stack)
            PreparedDraftWeights(operations, 'mesh', self.layers[:4], self.projection, self.selector, block_rows=16)
        self.assertEqual(operations.live, [])

    def test_closing_lent_weights_frees_them_and_names_the_borrowers(self):
        operations = FakeOperations()
        weights = self.prepare(operations)
        device = self.build(operations, weights=weights)
        with self.assertRaisesRegex(ValueError, device.name):
            weights.close()
        self.assertTrue(all(any(value is weight for value in operations.deallocated) for weight in weights.tensors))

    def test_a_device_that_fails_after_borrowing_returns_the_weights_unfreed(self):
        operations = FakeOperations()
        weights = self.prepare(operations)
        with self.assertRaisesRegex(RuntimeError, 'projection failed'):
            self.build(operations, weights=weights, project=Mock(side_effect=RuntimeError('projection failed')))
        self.assertEqual(weights.borrowers, [])
        self.assertFalse(any(value is weight for value in operations.deallocated for weight in weights.tensors))

    def test_failed_preparation_frees_the_uploads_it_made(self):
        operations = FakeOperations()
        with self.assertRaisesRegex(RuntimeError, 'no DRAM'), ExitStack() as stack:
            self.patches(stack)
            stack.enter_context(patch('dflash_device.prepare_mlp_branch', side_effect=RuntimeError('no DRAM')))
            PreparedDraftWeights(operations, 'mesh', self.layers, self.projection, self.selector, block_rows=16)
        self.assertEqual(operations.deallocated, operations.live)
        self.assertEqual(len(operations.live), 1)

    def test_describe_names_every_weight_by_its_place_in_the_prepared_dicts(self):
        operations = FakeOperations()
        weights = self.prepare(operations)
        report = weights.describe()
        self.assertEqual(report['tensors'], WEIGHT_UPLOADS)
        self.assertEqual([entry['name'] for entry in report['weights']],
                         [name for index in range(5) for name in ('layer%d.attention.norm' % index, 'layer%d.mlp.device_norm' % index)]
                         + ['projection', 'feature_norm', 'final_norm', 'selector_projection'])
        self.assertEqual([entry['addresses'] for entry in report['weights']],
                         [[shard.address for shard in tensor.shards] for tensor in weights.tensors])
        self.assertEqual(report['borrowers'], [])
        device = self.build(operations, weights=weights)
        self.assertEqual(weights.describe()['borrowers'], [device.name])
        device.close()

    def test_loans_and_returns_are_logged_with_the_device_name(self):
        operations = FakeOperations()
        weights = self.prepare(operations)
        with patch.object(dflash_device, 'pindiag') as log:
            device = self.build(operations, weights=weights)
            device.close()
        lines = [call.args[0].format(*call.args[1:]) for call in log.call_args_list]
        self.assertEqual(lines, ['[PINDIAG] draft weights lent to %s (borrowers=1 tensors=%d)' % (device.name, WEIGHT_UPLOADS),
                                 '[PINDIAG] draft weights returned by %s (borrowers=0)' % device.name])

    def test_pooled_history_and_shared_weights_leave_the_device_owning_nothing_persistent(self):
        operations = FakeOperations()
        weights = self.prepare(operations)
        pool = ServingBufferPool(operations, 'mesh', users=1)
        device = self.build(operations, pool=pool, weights=weights)
        self.assertEqual(device.owned, [])
        self.assertIs(device.history, pool.slots[0].history)
        self.assertIs(device.layers, weights.layers)
        device.close()
        self.assertEqual(weights.borrowers, [])
        self.assertFalse(pool.slots[0].lent)
        persistent = [*weights.tensors, *pool.slots[0].tensors]
        self.assertFalse(any(value is kept for value in operations.deallocated for kept in persistent))


if __name__ == '__main__':
    unittest.main()
