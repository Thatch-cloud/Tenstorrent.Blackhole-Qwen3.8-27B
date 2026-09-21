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
from pooled_attention_replay import bundle_batches, family_capacities
from serving_buffer_pool import (DRAFT_LAYERS, GDN_LAYERS, HISTORY_SHAPE, KV_SHAPE, QUERY_SHAPE, ServingBufferPool,
                                 bank_tensors, snapshot_tensors, tensor_bytes)

# A slot: the history pair plus, per draft layer, active and spare k and v.
SLOT_TENSORS = 2 + 4 * DRAFT_LAYERS
# One GDN slot-zero snapshot as gdn_snapshot.ActiveSnapshot.allocate hands it out: the
# recurrent state and four convolution taps (gdn_state_copy.page_counts).
SNAPSHOT_SHAPES = ((1, 24, 128, 128), (1, 1, 5120), (1, 1, 5120), (1, 1, 5120), (1, 1, 5120))
ROPE_DIM = 64
# The serving geometry: a T16 cap and a 256-token budget (verifier_engine.capture_bucket_rows).
BUCKET_ROWS = (1, 2, 4, 8, 8, 16, 16)
TAPS = 5
# The native chunk families a 68-page table can hold, i.e. the pool's default here.
REPLAY_CAPACITIES = (4096, 4352)


def replay_shapes(rows, capacities=REPLAY_CAPACITIES, group_rows=4):
    """The replay reader's page tables for a bucket: per family, one per bundle."""
    if rows < 8:
        return []
    return [(batches, capacity // 64) for capacity in capacities
            for batches in bundle_batches(rows, capacity, max_group_rows=group_rows)]


def fake_helpers(operations):
    return [SimpleNamespace(live=[FakeTensor(shape, []) for shape in SNAPSHOT_SHAPES],
                            allocate=Mock(side_effect=lambda: [operations.allocate(shape) for shape in SNAPSHOT_SHAPES]))
            for index in range(GDN_LAYERS)]


def fake_rope(operations):
    return Mock(side_effect=lambda positions: tuple(operations.allocate((1, len(positions), 1, ROPE_DIM), dtype='bf16', layout='tile')
                                                    for name in ('cos', 'sin')))


def verifier_pool(operations, users=1, **overrides):
    options = dict(helpers=fake_helpers(operations), page_width=68, bucket_rows=BUCKET_ROWS, feature_taps=TAPS,
                   rope=fake_rope(operations))
    options.update(overrides)
    return ServingBufferPool(operations, 'mesh', users=users, **options)


def snapshot_set_bytes():
    return GDN_LAYERS * sum(tensor_bytes(shape) for shape in SNAPSHOT_SHAPES)


def bucket_bytes(rows, page_width=68, taps=TAPS, capacities=REPLAY_CAPACITIES):
    integers = 4 * (rows + rows + rows * page_width + page_width + rows)
    replay = 4 * sum(batches * columns for batches, columns in replay_shapes(rows, capacities))
    return (snapshot_set_bytes() + taps * tensor_bytes((1, 1, rows, 5120)) + integers + replay
            + 2 * tensor_bytes((1, rows, 1, ROPE_DIM)))


def verifier_slot_bytes(bucket_rows=BUCKET_ROWS):
    return tensor_bytes(QUERY_SHAPE) + 2 * snapshot_set_bytes() + sum(bucket_bytes(rows) for rows in bucket_rows)


class FakeShard:
    def __init__(self, address):
        self.address = address

    def buffer_address(self):
        return self.address


class FakeTensor:
    def __init__(self, shape, shards, dtype=None, layout=None, mapper=None):
        self.shape, self.shards = tuple(shape), shards
        self.dtype, self.layout, self.mapper = dtype, layout, mapper


class FakeOperations:
    """Two independent bump allocators, one per chip, so chip addresses never coincide."""

    bfloat16, TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'tile', 'row_major', 'dram'
    uint32, int32 = 'uint32', 'int32'
    MathFidelity = SimpleNamespace(HiFi4='HiFi4')

    def __init__(self):
        self.next = [0x1000, 0x900000]
        self.live, self.deallocated, self.copies, self.fills = [], [], [], []
        self.zeros_like_calls = self.synchronized = 0

    def allocate(self, shape, **options):
        shards = []
        for chip in range(2):
            shards.append(FakeShard(self.next[chip]))
            self.next[chip] += 0x100
        tensor = FakeTensor(shape, shards, **options)
        self.live.append(tensor)
        return tensor

    def from_torch(self, value, **options):
        return self.allocate(value.shape, dtype=options.get('dtype'), layout=options.get('layout'),
                             mapper=options.get('mesh_mapper'))

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


class VerifierStorageTests(unittest.TestCase):
    """The per-request verifier buffers the 2026-09-20 audit found still allocated after
    an earlier request's traces: the draft cache's query, the initial and carried GDN
    state, one checkpoint set per capture, the feature taps and the fixture inputs."""

    def test_each_slot_holds_the_query_two_gdn_sets_and_one_bucket_per_capture_width(self):
        operations = FakeOperations()
        rope = fake_rope(operations)
        pool = verifier_pool(operations, users=2, rope=rope)
        self.assertEqual(len(pool.slots), 2)
        for slot in pool.slots:
            self.assertEqual((slot.query.shape, slot.query.dtype, slot.query.layout, slot.query.mapper),
                             (QUERY_SHAPE, 'bf16', 'tile', ('replicate', 'mesh')))
            verifier = slot.verifier
            for snapshots in (verifier.initial, verifier.carry):
                self.assertEqual([[value.shape for value in snapshot] for snapshot in snapshots],
                                 [list(SNAPSHOT_SHAPES)] * GDN_LAYERS)
            self.assertEqual([bucket.rows for bucket in verifier.buckets], list(BUCKET_ROWS))
            for bucket in verifier.buckets:
                rows = bucket.rows
                self.assertEqual([[value.shape for value in snapshot] for snapshot in bucket.checkpoints],
                                 [list(SNAPSHOT_SHAPES)] * GDN_LAYERS)
                self.assertEqual(len(bucket.target_features), TAPS)
                for feature in bucket.target_features:
                    self.assertEqual((feature.shape, feature.dtype, feature.layout, feature.mapper),
                                     ((1, 1, rows, 5120), 'bf16', 'tile', ('shard', 'mesh', 3)))
                self.assertIsNone(bucket.mtp_hidden)
                batch = bucket.batch
                self.assertEqual((batch.tokens.shape, batch.tokens.dtype, batch.tokens.layout), ((rows, 1), 'uint32', 'row_major'))
                self.assertEqual((batch.positions.shape, batch.positions.dtype, batch.positions.layout), ((rows,), 'int32', 'row_major'))
                self.assertEqual((batch.pages.shape, batch.pages.dtype), ((rows, 68), 'int32'))
                self.assertEqual((batch.singleton_pages.shape, batch.singleton_pages.dtype), ((1, 68), 'int32'))
                self.assertEqual([(value.shape, value.dtype, value.layout) for value in batch.singleton_positions],
                                 [((1,), 'int32', 'row_major')] * rows)
                for table in (batch.cos, batch.sin):
                    self.assertEqual((table.shape, table.mapper), ((1, rows, 1, ROPE_DIM), None))
                # The replay reader's page tables, per family the 68-page table can hold,
                # per bundle - at replay widths only; the pool's default enumerates the families.
                self.assertEqual(list(batch.replay_pages), list(REPLAY_CAPACITIES) if rows >= 8 else [])
                self.assertEqual([(value.shape, value.dtype, value.layout) for tables in batch.replay_pages.values()
                                  for value in tables],
                                 [(shape, 'int32', 'row_major') for shape in replay_shapes(rows)])
                self.assertEqual(list(bucket.replay_tables()), [value for tables in batch.replay_pages.values() for value in tables])
                self.assertFalse(bucket.taken)
            # Every tensor is in the slot's address record, once, on independent storage.
            expected = (2 + 4 * DRAFT_LAYERS + 1 + 2 * 5 * GDN_LAYERS
                        + sum(5 * GDN_LAYERS + TAPS + 6 + rows + len(replay_shapes(rows)) for rows in BUCKET_ROWS))
            self.assertEqual(len(slot.tensors), expected)
            for chip in range(2):
                self.assertEqual(len({address[chip] for address in slot.addresses}), expected)
            self.assertEqual(slot.bytes, 2 * tensor_bytes(HISTORY_SHAPE) + 4 * DRAFT_LAYERS * tensor_bytes(KV_SHAPE)
                             + verifier_slot_bytes())
        # The rotary builder was asked once per bucket per slot, for that bucket's rows.
        self.assertEqual([len(call.args[0]) for call in rope.call_args_list], list(BUCKET_ROWS) * 2)
        self.assertEqual([helper.allocate.call_count for helper in pool.helpers], [2 * (2 + len(BUCKET_ROWS))] * GDN_LAYERS)
        # Plus the packed block's per-user replay tables for the one shape two slots can fill
        # (M1, two T16 users): per family, per user, one table per bundle.
        packed = 2 * sum(len(bundle_batches(16, capacity)) for capacity in REPLAY_CAPACITIES)
        self.assertEqual(len(operations.live), 2 * expected + packed)
        self.assertEqual(operations.synchronized, 1)

    def test_describe_reports_the_verifier_bytes_and_layout_per_slot(self):
        operations = FakeOperations()
        pool = verifier_pool(operations)
        report = pool.describe()
        draft = 2 * tensor_bytes(HISTORY_SHAPE) + 4 * DRAFT_LAYERS * tensor_bytes(KV_SHAPE)
        self.assertEqual(report['draft_bytes_per_slot'], draft)
        self.assertEqual(report['query_bytes_per_slot'], tensor_bytes(QUERY_SHAPE))
        self.assertEqual(report['verifier_bytes_per_slot'], verifier_slot_bytes() - tensor_bytes(QUERY_SHAPE))
        self.assertEqual(report['bytes_per_slot'], draft + verifier_slot_bytes())
        self.assertEqual((report['page_width'], report['bucket_rows'], report['gdn_snapshot_sets'],
                          report['feature_taps'], report['mtp_hidden']), (68, list(BUCKET_ROWS), 9, TAPS, False))
        self.assertEqual((report['replay_group_rows'], report['replay_capacities']), (4, list(REPLAY_CAPACITIES)))
        self.assertEqual(report['replay_page_bytes_per_slot'],
                         4 * sum(batches * columns for rows in BUCKET_ROWS for batches, columns in replay_shapes(rows)))
        slot = pool.slots[0]
        described = report['slots'][0]
        self.assertEqual(described['query'], [shard.address for shard in slot.query.shards])
        verifier = described['verifier']
        self.assertEqual(verifier['initial'], [shard.address for shard in slot.verifier.initial[0][0].shards])
        self.assertEqual(verifier['carry'], [shard.address for shard in slot.verifier.carry[0][0].shards])
        self.assertEqual([bucket['rows'] for bucket in verifier['buckets']], list(BUCKET_ROWS))
        for bucket, described_bucket in zip(slot.verifier.buckets, verifier['buckets'], strict=True):
            self.assertEqual(described_bucket['checkpoints'], [shard.address for shard in bucket.checkpoints[0][0].shards])
            self.assertEqual(described_bucket['target_features'],
                             [[shard.address for shard in value.shards] for value in bucket.target_features])
            self.assertIsNone(described_bucket['mtp_hidden'])
            self.assertEqual(described_bucket['batch']['pages'], [shard.address for shard in bucket.batch.pages.shards])
            self.assertEqual(len(described_bucket['singleton_positions']), bucket.rows)
            self.assertEqual(described_bucket['replay_pages'],
                             {capacity: [[shard.address for shard in table.shards] for table in tables]
                              for capacity, tables in bucket.batch.replay_pages.items()})
            self.assertFalse(described_bucket['taken'])
        # A plain pool still reports exactly what it did.
        plain = ServingBufferPool(FakeOperations(), 'mesh', users=1).describe()
        self.assertEqual(plain['bytes_per_slot'], draft)
        self.assertNotIn('verifier', plain['slots'][0])
        self.assertNotIn('bucket_rows', plain)

    def test_the_slot_lends_each_width_bucket_once_and_refuses_a_width_it_lacks(self):
        operations = FakeOperations()
        pool = verifier_pool(operations, bucket_rows=(1, 8, 8))
        slot = pool.acquire()
        verifier = slot.verifier
        first, second = verifier.take(8), verifier.take(8)
        self.assertIsNot(first, second)
        self.assertEqual((first.rows, second.rows), (8, 8))
        with self.assertRaisesRegex(ValueError, r'no free 8-row bucket: the slot holds widths \[1, 8, 8\]'):
            verifier.take(8)
        with self.assertRaisesRegex(ValueError, 'no free 16-row bucket'):
            verifier.take(16)
        self.assertEqual(verifier.take(1).rows, 1)
        # Returned and lent again, every bucket is free.
        slot.release()
        self.assertFalse(any(bucket.taken for bucket in verifier.buckets))
        verifier.take(8)
        self.assertIs(pool.acquire(), slot)
        self.assertFalse(any(bucket.taken for bucket in verifier.buckets))

    def test_a_loan_zeroes_the_tiled_buffers_and_leaves_the_staged_integer_inputs(self):
        operations = FakeOperations()
        pool = verifier_pool(operations, bucket_rows=(1, 4))
        slot = pool.acquire()
        filled = [value for value, zero in operations.fills]
        self.assertEqual(filled, list(slot.zeroed))
        self.assertTrue(all(zero == 0.0 for value, zero in operations.fills))
        verifier = slot.verifier
        for value in (slot.history, slot.spare_history, *bank_tensors(slot.kv), slot.query,
                      *snapshot_tensors(verifier.initial), *snapshot_tensors(verifier.carry)):
            self.assertTrue(any(value is entry for entry in filled))
        for bucket in verifier.buckets:
            for value in (*snapshot_tensors(bucket.checkpoints), *bucket.target_features, bucket.batch.cos, bucket.batch.sin):
                self.assertTrue(any(value is entry for entry in filled))
            # Every integer input is fully restaged by the fixture before any read - the
            # replay reader's page tables by the reader at construction.
            for value in (bucket.batch.tokens, bucket.batch.positions, bucket.batch.pages, bucket.batch.singleton_pages,
                          *bucket.batch.singleton_positions, *bucket.replay_tables()):
                self.assertFalse(any(value is entry for entry in filled))
                self.assertTrue(any(value is entry for entry in slot.tensors))

    def test_mtp_hidden_is_allocated_per_bucket_only_when_asked(self):
        operations = FakeOperations()
        pool = verifier_pool(operations, bucket_rows=(2, 16), mtp_hidden=True)
        for bucket in pool.slots[0].verifier.buckets:
            hidden = bucket.mtp_hidden
            self.assertEqual((hidden.shape, hidden.dtype, hidden.layout, hidden.mapper),
                             ((1, 1, bucket.rows, 5120), 'bf16', 'tile', ('replicate', 'mesh')))
            self.assertTrue(any(hidden is value for value in bucket.zeroed))
        self.assertEqual(pool.describe()['mtp_hidden'], True)
        self.assertEqual(pool.slots[0].bytes - verifier_pool(FakeOperations(), bucket_rows=(2, 16)).slots[0].bytes,
                         tensor_bytes((1, 1, 2, 5120)) + tensor_bytes((1, 1, 16, 5120)))

    def test_verifier_geometry_is_checked_before_anything_is_allocated(self):
        operations = FakeOperations()
        helpers = fake_helpers(operations)
        invalid = [dict(helpers=helpers[:47]), dict(helpers=[object()] * 48), dict(page_width=0), dict(page_width=68.0),
                   dict(bucket_rows=()), dict(bucket_rows=(1, 3)), dict(bucket_rows=(True,)), dict(feature_taps=-1),
                   dict(feature_taps=5.0), dict(rope=None), dict(mtp_hidden=1),
                   # Replay families: distinct, admitted, and within the page table (4608 needs 72 pages).
                   dict(replay_group_rows=6), dict(replay_group_rows=4.0), dict(replay_capacities=(4096, 4096)),
                   dict(replay_capacities=(4608,)), dict(replay_capacities=(4000,)), dict(replay_capacities=(4096.0,)),
                   dict(replay_capacities=(256,))]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                verifier_pool(operations, **overrides)
        # Geometry without the helpers that shape it is a mistake, not a plain pool.
        for options in (dict(page_width=68), dict(bucket_rows=(1,)), dict(feature_taps=5),
                        dict(rope=fake_rope(operations)), dict(mtp_hidden=True), dict(replay_group_rows=8),
                        dict(replay_capacities=(4096,))):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'without the GDN helpers'):
                ServingBufferPool(operations, 'mesh', users=1, **options)
        self.assertEqual(operations.live, [])

    def test_partial_verifier_allocation_is_freed_when_a_bucket_fails(self):
        operations = FakeOperations()
        working, calls = fake_rope(operations), []

        def rope(positions):
            if calls:
                raise RuntimeError('out of DRAM')
            calls.append(positions)
            return working(positions)

        with self.assertRaisesRegex(RuntimeError, 'out of DRAM'):
            verifier_pool(operations, bucket_rows=(1, 2), rope=rope)
        self.assertEqual(operations.deallocated, operations.live)
        self.assertGreater(len(operations.live), 2 + 4 * DRAFT_LAYERS + 1 + 2 * 5 * GDN_LAYERS)

    def test_close_frees_every_verifier_buffer_once(self):
        operations = FakeOperations()
        pool = verifier_pool(operations, users=2, bucket_rows=(1, 8))
        pool.close()
        self.assertEqual(len(operations.deallocated), len(operations.live))
        self.assertEqual(len({id(value) for value in operations.deallocated}), len(operations.live))

    def test_a_moved_verifier_buffer_is_refused_at_both_ends_of_the_loan(self):
        operations = FakeOperations()
        pool = verifier_pool(operations, bucket_rows=(1,))
        slot = pool.slots[0]
        moved = slot.verifier.buckets[0].batch.pages.shards[1]
        moved.address += 1
        with self.assertRaisesRegex(AssertionError, 'slot 0 moved'):
            pool.acquire()
        moved.address -= 1
        self.assertIs(pool.acquire(), slot)
        slot.verifier.carry[7][0].shards[0].address += 1
        with self.assertRaisesRegex(AssertionError, 'slot 0 moved'):
            slot.release()
        slot.verifier.carry[7][0].shards[0].address -= 1
        slot.release()


class ReplayPageTableTests(unittest.TestCase):
    """The replay reader's per-bundle page tables - the last per-request buffer that was
    static across steps and allocated after an earlier request's traces (runs 35492676194
    and 35493208438) - come from the slot: per replay-width bucket, one set per native
    chunk family the request can be captured in, shaped exactly as the reader shapes its own."""

    def test_replay_buckets_hold_one_table_set_per_family_the_page_table_can_hold(self):
        operations = FakeOperations()
        with patch.dict('os.environ', {}, clear=True):
            pool = verifier_pool(operations, bucket_rows=(4, 8, 16), page_width=72)
            families = family_capacities(page_width=72)
        self.assertEqual(families, (4096, 4352, 4608))
        self.assertEqual(pool.replay_capacities, families)
        for bucket in pool.slots[0].verifier.buckets:
            tables = bucket.batch.replay_pages
            if bucket.rows < 8:
                self.assertEqual(tables, {})
                self.assertEqual(bucket.replay_tables(), ())
                continue
            self.assertEqual(list(tables), list(families))
            for capacity, values in tables.items():
                self.assertEqual([value.shape for value in values],
                                 [(batches, capacity // 64) for batches in bundle_batches(bucket.rows, capacity)])
                for value in values:
                    self.assertEqual((value.dtype, value.layout, value.mapper), ('int32', 'row_major', ('replicate', 'mesh')))
            # In the slot's record, and a moved one is refused like any other pooled buffer.
            self.assertTrue(all(any(value is entry for entry in pool.slots[0].tensors) for value in bucket.replay_tables()))
        moved = pool.slots[0].verifier.buckets[2].batch.replay_pages[4352][1].shards[0]
        moved.address += 1
        with self.assertRaisesRegex(AssertionError, 'slot 0 moved'):
            pool.acquire()
        moved.address -= 1
        slot = pool.acquire()
        # Not zeroed on loan: the reader stages the request's table before any read.
        self.assertFalse(any(value is filled for value, zero in operations.fills
                             for bucket in slot.verifier.buckets for filled in bucket.replay_tables()))

    def test_the_families_and_grouping_can_be_given_explicitly_and_an_empty_set_pools_none(self):
        operations = FakeOperations()
        pool = verifier_pool(operations, bucket_rows=(8, 16, 32), replay_capacities=(4352,), replay_group_rows=8)
        self.assertEqual((pool.replay_capacities, pool.replay_group_rows), ((4352,), 8))
        for bucket in pool.slots[0].verifier.buckets:
            self.assertEqual(list(bucket.batch.replay_pages), [4352])
            self.assertEqual([value.shape for value in bucket.batch.replay_pages[4352]],
                             [(batches, 68) for batches in bundle_batches(bucket.rows, 4352, max_group_rows=8)])
        report = pool.describe()
        self.assertEqual((report['replay_capacities'], report['replay_group_rows']), ([4352], 8))
        self.assertEqual(report['replay_page_bytes_per_slot'],
                         4 * 68 * sum(sum(bundle_batches(rows, 4352, max_group_rows=8)) for rows in (8, 16, 32)))
        none = verifier_pool(FakeOperations(), bucket_rows=(8, 16), replay_capacities=())
        self.assertEqual([bucket.batch.replay_pages for bucket in none.slots[0].verifier.buckets], [{}, {}])
        self.assertEqual(none.describe()['replay_page_bytes_per_slot'], 0)
        self.assertEqual(none.slots[0].bytes, verifier_pool(FakeOperations(), bucket_rows=(8, 16)).slots[0].bytes
                         - 4 * sum(batches * columns for rows in (8, 16) for batches, columns in replay_shapes(rows)))

    def test_the_tables_are_counted_in_the_slot_bytes_and_freed_with_the_pool(self):
        operations = FakeOperations()
        pool = verifier_pool(operations, users=2, bucket_rows=(8, 16))
        self.assertEqual(pool.slots[0].bytes, 2 * tensor_bytes(HISTORY_SHAPE) + 4 * DRAFT_LAYERS * tensor_bytes(KV_SHAPE)
                         + verifier_slot_bytes((8, 16)))
        self.assertEqual(pool.describe()['replay_page_bytes_per_slot'],
                         4 * sum(batches * columns for rows in (8, 16) for batches, columns in replay_shapes(rows)))
        pool.close()
        self.assertEqual(len(operations.deallocated), len(operations.live))


class PackedReplayTableTests(unittest.TestCase):
    """The packed block's per-user replay page tables (M1b): allocated with the pool for
    every packed shape its slots can fill, per family, per user, per bundle of a
    rows_per_user-row reader; lent once to the block; never per slot, never zeroed."""

    def test_the_default_shapes_follow_the_slots_and_each_set_is_shaped_per_user_per_bundle(self):
        # One narrow bucket per slot: the pool's overlap check is quadratic in its tensors.
        with patch.dict('os.environ', {}, clear=True):
            # M1 (2, 16) from two slots, M3 (4, 16) from four; M2's (4, 8) only when named.
            for users, shapes in ((1, ()), (2, ((2, 16),)), (3, ((2, 16),)), (4, ((2, 16), (4, 16)))):
                operations = FakeOperations()
                pool = verifier_pool(operations, users=users, bucket_rows=(8,))
                self.assertEqual(pool.packed_shapes, shapes, users)
                self.assertEqual(sorted(pool.packed), sorted(shapes))
                for count, rows in shapes:
                    tables = pool.packed_replay(count, rows)
                    self.assertEqual((tables.users, tables.rows, tables.taken), (count, rows, False))
                    self.assertEqual(list(tables.replay_pages), list(REPLAY_CAPACITIES))
                    for capacity, per_user in tables.replay_pages.items():
                        self.assertEqual(len(per_user), count)
                        for user_tables in per_user:
                            self.assertEqual([value.shape for value in user_tables],
                                             [(batches, capacity // 64) for batches in bundle_batches(rows, capacity)])
                            for value in user_tables:
                                self.assertEqual((value.dtype, value.layout, value.mapper), ('int32', 'row_major', ('replicate', 'mesh')))
                    # In the pool's own record, on independent chip storage, and no two sets share a table.
                    self.assertTrue(all(any(value is owned for owned in pool.owned) for value in tables.tensors))
                    self.assertEqual(len(tables.tensors), count * sum(len(bundle_batches(rows, capacity)) for capacity in REPLAY_CAPACITIES))
                every = [value for group in pool.packed.values() for tables in group for value in tables.tensors]
                self.assertEqual(len({id(value) for value in every}), len(every))
                self.assertEqual(pool.packed_bytes, sum(4 * value.shape[0] * value.shape[1] for value in every))
                # Not in any slot: one block serves every slot, and the slot loan zeroes nothing of it.
                self.assertFalse(any(any(value is entry for entry in slot.tensors) for slot in pool.slots for value in every))

    def test_the_shapes_can_be_given_explicitly_and_a_set_is_lent_once(self):
        operations = FakeOperations()
        pool = verifier_pool(operations, users=2, packed_shapes=((4, 8),))
        self.assertEqual(pool.packed_shapes, ((4, 8),))
        with self.assertRaisesRegex(ValueError, r'shapes \[\(4, 8\)\]; \(2, 16\) was asked for'):
            pool.packed_replay(2, 16)
        tables = pool.packed_replay(4, 8)
        self.assertIs(tables.take(), tables)
        self.assertTrue(tables.taken)
        with self.assertRaisesRegex(ValueError, 'already lent'):
            tables.take()
        tables.release()
        self.assertIs(tables.take(), tables)
        none = verifier_pool(FakeOperations(), users=2, packed_shapes=())
        self.assertEqual((none.packed_shapes, none.packed, none.packed_bytes), ((), {}, 0))
        with self.assertRaisesRegex(ValueError, r'shapes \[\]'):
            none.packed_replay(2, 16)
        slot = pool.acquire()
        self.assertFalse(any(table is value for value, zero in operations.fills for table in tables.tensors))
        slot.release()
        pool.close()
        with self.assertRaisesRegex(ValueError, 'Closed'):
            pool.packed_replay(4, 8)

    def test_describe_and_close_cover_the_packed_sets(self):
        operations = FakeOperations()
        pool = verifier_pool(operations, users=2)
        report = pool.describe()
        self.assertEqual(report['packed_shapes'], [[2, 16]])
        tables = pool.packed_replay(2, 16)
        self.assertEqual(report['packed_replay_bytes'], pool.packed_bytes)
        self.assertEqual(report['packed_replay_bytes'], sum(4 * value.shape[0] * value.shape[1] for value in tables.tensors))
        (described,) = report['packed_replay']
        self.assertEqual((described['users'], described['rows'], described['taken'], described['bytes']),
                         (2, 16, False, pool.packed_bytes))
        self.assertEqual(described['replay_pages'],
                         {capacity: [[[shard.address for shard in value.shards] for value in user_tables] for user_tables in per_user]
                          for capacity, per_user in tables.replay_pages.items()})
        tables.take()
        self.assertTrue(pool.describe()['packed_replay'][0]['taken'])
        pool.close()
        self.assertEqual(len(operations.deallocated), len(operations.live))
        self.assertTrue(all(any(value is freed for freed in operations.deallocated) for value in tables.tensors))
        self.assertEqual(pool.packed, {})
        plain = ServingBufferPool(FakeOperations(), 'mesh', users=2)
        self.assertEqual((plain.packed_shapes, plain.packed), ((), {}))
        self.assertNotIn('packed_replay', plain.describe())

    def test_shapes_are_checked_before_anything_is_allocated(self):
        operations = FakeOperations()
        # (3, 16) and (8, 16) fill no legal block width (48 and 128 rows); 64 is the widest.
        for shapes in (((2, 16), (2, 16)), ((3, 12),), ((2, 4),), ((3, 16),), ((8, 16),), ((0, 16),), ((2.0, 16),),
                       ((2, 16, 32),), (2, 16)):
            with self.subTest(shapes=shapes), self.assertRaisesRegex(ValueError, 'Packed replay shapes'):
                verifier_pool(operations, users=4, packed_shapes=shapes)
        with self.assertRaisesRegex(ValueError, 'without the GDN helpers'):
            ServingBufferPool(operations, 'mesh', users=2, packed_shapes=((2, 16),))
        self.assertEqual(operations.live, [])

    def test_the_m3_shape_lends_four_sixteen_row_table_sets_per_family(self):
        """The 64-row block's readers: four users, each a 16-row reader's bundles, per family
        (the serving runtime names exactly this shape for four scheduler requests)."""
        with patch.dict('os.environ', {}, clear=True):
            operations = FakeOperations()
            pool = verifier_pool(operations, users=4, bucket_rows=(8,), packed_shapes=((4, 16),))
            self.assertEqual((pool.packed_shapes, sorted(pool.packed)), (((4, 16),), [(4, 16)]))
            tables = pool.packed_replay(4, 16)
            self.assertEqual((tables.users, tables.rows), (4, 16))
            self.assertEqual(list(tables.replay_pages), list(REPLAY_CAPACITIES))
            for capacity, per_user in tables.replay_pages.items():
                self.assertEqual(len(per_user), 4)
                for user_tables in per_user:
                    self.assertEqual([value.shape for value in user_tables],
                                     [(batches, capacity // 64) for batches in bundle_batches(16, capacity)])
            self.assertEqual(len(tables.tensors), 4 * sum(len(bundle_batches(16, capacity)) for capacity in REPLAY_CAPACITIES))
            self.assertEqual(len({id(value) for value in tables.tensors}), len(tables.tensors))
            self.assertEqual(pool.packed_bytes, sum(4 * value.shape[0] * value.shape[1] for value in tables.tensors))
            self.assertIs(tables.take(), tables)
            with self.assertRaisesRegex(ValueError, r'\(2, 16\) was asked for'):
                pool.packed_replay(2, 16)
            self.assertEqual(pool.describe()['packed_shapes'], [[4, 16]])
            # a pool of two slots cannot be asked for the four-user tables by default, but
            # can hold them when named (the block's construction checks the slot count)
            self.assertEqual(verifier_pool(FakeOperations(), users=2, bucket_rows=(8,)).packed_shapes, ((2, 16),))
            self.assertEqual(verifier_pool(FakeOperations(), users=2, bucket_rows=(8,), packed_shapes=((4, 16),)).packed_shapes,
                             ((4, 16),))

    def test_packed_replicas_lend_independent_sets_of_the_same_shape(self):
        """QWEN_FAST_FOUR_AS_TWO's pair of 32-row blocks: `packed_shapes` still names the
        (2, 16) shape once (a shape repeated there is still refused - see
        test_shapes_are_checked_before_anything_is_allocated), and the two independent
        sets come from `packed_replicas` instead. `packed_replay(2, 16)` lends the first
        untaken one each call, so each block's own PackedVerifierEngine.__init__ (which
        immediately `take()`s what it is given) ends up with its OWN set without either
        naming which."""
        operations = FakeOperations()
        pool = verifier_pool(operations, users=4, bucket_rows=(8,), packed_shapes=((2, 16),),
                             packed_replicas={(2, 16): 2})
        self.assertEqual((pool.packed_shapes, pool.packed_replicas), (((2, 16),), {(2, 16): 2}))
        self.assertEqual(len(pool.packed[(2, 16)]), 2)
        first = pool.packed_replay(2, 16)
        self.assertIs(first.take(), first)
        second = pool.packed_replay(2, 16)
        self.assertIsNot(second, first)
        self.assertIs(second.take(), second)
        # Independent chip storage: no table is shared between the two lent sets.
        every = [value for tables in (first, second) for value in tables.tensors]
        self.assertEqual(len({id(value) for value in every}), len(every))
        # Both taken: packed_replay still returns one of them (the single-replica case
        # always returned its one set regardless of taken state), and its own take() raises.
        third = pool.packed_replay(2, 16)
        self.assertIn(third, (first, second))
        with self.assertRaisesRegex(ValueError, 'already lent'):
            third.take()
        second.release()
        self.assertIs(pool.packed_replay(2, 16), second, 'the freed set comes back once released')
        report = pool.describe()
        self.assertEqual(len(report['packed_replay']), 2)
        pool.close()
        self.assertEqual(len(operations.deallocated), len(operations.live))

    def test_packed_replicas_default_to_one_and_reject_unknown_shapes_or_bad_counts(self):
        self.assertEqual(verifier_pool(FakeOperations(), users=2).packed_replicas, {})
        operations = FakeOperations()
        for replicas in ({(4, 16): 2}, {(2, 16): 0}, {(2, 16): 1.0}):
            with self.subTest(replicas=replicas), self.assertRaisesRegex(ValueError, 'Packed replay replica counts'):
                verifier_pool(operations, users=4, packed_shapes=((2, 16),), packed_replicas=replicas)
        with self.assertRaisesRegex(ValueError, 'without the GDN helpers'):
            ServingBufferPool(operations, 'mesh', users=2, packed_replicas={(2, 16): 2})
        # Every case raised before allocating anything.
        self.assertEqual(operations.live, [])


class DramStatisticsTests(unittest.TestCase):
    """The allocator's DRAM figures per chip for the [PINDIAG] dram lines, read through a
    pooled buffer: a diagnostic that never raises."""

    def view(self, allocated, free, largest, total=4138123648, banks=8):
        return SimpleNamespace(num_banks=banks, total_bytes_per_bank=total, total_bytes_allocated_per_bank=allocated,
                               total_bytes_free_per_bank=free, largest_contiguous_bytes_free_per_bank=largest)

    def operations(self, views):
        return SimpleNamespace(BufferType=SimpleNamespace(DRAM='dram'),
            get_device_tensors=lambda tensor: [SimpleNamespace(device=lambda name=name: name) for name in views],
            get_memory_view=Mock(side_effect=lambda device, kind: views[device] if kind == 'dram' else None))

    def test_the_figures_are_summed_over_the_banks_per_chip_and_formatted_for_the_log(self):
        from serving_buffer_pool import dram_line, dram_statistics, format_dram

        # run 35509307389's chip 0 as the allocator reported it, per bank
        views = {'d0': self.view(4111316352, 26807296, 712896), 'd1': self.view(4000000000, 138123648, 1000000)}
        report = dram_statistics(self.operations(views), 'tensor')
        self.assertEqual(report, [
            dict(chip=0, banks=8, allocated=8 * 4111316352, free=8 * 26807296, largest_free=8 * 712896, total=8 * 4138123648),
            dict(chip=1, banks=8, allocated=8 * 4000000000, free=8 * 138123648, largest_free=8 * 1000000, total=8 * 4138123648)])
        self.assertEqual(format_dram(report),
                         'chip0 allocated=32.89GB free=0.21GB largest_free=5.7MB of 33.10GB; '
                         'chip1 allocated=32.00GB free=1.10GB largest_free=8.0MB of 33.10GB')
        # a ttnn without the view, or one that refuses it, reports the reason instead of raising
        self.assertEqual(dram_statistics(SimpleNamespace(get_device_tensors=lambda tensor: [SimpleNamespace(device=lambda: 'd0')]), 'tensor'),
                         dict(unavailable="AttributeError: 'types.SimpleNamespace' object has no attribute 'get_memory_view'"))
        refusing = self.operations(views)
        refusing.get_memory_view = Mock(side_effect=RuntimeError('no allocator on this device'))
        self.assertEqual(format_dram(dram_statistics(refusing, 'tensor')), 'unavailable (RuntimeError: no allocator on this device)')
        self.assertEqual(dram_line(SimpleNamespace()), 'unavailable (pool without device statistics)')
        self.assertEqual(dram_line(SimpleNamespace(dram_statistics=Mock(side_effect=RuntimeError('boom')))), 'unavailable (RuntimeError: boom)')
        self.assertEqual(dram_line(SimpleNamespace(dram_statistics=Mock(return_value=report[:1]))),
                         'chip0 allocated=32.89GB free=0.21GB largest_free=5.7MB of 33.10GB')

    def test_the_pool_reads_the_chips_through_its_first_buffer_and_says_so_when_it_cannot(self):
        from serving_buffer_pool import dram_statistics as statistics

        operations = FakeOperations()
        pool = ServingBufferPool(operations, 'mesh', users=1)
        first = pool.owned[0]
        views = {'d0': self.view(1, 2, 3), 'd1': self.view(4, 5, 6)}
        shards = operations.get_device_tensors
        operations.BufferType = SimpleNamespace(DRAM='dram')
        operations.get_device_tensors = lambda tensor: [SimpleNamespace(device=lambda name=name: name) for name in views] if tensor is first else []
        operations.get_memory_view = lambda device, kind: views[device]
        self.assertEqual([chip['allocated'] for chip in pool.dram_statistics()], [8, 32])
        self.assertEqual(pool.dram_statistics(), statistics(operations, first))
        operations.get_device_tensors = shards
        pool.close()
        self.assertEqual(pool.dram_statistics(), dict(unavailable='no pooled buffer to read the chips through'))


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
            self.assertNotIn('query', plain.kv_history.options)
            plain.close()
            self.assertTrue(plain.kv_history.closed)

    def test_a_pool_with_verifier_storage_lends_its_query_to_the_draft_cache_too(self):
        serving = dict(cache_history=True, proposal_capture=True, defer_proposal_capture=True,
                       native_proposal_attention=True)
        operations = FakeOperations()
        pool = verifier_pool(operations, users=1)
        with patch('draft_kv_history.DraftKVHistory', FakeKVHistory):
            device = self.build(operations, pool, **serving)
            cache = device.kv_history
            self.assertIs(cache.options['storage'], pool.slots[0].kv)
            self.assertIs(cache.options['query'], pool.slots[0].query)
            device.close()
            self.assertFalse(pool.slots[0].lent)
            self.assertFalse(any(value is pool.slots[0].query for value in operations.deallocated))
            # A plain pool has no query to lend; the cache uploads its own as before.
            plain_pool = ServingBufferPool(operations, 'mesh', users=1)
            plain = self.build(operations, plain_pool, **serving)
            self.assertIs(plain.kv_history.options['storage'], plain_pool.slots[0].kv)
            self.assertNotIn('query', plain.kv_history.options)
            plain.close()


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
