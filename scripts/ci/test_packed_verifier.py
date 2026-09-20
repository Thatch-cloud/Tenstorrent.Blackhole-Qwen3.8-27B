"""One packed verify block serves every user: one trace per round, each user's rows staged
into its own segment, each user's taps at its own row offset, each user's commit through
its own prefix trace into its own carry - and every result indexed by the scheduler's
ENTRIES, never by pool slot or registry order.

The fixture is mocked the way test_target_packed_pages and test_gdn_packed_segments mock
theirs: a fake ttnn whose tensors carry their values and addresses, a fake ModelBatch
that records the pack it was built with and retains one record per GDN layer, and the
commit DMA preparation recorded rather than run. The fake ModelBatch builds the REAL
per-user replay reader (pooled_attention_replay.PackedReplayAttentionReader, M1b) over
the pool's lent tables, so what the block stages into each user's positions word and
bundle tables, and which tickets it refuses as outside the family, is the real thing."""

from contextlib import contextmanager
from itertools import count
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import packed_verifier
from packed_verifier import PackedFeatureTaps, PackedShape, PackedVerifierEngine, m1_shape, segment_rows, validate_shape
from pooled_attention_replay import bundle_batches
from serving_buffer_pool import PackedReplayTables
import verifier_engine
from verifier_pack import GDN_LAYERS

_addresses = count(0x1000)
PAGE_WIDTH = 68
TAPS = (5, 19, 33, 47, 61)
# The block's native chunk family at 68 pages: capture position 4096, family 4352; the
# pool holds the families the table can hold, 4096 and 4352.
FAMILY = 4352
FAMILIES = (4096, 4352)


class FakeShard:
    def __init__(self, tensor, address):
        self.tensor, self.address = tensor, address

    def buffer_address(self):
        return self.address


class FakeTTNN:
    """Device tensors carry their values, so what the block staged into which rows can be
    checked exactly; traces are numbered, so which one ran can be too."""

    uint32, int32, bfloat16, float32 = 'uint32', 'int32', 'bf16', 'f32'
    ROW_MAJOR_LAYOUT, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'row_major', 'tile', 'dram'

    def __init__(self):
        self.hosts, self.host_copies, self.executed, self.released, self.deallocated = [], [], [], [], []
        self.device_uploads, self.zeroed = [], []
        self.synchronized = 0
        self.SDPAProgramConfig = Mock(return_value='sdpa-config')

    def full_like(self, tensor, fill, optional_tensor=None):
        if optional_tensor is not tensor:
            raise AssertionError('In-place fill expected')
        self.zeroed.append(tensor)

    def allocate(self, shape, dtype='bf16', layout='tile', value=None, mapper=None):
        tensor = SimpleNamespace(shape=tuple(shape), dtype=dtype, layout=layout, value=value, mapper=mapper)
        tensor.shards = [FakeShard(tensor, next(_addresses)) for chip in range(2)]
        return tensor

    def from_torch(self, value, device=None, dtype=None, layout=None, memory_config=None, mesh_mapper=None):
        if device is None:
            host = SimpleNamespace(shape=tuple(value.shape), dtype=dtype, layout=layout, value=value.clone())
            self.hosts.append(host)
            return host
        tensor = self.allocate(value.shape, dtype, layout, value.clone(), mesh_mapper)
        self.device_uploads.append(tensor)
        return tensor

    def copy_host_to_device_tensor(self, host, destination):
        destination.value = host.value.clone()
        self.host_copies.append((host, destination))

    def copy(self, source, destination):
        destination.value = source.value

    def get_device_tensors(self, tensor):
        return tensor.shards

    def to_torch(self, shard):
        return shard.tensor.value

    def execute_trace(self, mesh, trace, cq_id=0, blocking=True):
        self.executed.append(trace)

    def release_trace(self, mesh, trace):
        self.released.append(trace)

    def deallocate(self, tensor):
        self.deallocated.append(tensor)

    def synchronize_device(self, mesh):
        self.synchronized += 1

    def ShardTensorToMesh(self, mesh, dim):
        return ('shard', dim)

    def ReplicateTensorToMesh(self, mesh):
        return 'replicate'


def tensor(ttnn, shape=(5,), dtype='bf16'):
    return ttnn.allocate(shape, dtype)


def helpers(ttnn):
    """The 48 shared layer helpers over fake native state; allocate hands out fresh snapshots."""
    made = []
    for index in range(GDN_LAYERS):
        gdn = SimpleNamespace(B=8, _stable_state=True, rec_state=tensor(ttnn, (8, 24, 128, 128)),
                              conv_states=[tensor(ttnn, (1, 8, 5120)) for tap in range(4)])
        made.append(SimpleNamespace(gdn=gdn, live=[gdn.rec_state, *gdn.conv_states],
                                    allocate=Mock(side_effect=lambda: [tensor(ttnn) for part in range(5)]),
                                    save=Mock(), restore=Mock()))
    return made


def snapshot_set(ttnn):
    return [[tensor(ttnn) for part in range(5)] for layer in range(GDN_LAYERS)]


def packed_tables(ttnn, users=2, rows=16, families=FAMILIES):
    """The pool's packed replay page tables for one shape (serving_buffer_pool.PackedReplayTables):
    per family, per user, one (batches, capacity // 64) table per bundle of a rows-row reader."""
    return PackedReplayTables(users, rows, {
        capacity: [[ttnn.allocate((batches, capacity // 64), 'int32', 'row_major', torch.zeros(batches, capacity // 64, dtype=torch.int32))
                    for batches in bundle_batches(rows, capacity)] for user in range(users)]
        for capacity in families})


def pool(ttnn, shared, users=2, page_width=PAGE_WIDTH, packed=None):
    slots = [SimpleNamespace(index=index, lent=False, verifier=SimpleNamespace(carry=snapshot_set(ttnn)))
             for index in range(users)]
    packed = {(2, 16): packed_tables(ttnn)} if packed is None else packed

    def packed_replay(count, rows):
        if (count, rows) not in packed:
            raise ValueError('The pool holds packed replay page tables for shapes %r' % sorted(packed))
        return packed[(count, rows)]

    return SimpleNamespace(closed=False, helpers=shared, page_width=page_width, slots=slots, packed=packed,
                           packed_replay=packed_replay)


def weights():
    return SimpleNamespace(closed=False, tensors=['weight'], lend=Mock())


def model():
    return SimpleNamespace(mesh_device=SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10)),
                           layers=[object()] * 64,
                           args=SimpleNamespace(rope_head_dim=64, rope_theta=1e6, vocab_size=100, max_seq_len=65536))


class FakeRetained:
    """gdn_records.RetainedGDNBlock as the block uses it: records of (state, result, carries),
    segment_layers built from them, commit_user publishing nothing at prefix 0."""

    def __init__(self, rows):
        self.rows, self.records, self.closed = rows, [], False
        self.commit_user = Mock(side_effect=self.commit)
        self.replay = Mock(side_effect=lambda operation: operation())
        self.commits = []

    def segment_layers(self, segment):
        return [[*state.segment_entries[segment], piece['states'], *piece['packed_conv_states'],
                 state.gdn.rec_state, *state.gdn.conv_states, *carries[segment]]
                for state, result, carries in self.records for piece in (result['segment_results'][segment],)]

    def commit(self, segment, prefix, *, dma, synchronize, publication):
        self.commits.append((segment, prefix))
        if prefix:
            publication(prefix)

    def close(self):
        self.closed = True


class FakeModelBatch:
    """Records the pack it was built with; run() retains one record per GDN layer whose
    per-segment results are named after the segment, so commit layers can be checked."""

    ttnn = None
    instances = []

    def __init__(self, model, tokens, start, pages, helpers, checkpoints, prefix, **options):
        ttnn = type(self).ttnn
        self.rows, self.start, self.positional_pages, self.helpers = len(tokens), start, pages, helpers
        self.positional_checkpoints, self.prefix, self.options = checkpoints, prefix, options
        self.pack = options.get('pack')
        rows, page_width = self.rows, pages.shape[1]

        def integers(shape, dtype='int32'):
            return ttnn.allocate(shape, dtype, 'row_major', torch.zeros(shape, dtype=torch.int32))

        self.tokens, self.positions = integers((rows, 1), 'uint32'), integers((rows,))
        self.pages, self.singleton_pages = integers((rows, page_width)), integers((1, page_width))
        self.singleton_positions = [integers((1,)) for row in range(rows)]
        self.row_pages = [integers((1, page_width)) for row in range(rows)]
        self.cos = ttnn.allocate((1, rows, 1, 64), 'bf16', 'tile', torch.zeros(1, rows, 1, 64))
        self.sin = ttnn.allocate((1, rows, 1, 64), 'bf16', 'tile', torch.zeros(1, rows, 1, 64))
        self.retained = FakeRetained(rows) if options.get('retain_records') else None
        # The real per-user replay reader over the block's lent tables, as model_batch
        # builds it packed: one pooled reader per segment in the capture's family.
        self.replay_reader, self.readers, self.grouped_readers = None, [], []
        if options.get('attention_replay'):
            from pooled_attention_replay import PackedReplayAttentionReader
            from target_packed_pages import segments

            spans, total = segments(self.pack)
            self.replay_capacity = (start // 256 + 1) * 256
            self.replay_reader = PackedReplayAttentionReader(ttnn, model.mesh_device, spans, self.replay_capacity,
                [user['pages'] for user in self.pack], self.upload_replay, storage=options.get('packed_replay_pages'),
                max_group_rows=options.get('replay_group_rows', 4))
            self.grouped_readers.append(self.replay_reader)
            self.readers = [self.replay_reader] * 16
        self.run = Mock(side_effect=self.forward)
        self.close = Mock(side_effect=self.release)
        type(self).instances.append(self)

    def upload_replay(self, value, dtype='bf16'):
        return type(self).ttnn.allocate(value.shape, dtype, 'row_major' if dtype == 'int32' else 'tile', value.clone())

    def release(self):
        for reader in self.grouped_readers:
            reader.close()

    def forward(self, *, sharded_logits):
        ttnn = type(self).ttnn
        if self.retained is not None and not self.retained.records:
            users = len(self.pack)
            for layer, helper in enumerate(self.helpers):
                pieces = tuple(dict(states=SimpleNamespace(name='states%d.%d' % (user, layer)),
                                    packed_conv_states=[SimpleNamespace(name='conv%d.%d.%d' % (user, layer, tap)) for tap in range(4)])
                               for user in range(users))
                state = SimpleNamespace(gdn=helper.gdn,
                                        segment_entries=tuple([SimpleNamespace(name='entry%d.%d' % (user, layer))] * 5
                                                              for user in range(users)))
                # model_batch appends each user's carry for this layer as the record's checkpoint
                carries = tuple(user['slots'][layer] for user in self.pack)
                self.retained.records.append((state, dict(segment_results=pieces, segments=((0, 16), (16, 32))), carries))
        return ttnn.allocate((1, 1, self.rows, 124160), 'bf16', 'tile')


class FakeFeatures:
    instances = []

    def __init__(self, model, tap_ids, destinations, *, copy, storage_ids):
        self.tap_ids, self.destinations = tuple(tap_ids), tuple(destinations)
        self.captures, self.closed = 0, False
        type(self).instances.append(self)

    @contextmanager
    def capture(self):
        self.captures += 1
        yield self

    def outputs(self):
        return self.destinations

    def close(self):
        self.closed = True


def request(request_id, slot, position, page_value, page_width=PAGE_WIDTH):
    """A request whose engine borrowed `slot`'s carry, the way VerifierEngine.allocate_carry does."""
    session = SimpleNamespace(request_id=request_id, phase='pending', pending=None, check_ticket=Mock(),
                              fail_verification=Mock())
    engine = SimpleNamespace(carry=[list(snapshot) for snapshot in slot.verifier.carry], position=position,
                             pages=torch.full((1, page_width), page_value, dtype=torch.int32),
                             phase='idle', pending=None, session=session)
    return SimpleNamespace(engine=engine, session=session)


def entry(owner, tokens):
    ticket = SimpleNamespace(request_id=owner.session.request_id, position=owner.engine.position, tokens=tuple(tokens))
    owner.session.pending, owner.session.phase = ticket, 'pending'
    return dict(request_id=owner.session.request_id, request=owner, ticket=ticket)


class BlockFixture(unittest.TestCase):
    def setUp(self):
        verifier_engine.note_prefill()
        self.ttnn = FakeTTNN()
        FakeModelBatch.ttnn = self.ttnn
        FakeModelBatch.instances, FakeFeatures.instances = [], []
        self.helpers = helpers(self.ttnn)
        self.pool, self.weights, self.model = pool(self.ttnn, self.helpers), weights(), model()
        self.prepared, self.traces = [], count(1)
        self.ids = self.ttnn.allocate((32,), 'uint32', 'row_major', torch.arange(1000, 1032, dtype=torch.int32))

        def prepare(mesh, layers, prefix):
            self.prepared.append((layers, prefix))
            return Mock(name='publication')

        def capture_operation(operations, mesh, operation):
            return 'trace%d' % next(self.traces), operation()

        for target, value in (('ModelBatch', FakeModelBatch), ('PreparedTargetFeatures', FakeFeatures),
                              ('prepare', Mock(side_effect=prepare)),
                              ('capture_operation', Mock(side_effect=capture_operation)),
                              ('sample_rows', Mock(return_value=self.ids))):
            patcher = patch.object(packed_verifier, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # The pinned reader's mask program needs the real ttnn; the readers are otherwise real.
        patcher = patch('attention_replay.prepare', return_value='mask-program')
        patcher.start()
        self.addCleanup(patcher.stop)

    def build(self, **options):
        return PackedVerifierEngine(self.ttnn, self.model, self.helpers, 'sampler', pool=self.pool,
            shared_weights=self.weights, shape=m1_shape(PAGE_WIDTH), feature_taps=TAPS, **options)

    def two(self, reversed_order=True, rows=16):
        """Two admitted requests: A in pool slot 0, B in pool slot 1, presented B first
        (probe 35436807668 saw the scheduler present the pair as ['B', 'A']). Both inside
        the block's native chunk family [4096, 4352), as the serving pin keeps them."""
        first = request('A', self.pool.slots[0], 4100, 7)
        second = request('B', self.pool.slots[1], 4200, 11)
        entries = [entry(second, range(50, 50 + rows)), entry(first, range(10, 10 + rows))]
        return entries if reversed_order else entries[::-1]


class ConstructionTests(BlockFixture):
    def test_the_block_is_the_retained_per_user_replay_pack_over_the_pool_carries(self):
        block = self.build()
        self.assertEqual(block.phase, 'idle')
        self.assertEqual(block.capture_position, PAGE_WIDTH * 64 - 256)
        # the initial snapshot, two users x 48 checkpoint sets and the five 32-row taps,
        # allocated before any capture
        self.assertEqual([helper.allocate.call_count for helper in self.helpers], [3] * GDN_LAYERS)
        self.assertEqual([len(checkpoints) for checkpoints in block.checkpoints], [GDN_LAYERS, GDN_LAYERS])
        # slot 0 as attach found it was saved first and put back after the commit warming,
        # and every pooled carry the warming wrote was zeroed again
        for helper, snapshot in zip(self.helpers, block.initial, strict=True):
            helper.save.assert_called_once_with(snapshot)
            helper.restore.assert_called_once_with(snapshot)
        pooled = [value for slot in self.pool.slots for snapshot in slot.verifier.carry for value in snapshot]
        self.assertEqual(len(self.ttnn.zeroed), len(pooled))
        self.assertTrue(all(any(value is entry for entry in self.ttnn.zeroed) for value in pooled))
        self.assertIs(self.ttnn.zeroed[0], pooled[0])
        self.assertEqual([(tap.shape, tap.dtype, tap.layout, tap.mapper) for tap in block.taps],
                         [((1, 1, 32, 5120), 'bf16', 'tile', ('shard', 3))] * 5)
        (features,) = FakeFeatures.instances
        self.assertEqual((features.tap_ids, features.destinations), (TAPS, tuple(block.taps)))
        # a warm fixture, run once and closed, then the captured one
        warm, captured = FakeModelBatch.instances
        warm.run.assert_called_once_with(sharded_logits=True)
        warm.close.assert_called_once()
        captured.run.assert_called_once_with(sharded_logits=True)
        captured.close.assert_not_called()
        self.assertIs(block.fixture, captured)
        self.assertEqual(features.captures, 2)
        for fixture in (warm, captured):
            options = fixture.options
            self.assertEqual((fixture.rows, fixture.start, fixture.prefix), (32, block.capture_position, 32))
            self.assertIs(fixture.positional_checkpoints, block.checkpoints[0])
            self.assertTrue(options['retain_records'])
            # the attention is one bundled replay reader per user (M1b), four-row groups,
            # masks refreshed per layer in-trace, over the pool's lent table sets
            self.assertTrue(options['attention_replay'])
            self.assertFalse(options['attention_mask_once'])
            self.assertEqual(options['replay_group_rows'], 4)
            self.assertIs(options['packed_replay_pages'], block.replay_tables)
            # commit-only is the deferred packed decode: per-user entries, decided by commit_user
            self.assertTrue(options['commit_only_gdn'])
            self.assertTrue(all(options[name] for name in ('serial_sdpa', 'ordered_cache', 'norm_batch', 'device_loop_gdn',
                                                            'packed_checkpoints', 'batch_conv')))
            self.assertEqual([user['rows'] for user in fixture.pack], [16, 16])
            self.assertEqual([user['prefix'] for user in fixture.pack], [0, 0])
            for user, participant in enumerate(fixture.pack):
                self.assertEqual(list(participant['checkpoints']), block.checkpoints[user])
                lent = self.pool.slots[user].verifier.carry
                self.assertTrue(all(a is b for mine, theirs in zip(participant['slots'], lent, strict=True)
                                    for a, b in zip(mine, theirs, strict=True)))
        # the block took the pool's table set for its shape, in its family (the capture
        # position's); the captured fixture's readers are one per user, each over that
        # user's tables of that family and at the family start, each with its own word
        tables = self.pool.packed[(2, 16)]
        self.assertTrue(tables.taken)
        self.assertIs(block.replay, tables)
        self.assertEqual(block.replay_capacity, FAMILY)
        self.assertEqual([[table is lent for table, lent in zip(mine, theirs, strict=True)]
                          for mine, theirs in zip(block.replay_tables, tables.replay_pages[FAMILY], strict=True)],
                         [[True, True], [True, True]])
        reader = captured.replay_reader
        self.assertEqual((reader.rows, reader.capacity, reader.starts, len(reader.readers)), (32, FAMILY, (4096, 4096), 2))
        self.assertIsNot(reader.readers[0].positions, reader.readers[1].positions)
        for user, own in enumerate(reader.readers):
            self.assertEqual([entry[1] for entry in own.metadata], tables.replay_pages[FAMILY][user])
            self.assertEqual([tuple(entry[1].shape) for entry in own.metadata], [(3, 68), (1, 68)])
            self.assertEqual(own.positions.value.tolist(), [4096, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(warm.replay_reader.closed)
        self.assertFalse(reader.closed)
        self.assertEqual(block.describe()['attention'],
                         dict(reader='per-user bundled replay', family=FAMILY, replay_group_rows=4, bundles_per_user=[2, 2],
                              tables=[[[shard.address for shard in table.shards] for table in user] for user in block.replay_tables]))
        # one verify trace, then users x rows_per_user commit traces: prefix 0 runs nothing
        self.assertEqual(block.trace, 'trace1')
        self.assertEqual([sorted(commits) for commits in block.commits], [list(range(1, 17))] * 2)
        self.assertEqual([prefix for layers, prefix in self.prepared], [*range(1, 17)] * 2)
        for index, (layers, prefix) in enumerate(self.prepared):
            user = index // 16
            self.assertEqual(len(layers), GDN_LAYERS)
            for layer, (record, helper, slot) in enumerate(zip(layers, self.helpers, self.pool.slots[user].verifier.carry, strict=True)):
                self.assertEqual(len(record), 20)
                self.assertEqual([value.name for value in record[:5]], ['entry%d.%d' % (user, layer)] * 5)
                self.assertEqual(record[5].name, 'states%d.%d' % (user, layer))
                self.assertEqual([value.name for value in record[6:10]], ['conv%d.%d.%d' % (user, layer, tap) for tap in range(4)])
                self.assertEqual(record[10:15], [helper.gdn.rec_state, *helper.gdn.conv_states])
                self.assertTrue(all(a is b for a, b in zip(record[15:], slot, strict=True)),
                                'the commit writes the accepted state straight into the user carry')
        self.assertEqual(block.describe()['commit_traces'], 32)
        self.assertIsNone(verifier_engine._resident)

    def test_commit_layers_that_do_not_end_in_the_users_carry_are_refused(self):
        from packed_verifier import validate_commit_layers

        carry = snapshot_set(self.ttnn)
        layers = [[*(tensor(self.ttnn) for part in range(15)), *snapshot] for snapshot in carry]
        self.assertEqual(len(validate_commit_layers(layers, carry)), GDN_LAYERS)
        for broken in (layers[:47], [layer[:19] for layer in layers],
                       [[*layer[:15], *snapshot] for layer, snapshot in zip(layers, snapshot_set(self.ttnn))]):
            with self.assertRaises(ValueError):
                validate_commit_layers(broken, carry)

    def test_construction_is_refused_once_a_request_could_exist_and_before_anything_is_allocated(self):
        cases = {}
        lent = pool(self.ttnn, self.helpers)
        lent.slots[1].lent = True
        cases['a lent pool slot'] = dict(pool=lent)
        closed = pool(self.ttnn, self.helpers)
        closed.closed = True
        cases['a closed pool'] = dict(pool=closed)
        cases['a pool without verifier storage'] = dict(pool=SimpleNamespace(closed=False, helpers=None, page_width=PAGE_WIDTH, slots=[]))
        cases['a narrower pool'] = dict(pool=pool(self.ttnn, self.helpers, page_width=67))
        cases['a one-user pool'] = dict(pool=pool(self.ttnn, self.helpers, users=1))
        stale = weights()
        stale.closed = True
        cases['closed draft weights'] = dict(shared_weights=stale)
        cases['draft weights not yet uploaded'] = dict(shared_weights=SimpleNamespace(closed=False, tensors=[], lend=Mock()))
        cases['no sampler'] = dict(sampler=None)
        cases['a capture position past the pages'] = dict(capture_position=PAGE_WIDTH * 64 - 15)
        # the readers' tables: the pool must hold a set for the shape, in the block's family, unlent
        cases['a pool without packed replay tables for the shape'] = dict(pool=pool(self.ttnn, self.helpers, packed={}))
        cases['packed replay tables lacking the block family'] = dict(
            pool=pool(self.ttnn, self.helpers, packed={(2, 16): packed_tables(self.ttnn, families=(4096,))}))
        lent = pool(self.ttnn, self.helpers)
        lent.packed[(2, 16)].take()
        cases['packed replay tables already lent'] = dict(pool=lent)
        cases['a capture position outside every family'] = dict(capture_position=4090)
        for name, overrides in cases.items():
            with self.subTest(name=name):
                options = dict(pool=self.pool, shared_weights=self.weights, sampler='sampler', capture_position=None)
                options.update(overrides)
                sampler = options.pop('sampler')
                with self.assertRaises(ValueError):
                    PackedVerifierEngine(self.ttnn, self.model, self.helpers, sampler, shape=m1_shape(PAGE_WIDTH),
                                         feature_taps=TAPS, **options)
        verifier_engine._resident = object()
        with self.assertRaisesRegex(ValueError, 'resident'):
            self.build()
        verifier_engine.note_prefill()
        for helper in self.helpers:
            helper.allocate.assert_not_called()
        self.assertEqual((self.ttnn.device_uploads, FakeModelBatch.instances, self.prepared), ([], [], []))
        self.assertFalse(self.pool.packed[(2, 16)].taken, 'nothing refused took the pool set')

    def test_shapes_are_keyed_on_the_whole_block(self):
        self.assertEqual(m1_shape(PAGE_WIDTH), PackedShape(2, 16, 32, PAGE_WIDTH, PAGE_WIDTH * 64))
        validate_shape(PackedShape(4, 8, 32, 512, 512 * 64))
        for broken in (PackedShape(2, 16, 32, PAGE_WIDTH, PAGE_WIDTH * 64 + 1), PackedShape(3, 16, 32, 68, 68 * 64),
                       PackedShape(2, 16, 64, 68, 68 * 64), PackedShape(2, 16, 32, 67, 67 * 64), (2, 16, 32, 68, 68 * 64)):
            with self.assertRaises(ValueError):
                validate_shape(broken)
        self.assertEqual([segment_rows(m1_shape(PAGE_WIDTH), user) for user in (0, 1)], [(0, 16), (16, 32)])
        with self.assertRaises(ValueError):
            segment_rows(m1_shape(PAGE_WIDTH), 2)


class RoundTests(BlockFixture):
    def test_one_trace_serves_both_users_and_every_result_follows_the_entries(self):
        block = self.build()
        for helper in self.helpers:
            helper.restore.reset_mock()
            helper.save.reset_mock()
        entries = self.two()
        executed, synchronized = len(self.ttnn.executed), self.ttnn.synchronized
        copies = len(self.ttnn.host_copies)
        predictions, metrics = block.verify(entries)
        # B (pool slot 1, entry 0) took segment 1; A (pool slot 0, entry 1) took segment 0
        self.assertEqual(metrics['segments'], (1, 0))
        self.assertEqual(predictions, [list(range(1016, 1032)), list(range(1000, 1016))])
        fixture = block.fixture
        self.assertEqual(fixture.tokens.value[16:32, 0].tolist(), list(range(50, 66)))
        self.assertEqual(fixture.tokens.value[:16, 0].tolist(), list(range(10, 26)))
        self.assertEqual(fixture.positions.value.tolist(), [*range(4100, 4116), *range(4200, 4216)])
        self.assertTrue(bool((fixture.pages.value[:16] == 7).all()))
        self.assertTrue(bool((fixture.pages.value[16:] == 11).all()))
        self.assertEqual([table.value[0, 0].item() for table in fixture.row_pages], [7] * 16 + [11] * 16)
        self.assertEqual([position.value.item() for position in fixture.singleton_positions],
                         [*range(4100, 4116), *range(4200, 4216)])
        self.assertEqual((fixture.cos.value.shape, fixture.sin.value.shape), ((1, 32, 1, 64), (1, 32, 1, 64)))
        # and each user's own reader: its positions word at that user's start, its per-bundle
        # tables holding that user's page table - B's pages in segment 1's reader, A's in segment 0's
        reader = fixture.replay_reader
        self.assertEqual(reader.starts, (4100, 4200))
        self.assertEqual([own.positions.value.tolist() for own in reader.readers],
                         [[4100, 0, 0, 0, 0, 0, 0, 0], [4200, 0, 0, 0, 0, 0, 0, 0]])
        for own, page in zip(reader.readers, (7, 11), strict=True):
            for bundle, table, mask, config in own.metadata:
                self.assertEqual(tuple(table.value.shape), (len(bundle), 68))
                self.assertTrue(bool((table.value == page).all()))
        self.assertFalse(reader.failed)
        # tokens, positions, cos, sin, pages, 32 singleton positions, 32 row tables, two positions
        # words and two users x two bundle tables; one fence for all of them
        self.assertEqual(len(self.ttnn.host_copies) - copies, 75)
        self.assertEqual(metrics['staged_buffers'], 75)
        # exactly one weight pass: the verify trace once, eagerly then fenced on the first round
        self.assertEqual(self.ttnn.executed[executed:], ['trace1'])
        self.assertEqual(self.ttnn.synchronized - synchronized, 2)
        fixture.retained.replay.assert_not_called()
        # and no host carry copies at all: the segment restores are inside the trace
        for helper in self.helpers:
            helper.restore.assert_not_called()
            helper.save.assert_not_called()
        for item in entries:
            item['request'].session.check_ticket.assert_called_once_with(item['request_id'], item['ticket'])
        self.assertEqual((block.phase, block.pending_segments), ('verified', {0, 1}))
        self.assertIsNone(verifier_engine._resident)
        # the next round replays through the retained block, again once
        block.commit_user(1, 9)
        block.commit_user(0, 16)
        self.assertEqual(block.phase, 'idle')
        entries = self.two(reversed_order=False)
        predictions, metrics = block.verify(entries)
        self.assertEqual(metrics['segments'], (0, 1))
        self.assertEqual(predictions, [list(range(1000, 1016)), list(range(1016, 1032))])
        fixture.retained.replay.assert_called_once()
        self.assertEqual(self.ttnn.executed[executed:], ['trace1', block.commits[1][9], block.commits[0][16], 'trace1'])
        self.assertEqual(block.rounds, 2)

    def test_taps_come_at_the_users_row_offset_only_while_its_segment_is_verified(self):
        block = self.build()
        with self.assertRaises(ValueError):
            block.features(0)
        block.verify(self.two())
        for segment in (0, 1):
            taps = block.features(segment)
            self.assertIsInstance(taps, PackedFeatureTaps)
            self.assertEqual(tuple(taps), tuple(block.taps))
            self.assertEqual((taps.row_offset, taps.rows), (16 * segment, 16))
        with self.assertRaises(ValueError):
            block.features(2)
        block.commit_user(1, 4)
        with self.assertRaises(ValueError):
            block.features(1)
        self.assertEqual(block.features(0).row_offset, 0)

    def test_each_user_commits_through_its_own_prefix_trace_and_prefix_zero_runs_nothing(self):
        block = self.build()
        block.verify(self.two())
        retained = block.fixture.retained
        executed = len(self.ttnn.executed)
        block.commit_user(1, 9)
        self.assertEqual(self.ttnn.executed[executed:], [block.commits[1][9]])
        self.assertEqual(retained.commits, [(1, 9)])
        call = retained.commit_user.call_args
        # not the last user: the trace blocked the host already, no fence yet
        self.assertEqual((call.args, call.kwargs['dma'], call.kwargs['synchronize']), ((1, 9), True, False))
        self.assertTrue(callable(call.kwargs['publication']))
        self.assertEqual((block.phase, block.pending_segments), ('verified', {0}))
        with self.assertRaises(ValueError):
            block.commit_user(1, 9)
        with self.assertRaises(ValueError):
            block.commit_user(0, 17)
        block.commit_user(0, 0)
        self.assertEqual(self.ttnn.executed[executed:], [block.commits[1][9]], 'prefix 0 executes no trace')
        self.assertEqual(retained.commits, [(1, 9), (0, 0)])
        # the last user's commit carries the one fence that arms the next round's replay
        self.assertTrue(retained.commit_user.call_args.kwargs['synchronize'])
        self.assertEqual((block.phase, block.pending_segments), ('idle', set()))
        self.assertIsNone(verifier_engine._resident)
        with self.assertRaises(ValueError):
            block.commit_user(0, 1)

    def test_a_partial_pack_a_narrow_ticket_or_a_foreign_engine_is_refused_before_the_trace(self):
        block = self.build()
        executed, copies = len(self.ttnn.executed), len(self.ttnn.host_copies)
        cases = {}
        cases['one entry'] = self.two()[:1]
        cases['a narrow ticket'] = [self.two()[0], entry(request('A', self.pool.slots[0], 100, 7), range(8))]
        stranger = request('C', SimpleNamespace(verifier=SimpleNamespace(carry=snapshot_set(self.ttnn))), 7, 3)
        cases['an engine outside the pool'] = [self.two()[0], entry(stranger, range(16))]
        twice = request('C', self.pool.slots[1], 7, 3)
        cases['two entries through one slot'] = [self.two()[0], entry(twice, range(16))]
        moved = self.two()
        moved[0]['request'].engine.position += 1
        cases['a ticket off the frontier'] = moved
        busy = self.two()
        busy[1]['request'].engine.phase = 'verifying'
        cases['a busy engine'] = busy
        mismatched = self.two()
        mismatched[0]['request_id'] = 'Z'
        cases['a ticket of another request'] = mismatched
        # each user's reader is captured in the block's family [4096, 4352): a ticket
        # outside it has no trace here, and is refused before anything is staged
        cases['a ticket past the block family'] = [self.two()[0], entry(request('A', self.pool.slots[0], 4340, 7), range(16))]
        cases['a ticket below the block family'] = [self.two()[0], entry(request('A', self.pool.slots[0], 4000, 7), range(16))]
        for name, entries in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                block.verify(entries)
            self.assertEqual(block.phase, 'idle', name)
        with self.assertRaisesRegex(ValueError, r'request A at 4340 leaves the block.s native chunk family \[4096, 4352\)'):
            block.verify(cases['a ticket past the block family'])
        self.assertEqual((len(self.ttnn.executed), len(self.ttnn.host_copies)), (executed, copies))
        self.assertFalse(block.fixture.replay_reader.failed)
        with self.assertRaises(ValueError):
            block.segment_of(SimpleNamespace(carry=[]))

    def test_a_failed_round_fails_every_session_and_a_moved_buffer_is_caught(self):
        block = self.build()
        entries = self.two()
        original = self.ttnn.execute_trace
        self.ttnn.execute_trace = Mock(side_effect=RuntimeError('device failure'))
        with self.assertRaises(RuntimeError):
            block.verify(entries)
        self.assertEqual(block.phase, 'failed')
        for item in entries:
            item['request'].session.fail_verification.assert_called_once_with(item['request_id'], item['ticket'])
        self.assertIsNone(verifier_engine._resident)
        with self.assertRaises(ValueError):
            block.verify(entries)
        self.ttnn.execute_trace = original
        block.phase = 'idle'
        # a carried state moved under the block: refused before anything is staged or run
        copies, executed = len(self.ttnn.host_copies), len(self.ttnn.executed)
        self.pool.slots[0].verifier.carry[5][2].shards[0].address = -1
        with self.assertRaisesRegex(ValueError, 'moved'):
            block.verify(self.two())
        self.assertEqual((block.phase, len(self.ttnn.host_copies), len(self.ttnn.executed)), ('failed', copies, executed))
        self.pool.slots[0].verifier.carry[5][2].shards[0].address = block.carry_addresses[0][5][2][0]
        block.phase = 'idle'
        # a pooled replay page table moved under the block: refused the same way
        table = block.replay_tables[1][0]
        table.shards[1].address = -3
        with self.assertRaisesRegex(ValueError, 'replay page table moved'):
            block.verify(self.two())
        self.assertEqual((block.phase, len(self.ttnn.host_copies), len(self.ttnn.executed)), ('failed', copies, executed))
        table.shards[1].address = block.replay_addresses[1][0][1]
        block.phase = 'idle'
        # an input buffer replaced while staging: caught after the fence, before the trace,
        # and the readers are poisoned with it
        stage = self.ttnn.copy_host_to_device_tensor

        def replace(host, destination):
            stage(host, destination)
            if destination is block.fixture.row_pages[3]:
                destination.shards[1].address = -2

        self.ttnn.copy_host_to_device_tensor = replace
        with self.assertRaisesRegex(AssertionError, 'replaced'):
            block.verify(self.two())
        self.assertEqual((block.phase, len(self.ttnn.executed)), ('failed', executed))
        self.assertTrue(all(own.failed for own in block.fixture.replay_reader.readers))

    def test_close_releases_every_trace_and_owned_buffer_but_never_a_pooled_carry(self):
        block = self.build()
        block.verify(self.two())
        with self.assertRaises(ValueError):
            block.close()
        block.commit_user(0, 3)
        block.commit_user(1, 0)
        owned = [*block.taps, *(value for checkpoints in block.checkpoints for snapshot in checkpoints for value in snapshot),
                 *(value for snapshot in block.initial for value in snapshot), *block.output]
        fixture = block.fixture
        # the readers' own uploads (each user's positions word and masks) go with the fixture;
        # their page tables are the pool's, handed back
        uploads = [value for own in fixture.replay_reader.readers for value in own.owned]
        self.assertEqual(len(uploads), 6)
        tables = self.pool.packed[(2, 16)]
        block.close()
        self.assertEqual(block.phase, 'closed')
        # the verify trace and the 32 commit traces, each once
        self.assertEqual(sorted(self.ttnn.released), sorted('trace%d' % index for index in range(1, 34)))
        fixture.close.assert_called_once()
        self.assertTrue(fixture.replay_reader.closed)
        self.assertTrue(FakeFeatures.instances[0].closed)
        freed = self.ttnn.deallocated
        self.assertTrue(all(any(value is entry for entry in freed) for value in owned))
        self.assertTrue(all(any(value is entry for entry in freed) for value in uploads))
        pooled = [value for slot in self.pool.slots for snapshot in slot.verifier.carry for value in snapshot]
        self.assertFalse(any(any(value is entry for entry in freed) for value in pooled))
        self.assertFalse(any(any(value is entry for entry in freed) for value in tables.tensors))
        self.assertFalse(tables.taken)
        self.assertEqual((block.taps, block.checkpoints, block.initial, block.carries, block.trace), ([], [], [], [], None))
        self.assertEqual((block.replay, block.replay_tables, block.replay_addresses), (None, [], []))
        block.close()


class ProjectionOffsetTests(unittest.TestCase):
    """DFlashDevice.project_features slices a packed block's taps at the user's row offset."""

    def device(self):
        from dflash_device import DFlashDevice

        made, slices = [], []

        def stub(shape):
            value = SimpleNamespace(shape=tuple(shape), dtype='bf16')
            value.shards = [FakeShard(value, next(_addresses)) for chip in range(2)]
            made.append(value)
            return value

        operations = SimpleNamespace(bfloat16='bf16', float32='f32', DRAM_MEMORY_CONFIG='dram',
            get_device_tensors=lambda value: value.shards,
            MatmulMultiCoreReuseMultiCast1DProgramConfig=lambda **options: 'program',
            pad=lambda value, padding, fill: stub((1, 1, 32, value.shape[3])),
            matmul=lambda left, right, **options: stub((1, 1, 32, 5120)),
            typecast=lambda value, dtype: stub(value.shape), rms_norm=lambda value, **options: stub(value.shape),
            concat=lambda parts, dim: stub((1, 1, sum(part.shape[2] for part in parts), 5120)),
            synchronize_device=Mock(), deallocate=Mock())

        def slice_rows(value, start, stop, **options):
            slices.append((start, stop))
            return stub((1, 1, stop[2] - start[2], stop[3] - start[3]))

        operations.slice = slice_rows
        device = DFlashDevice.__new__(DFlashDevice)
        device.operations, device.mesh, device.collectives, device.kernel = operations, 'mesh', None, 'kernel'
        device.projection, device.feature_norm, device.borrowed = stub((2560, 5120)), stub((1, 1, 160, 32)), []
        return device, stub, slices

    def project(self, features, count, **options):
        device, stub, slices = self.device()
        with patch('dflash_device.concatenate_local_features', side_effect=lambda operations, parts: stub((1, 1, 32, 12800))), \
                patch('dflash_device.gather_add_projection', side_effect=lambda *args, **options: stub((1, 1, 32, 5120))):
            output = device.project_features(features, count, **options)
        return output, [entry for entry in slices if entry[1][3] == 2560]

    def taps(self, rows=32):
        made = []
        for tap in range(5):
            value = SimpleNamespace(shape=(1, 1, rows, 2560), dtype='bf16')
            value.shards = [FakeShard(value, next(_addresses)) for chip in range(2)]
            made.append(value)
        return made

    def test_packed_taps_are_sliced_at_the_users_offset_and_plain_taps_at_zero(self):
        taps = self.taps()
        output, slices = self.project(PackedFeatureTaps(taps, row_offset=16, rows=16), 9)
        self.assertEqual(slices, [((0, 0, 16, 0), (1, 1, 25, 2560))] * 5)
        self.assertEqual(output.shape, (1, 1, 9, 5120))
        output, slices = self.project(tuple(taps), 9)
        self.assertEqual(slices, [((0, 0, 0, 0), (1, 1, 9, 2560))] * 5)
        output, slices = self.project(tuple(taps), 9, row_offset=16)
        self.assertEqual(slices, [((0, 0, 16, 0), (1, 1, 25, 2560))] * 5)

    def test_an_offset_past_the_taps_is_refused(self):
        taps = self.taps()
        for features, count, options in ((PackedFeatureTaps(taps, row_offset=16, rows=16), 17, {}),
                                         (tuple(taps), 1, dict(row_offset=32)), (tuple(taps), 1, dict(row_offset=-1)),
                                         (tuple(taps), 1, dict(row_offset=True))):
            with self.subTest(count=count, options=options), self.assertRaises(ValueError):
                self.project(features, count, **options)


if __name__ == '__main__':
    unittest.main()
