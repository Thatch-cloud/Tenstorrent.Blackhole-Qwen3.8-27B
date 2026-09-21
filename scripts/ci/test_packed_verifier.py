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

from packed_cache_writer import tile as cache_tile, tile_rows
import packed_verifier
from packed_verifier import (PackedFeatureTaps, PackedShape, PackedVerifierEngine, m1_shape, m3_shape, segment_rows,
                             validate_shape)
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
        # Parallel to .executed (unchanged: still one entry per execute_trace call, the
        # trace itself, so every existing `self.ttnn.executed == [...]` assertion is
        # untouched) - the blocking flag that call was made with.
        self.execute_blocking = []
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
        self.execute_blocking.append(blocking)

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
    segment_layers built from them, commit_user publishing nothing at prefix 0.

    `commit`'s own `if synchronize: self.operations.synchronize_device(mesh)`, AFTER
    `publication(prefix)`, mirrors gdn_records.RetainedGDNBlock.commit_user exactly (its
    real `if synchronize:` branch runs the same way, in the same order, for the same
    reason - PackedVerifierEngine.commit_user's pipelined mode relies on this to fence the
    round's enqueued commit traces without a separate sync call of its own)."""

    def __init__(self, rows, operations=None, mesh=None):
        self.rows, self.records, self.closed = rows, [], False
        self.operations, self.mesh = operations, mesh
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
        if synchronize and self.operations is not None:
            self.operations.synchronize_device(self.mesh)

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
        # Beyond one 32-row tile, the K/V write's per-tile positions and page-table rows
        # (model_batch.prepare_inputs, packed_cache_writer.py).
        self.cache_tiles = [cache_tile((first, last), integers((32,)), integers((32, page_width)))
                            for first, last in (tile_rows(rows) if rows > 32 else ())]
        self.cos = ttnn.allocate((1, rows, 1, 64), 'bf16', 'tile', torch.zeros(1, rows, 1, 64))
        self.sin = ttnn.allocate((1, rows, 1, 64), 'bf16', 'tile', torch.zeros(1, rows, 1, 64))
        self.retained = FakeRetained(rows, ttnn, model.mesh_device) if options.get('retain_records') else None
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
        from target_packed_pages import segments

        ttnn = type(self).ttnn
        if self.retained is not None and not self.retained.records:
            users = len(self.pack)
            spans, total = segments(self.pack)
            for layer, helper in enumerate(self.helpers):
                pieces = tuple(dict(states=SimpleNamespace(name='states%d.%d' % (user, layer)),
                                    packed_conv_states=[SimpleNamespace(name='conv%d.%d.%d' % (user, layer, tap)) for tap in range(4)])
                               for user in range(users))
                state = SimpleNamespace(gdn=helper.gdn,
                                        segment_entries=tuple([SimpleNamespace(name='entry%d.%d' % (user, layer))] * 5
                                                              for user in range(users)))
                # model_batch appends each user's carry for this layer as the record's checkpoint
                carries = tuple(user['slots'][layer] for user in self.pack)
                self.retained.records.append((state, dict(segment_results=pieces, segments=spans), carries))
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
    """The M1 block: two T16 users in 32 rows over a two-slot pool. FourUserFixture below
    is the same fixture at the M3 shape; everything here scales from USERS."""

    USERS = 2

    def shape(self):
        return {2: m1_shape, 4: m3_shape}[self.USERS](PAGE_WIDTH)

    def setUp(self):
        verifier_engine.note_prefill()
        self.ttnn = FakeTTNN()
        FakeModelBatch.ttnn = self.ttnn
        FakeModelBatch.instances, FakeFeatures.instances = [], []
        self.helpers = helpers(self.ttnn)
        self.pool = pool(self.ttnn, self.helpers, users=self.USERS,
                         packed={(self.USERS, 16): packed_tables(self.ttnn, users=self.USERS)})
        self.weights, self.model = weights(), model()
        self.prepared, self.traces = [], count(1)
        rows = self.shape().block_rows
        self.ids = self.ttnn.allocate((rows,), 'uint32', 'row_major', torch.arange(1000, 1000 + rows, dtype=torch.int32))

        def prepare(mesh, layers, prefix):
            self.prepared.append((layers, prefix))
            return Mock(name='publication')

        def capture_operation(operations, mesh, operation):
            return 'trace%d' % next(self.traces), operation()

        for target, value in (('ModelBatch', FakeModelBatch), ('PreparedTargetFeatures', FakeFeatures),
                              ('prepare', Mock(side_effect=prepare)),
                              ('capture_operation', Mock(side_effect=capture_operation)),
                              ('sample_rows', Mock(side_effect=lambda *args, **options: self.ids))):
            patcher = patch.object(packed_verifier, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # The pinned reader's mask program needs the real ttnn; the readers are otherwise real.
        patcher = patch('attention_replay.prepare', return_value='mask-program')
        patcher.start()
        self.addCleanup(patcher.stop)

    def build(self, **options):
        return PackedVerifierEngine(self.ttnn, self.model, self.helpers, 'sampler', pool=self.pool,
            shared_weights=self.weights, shape=self.shape(), feature_taps=TAPS, **options)

    def two(self, reversed_order=True, rows=16):
        """Two admitted requests: A in pool slot 0, B in pool slot 1, presented B first
        (probe 35436807668 saw the scheduler present the pair as ['B', 'A']). Both inside
        the block's native chunk family [4096, 4352), as the serving pin keeps them."""
        first = request('A', self.pool.slots[0], 4100, 7)
        second = request('B', self.pool.slots[1], 4200, 11)
        entries = [entry(second, range(50, 50 + rows)), entry(first, range(10, 10 + rows))]
        return entries if reversed_order else entries[::-1]


class ConstructionFailureTests(BlockFixture):
    """A construction that fails after device work was enqueued (run 35505708710, image v50:
    the 64-row warm forward raised on the host with the device hung on one of its ops) logs
    the cause and its traceback BEFORE closing, and closes without the device fence: no
    synchronize, no trace release, everything else released as usual."""

    def failing_build(self, failure, **patches):
        lines, seen = [], {}
        original = PackedVerifierEngine.close

        def spy(block, *, wait=True):
            seen.update(wait=wait, phase=block.phase, stage=block.stage, taps=list(block.taps), trace=block.trace,
                        commits=sum(len(commits) for commits in block.commits), fences=self.ttnn.synchronized)
            original(block, wait=wait)
            # the fences close itself added (the readers' staging fences during construction are not its)
            seen.update(closed=block.phase, fences=self.ttnn.synchronized - seen['fences'])

        with patch.object(packed_verifier, 'diagnostic', Mock(side_effect=lines.append)), \
                patch.object(PackedVerifierEngine, 'close', spy):
            with self.assertRaisesRegex(type(failure), str(failure)):
                self.build()
        return lines, seen

    def test_a_failed_warm_forward_is_logged_with_its_traceback_before_a_close_that_never_fences(self):
        failure = RuntimeError('device refused the 64-row op')
        with patch.object(FakeModelBatch, 'forward', Mock(side_effect=failure)):
            lines, seen = self.failing_build(failure)
        self.assertEqual(lines[0], '[PINDIAG] packed block warm failed with RuntimeError: device refused the 64-row op; '
                                   'stage warm forward; closing without the device fence')
        self.assertTrue(lines[1].startswith('[PINDIAG] packed block failure traceback\nTraceback (most recent call last):'))
        self.assertIn('RuntimeError: device refused the 64-row op', lines[1])
        self.assertIn('in operation', lines[1], 'the traceback names the frame that raised')
        self.assertEqual(lines[2], '[PINDIAG] packed block closed without the device fence at stage warm forward; '
                                   '0 captured trace(s) abandoned')
        self.assertEqual(len(lines), 3)
        self.assertEqual((seen['wait'], seen['phase'], seen['closed'], seen['trace'], seen['commits']),
                         (False, 'failed', 'closed', None, 0))
        # no fence, no trace release; the frees and the pool hand-back happen as they always did
        self.assertEqual((seen['fences'], self.ttnn.released), (0, []))
        freed = self.ttnn.deallocated
        self.assertEqual(len(seen['taps']), 5)
        self.assertTrue(all(any(tap is entry for entry in freed) for tap in seen['taps']))
        self.assertFalse(self.pool.packed[(2, 16)].taken)
        (warm,) = FakeModelBatch.instances
        warm.close.assert_called_once()
        self.assertTrue(FakeFeatures.instances[0].closed)
        pooled = [value for slot in self.pool.slots for snapshot in slot.verifier.carry for value in snapshot]
        self.assertFalse(any(any(value is entry for entry in freed) for value in pooled))

    def test_a_failure_after_the_verify_trace_abandons_it_and_names_its_stage(self):
        failure = RuntimeError('commit capture refused')
        with patch.object(packed_verifier, 'prepare', Mock(side_effect=failure)):
            lines, seen = self.failing_build(failure)
        self.assertEqual(lines[0], '[PINDIAG] packed block warm failed with RuntimeError: commit capture refused; '
                                   'stage commit trace capture; closing without the device fence')
        self.assertEqual(lines[2], '[PINDIAG] packed block closed without the device fence at stage commit trace capture; '
                                   '1 captured trace(s) abandoned')
        self.assertEqual((seen['wait'], seen['trace'], seen['closed']), (False, 'trace1', 'closed'))
        # close added no fence, and the captured verify trace is abandoned, not released
        self.assertEqual((seen['fences'], self.ttnn.released), (0, []))
        warm, captured = FakeModelBatch.instances
        warm.close.assert_called_once()
        captured.close.assert_called_once()
        self.assertFalse(self.pool.packed[(2, 16)].taken)

    def test_a_healthy_close_still_fences_and_releases_the_traces(self):
        block = self.build()
        synchronized = self.ttnn.synchronized
        block.close()
        self.assertEqual(self.ttnn.synchronized - synchronized, 1)
        self.assertEqual(sorted(self.ttnn.released), sorted('trace%d' % index for index in range(1, 34)))

    def test_the_diagnostic_line_goes_to_loguru_when_present_and_to_stderr_otherwise(self):
        import io
        import sys
        from contextlib import redirect_stderr

        logger = SimpleNamespace(info=Mock())
        with patch.dict(sys.modules, {'loguru': SimpleNamespace(logger=logger)}):
            packed_verifier.diagnostic('[PINDIAG] via loguru')
        logger.info.assert_called_once_with('{}', '[PINDIAG] via loguru')
        captured = io.StringIO()
        with patch.dict(sys.modules, {'loguru': None}), redirect_stderr(captured):
            packed_verifier.diagnostic('[PINDIAG] via stderr')
        self.assertEqual(captured.getvalue(), '[PINDIAG] via stderr\n')
        # a failing sink never masks the failure being reported
        with patch.dict(sys.modules, {'loguru': SimpleNamespace(logger=SimpleNamespace(info=Mock(side_effect=OSError('sink'))))}):
            packed_verifier.diagnostic('[PINDIAG] lost')


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
        import packed_shapes

        # One definition: the engine takes its shape and validation from packed_shapes.
        self.assertIs(PackedShape, packed_shapes.PackedShape)
        self.assertIs(validate_shape, packed_shapes.validate_shape)
        self.assertEqual(m1_shape(PAGE_WIDTH), PackedShape(2, 16, 32, PAGE_WIDTH, PAGE_WIDTH * 64))
        self.assertEqual(m3_shape(PAGE_WIDTH), PackedShape(4, 16, 64, PAGE_WIDTH, PAGE_WIDTH * 64))
        validate_shape(PackedShape(4, 8, 32, 512, 512 * 64))
        for broken in (PackedShape(2, 16, 32, PAGE_WIDTH, PAGE_WIDTH * 64 + 1), PackedShape(3, 16, 32, 68, 68 * 64),
                       PackedShape(2, 16, 64, 68, 68 * 64), PackedShape(2, 16, 32, 67, 67 * 64), (2, 16, 32, 68, 68 * 64),
                       PackedShape(8, 16, 128, 68, 68 * 64), PackedShape(4, 16, 64, 68, 68 * 64 + 1)):
            with self.assertRaises(ValueError):
                validate_shape(broken)
        self.assertEqual([segment_rows(m1_shape(PAGE_WIDTH), user) for user in (0, 1)], [(0, 16), (16, 32)])
        self.assertEqual([segment_rows(m3_shape(PAGE_WIDTH), user) for user in range(4)],
                         [(0, 16), (16, 32), (32, 48), (48, 64)])
        with self.assertRaises(ValueError):
            segment_rows(m1_shape(PAGE_WIDTH), 2)
        with self.assertRaises(ValueError):
            segment_rows(m3_shape(PAGE_WIDTH), 4)


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


class FourUserFixture(BlockFixture):
    """The M3 block: four T16 users in 64 rows over a four-slot pool holding the (4, 16)
    replay tables. Same fakes, same block class; only the shape differs."""

    USERS = 4
    # A in slot 0, B in slot 1, C in slot 2, D in slot 3: their pages, positions and tokens.
    PAGES = (7, 11, 13, 17)
    POSITIONS = (4100, 4200, 4150, 4300)
    TOKENS = (10, 50, 30, 70)

    def four(self, order=(2, 0, 3, 1), rows=16):
        """Four admitted requests, presented in `order` (C, A, D, B by default), every one
        inside the block's native chunk family [4096, 4352)."""
        owners = [request(name, self.pool.slots[index], self.POSITIONS[index], self.PAGES[index])
                  for index, name in enumerate('ABCD')]
        return [entry(owners[index], range(self.TOKENS[index], self.TOKENS[index] + rows)) for index in order]


class FourUserConstructionTests(FourUserFixture):
    def test_every_per_segment_structure_scales_to_four_users_and_sixty_four_rows(self):
        block = self.build()
        self.assertEqual((block.phase, block.users, block.rows_per_user, block.block_rows), ('idle', 4, 16, 64))
        self.assertEqual(block.shape, m3_shape(PAGE_WIDTH))
        # the initial snapshot plus one checkpoint set per user, before any capture
        self.assertEqual([helper.allocate.call_count for helper in self.helpers], [5] * GDN_LAYERS)
        self.assertEqual([len(checkpoints) for checkpoints in block.checkpoints], [GDN_LAYERS] * 4)
        self.assertEqual(len(block.carries), 4)
        # the five taps hold the whole 64-row block
        self.assertEqual([(tap.shape, tap.dtype, tap.layout, tap.mapper) for tap in block.taps],
                         [((1, 1, 64, 5120), 'bf16', 'tile', ('shard', 3))] * 5)
        warm, captured = FakeModelBatch.instances
        self.assertIs(block.fixture, captured)
        for fixture in (warm, captured):
            self.assertEqual((fixture.rows, fixture.start, fixture.prefix), (64, block.capture_position, 64))
            self.assertEqual([user['rows'] for user in fixture.pack], [16] * 4)
            self.assertEqual([user['prefix'] for user in fixture.pack], [0] * 4)
            self.assertIs(fixture.options['packed_replay_pages'], block.replay_tables)
            self.assertTrue(fixture.options['commit_only_gdn'] and fixture.options['attention_replay']
                            and fixture.options['ordered_cache'])
            # segment u restores from and commits into pool slot u's carry
            for user, participant in enumerate(fixture.pack):
                self.assertEqual(list(participant['checkpoints']), block.checkpoints[user])
                lent = self.pool.slots[user].verifier.carry
                self.assertTrue(all(a is b for mine, theirs in zip(participant['slots'], lent, strict=True)
                                    for a, b in zip(mine, theirs, strict=True)))
            # two 32-row K/V cache tiles, each with its own positions word and page-table rows
            self.assertEqual([tile.rows for tile in fixture.cache_tiles], [(0, 32), (32, 64)])
            self.assertEqual([(tile.positions.shape, tile.pages.shape) for tile in fixture.cache_tiles],
                             [((32,), (32, PAGE_WIDTH))] * 2)
        # the (4, 16) table set, one 16-row reader per user in the block's family
        tables = self.pool.packed[(4, 16)]
        self.assertTrue(tables.taken)
        self.assertIs(block.replay, tables)
        self.assertEqual(block.replay_capacity, FAMILY)
        reader = captured.replay_reader
        self.assertEqual((reader.rows, reader.capacity, reader.starts, len(reader.readers)), (64, FAMILY, (4096,) * 4, 4))
        self.assertEqual(reader.segments, ((0, 16), (16, 32), (32, 48), (48, 64)))
        self.assertEqual(len({id(own.positions) for own in reader.readers}), 4)
        for user, own in enumerate(reader.readers):
            self.assertEqual(own.rows, 16)
            self.assertEqual([entry[1] for entry in own.metadata], tables.replay_pages[FAMILY][user])
            self.assertEqual([tuple(entry[1].shape) for entry in own.metadata], [(3, 68), (1, 68)])
        self.assertEqual(block.describe()['attention']['bundles_per_user'], [2, 2, 2, 2])
        # one verify trace, then sixteen commit traces per user: sixty-four, each ending in
        # that user's own carry (packed_shapes.commit_traces)
        self.assertEqual(block.trace, 'trace1')
        self.assertEqual([sorted(commits) for commits in block.commits], [list(range(1, 17))] * 4)
        self.assertEqual(block.describe()['commit_traces'], 64)
        self.assertEqual([prefix for layers, prefix in self.prepared], [*range(1, 17)] * 4)
        for index, (layers, prefix) in enumerate(self.prepared):
            user = index // 16
            self.assertEqual(len(layers), GDN_LAYERS)
            for layer, (record, slot) in enumerate(zip(layers, self.pool.slots[user].verifier.carry, strict=True)):
                self.assertEqual(len(record), 20)
                self.assertEqual([value.name for value in record[:5]], ['entry%d.%d' % (user, layer)] * 5)
                self.assertEqual(record[5].name, 'states%d.%d' % (user, layer))
                self.assertTrue(all(a is b for a, b in zip(record[15:], slot, strict=True)))
        # every carry of the four slots zeroed again after the commit warming
        pooled = [value for slot in self.pool.slots for snapshot in slot.verifier.carry for value in snapshot]
        self.assertEqual(len(self.ttnn.zeroed), len(pooled))
        self.assertIsNone(verifier_engine._resident)

    def test_a_pool_short_of_slots_or_of_the_four_user_tables_refuses_the_block(self):
        cases = {'a two-slot pool': pool(self.ttnn, self.helpers, users=2, packed={(4, 16): packed_tables(self.ttnn, users=4)}),
                 'a four-slot pool holding only the M1 tables': pool(self.ttnn, self.helpers, users=4),
                 'four-user tables lacking the block family': pool(self.ttnn, self.helpers, users=4,
                     packed={(4, 16): packed_tables(self.ttnn, users=4, families=(4096,))})}
        for name, refused in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                PackedVerifierEngine(self.ttnn, self.model, self.helpers, 'sampler', pool=refused,
                                     shared_weights=self.weights, shape=m3_shape(PAGE_WIDTH), feature_taps=TAPS)
        for helper in self.helpers:
            helper.allocate.assert_not_called()
        self.assertEqual((FakeModelBatch.instances, self.prepared), ([], []))


class FourUserRoundTests(FourUserFixture):
    def test_one_trace_serves_four_users_each_from_the_segment_its_carry_binds(self):
        block = self.build()
        for helper in self.helpers:
            helper.restore.reset_mock()
            helper.save.reset_mock()
        entries = self.four()
        executed, copies = len(self.ttnn.executed), len(self.ttnn.host_copies)
        predictions, metrics = block.verify(entries)
        # presented C, A, D, B: segments 2, 0, 3, 1, never the entry index
        self.assertEqual(metrics['segments'], (2, 0, 3, 1))
        self.assertEqual(metrics['users'], 4)
        self.assertEqual(predictions, [list(range(1032, 1048)), list(range(1000, 1016)),
                                       list(range(1048, 1064)), list(range(1016, 1032))])
        fixture = block.fixture
        # each user's tokens, positions and pages in its own segment's rows
        for user in range(4):
            rows = slice(16 * user, 16 * user + 16)
            self.assertEqual(fixture.tokens.value[rows, 0].tolist(), list(range(self.TOKENS[user], self.TOKENS[user] + 16)))
            self.assertEqual(fixture.positions.value[rows].tolist(), list(range(self.POSITIONS[user], self.POSITIONS[user] + 16)))
            self.assertTrue(bool((fixture.pages.value[rows] == self.PAGES[user]).all()))
            self.assertEqual([table.value[0, 0].item() for table in fixture.row_pages[rows]], [self.PAGES[user]] * 16)
            self.assertEqual([position.value.item() for position in fixture.singleton_positions[rows]],
                             list(range(self.POSITIONS[user], self.POSITIONS[user] + 16)))
        self.assertEqual((fixture.cos.value.shape, fixture.sin.value.shape), ((1, 64, 1, 64), (1, 64, 1, 64)))
        # each user's own reader: its word at its start, its bundle tables holding its pages
        reader = fixture.replay_reader
        self.assertEqual(reader.starts, self.POSITIONS)
        self.assertEqual([own.positions.value.tolist()[0] for own in reader.readers], list(self.POSITIONS))
        for own, page in zip(reader.readers, self.PAGES, strict=True):
            for bundle, table, mask, config in own.metadata:
                self.assertEqual(tuple(table.value.shape), (len(bundle), PAGE_WIDTH))
                self.assertTrue(bool((table.value == page).all()))
        # and the two cache tiles: rows [0, 32) are A's and B's, rows [32, 64) C's and D's
        first, second = fixture.cache_tiles
        self.assertEqual(first.positions.value.tolist(), [*range(4100, 4116), *range(4200, 4216)])
        self.assertEqual(second.positions.value.tolist(), [*range(4150, 4166), *range(4300, 4316)])
        self.assertEqual(first.pages.value[:, 0].tolist(), [7] * 16 + [11] * 16)
        self.assertEqual(second.pages.value[:, 0].tolist(), [13] * 16 + [17] * 16)
        # tokens, positions, cos, sin, pages, 64 singleton positions, 64 row tables, four
        # positions words, four users x two bundle tables, two tiles x (positions, pages)
        self.assertEqual(len(self.ttnn.host_copies) - copies, 149)
        self.assertEqual(metrics['staged_buffers'], 149)
        self.assertEqual(self.ttnn.executed[executed:], ['trace1'])
        for helper in self.helpers:
            helper.restore.assert_not_called()
        self.assertEqual((block.phase, block.pending_segments), ('verified', {0, 1, 2, 3}))
        # taps at each user's row offset
        self.assertEqual([(block.features(segment).row_offset, block.features(segment).rows) for segment in range(4)],
                         [(0, 16), (16, 16), (32, 16), (48, 16)])
        # commits in entries order, each through its own prefix trace, the last one the fence
        retained = fixture.retained
        for segment, prefix in zip(metrics['segments'], (9, 16, 0, 4)):
            block.commit_user(segment, prefix)
        self.assertEqual(retained.commits, [(2, 9), (0, 16), (3, 0), (1, 4)])
        self.assertEqual([call.kwargs['synchronize'] for call in retained.commit_user.call_args_list],
                         [False, False, False, True])
        self.assertEqual(self.ttnn.executed[executed + 1:], [block.commits[2][9], block.commits[0][16], block.commits[1][4]])
        self.assertEqual((block.phase, block.pending_segments), ('idle', set()))
        # the next round, in another order, replays the same trace
        predictions, metrics = block.verify(self.four(order=(0, 1, 2, 3)))
        self.assertEqual(metrics['segments'], (0, 1, 2, 3))
        self.assertEqual(predictions[3], list(range(1048, 1064)))
        retained.replay.assert_called_once()
        self.assertEqual(block.rounds, 2)

    def test_fewer_than_four_entries_or_a_foreign_engine_is_refused_before_the_trace(self):
        block = self.build()
        executed, copies = len(self.ttnn.executed), len(self.ttnn.host_copies)
        cases = {'three survivors': self.four()[:3], 'two survivors': self.four()[:2], 'one survivor': self.four()[:1]}
        stranger = request('E', SimpleNamespace(verifier=SimpleNamespace(carry=snapshot_set(self.ttnn))), 4100, 3)
        cases['an engine outside the pool'] = [*self.four()[:3], entry(stranger, range(16))]
        # the first three entries are C, A and D (slots 2, 0 and 3): E through slot 0 again
        twice = request('E', self.pool.slots[0], 4100, 3)
        cases['two entries through one slot'] = [*self.four()[:3], entry(twice, range(16))]
        cases['a narrow ticket'] = [*self.four()[:3], entry(request('B', self.pool.slots[1], 4200, 11), range(8))]
        for name, entries in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                block.verify(entries)
            self.assertEqual(block.phase, 'idle', name)
        self.assertEqual((len(self.ttnn.executed), len(self.ttnn.host_copies)), (executed, copies))

    def test_close_releases_the_verify_trace_and_sixty_four_commit_traces(self):
        block = self.build()
        block.verify(self.four())
        for segment in range(4):
            block.commit_user(segment, 0)
        tables = self.pool.packed[(4, 16)]
        block.close()
        self.assertEqual(block.phase, 'closed')
        self.assertEqual(sorted(self.ttnn.released), sorted('trace%d' % index for index in range(1, 66)))
        self.assertFalse(tables.taken)
        freed = self.ttnn.deallocated
        self.assertFalse(any(any(value is entry for entry in freed) for value in tables.tensors))
        pooled = [value for slot in self.pool.slots for snapshot in slot.verifier.carry for value in snapshot]
        self.assertFalse(any(any(value is entry for entry in freed) for value in pooled))


class PipelinedCommitTests(FourUserFixture):
    """QWEN_FAST_PIPELINED_COMMITS=1 (task #45): each user's commit trace is enqueued
    (blocking=False) instead of replayed one at a time. The round's one trailing fence is
    NOT a new call here - it is the existing `synchronize=last` plumbing
    (gdn_records.RetainedGDNBlock.commit_user calls `self.operations.synchronize_device`
    right after the publication callback returns, on the round's last segment only; mirrored
    faithfully by FakeRetained.commit above), which already fences the whole command queue
    every one of the round's four commits was enqueued to, in this same per-segment call
    order. Read once at construction (self.pipelined_commits), like every other packed
    engine flag."""

    def pipelined_build(self, **options):
        with patch.dict('os.environ', {'QWEN_FAST_PIPELINED_COMMITS': '1'}):
            return self.build(**options)

    def test_default_mode_is_unchanged_four_blocking_replays_one_trailing_sync(self):
        block = self.build()
        self.assertFalse(block.pipelined_commits)
        block.verify(self.four())
        synchronized = self.ttnn.synchronized
        for segment, prefix in zip((2, 0, 3, 1), (9, 16, 0, 4)):
            block.commit_user(segment, prefix)
        # three traces ran (segment 3's prefix 0 runs none), every one blocking - exactly
        # as before this change
        self.assertEqual(self.ttnn.execute_blocking[-3:], [True, True, True])
        # the pre-existing `synchronize=last` fence still fires exactly once
        self.assertEqual(self.ttnn.synchronized - synchronized, 1)
        self.assertEqual(block.phase, 'idle')

    def test_pipelined_mode_enqueues_all_four_and_synchronizes_once_after_the_last(self):
        block = self.pipelined_build()
        self.assertTrue(block.pipelined_commits)
        block.verify(self.four())
        executed, synchronized = len(self.ttnn.executed), self.ttnn.synchronized
        decisions = list(zip((2, 0, 3, 1), (9, 16, 3, 4)))  # every prefix here is nonzero
        for segment, prefix in decisions[:-1]:
            block.commit_user(segment, prefix)
            # not yet the last segment: enqueued, not fenced, the round still open
            self.assertEqual(self.ttnn.synchronized, synchronized, 'no sync before the last segment')
            self.assertEqual(block.phase, 'verified')
        last_segment, last_prefix = decisions[-1]
        block.commit_user(last_segment, last_prefix)
        # four non-blocking enqueues
        self.assertEqual(self.ttnn.execute_blocking[executed:], [False, False, False, False])
        # exactly one synchronize_device call fences all four - and by the time this method's
        # call into RetainedGDNBlock.commit_user returns, that fence has already happened, so
        # the block is idle only once every enqueued trace has completed
        self.assertEqual(self.ttnn.synchronized - synchronized, 1)
        self.assertEqual(block.phase, 'idle')

    def test_a_pipelined_prefix_zero_commit_enqueues_nothing(self):
        block = self.pipelined_build()
        block.verify(self.four())
        executed = len(self.ttnn.executed)
        block.commit_user(2, 0)
        self.assertEqual(len(self.ttnn.executed), executed, 'prefix 0 enqueues no trace')
        for segment, prefix in ((0, 16), (3, 4), (1, 9)):
            block.commit_user(segment, prefix)
        self.assertEqual(self.ttnn.execute_blocking[executed:], [False, False, False])
        self.assertEqual(block.phase, 'idle')

    def test_verify_is_refused_until_every_segment_of_the_round_is_committed(self):
        block = self.pipelined_build()
        block.verify(self.four())
        for segment, prefix in zip((2, 0, 3), (9, 16, 4)):
            block.commit_user(segment, prefix)
        self.assertEqual(block.phase, 'verified')
        with self.assertRaisesRegex(ValueError, 'idle packed block is required'):
            block.verify(self.four(order=(0, 1, 2, 3)))
        block.commit_user(1, 0)
        self.assertEqual(block.phase, 'idle')
        # now accepted
        block.verify(self.four(order=(0, 1, 2, 3)))

    def test_audit_line_reports_pipelined_mode_per_commit_timings_and_the_trailing_sync(self):
        block = self.pipelined_build()
        block.verify(self.four())
        lines = []
        with patch.dict('os.environ', {'QWEN_FAST_PACKED_AUDIT': '1'}), \
                patch.object(packed_verifier, 'diagnostic', Mock(side_effect=lines.append)):
            for segment, prefix in zip((2, 0, 3, 1), (9, 16, 0, 4)):
                block.commit_user(segment, prefix)
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], r'^\[PACKED-COMMIT\] round=1 mode=pipelined '
                                   r'commit_ms=\[[\d.]+,[\d.]+,[\d.]+,[\d.]+\] sync_ms=[\d.]+$')
        # commit_timings is segment-indexed; segment 3's prefix-0 commit cost nothing
        self.assertEqual(block.commit_timings[3], 0.0)

    def test_audit_line_reports_blocking_mode_by_name(self):
        block = self.build()
        block.verify(self.four())
        lines = []
        with patch.dict('os.environ', {'QWEN_FAST_PACKED_AUDIT': '1'}), \
                patch.object(packed_verifier, 'diagnostic', Mock(side_effect=lines.append)):
            for segment, prefix in zip((2, 0, 3, 1), (9, 16, 0, 4)):
                block.commit_user(segment, prefix)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith('[PACKED-COMMIT] round=1 mode=blocking commit_ms='))

    def test_audit_line_is_silent_without_the_flag(self):
        block = self.pipelined_build()
        block.verify(self.four())
        with patch.object(packed_verifier, 'diagnostic', Mock(side_effect=AssertionError('should not be called'))):
            for segment, prefix in zip((2, 0, 3, 1), (9, 16, 0, 4)):
                block.commit_user(segment, prefix)


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


class PoolSlotBindingTests(BlockFixture):
    """QWEN_FAST_FOUR_AS_TWO's pair of 32-row blocks: each bound to its OWN disjoint pool
    slots (block A over 0, 1; block B over 2, 3) instead of the pool's first `shape.users`
    slots. `pool_slots=` names them explicitly; left unnamed, a block still takes slots
    0..users-1 in order, exactly as it always has."""

    USERS = 4

    def setUp(self):
        super().setUp()
        # A second (2, 16) table set alongside the one an M1 block over this pool would
        # already find - one per block, as the pool built for QWEN_FAST_FOUR_AS_TWO would
        # hold (serving_buffer_pool.py, packed_replicas).
        self.pool.packed[(2, 16)] = packed_tables(self.ttnn, users=2)

    def shape(self):
        # The four-slot pool is FourUserFixture's; the block built over it here is M1's.
        return m1_shape(PAGE_WIDTH)

    def test_a_block_bound_to_the_upper_two_slots_restores_and_commits_through_them(self):
        block = self.build(pool_slots=(2, 3))
        self.assertEqual(block.pool_slots, (2, 3))
        self.assertEqual(block.describe()['pool_slots'], [2, 3])
        # segment u carries pool slot (2 + u)'s carry, never slot u's
        for segment in range(2):
            lent = self.pool.slots[2 + segment].verifier.carry
            self.assertTrue(all(a is b for mine, theirs in zip(block.carries[segment], lent, strict=True)
                                for a, b in zip(mine, theirs, strict=True)))
        first = request('A', self.pool.slots[2], 4100, 7)
        second = request('B', self.pool.slots[3], 4200, 11)
        self.assertEqual(block.segment_of(first.engine), 0)
        self.assertEqual(block.segment_of(second.engine), 1)
        # an engine borrowed from slot 0 or 1 - this block's segments are 2 and 3 - matches nothing
        foreign = request('X', self.pool.slots[0], 4100, 7)
        with self.assertRaises(ValueError):
            block.segment_of(foreign.engine)

    def test_left_unnamed_a_block_still_takes_slots_zero_and_one_in_order(self):
        block = self.build()
        self.assertEqual(block.pool_slots, (0, 1))
        first = request('A', self.pool.slots[0], 4100, 7)
        self.assertEqual(block.segment_of(first.engine), 0)

    def test_pool_slots_must_be_distinct_and_within_the_pool(self):
        for pool_slots in ((0,), (0, 0), (0, 4), (0, 'x'), (0, 1, 2)):
            with self.subTest(pool_slots=pool_slots), self.assertRaisesRegex(ValueError, 'One distinct pool slot'):
                self.build(pool_slots=pool_slots)


class ProfileDumpRoundTests(unittest.TestCase):
    """QWEN_FAST_PROFILE_DUMP_ROUND=N reads the device profiler back exactly once, after round N."""

    def setUp(self):
        from packed_verifier import dump_device_profiler_after_round
        self.dump = dump_device_profiler_after_round
        self.reads = []
        self.operations = SimpleNamespace(ReadDeviceProfiler=lambda mesh: self.reads.append(mesh))
        self.mesh = object()

    def test_inert_when_unset(self):
        self.assertFalse(self.dump(self.operations, self.mesh, 5, environ={}))
        self.assertEqual(self.reads, [])

    def test_reads_back_only_on_the_named_round(self):
        env = {'QWEN_FAST_PROFILE_DUMP_ROUND': '5'}
        self.assertFalse(self.dump(self.operations, self.mesh, 4, environ=env))
        self.assertTrue(self.dump(self.operations, self.mesh, 5, environ=env))
        self.assertFalse(self.dump(self.operations, self.mesh, 6, environ=env))
        self.assertEqual(self.reads, [self.mesh])

    def test_a_runtime_without_the_reader_is_reported_not_crashed(self):
        env = {'QWEN_FAST_PROFILE_DUMP_ROUND': '2'}
        self.assertFalse(self.dump(SimpleNamespace(), self.mesh, 2, environ=env))

    def test_a_non_integer_round_is_refused(self):
        with self.assertRaises(ValueError):
            self.dump(self.operations, self.mesh, 1, environ={'QWEN_FAST_PROFILE_DUMP_ROUND': 'five'})


if __name__ == '__main__':
    unittest.main()
