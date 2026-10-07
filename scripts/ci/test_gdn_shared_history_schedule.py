"""QWEN_FAST_GDN_SHARED_HISTORY: the schedule, with the REAL PackedVerifierEngine, two M3 blocks and a byte-level device model.

The question is the one the lever's exactness rests on: a block's history is read by its commits, so the other block's verify may not
overwrite it before those commits have run. The fake device here keeps real values: a verify trace writes, for every (layer, user), a
16-token chain of states derived from that user's carry and tokens into the history tensor the block's records name; a commit trace
copies history[prefix - 1] into the carry. Shared histories are the SAME tensor objects, so a verify of block B really overwrites
block A's rows. Every schedule runs the same rounds twice, once with private histories (the shipped engine) and once shared, and the
carries after every round must be identical to each other and to a pure-Python reference of the recurrence.

The negative control cuts the guard (`claim` records the writer and flushes nothing) under the deferred schedule and the carries come
out WRONG: the hazard is real and this model sees it.

Nothing touches a card."""

from contextlib import ExitStack, contextmanager
import os
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import gdn_shared_history as shared
import packed_verifier
from packed_verifier import PackedVerifierEngine
import serving_runtime
import test_packed_verifier as tpv
import verifier_engine
from verifier_pack import GDN_LAYERS

MASK = (1 << 61) - 1
LAYERS, USERS, TOKENS = GDN_LAYERS, 4, 16
BASE_ENV = {'QWEN_FAST_TP': '4', 'QWEN_FAST_M3_BLOCKS': '2'}
LAYERS_PER_COMMIT = 1          # one commit trace per (user, prefix); the fake counts one event per trace
DEFERRED_ENV = {'QWEN_FAST_ROUND_FENCES': '1', 'QWEN_FAST_EARLY_DRAFT': '1', 'QWEN_FAST_GDN_AFTER_PAIRS': '1'}


def mix(state, token, layer, step):
    """One token of the toy recurrence: the next state from the last, the token, the layer and the position."""
    return (state * 6364136223846793005 + token * 1442695040888963407 + layer * 40503 + step * 9973 + 1) & MASK


def chain(state, tokens, layer, count):
    states = []
    for step in range(count):
        state = mix(state, tokens[step], layer, step)
        states.append(state)
    return states


class Device:
    """The fake device's memory: one 16-slot history per history tensor (by identity), written by verify traces and read by commits."""

    def __init__(self):
        self.history = {}
        self.events = []

    def verify(self, label, fixture):
        tokens = fixture.tokens.value[:, 0].tolist()
        for layer, (state, result, carries) in enumerate(fixture.retained.records):
            for user in range(USERS):
                carry = carries[user][0]
                states = result['segment_results'][user]['states']
                slots = self.history.setdefault(id(states), [None] * TOKENS)
                slots[:] = chain(getattr(carry, 'sim', 0), tokens[TOKENS * user:TOKENS * (user + 1)], layer, TOKENS)
        self.events.append(('verify', label))

    def commit(self, label, layers, prefix):
        for layer in layers:
            slots = self.history.get(id(layer[5]))
            if slots is not None:
                layer[15].sim = slots[prefix - 1]
        self.events.append(('commit', label, prefix))


class Trace:
    def __init__(self, run):
        self.run = run


class SimTTNN(tpv.FakeTTNN):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def empty(self, shape, device=None, dtype=None, layout=None, memory_config=None):
        return self.allocate(shape, dtype, layout)

    def full_like(self, tensor, fill, optional_tensor=None):
        super().full_like(tensor, fill, optional_tensor)
        tensor.sim = 0

    def execute_trace(self, mesh, trace, cq_id=0, blocking=True):
        super().execute_trace(mesh, trace, cq_id=cq_id, blocking=blocking)
        if isinstance(trace, Trace):
            trace.run()


class SimRetained(tpv.FakeRetained):
    def commit(self, segment, prefix, *, dma, synchronize, publication, validated_this_round=False):
        super().commit(segment, prefix, dma=dma, synchronize=synchronize, publication=publication)

    def use_round_fences(self):
        self.round_fences = True

    def note_deferred_publications(self):
        pass


class SimModelBatch(tpv.FakeModelBatch):
    """The fake fixture whose retained records hold a real history tensor per (layer, user): the pool's inside a verify capture,
    a private one anywhere else - the choice gdn_user_batch_tp.execute makes."""

    last = None

    def __init__(self, model, tokens, start, pages, helpers, checkpoints, prefix, **options):
        super().__init__(model, tokens, start, pages, helpers, checkpoints, prefix, **options)
        self.mesh = model.mesh_device
        if self.retained is not None:
            self.retained = SimRetained(self.rows, type(self).ttnn, model.mesh_device)

    def forward(self, *, sharded_logits):
        from target_packed_pages import segments

        ttnn = type(self).ttnn
        type(self).last = self
        if self.retained is not None and not self.retained.records:
            users = len(self.pack)
            spans, total = segments(self.pack)
            module = sys.modules.get('gdn_shared_history')
            pool = module.active() if module is not None else None
            for layer, helper in enumerate(self.helpers):
                pieces = []
                for user in range(users):
                    if pool is not None:
                        states = pool.take(ttnn, self.mesh, TOKENS, 12)
                    else:
                        states = ttnn.empty((TOKENS, 12, 128, 128), device=self.mesh, dtype=ttnn.bfloat16,
                                            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    pieces.append(dict(states=states, packed_conv_states=[SimpleNamespace(name='conv%d.%d.%d' % (user, layer, tap))
                                                                          for tap in range(4)]))
                state = SimpleNamespace(gdn=helper.gdn, segment_entries=tuple(
                    [SimpleNamespace(name='entry%d.%d' % (user, layer))] * 5 for user in range(users)))
                carries = tuple(user['slots'][layer] for user in self.pack)
                self.retained.records.append((state, dict(segment_results=tuple(pieces), segments=spans), carries))
        return ttnn.allocate((1, 1, self.rows, 124160), 'bf16', 'tile')


class Rig(unittest.TestCase):
    """Two M3 blocks over one eight-slot pool, built through the real two-phase construction."""

    def setUp(self):
        shared.reset()
        self.addCleanup(shared.reset)
        verifier_engine.note_prefill()
        self.device = Device()
        self.ttnn = SimTTNN(self.device)
        SimModelBatch.ttnn = tpv.FakeModelBatch.ttnn = self.ttnn
        tpv.FakeModelBatch.instances, tpv.FakeFeatures.instances = [], []
        self.helpers = tpv.helpers(self.ttnn)
        self.pool = tpv.pool(self.ttnn, self.helpers, users=8, packed={(4, 16): tpv.packed_tables(self.ttnn, users=4)})
        self.sets = [tpv.packed_tables(self.ttnn, users=4), tpv.packed_tables(self.ttnn, users=4)]
        self.pool.packed_replay = lambda count, rows: next((tables for tables in self.sets if not tables.taken), self.sets[-1])
        self.weights, self.model = tpv.weights(), tpv.model()
        self.traces = iter(range(1, 10 ** 6))
        self.ids = self.ttnn.allocate((64,), 'uint32', 'row_major', torch.arange(1000, 1064, dtype=torch.int32))
        self.lines = []

        self.slot_of = {id(self.pool.slots[slot].verifier.carry[0][0]): slot for slot in range(8)}

        def prepare(mesh, layers, prefix):
            label = 'AB'[self.slot_of[id(layers[0][15])] // 4]
            return lambda: self.device.commit(label, layers, prefix)

        def capture_operation(operations, mesh, operation):
            result = operation()
            if isinstance(result, tuple):       # the verify forward returns (logits, ids...); a commit publication returns None
                fixture = SimModelBatch.last
                return Trace(lambda: self.device.verify(self.label_of(fixture), fixture)), result
            return Trace(operation), result

        for target, value in (('ModelBatch', SimModelBatch), ('PreparedTargetFeatures', tpv.FakeFeatures),
                              ('prepare', Mock(side_effect=prepare)),
                              ('capture_operation', Mock(side_effect=capture_operation)),
                              ('diagnostic', Mock(side_effect=self.lines.append)),
                              ('sample_rows', Mock(side_effect=lambda *args, **options: self.ids))):
            patcher = patch.object(packed_verifier, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch('attention_replay.prepare', return_value='mask-program')
        patcher.start()
        self.addCleanup(patcher.stop)
        self.blocks = []
        self.owners = [tpv.request('RQ%d' % slot, self.pool.slots[slot], 4100 + 8 * slot, 7 + slot) for slot in range(8)]
        self.labels = {}

    def label_of(self, fixture):
        return self.labels.get(id(fixture), '?')

    def build(self, environ=None):
        """Both blocks, in the order serving_runtime builds them; `environ` is in force while they are constructed (flags are read there)."""
        with patch.dict(os.environ, {**BASE_ENV, **(environ or {})}):
            self.blocks = [self.block((0, 1, 2, 3)), self.block((4, 5, 6, 7))]
            serving_runtime.complete_blocks_two_phase(self.blocks, self.model, log=lambda *args: None)
        for index, block in enumerate(self.blocks):
            self.labels[id(block.fixture)] = 'AB'[index]
        return self.blocks

    def block(self, slots):
        return PackedVerifierEngine(self.ttnn, self.model, self.helpers, 'sampler', pool=self.pool, shared_weights=self.weights,
                                    shape=tpv.m3_shape(tpv.PAGE_WIDTH), feature_taps=tpv.TAPS, pool_slots=slots, defer_capture=True)

    # -- rounds
    def entries(self, block_index, round_number):
        owners = self.owners[4 * block_index:4 * block_index + 4]
        entries = []
        for user, owner in enumerate(owners):
            entries.append(tpv.entry(owner, tokens_for(round_number, block_index, user)))
        return entries

    def verify(self, block_index, round_number):
        predictions, metrics = self.blocks[block_index].verify(self.entries(block_index, round_number))
        return metrics

    def decide(self, block_index, prefixes):
        block = self.blocks[block_index]
        for segment in range(USERS):
            block.commit_user(segment, prefixes[segment])

    def step(self, round_number, prefixes, deferred=False, flush_site='end'):
        """One scheduler step over eight seats: block A's round, then block B's, as packed_device_rounds runs them."""
        if deferred:
            for block in self.blocks:
                self.assertTrue(block.arm_deferred_commits())
        for index in (0, 1):
            self.verify(index, round_number)
            self.decide(index, prefixes[index])
        if deferred:
            from serving_packed_step import flush_blocks_deferred

            flush_blocks_deferred(self.blocks, flush_site)

    def carries(self):
        return {(slot, layer): getattr(self.pool.slots[slot].verifier.carry[layer][0], 'sim', 0)
                for slot in range(8) for layer in range(LAYERS)}


def reference(rounds):
    """The recurrence in plain Python: each (slot, layer) carry advanced by its accepted prefix every round."""
    state = {(slot, layer): 0 for slot in range(8) for layer in range(LAYERS)}
    for round_number, prefixes in enumerate(rounds, 1):
        for block_index in (0, 1):
            for user in range(USERS):
                slot, count = 4 * block_index + user, prefixes[block_index][user]
                for layer in range(LAYERS):
                    if count:
                        state[(slot, layer)] = chain(state[(slot, layer)], tokens_for(round_number, block_index, user), layer,
                                                     count)[-1]
    return state


def tokens_for(round_number, block_index, user):
    """One user's 16 tokens this round (the fake model's vocabulary is 100 wide)."""
    base = 7 * round_number + 31 * block_index + 13 * user
    return [(base + 5 * step) % 100 for step in range(TOKENS)]


def random_rounds(count, seed):
    generator = random.Random(seed)
    # the committed prefix distribution has mass at 1 and 2 and a tail at 16; zero is a user that commits nothing
    return [[[generator.choice((0, 1, 1, 2, 2, 3, 4, 5, 8, 16)) for user in range(USERS)] for block in (0, 1)] for rnd in range(count)]


class ModelSelfTests(unittest.TestCase):
    def test_the_toy_recurrence_has_a_distinct_state_at_every_position_and_depends_on_its_start(self):
        a, b = chain(0, list(range(16)), 3, 16), chain(1, list(range(16)), 3, 16)
        self.assertEqual(len(set(a)), 16)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(chain(0, list(range(16)), 3, 5), a[:5])


class ConstructionTests(Rig):
    def test_flag_off_the_blocks_have_private_histories_and_no_pool(self):
        blocks = self.build()
        self.assertIsNone(blocks[0].shared_history)
        self.assertIsNone(blocks[1].shared_history)
        self.assertNotIn('shared_history', blocks[0].describe())
        histories = [id(piece['states']) for block in blocks for state, result, carries in block.fixture.retained.records
                     for piece in result['segment_results']]
        self.assertEqual(len(set(histories)), 2 * 192)
        self.assertFalse([line for line in self.lines if 'shared history' in line])

    def test_flag_on_both_blocks_name_the_same_192_tensors_and_each_logs_its_role(self):
        blocks = self.build({shared.FLAG: '1'})
        pool = blocks[0].shared_history
        self.assertIsNotNone(pool)
        self.assertIs(blocks[1].shared_history, pool)
        tensors = [[piece['states'] for state, result, carries in block.fixture.retained.records
                    for piece in result['segment_results']] for block in blocks]
        self.assertEqual(len(tensors[0]), 192)
        self.assertTrue(all(a is b for a, b in zip(*tensors, strict=True)))
        self.assertEqual(len({id(value) for value in tensors[0]}), 192)
        self.assertTrue(all(shared.holds(value) for value in tensors[0]))
        engaged = [line for line in self.lines if line.startswith(shared.ENGAGED_MARKER)]
        self.assertEqual(len(engaged), 2)
        self.assertIn('role=owner blocks=2 users=4 layers=48 tensors=192 tensor_bytes=6291456 freed_per_chip=1207959552 '
                      'kv_blocks_gained=2168', engaged[0])
        self.assertIn('role=sharer', engaged[1])
        self.assertEqual(blocks[0].describe()['shared_history']['shared_bytes'], 192 * 6 * 2 ** 20)
        # the warm forward's histories were private: the pool took nothing from it
        self.assertEqual(pool.counts['captures'], 2)
        self.assertEqual(pool.counts['reused'], 192)

    def test_the_second_blocks_capture_allocates_no_history_so_it_holds_1_125_gib_less(self):
        def history_allocations(environ):
            self.setUp()
            before = len(self.ttnn.hosts)
            seen = []
            original = self.ttnn.empty
            self.ttnn.empty = lambda shape, **options: (seen.append(tuple(shape)), original(shape, **options))[1]
            self.build(environ)
            return sum(1 for shape in seen if shape == (16, 12, 128, 128))

        # the warm forwards allocate privately in both cases (2 x 192); the captures are 2 x 192 off, 1 x 192 on
        self.assertEqual(history_allocations({}), 4 * 192)
        self.assertEqual(history_allocations({shared.FLAG: '1'}), 3 * 192)

    def test_a_flag_the_environment_cannot_honour_stops_the_attach_with_the_reason_logged(self):
        for environ, message in (({shared.FLAG: '1', 'QWEN_FAST_TP': '2'}, 'QWEN_FAST_TP=4'),
                                 ({shared.FLAG: '1', 'QWEN_FAST_GDN_SEQ_BLOCK': '1', 'QWEN_FAST_GDN_USER_BATCH': '1'}, 'K5-A'),
                                 ({shared.GROW_FLAG: '1'}, 'needs ' + shared.FLAG)):
            with self.subTest(environ=environ):
                self.setUp()
                with patch.dict(os.environ, {**BASE_ENV, **environ}):
                    with self.assertRaisesRegex(ValueError, message):
                        self.block((0, 1, 2, 3))
                self.assertTrue(any(line.startswith(shared.REFUSED_MARKER) for line in self.lines))
                self.assertIsNone(shared.current())

    def test_a_malformed_flag_is_not_read_as_off(self):
        with patch.dict(os.environ, {**BASE_ENV, shared.FLAG: 'on'}):
            with self.assertRaisesRegex(ValueError, shared.FLAG):
                self.block((0, 1, 2, 3))


class ExactnessTests(Rig):
    ROUNDS = 12

    def run_schedule(self, environ, deferred=False, seed=1):
        self.setUp()
        self.build(environ)
        rounds = random_rounds(self.ROUNDS, seed)
        for number, prefixes in enumerate(rounds, 1):
            self.step(number, prefixes, deferred=deferred)
        return self.carries(), rounds

    def test_blocking_commits_share_and_stay_exact_at_no_cost(self):
        private, rounds = self.run_schedule({})
        sharing, unused = self.run_schedule({shared.FLAG: '1'})
        self.assertEqual(sharing, private)
        self.assertEqual(sharing, reference(rounds))
        self.assertNotEqual(set(sharing.values()), {0})
        pool = self.blocks[0].shared_history
        self.assertEqual(pool.counts['flushes'], 0, 'commits already run before the other block verifies: nothing to move')
        self.assertEqual(pool.counts['claims'], 2 * self.ROUNDS)

    def test_pipelined_commits_share_and_stay_exact(self):
        environ = {'QWEN_FAST_PIPELINED_COMMITS': '1'}
        private, rounds = self.run_schedule(environ)
        sharing, unused = self.run_schedule({**environ, shared.FLAG: '1'})
        self.assertEqual(sharing, private)
        self.assertEqual(sharing, reference(rounds))
        self.assertFalse(all(blocking for blocking in self.ttnn.execute_blocking[-50:]), 'the commits were enqueued, not blocking')

    def test_deferred_commits_are_flushed_before_the_other_blocks_verify_and_stay_exact(self):
        private, rounds = self.run_schedule(DEFERRED_ENV, deferred=True)
        sharing, unused = self.run_schedule({**DEFERRED_ENV, shared.FLAG: '1'}, deferred=True)
        self.assertEqual(private, reference(rounds))
        self.assertEqual(sharing, private)
        pool = self.blocks[0].shared_history
        self.assertGreater(pool.counts['flushes'], 0, 'block A\'s deferred commits were enqueued by block B\'s claim')
        self.assertTrue(any('site=shared-history' in line for line in self.lines) or pool.counts['flushed_commits'] > 0)
        # the order on the device queue: after each verify, exactly that block's commits run before the other block's verify
        self.assertEqual(self.commit_groups_are_the_verifying_blocks(self.device.events, rounds), [])

    def commit_groups_are_the_verifying_blocks(self, events, rounds):
        """Problems with the event order: a commit that ran after the OTHER block's verify, or a group of the wrong size."""
        problems, groups = [], []
        for event in events:
            if event[0] == 'verify':
                groups.append((event[1], []))
            elif groups:                         # the commit traces the construction ran once (warming) precede every verify
                groups[-1][1].append(event[1])
        for index, (label, commits) in enumerate(groups):
            block_index, round_number = 'AB'.index(label), index // 2
            wanted = sum(1 for prefix in rounds[round_number][block_index] if prefix)
            if commits != [label] * wanted * LAYERS_PER_COMMIT:
                problems.append((index, label, sorted(set(commits)), len(commits), wanted))
        return problems

    def test_the_hazard_is_real_the_deferred_schedule_without_the_guard_reads_the_other_blocks_rows(self):
        reference_run, rounds = self.run_schedule(DEFERRED_ENV, deferred=True)

        def unguarded(pool, block):
            pool.writer = block
            pool.counts['claims'] += 1

        with patch.object(shared.SharedHistory, 'claim', unguarded):
            broken, unused = self.run_schedule({**DEFERRED_ENV, shared.FLAG: '1'}, deferred=True)
        self.assertEqual(reference_run, reference(rounds))
        self.assertNotEqual(broken, reference_run, 'without commit-before-reuse the sharing corrupts the carries')
        wrong = [key for key in broken if broken[key] != reference_run[key]]
        self.assertTrue(wrong)
        self.assertTrue(all(slot < 4 for slot, layer in wrong), 'only block A\'s users are read after block B overwrote their rows')

    def test_a_round_where_one_block_sits_out_or_a_user_commits_nothing_stays_exact(self):
        self.build({shared.FLAG: '1'})
        rounds = [[[0, 0, 0, 0], [3, 16, 1, 2]], [[5, 1, 0, 16], [0, 0, 0, 0]], [[2, 2, 2, 2], [4, 4, 4, 4]]]
        for number, prefixes in enumerate(rounds, 1):
            self.step(number, prefixes)
        self.assertEqual(self.carries(), reference(rounds))

    def test_a_block_that_verifies_twice_in_a_row_reclaims_without_a_flush(self):
        self.build({shared.FLAG: '1'})
        self.verify(0, 1)
        self.decide(0, [3, 3, 3, 3])
        self.verify(0, 2)
        self.decide(0, [1, 2, 3, 4])
        pool = self.blocks[0].shared_history
        self.assertEqual((pool.counts['flushes'], pool.counts['claims']), (0, 2))
        rounds = [[[3, 3, 3, 3], [0, 0, 0, 0]], [[1, 2, 3, 4], [0, 0, 0, 0]]]
        self.assertEqual(self.carries(), reference(rounds))


class RefusalTests(Rig):
    def test_a_verify_while_the_other_block_has_undecided_segments_is_refused_and_leaves_the_block_idle(self):
        blocks = self.build({shared.FLAG: '1'})
        self.verify(0, 1)                      # A verified, nothing decided
        with self.assertRaisesRegex(ValueError, 'Commit-before-reuse refused'):
            self.verify(1, 1)
        self.assertEqual(blocks[1].phase, 'idle')
        self.assertEqual(blocks[0].phase, 'verified')
        self.decide(0, [2, 2, 2, 2])
        self.verify(1, 1)                      # now it may
        self.decide(1, [1, 1, 1, 1])
        self.assertEqual(self.carries(), reference([[[2, 2, 2, 2], [1, 1, 1, 1]]]))

    def test_deferred_commits_that_cannot_be_enqueued_leave_the_other_block_refused(self):
        blocks = self.build({**DEFERRED_ENV, shared.FLAG: '1'})
        for block in blocks:
            block.arm_deferred_commits()
        self.verify(0, 1)
        self.decide(0, [2, 2, 2, 2])
        self.assertEqual(len(blocks[0].deferred_commits), 4)
        with patch.object(PackedVerifierEngine, 'flush_commits', Mock(return_value=0)):
            with self.assertRaisesRegex(ValueError, '4 deferred commit traces not enqueued'):
                self.verify(1, 1)
        self.assertEqual(blocks[1].phase, 'idle')

    def test_a_failed_block_never_holds_the_other_one_up(self):
        blocks = self.build({shared.FLAG: '1'})
        self.verify(0, 1)
        blocks[0].phase = 'failed'
        self.verify(1, 1)
        self.assertEqual(blocks[1].phase, 'verified')


class OwnershipTests(Rig):
    def test_closing_one_block_frees_no_history_and_closing_the_last_frees_each_tensor_once(self):
        blocks = self.build({shared.FLAG: '1'})
        pool = blocks[0].shared_history
        names = [id(value) for value in pool.tensors]
        # the retained records' own release (FakeRetained does none; the real one calls release_owned on `owned`, which skips the pool's)
        import tp_addresses

        for block in blocks:
            owned = [piece['states'] for state, result, carries in block.fixture.retained.records for piece in result['segment_results']]
            tp_addresses.release_owned(self.ttnn, owned)
        self.assertEqual([entry for entry in self.ttnn.deallocated if id(entry) in set(names)], [])
        blocks[0].close()
        self.assertEqual([entry for entry in self.ttnn.deallocated if id(entry) in set(names)], [])
        self.assertEqual(blocks[0].phase, 'closed')
        self.assertIsNone(blocks[0].shared_history)
        blocks[1].close()
        freed = [id(entry) for entry in self.ttnn.deallocated if id(entry) in set(names)]
        self.assertEqual(sorted(freed), sorted(names))
        self.assertTrue(any(line.startswith(shared.CLOSED_MARKER) for line in self.lines))

    def test_a_block_that_fails_its_capture_leaves_the_others_history_alone_and_the_pool_survives_until_the_last_close(self):
        self.setUp()
        with patch.dict(os.environ, {**BASE_ENV, shared.FLAG: '1'}):
            blocks = [self.block((0, 1, 2, 3)), self.block((4, 5, 6, 7))]
            self.blocks = blocks
            for block in blocks:
                block.warm_and_fixture()
            blocks[0].capture_traces()
            pool = blocks[0].shared_history
            tensors = list(pool.tensors)
            failure = RuntimeError('capture refused')
            with patch.object(packed_verifier, 'prepare', Mock(side_effect=failure)):
                with self.assertRaisesRegex(RuntimeError, 'capture refused'):
                    blocks[1].capture_traces()
        self.assertEqual(blocks[1].phase, 'closed')
        self.assertFalse(pool.closed)
        self.assertEqual(pool.tensors, tensors)
        self.assertFalse([entry for entry in self.ttnn.deallocated if any(entry is value for value in tensors)])
        self.assertFalse(shared.holds(object()))
        self.assertTrue(all(shared.holds(value) for value in tensors))
        blocks[0].close(wait=False)
        self.assertTrue(pool.closed)


if __name__ == '__main__':
    unittest.main()
