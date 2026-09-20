"""Each engine carries its own GDN slot-zero state across the other engine's blocks."""

import io
from itertools import count
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding' / 'harness'))
from greedy_session import GreedySession
import verifier_engine
from verifier_engine import VerifierEngine, capture_bucket_rows, carry_log, note_packed_step, note_prefill
from verifier_pack import GDN_LAYERS

_addresses = count(1)


def tensor():
    """A device tensor stand-in with one stable address on both chips."""
    return SimpleNamespace(buffer_address=Mock(return_value=next(_addresses)))


def helpers():
    """The 48 shared layer helpers; allocate hands out a fresh snapshot each call."""
    return [SimpleNamespace(live=[tensor() for index in range(5)],
                            allocate=Mock(side_effect=lambda: [tensor() for index in range(5)]),
                            save=Mock(), restore=Mock()) for index in range(GDN_LAYERS)]


def engine(request_id, shared, rows=4):
    """A prepared engine over the shared helpers, ending the way __init__ ends."""
    session = GreedySession(request_id, [0, 1, 2] * 12, 0, vocab_size=100, max_new_tokens=64)
    engine = VerifierEngine.__new__(VerifierEngine)
    engine.session, engine.position = session, session.position
    engine.phase, engine.pending = 'idle', None
    engine.mesh = object()
    engine.model = SimpleNamespace(args=SimpleNamespace(vocab_size=100))
    engine.operations = SimpleNamespace(execute_trace=Mock(return_value=None), synchronize_device=Mock(),
        release_trace=Mock(), get_device_tensors=lambda value: [value, value], to_torch=lambda value: value)
    engine.validate_bindings = Mock()
    engine.helpers, engine.initial = shared, []
    retained = Mock() if rows > 1 else None
    if retained is not None:
        retained.replay.side_effect = lambda operation: operation()
        retained.commit.side_effect = lambda prefix, **kwargs: kwargs['publication'](prefix)
    bucket = dict(first=True, fixture=SimpleNamespace(retained=retained, close=Mock()), trace=7,
        output=(None, torch.tensor([1, 2, 0, 1])), commits={prefix: 20 + prefix for prefix in range(5)},
        checkpoints=[[tensor()] for index in range(GDN_LAYERS)])
    engine.buckets = {width: bucket for width in (1, 2, 4) if width <= rows}
    engine.widths = tuple(engine.buckets)
    engine.allocate_carry()
    engine.save_carry()
    return engine, session


class ProposalRowsTests(unittest.TestCase):
    """The engine's ticket width: its widest capture, or the packed block's rows for a round
    the block will serve (the hint the worker hook threads down), and whether a ticket is one
    its own captures hold."""

    def setUp(self):
        note_prefill()

    def test_the_hint_names_the_blocks_rows_while_the_budget_holds_them_and_nothing_otherwise(self):
        shared = helpers()
        engine_, session = engine('A', shared)
        # without the hint: the widest capture, exactly as before
        self.assertEqual(engine_.proposal_rows(), 4)
        # for a block-served round: the block's rows, which this engine never verifies itself
        self.assertEqual(engine_.proposal_rows(packed_rows=16), 16)
        self.assertEqual(engine_.proposal_rows(packed_rows=4), 4)
        # once the budget no longer holds a block, the widest capture again (the seed
        # counts as emitted, so 63 remain at the start)
        session.emitted.extend([1] * 54)
        self.assertEqual(session.max_new_tokens - len(session.emitted), 9)
        self.assertEqual(engine_.proposal_rows(packed_rows=16), 4)
        self.assertEqual(engine_.proposal_rows(), 4)
        session.emitted.extend([1] * 6)
        self.assertEqual(engine_.proposal_rows(packed_rows=16), 2)
        for rows in (3, 32, True, 4.0, 0):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, 'packed block rows'):
                engine_.proposal_rows(packed_rows=rows)

    def test_serves_says_whether_the_engines_own_captures_hold_the_ticket(self):
        shared = helpers()
        engine_, session = engine('A', shared)
        ticket = session.propose(session.request_id, max_rows=4)
        self.assertTrue(engine_.serves(ticket))
        self.assertFalse(engine_.serves(SimpleNamespace(tokens=(0,) * 16, position=session.position)))
        # with a replay plan the plan decides, and a width it lacks is not served
        engine_.replay_plan = SimpleNamespace(select=Mock(side_effect=ValueError('no bucket')))
        self.assertFalse(engine_.serves(ticket))
        engine_.replay_plan = SimpleNamespace(select=Mock(return_value=SimpleNamespace(key=4)))
        self.assertTrue(engine_.serves(ticket))
        session.fail_verification(session.request_id, ticket)


def cycle(engine, session, rows=4, abort=False):
    """One verify and its publication, the way FastRequest.step drives them."""
    with patch('verifier_engine.stage_inputs'):
        ticket = session.propose(session.request_id, max_rows=rows)
        predictions, metrics = engine.verify(ticket)
    if abort:
        session.abort(session.request_id, ticket, engine.publish)
    else:
        session.commit(session.request_id, ticket, predictions, engine.publish)
    return metrics


def reset(shared):
    for helper in shared:
        helper.restore.reset_mock()
        helper.save.reset_mock()


class CarryTests(unittest.TestCase):
    def setUp(self):
        note_prefill()

    def test_a_lone_engine_never_restores_and_saves_after_each_publication(self):
        shared = helpers()
        alone, session = engine('A', shared)
        reset(shared)
        for index in range(2):
            self.assertFalse(cycle(alone, session)['carry_restored'])
        for helper, slot in zip(shared, alone.carry, strict=True):
            helper.restore.assert_not_called()
            self.assertEqual(helper.save.call_args_list, [call(slot), call(slot)])

    def test_the_single_row_branch_saves_on_commit_and_on_abort(self):
        shared = helpers()
        alone, session = engine('A', shared, rows=1)
        reset(shared)
        cycle(alone, session, rows=1)
        cycle(alone, session, rows=1, abort=True)
        for helper, slot, checkpoint in zip(shared, alone.carry, alone.buckets[1]['checkpoints'], strict=True):
            # the abort put the pre-block checkpoint back, and the carry followed it
            helper.restore.assert_called_once_with(checkpoint)
            self.assertEqual(helper.save.call_args_list, [call(slot), call(slot)])

    def test_alternating_engines_each_restore_their_own_carry(self):
        shared = helpers()
        first, first_session = engine('A', shared)
        cycle(first, first_session)
        # B's prefill overwrites the slot; B is seeded from it and is the resident.
        note_prefill()
        second, second_session = engine('B', shared)
        reset(shared)
        self.assertFalse(cycle(second, second_session)['carry_restored'])
        self.assertTrue(cycle(first, first_session)['carry_restored'])
        self.assertTrue(cycle(second, second_session)['carry_restored'])
        for helper, own_first, own_second in zip(shared, first.carry, second.carry, strict=True):
            self.assertIsNot(own_first, own_second)
            self.assertEqual(helper.restore.call_args_list, [call(own_first), call(own_second)])
            self.assertEqual(helper.save.call_args_list, [call(own_second), call(own_first), call(own_second)])

    def test_a_prefill_displaces_the_resident_engine(self):
        shared = helpers()
        alone, session = engine('A', shared)
        cycle(alone, session)
        note_prefill()
        reset(shared)
        self.assertTrue(cycle(alone, session)['carry_restored'])
        self.assertFalse(cycle(alone, session)['carry_restored'])
        for helper, slot in zip(shared, alone.carry, strict=True):
            helper.restore.assert_called_once_with(slot)

    def test_a_failed_verify_after_restoring_still_displaces_the_other_engine(self):
        """A restored itself into the slot and then died; B must not trust its old residency."""
        shared = helpers()
        first, first_session = engine('A', shared)
        cycle(first, first_session)
        note_prefill()
        second, second_session = engine('B', shared)
        cycle(second, second_session)
        first.operations.execute_trace.side_effect = RuntimeError('device failure')
        with patch('verifier_engine.stage_inputs'), self.assertRaises(RuntimeError):
            first.verify(first_session.propose('A', max_rows=4))
        reset(shared)
        self.assertTrue(cycle(second, second_session)['carry_restored'])
        for helper, slot in zip(shared, second.carry, strict=True):
            helper.restore.assert_called_once_with(slot)

    def test_a_moved_carry_is_refused_before_any_copy(self):
        shared = helpers()
        first, first_session = engine('A', shared)
        note_prefill()
        second, second_session = engine('B', shared)
        first.carry[3][0].buffer_address.return_value = -1
        reset(shared)
        with patch('verifier_engine.stage_inputs'), self.assertRaisesRegex(ValueError, 'moved'):
            first.verify(first_session.propose('A', max_rows=4))
        self.assertEqual((first.phase, first_session.phase), ('failed', 'failed'))
        for helper in shared:
            helper.restore.assert_not_called()
        # the same check guards the save that follows publication
        with patch('verifier_engine.stage_inputs'):
            ticket = second_session.propose('B', max_rows=4)
            predictions, unused = second.verify(ticket)
        second.carry[0][4].buffer_address.return_value = -2
        with self.assertRaisesRegex(ValueError, 'moved'):
            second_session.commit('B', ticket, predictions, second.publish)
        self.assertEqual(second.phase, 'failed')
        for helper in shared:
            helper.save.assert_not_called()

    def test_log_lines_are_fenced_bounded_and_only_emitted_when_asked(self):
        """Ids are cut to 48 characters so two of them and the fields stay under 180."""
        first_id, second_id = 'A' * 100, 'B' * 100

        def round_trip(enabled):
            note_prefill()
            shared = helpers()
            environment = {'QWEN_FAST_CARRY_LOG': '1' if enabled else '0'}
            with patch.dict('os.environ', environment), patch('verifier_engine.carry_log') as log:
                first, first_session = engine(first_id, shared)
                cycle(first, first_session)
                note_prefill()
                second, second_session = engine(second_id, shared)
                cycle(second, second_session)
                fences = first.operations.synchronize_device.call_count
                cycle(first, first_session)
                fences = first.operations.synchronize_device.call_count - fences
            return [entry.args[0].format(**entry.kwargs) for entry in log.call_args_list], fences

        lines, fences = round_trip(False)
        self.assertEqual((lines, fences), ([], 0))
        lines, fences = round_trip(True)
        # A's last cycle: one restore and one save, each fenced
        self.assertEqual(fences, 2)
        self.assertTrue(all(len(line) < 180 for line in lines), max(map(len, lines)))
        request, source = 'A' * 48, 'B' * 48
        self.assertEqual(lines[-4], f'[CARRY] op=restore request={request} from={source} layers=48 begin')
        self.assertRegex(lines[-3], rf'^\[CARRY\] op=restore request={request} layers=48 enqueue_ms=\d+\.\d fence_ms=\d+\.\d$')
        self.assertEqual(lines[-2], f'[CARRY] op=save request={request} layers=48 begin')
        self.assertRegex(lines[-1], rf'^\[CARRY\] op=save request={request} layers=48 enqueue_ms=\d+\.\d fence_ms=\d+\.\d$')
        # seeds and lone-engine saves log too, and a resident engine restores nothing
        self.assertEqual([line.split()[1] for line in lines],
                         ['op=save'] * 8 + ['op=restore'] * 2 + ['op=save'] * 2)

    def test_carry_log_prints_when_loguru_is_absent(self):
        with patch.dict(sys.modules, loguru=None), patch('sys.stdout', new_callable=io.StringIO) as output:
            carry_log('[CARRY] op={op} request={request}{origin} layers={layers} begin',
                      op='save', request='A', origin='', layers=48)
        self.assertEqual(output.getvalue().strip(), '[CARRY] op=save request=A layers=48 begin')

    def test_the_carry_is_allocated_beside_initial_before_any_fixture_exists(self):
        """Allocate before capture: the slots must exist before the first fixture, let alone its trace."""
        session = GreedySession('request', [0, 1, 2], 0, vocab_size=100, max_new_tokens=33)
        shared = helpers()
        seen = []

        def fail(engine, *args, **kwargs):
            seen.append((list(engine.carry), list(engine.carry_addresses), [helper.save.call_count for helper in shared]))
            raise RuntimeError('fixture failure')

        operations = SimpleNamespace(synchronize_device=Mock(), get_device_tensors=lambda value: [value, value])
        with patch.dict(sys.modules, ttnn=operations), patch('verifier_engine.release_owned') as release, \
             patch.object(VerifierEngine, 'fixture', autospec=True, side_effect=fail):
            with self.assertRaises(RuntimeError):
                VerifierEngine(SimpleNamespace(mesh_device=object()), session, SimpleNamespace(shape=(1, 1024)), shared)
        (carry, recorded, saves), = seen
        self.assertEqual(len(carry), GDN_LAYERS)
        self.assertEqual(recorded, [[(value.buffer_address(),) * 2 for value in slot] for slot in carry])
        # only the initial snapshot had been saved: the carry is seeded after the last restore
        self.assertEqual(saves, [1] * GDN_LAYERS)
        self.assertIn(call(operations, [value for slot in carry for value in slot]), release.call_args_list)
        self.assertIsNone(verifier_engine._resident)

    def test_close_releases_the_carry_and_gives_up_residency(self):
        shared = helpers()
        alone, session = engine('A', shared)
        carried = [value for slot in alone.carry for value in slot]
        self.assertEqual(len(carried), 5 * GDN_LAYERS)
        self.assertIs(verifier_engine._resident, alone)
        with patch('verifier_engine.release_owned') as release:
            alone.close()
        self.assertIn(call(alone.operations, carried), release.call_args_list)
        self.assertEqual((alone.carry, alone.carry_addresses), ([], []))
        self.assertIsNone(verifier_engine._resident)
        self.assertEqual(alone.phase, 'closed')


def block(rows=4):
    """A packed block (packed_verifier.PackedVerifierEngine) as adopt_packed sees it."""
    taps = ('tap0', 'tap1', 'tap2', 'tap3', 'tap4')
    return SimpleNamespace(rows_per_user=rows, taps=taps, features=Mock(return_value=taps), commit_user=Mock())


class PackedAdoptionTests(unittest.TestCase):
    """A ticket verified by the packed block publishes through the block's per-user commit,
    which writes the accepted state straight into this engine's carry: no save follows,
    no engine is resident afterwards, and the next sequential step restores first."""

    def setUp(self):
        note_prefill()

    def adopt(self, engine, session, packed, segment=1):
        ticket = session.propose(session.request_id, max_rows=4)
        engine.adopt_packed(ticket, packed, segment)
        return ticket

    def test_an_adopted_ticket_publishes_through_the_block_and_saves_no_carry(self):
        shared = helpers()
        alone, session = engine('A', shared)
        alone.retain_feature_taps = (5, 19, 33, 47, 61)
        cycle(alone, session)
        self.assertIs(verifier_engine._resident, alone)
        reset(shared)
        packed = block()
        traces = alone.operations.execute_trace.call_count
        ticket = self.adopt(alone, session, packed)
        self.assertEqual((alone.phase, alone.pending, alone.pending_key), ('verified', ticket, ('packed', 1)))
        seen = []

        def publish(prefix):
            # DFlashRequestRuntime.publish: the features first, then the target publication
            seen.append(alone.verified_features_for_publication(ticket))
            return alone.publish(prefix)

        position = alone.position
        decision = session.commit('A', ticket, [1, 2, 0, 1], publish)
        packed.features.assert_called_once_with(1)
        self.assertEqual(seen, [packed.taps])
        packed.commit_user.assert_called_once_with(1, decision.state_rows)
        self.assertEqual(alone.position, position + decision.state_rows)
        self.assertEqual(alone.position, session.position)
        self.assertEqual((alone.phase, alone.pending, alone.pending_key), ('idle', None, None))
        self.assertNotIn(('packed', 1), alone.buckets)
        for helper in shared:
            helper.save.assert_not_called()
            helper.restore.assert_not_called()
        # no per-request commit trace either: the block's commit was the publication
        self.assertEqual(alone.operations.execute_trace.call_count, traces)
        self.assertIsNone(verifier_engine._resident)
        # slot 0 is not trusted: the next sequential step restores this engine's carry
        self.assertTrue(cycle(alone, session)['carry_restored'])
        for helper, slot in zip(shared, alone.carry, strict=True):
            helper.restore.assert_called_once_with(slot)

    def test_an_abort_after_adoption_publishes_prefix_zero_through_the_block(self):
        shared = helpers()
        alone, session = engine('A', shared)
        reset(shared)
        packed = block()
        ticket = self.adopt(alone, session, packed, segment=0)
        position = alone.position
        session.abort('A', ticket, alone.publish)
        packed.commit_user.assert_called_once_with(0, 0)
        self.assertEqual((alone.position, alone.phase, alone.pending), (position, 'idle', None))
        for helper in shared:
            helper.save.assert_not_called()
        self.assertIsNone(verifier_engine._resident)

    def test_adoption_needs_an_idle_engine_at_the_frontier_and_a_block_holding_its_rows(self):
        shared = helpers()
        alone, session = engine('A', shared)
        ticket = session.propose('A', max_rows=4)
        for name, packed, segment, phase, offset in (('a busy engine', block(), 1, 'verifying', 0),
                                                     ('a moved frontier', block(), 1, 'idle', 1),
                                                     ('a block of other rows', block(rows=16), 1, 'idle', 0),
                                                     ('a block without commits', SimpleNamespace(rows_per_user=4, features=Mock()), 1, 'idle', 0),
                                                     ('a segment index that is not one', block(), '1', 'idle', 0)):
            with self.subTest(name=name):
                alone.phase, alone.position = phase, session.position + offset
                with self.assertRaises(ValueError):
                    alone.adopt_packed(ticket, packed, segment)
                self.assertEqual((alone.pending, getattr(alone, 'pending_key', None)), (None, None))
                self.assertFalse(any(isinstance(key, tuple) for key in alone.buckets))
        alone.phase, alone.position = 'idle', session.position
        with self.assertRaises(ValueError):
            alone.adopt_packed(SimpleNamespace(position=session.position, tokens=ticket.tokens), block(), 1)

    def test_a_failed_packed_commit_fails_the_engine_and_close_still_releases_it(self):
        shared = helpers()
        alone, session = engine('A', shared)
        packed = block()
        packed.commit_user.side_effect = RuntimeError('commit trace failure')
        ticket = self.adopt(alone, session, packed)
        with self.assertRaises(RuntimeError):
            session.commit('A', ticket, [1, 2, 0, 1], alone.publish)
        self.assertEqual((alone.phase, session.phase), ('failed', 'failed'))
        self.assertIn(('packed', 1), alone.buckets)
        with patch('verifier_engine.release_owned'):
            alone.close()
        self.assertEqual((alone.phase, alone.buckets), ('closed', {}))

    def test_note_packed_step_displaces_the_resident_engine(self):
        shared = helpers()
        alone, session = engine('A', shared)
        cycle(alone, session)
        self.assertIs(verifier_engine._resident, alone)
        note_packed_step()
        self.assertIsNone(verifier_engine._resident)
        reset(shared)
        self.assertTrue(cycle(alone, session)['carry_restored'])
        self.assertFalse(cycle(alone, session)['carry_restored'])
        for helper, slot in zip(shared, alone.carry, strict=True):
            helper.restore.assert_called_once_with(slot)


def shaped(shape):
    value = tensor()
    value.shape = shape
    return value


def snapshot_set():
    return [[tensor() for index in range(5)] for layer in range(GDN_LAYERS)]


def verifier_storage(bucket_rows, taps=5, page_width=68):
    """A pool slot's verifier storage (serving_buffer_pool.VerifierSlot) over address stand-ins."""
    from serving_buffer_pool import BucketSlot, VerifierSlot

    buckets = []
    for rows in bucket_rows:
        batch = SimpleNamespace(tokens=shaped((rows, 1)), positions=shaped((rows,)), pages=shaped((rows, page_width)),
                                singleton_pages=shaped((1, page_width)), cos=shaped((1, rows, 1, 64)), sin=shaped((1, rows, 1, 64)),
                                singleton_positions=[shaped((1,)) for row in range(rows)])
        buckets.append(BucketSlot(rows, snapshot_set(), [shaped((1, 1, rows, 5120)) for tap in range(taps)], None, batch))
    return VerifierSlot(snapshot_set(), snapshot_set(), buckets)


class PooledStorageTests(unittest.TestCase):
    """Pooled, the engine keeps nothing it allocated itself: the initial and carried GDN
    state, every bucket's checkpoints and feature taps, and the fixtures' inputs are the
    pool slot's, allocated at attach before any request's trace, and close() frees none
    of them. Unpooled, it allocates exactly as it always has."""

    def setUp(self):
        note_prefill()

    def construct(self, storage=None, taps=(5, 19), budget=3):
        """The constructor over fakes: two widths (1 and 2), no replay, features retained."""
        session = GreedySession('request', [0, 1, 2], 0, vocab_size=100, max_new_tokens=budget)
        shared = helpers()
        fixtures = []

        def fixture(engine, rows, checkpoints, *, retain, position=None, pack=None, storage=None):
            built = SimpleNamespace(rows=rows, checkpoints=checkpoints, storage=storage, close=Mock(), replay_reader=None,
                                    retained=SimpleNamespace(records=[]) if rows > 1 else None)
            fixtures.append(built)
            return built

        def upload(value, **options):
            return shaped(tuple(value.shape))

        operations = SimpleNamespace(synchronize_device=Mock(), get_device_tensors=lambda value: [value, value],
            copy=Mock(), from_torch=Mock(side_effect=upload), deallocate=Mock(), release_trace=Mock(),
            bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
            ShardTensorToMesh=lambda mesh, dim: ('shard', dim), ReplicateTensorToMesh=lambda mesh: 'replicate')
        model = SimpleNamespace(mesh_device=object(), layers=[object()] * 64)
        with patch.dict(sys.modules, ttnn=operations), \
             patch.object(VerifierEngine, 'fixture', autospec=True, side_effect=fixture), \
             patch.object(VerifierEngine, 'operation', return_value=(None, None)), \
             patch.object(VerifierEngine, 'validate_bindings'), \
             patch('verifier_engine.capture_operation', return_value=('trace', (None, None))), \
             patch('verifier_engine.prepare', return_value=Mock()):
            engine = VerifierEngine(model, session, SimpleNamespace(shape=(1, 1024)), shared, retain_feature_taps=taps,
                                    **(dict(storage=storage) if storage is not None else {}))
            return SimpleNamespace(engine=engine, session=session, helpers=shared, fixtures=fixtures,
                                   operations=operations)

    def test_pooled_engine_borrows_every_persistent_buffer_and_allocates_none(self):
        storage = verifier_storage((1, 2, 4, 8))
        built = self.construct(storage)
        engine = built.engine
        self.assertEqual(engine.phase, 'idle')
        self.assertEqual(sorted(engine.buckets), [1, 2])
        for mine, lent in ((engine.initial, storage.initial), (engine.carry, storage.carry)):
            self.assertEqual(len(mine), GDN_LAYERS)
            self.assertTrue(all(a is b for own, slot in zip(mine, lent, strict=True) for a, b in zip(own, slot, strict=True)))
        taken = [bucket for bucket in storage.buckets if bucket.taken]
        self.assertEqual([bucket.rows for bucket in taken], [1, 2])
        for slot in taken:
            bucket = engine.buckets[slot.rows]
            self.assertTrue(all(a is b for own, lent in zip(bucket['checkpoints'], slot.checkpoints, strict=True)
                                for a, b in zip(own, lent, strict=True)))
            # The first two of the slot's five taps, in order.
            self.assertEqual(bucket['target_features'], list(slot.target_features[:2]))
            self.assertIs(bucket['fixture'].storage, slot.batch)
            self.assertIs(bucket['fixture'].checkpoints, bucket['checkpoints'])
        # No per-request allocation: not one helper snapshot, not one upload.
        for helper in built.helpers:
            helper.allocate.assert_not_called()
        built.operations.from_torch.assert_not_called()
        # The carry was seeded and its addresses recorded from the pooled slots.
        self.assertEqual(engine.carry_addresses, engine.slot_addresses())
        self.assertIs(verifier_engine._resident, engine)
        # Closing frees nothing the pool lent - nothing at all, since it owns nothing.
        built.operations.deallocate.reset_mock()
        engine.close()
        self.assertEqual(engine.phase, 'closed')
        built.operations.deallocate.assert_not_called()
        self.assertEqual((engine.borrowed, engine.carry, engine.initial, engine.buckets), ([], [], [], {}))
        for fixture in built.fixtures:
            fixture.close.assert_called_once()

    def test_unpooled_engine_allocates_exactly_as_before_and_frees_it_all(self):
        built = self.construct()
        engine = built.engine
        # initial, carry, and one checkpoint set per bucket
        self.assertEqual([helper.allocate.call_count for helper in built.helpers], [4] * GDN_LAYERS)
        # two taps per bucket, uploaded
        self.assertEqual(built.operations.from_torch.call_count, 4)
        self.assertTrue(all(bucket['fixture'].storage is None for bucket in engine.buckets.values()))
        self.assertEqual(engine.borrowed, [])
        owned = [*(value for snapshot in engine.initial for value in snapshot),
                 *(value for slot in engine.carry for value in slot),
                 *(value for bucket in engine.buckets.values() for snapshot in bucket['checkpoints'] for value in snapshot),
                 *(value for bucket in engine.buckets.values() for value in bucket['target_features'])]
        built.operations.deallocate.reset_mock()
        engine.close()
        freed = [call.args[0] for call in built.operations.deallocate.call_args_list]
        self.assertEqual(len(freed), len(owned))
        self.assertTrue(all(any(value is entry for entry in freed) for value in owned))

    def test_a_request_the_pool_cannot_hold_is_refused_before_any_allocation_or_capture(self):
        storage = verifier_storage((1, 4))
        with self.assertRaisesRegex(ValueError, r'no free 2-row bucket: the slot holds widths \[1, 4\]'):
            self.construct(storage)
        self.assertIsNone(verifier_engine._resident)
        # Refused at take(), so a width that fits is still lent... and released with the slot.
        self.assertEqual([bucket.taken for bucket in storage.buckets], [True, False])
        # Too few taps in the bucket, likewise.
        storage = verifier_storage((1, 2), taps=1)
        with self.assertRaisesRegex(ValueError, 'holds 1 feature taps; this request retains 2'):
            self.construct(storage)
        # A pooled snapshot set that does not match the helpers.
        storage = verifier_storage((1, 2))
        storage.initial = storage.initial[:47]
        with self.assertRaisesRegex(ValueError, 'initial snapshots'):
            self.construct(storage)
        with self.assertRaisesRegex(ValueError, 'lend width buckets'):
            self.construct(object())

    def test_a_pooled_carry_is_still_checked_against_its_recorded_addresses(self):
        storage = verifier_storage((1, 2))
        built = self.construct(storage)
        engine, session = built.engine, built.session
        engine.buckets[2].update(trace=7, output=(None, torch.tensor([1, 2])), commits={prefix: 20 + prefix for prefix in range(3)})
        engine.buckets[2]['fixture'].retained = Mock()
        engine.buckets[2]['fixture'].retained.replay.side_effect = lambda operation: operation()
        engine.operations = SimpleNamespace(execute_trace=Mock(return_value=None), synchronize_device=Mock(),
            get_device_tensors=lambda value: [value, value], to_torch=lambda value: value)
        engine.model = SimpleNamespace(args=SimpleNamespace(vocab_size=100))
        engine.validate_bindings = Mock()
        storage.carry[5][2].buffer_address.return_value = -1
        # Another engine's prefill displaced this one, so the next verify restores the carry.
        note_prefill()
        with patch('verifier_engine.stage_inputs'), self.assertRaisesRegex(ValueError, 'moved'):
            engine.verify(session.propose('request', max_rows=2))
        self.assertEqual(engine.phase, 'failed')

    def test_capture_bucket_rows_bounds_every_plan_the_engine_can_build(self):
        from collections import Counter
        from attention_request_plan import capture_plan

        self.assertEqual(capture_bucket_rows(16, 256, 16), (1, 2, 4, 8, 8, 16, 16))
        self.assertEqual(capture_bucket_rows(16, 3, 16), (1, 2))
        self.assertEqual(capture_bucket_rows(16, 8, 16), (1, 2, 4, 8))
        self.assertEqual(capture_bucket_rows(16, 24, 16), (1, 2, 4, 8, 8, 16))
        self.assertEqual(capture_bucket_rows(32, 512, 32), (1, 2, 4) + (8,) * 3 + (16,) * 3 + (32,) * 3)
        # M3: a 64-row block over the output budget touches two families per width, and
        # the pool's T16 cap leaves today's per-request widths exactly as they were.
        self.assertEqual(capture_bucket_rows(64, 256, 64), (1, 2, 4, 8, 8, 16, 16, 32, 32, 64, 64))
        self.assertEqual(capture_bucket_rows(64, 256, 16), (1, 2, 4, 8, 8, 16, 16))
        for remaining in (1, 2, 7, 8, 9, 24, 100, 255, 256, 257, 300):
            bound = Counter(capture_bucket_rows(16, remaining, 16))
            tight = Counter()
            # Every offset within a family, plus the seam positions of the next one.
            for position in [*range(4096, 4096 + 256), 4607, 4608]:
                plan = capture_plan(position, 65536, 16, remaining, max_verify_rows=16)
                counts = Counter(capture.rows for capture in plan.captures)
                for rows, count in counts.items():
                    self.assertLessEqual(count, bound[rows], (remaining, position, rows))
                    tight[rows] = max(tight[rows], count)
                unreplayed = Counter(verifier_engine.capture_widths(position, 65536, 16, remaining, 16))
                self.assertTrue(all(count <= bound[rows] for rows, count in unreplayed.items()))
            # Reached somewhere, for every width: the bound is not slack by a whole bucket.
            self.assertEqual(tight, bound, remaining)
        for remaining in (0, -1, True, 2.0):
            with self.assertRaises(ValueError):
                capture_bucket_rows(16, remaining, 16)


if __name__ == '__main__':
    unittest.main()
