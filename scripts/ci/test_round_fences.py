"""Round-fence plan H1a, the fence diet (QWEN_FAST_ROUND_FENCES=1, default off): on fake operations,
gdn_records.RetainedGDNBlock drops the fence and the second validate after the blocking trace (F3),
its last commit owes its fence (F8) to the drafts' fence F9 (note_round_fence) or to the next replay,
and its first commit skips the validate the round's replay already ran (validated_this_round) - while
a replay still needs every user decided and a fence after the last decision, and a poisoned block
still refuses. Then the same through the real packed block (test_packed_verifier's four-user fixture
over a retained block of this class), through the drafts' window (verify_prestage.WhileWaiting) and
the coordinator's fence, and the step's FENCES_MARKER line.

With the flag off every module this touches is proved call for call its PARENT copy: gdn_records
over the same sequence of decisions and replays, the packed block's rounds (synchronize and validate
counts, traces, copies), and the coordinator's prepare()."""

from itertools import count
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from dflash_packed_proposal_coordinator import note_fenced, run_while_waiting
import gdn_records
from gdn_records import RetainedGDNBlock
import packed_verifier
import serving_packed_step
import test_gdn_records as tgr
import test_packed_verifier as tpv
import verifier_engine
import verify_prestage

# H1a's parent: the gate-only commit after 2f047a02; every module H1a touches is 2f047a02's there.
PARENT = '81acfec9'
M3 = tgr.M3_SEGMENTS


def parent_module(relative):
    from test_padded_block import parent_module as load

    return load(relative, PARENT)


def clean_environment(**flags):
    environ = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
    environ.update(flags)
    return patch.dict(os.environ, environ, clear=True)


def counted(block):
    """The block with its validate_bindings counted (the real check still runs)."""
    block.validate_bindings = Mock(side_effect=block.validate_bindings)
    return block


def decide_all(block, prefixes=(3, 0, 16, 5), *, validated=None, publication=None):
    """Every user decides, the last with synchronize=True, as PackedVerifierEngine.commit_user does."""
    for segment, prefix in enumerate(prefixes):
        options = dict(dma=True, synchronize=segment == len(prefixes) - 1,
                       publication=publication or Mock(return_value=None))
        if validated is not None:
            options['validated_this_round'] = validated
        block.commit_user(segment, prefix, **options)


class FlagTests(unittest.TestCase):
    def test_the_flag_is_zero_or_one(self):
        self.assertFalse(verify_prestage.round_fences_enabled({}))
        self.assertFalse(verify_prestage.round_fences_enabled({'QWEN_FAST_ROUND_FENCES': '0'}))
        self.assertTrue(verify_prestage.round_fences_enabled({'QWEN_FAST_ROUND_FENCES': '1'}))
        for value in ('true', '2', ''):
            with self.subTest(value=value), self.assertRaises(ValueError):
                verify_prestage.round_fences_enabled({'QWEN_FAST_ROUND_FENCES': value})

    def test_a_block_is_off_until_its_owner_turns_it_on(self):
        block = tgr.packed_block(segments=M3)
        self.assertFalse(block.round_fences)
        with clean_environment(QWEN_FAST_ROUND_FENCES='1'):
            # Never read from the environment: a sequential engine's retained block keeps its fences.
            self.assertFalse(tgr.packed_block(segments=M3).round_fences)
        block.use_round_fences()
        self.assertTrue(block.round_fences)


class RetainedFlagOffTests(unittest.TestCase):
    """Without use_round_fences: today's synchronize and validate counts."""

    def test_the_last_commit_fences_and_the_replay_fences_and_validates_twice(self):
        block = counted(tgr.packed_block(segments=M3))
        sync = block.operations.synchronize_device
        decide_all(block)
        self.assertEqual((sync.call_count, block.validate_bindings.call_count), (1, 4))
        self.assertTrue(block.replay_ready)
        block.replay(Mock(return_value=None))
        self.assertEqual((sync.call_count, block.validate_bindings.call_count), (2, 6))
        self.assertIsNone(block.replay_fence)

    def test_validated_this_round_changes_nothing_without_the_flag(self):
        block = counted(tgr.packed_block(segments=M3))
        decide_all(block, validated=True)
        self.assertEqual((block.operations.synchronize_device.call_count, block.validate_bindings.call_count), (1, 4))

    def test_the_sequence_is_the_parents_call_for_call(self):
        parent = parent_module('gdn_records.py')
        if parent is None:
            self.skipTest('no git history for %s' % PARENT)

        def run(module):
            with clean_environment(QWEN_FAST_FAST_COMMIT='1'), patch.object(tgr, 'RetainedGDNBlock',
                                                                               module.RetainedGDNBlock):
                block = counted(tgr.packed_block(segments=M3))
            record = []
            block.operations.synchronize_device.side_effect = lambda mesh: record.append('sync')
            block.validate_bindings.side_effect = lambda: record.append('validate')
            for epoch in range(3):
                publication = Mock(side_effect=lambda prefix: record.append(('publish', prefix)))
                decide_all(block, prefixes=(epoch, 16, 0, 7), publication=publication)
                record.append(('ready', block.replay_ready, block.selected_prefix))
                block.replay(Mock(side_effect=lambda: record.append('trace')))
                record.append(('epoch', block.replay_epoch, block.selected_prefix, dict(block.decisions)))
            return record

        self.assertEqual(run(gdn_records), run(parent))


class RetainedRoundFenceTests(unittest.TestCase):
    def fenced(self, fast_commit=True):
        with clean_environment(**({'QWEN_FAST_FAST_COMMIT': '1'} if fast_commit else {})):
            block = counted(tgr.packed_block(segments=M3))
        block.use_round_fences()
        return block

    def test_the_last_commit_owes_its_fence_and_the_replay_pays_it_once(self):
        block = self.fenced()
        sync = block.operations.synchronize_device
        decide_all(block)
        self.assertEqual(sync.call_count, 0, 'F8 is gone from the last commit')
        self.assertTrue(block.fence_owed)
        self.assertFalse(block.replay_ready)
        operation = Mock(return_value=None)
        block.replay(operation)
        operation.assert_called_once_with()
        self.assertEqual(sync.call_count, 1, 'one fence at replay, none after the blocking trace (F3)')
        self.assertEqual((block.replay_fence, block.fence_owed, block.replay_ready), ('replay', False, False))
        self.assertEqual(block.replay_epoch, 1)

    def test_the_drafts_fence_arms_the_replay_and_the_replay_fences_nothing(self):
        block = self.fenced()
        sync = block.operations.synchronize_device
        decide_all(block)
        token = block.commit_serial
        self.assertTrue(block.note_round_fence(token))
        self.assertEqual((block.replay_ready, block.fence_owed, block.replay_fence), (True, False, 'f9'))
        block.replay(Mock(return_value=None))
        self.assertEqual(sync.call_count, 0)
        self.assertFalse(block.note_round_fence(token), 'nothing owed after the replay')

    def test_f3_the_replay_validates_once_before_its_trace(self):
        block = self.fenced()
        decide_all(block)
        before = block.validate_bindings.call_count
        order = []
        block.validate_bindings.side_effect = lambda: order.append('validate')
        block.replay(Mock(side_effect=lambda: order.append('trace')))
        self.assertEqual(order, ['validate', 'trace'])
        self.assertEqual(block.validate_bindings.call_count, before + 1)

    def test_the_first_commit_skips_the_validate_its_rounds_replay_ran(self):
        block = self.fenced()
        decide_all(block, validated=True)
        self.assertEqual(block.validate_bindings.call_count, 0, 'FAST_COMMIT skips the rest, the round the first')
        slow = self.fenced(fast_commit=False)
        decide_all(slow, validated=True)
        self.assertEqual(slow.validate_bindings.call_count, 3, 'only the first is skipped without FAST_COMMIT')
        unvalidated = self.fenced()
        decide_all(unvalidated, validated=False)
        self.assertEqual(unvalidated.validate_bindings.call_count, 1)

    def test_a_replay_without_a_decided_commit_is_refused(self):
        block = self.fenced()
        for segment, prefix in enumerate((3, 0, 16)):
            block.commit_user(segment, prefix, dma=True, synchronize=segment == 2, publication=Mock())
        self.assertFalse(block.fence_owed, 'three of four decided owe nothing')
        self.assertFalse(block.note_round_fence(block.commit_serial))
        with self.assertRaisesRegex(ValueError, 'synchronized commit is required'):
            block.replay(Mock(return_value=None))
        self.assertEqual(block.operations.synchronize_device.call_count, 0)
        # and a block that never decided at all
        fresh = self.fenced()
        with self.assertRaisesRegex(ValueError, 'synchronized commit is required'):
            fresh.replay(Mock(return_value=None))

    def test_a_decision_after_the_token_is_not_armed_by_that_fence(self):
        block = self.fenced()
        for segment, prefix in enumerate((3, 0, 16)):
            block.commit_user(segment, prefix, dma=True, publication=Mock())
        token = block.commit_serial
        block.commit_user(3, 5, dma=True, synchronize=True, publication=Mock())
        self.assertFalse(block.note_round_fence(token))
        self.assertTrue(block.fence_owed)
        block.replay(Mock(return_value=None))
        self.assertEqual((block.replay_fence, block.operations.synchronize_device.call_count), ('replay', 1))

    def test_a_poisoned_block_still_refuses(self):
        block = self.fenced()
        for segment, prefix in enumerate((3, 0, 16)):
            block.commit_user(segment, prefix, dma=True, publication=Mock())
        with self.assertRaises(RuntimeError):
            block.commit_user(3, 5, dma=True, synchronize=True, publication=Mock(side_effect=RuntimeError('dma')))
        self.assertTrue(block.poisoned)
        self.assertFalse(block.note_round_fence(block.commit_serial))
        with self.assertRaisesRegex(ValueError, 'synchronized commit is required'):
            block.replay(Mock(return_value=None))
        self.assertEqual(block.operations.synchronize_device.call_count, 0)
        with self.assertRaisesRegex(ValueError, 'Exactly one decision per user'):
            block.commit_user(0, 1, dma=True, publication=Mock())
        # owed and then poisoned (a later failure on the same block): still refused
        owed = self.fenced()
        decide_all(owed)
        owed.poisoned = True
        self.assertFalse(owed.note_round_fence(owed.commit_serial))
        with self.assertRaisesRegex(ValueError, 'synchronized commit is required'):
            owed.replay(Mock(return_value=None))

    def test_a_closed_block_owes_nothing(self):
        block = self.fenced()
        decide_all(block)
        with patch('gdn_records.release_owned'):
            block.close()
        self.assertFalse(block.fence_owed)
        self.assertFalse(block.note_round_fence(block.commit_serial))


class CountingRetained(RetainedGDNBlock):
    """The real retained block over test_packed_verifier's fake layer records: its binding check
    counted instead of read, its mesh the fake model's, its spans from the records."""

    def __init__(self, rows, operations, mesh):
        super().__init__(rows, operations)
        self.mesh, self.validations = mesh, 0

    def validate_bindings(self):
        self.validations += 1

    def bound_mesh(self):
        return self.mesh

    def validate_segment(self, segment):
        if self.segments is None and self.records:
            self.segments = tuple(tuple(span) for span in self.records[0][1]['segments'])
        return super().validate_segment(segment)


def fenced_model_batch(retained_class):
    class FencedModelBatch(tpv.FakeModelBatch):
        def __init__(self, model, tokens, start, pages, helpers, checkpoints, prefix, **options):
            super().__init__(model, tokens, start, pages, helpers, checkpoints, prefix, **options)
            if self.retained is not None:
                self.retained = retained_class(self.rows, type(self).ttnn, model.mesh_device)

    return FencedModelBatch


class BlockRoundFenceTests(tpv.FourUserFixture):
    """The packed block over the real retained block: what F1, F3, F8 and F9 cost per round."""

    def setUp(self):
        super().setUp()
        environment = clean_environment(QWEN_FAST_FAST_COMMIT='1', QWEN_FAST_PIPELINED_COMMITS='1')
        environment.start()
        self.addCleanup(environment.stop)
        self.lines = []
        for patcher in (patch.object(packed_verifier, 'ModelBatch', fenced_model_batch(CountingRetained)),
                        patch.object(packed_verifier, 'diagnostic', side_effect=self.lines.append),
                        patch.object(serving_packed_step, 'audit_log',
                                     side_effect=lambda message, **values: self.lines.append(message.format(**values)))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def serve(self, block, rounds, window=None):
        """`rounds` verify+commit rounds; `window(block)` between them (the drafts). Per round:
        (fences the verify issued, fences the commits issued, validates of the retained block)."""
        record = []
        for number in range(rounds):
            retained = block.fixture.retained
            synced, validated = self.ttnn.synchronized, retained.validations
            predictions, metrics = block.verify(self.four())
            verified = self.ttnn.synchronized
            for segment, prefix in zip(metrics['segments'], (9, 16, 0, 4)):
                block.commit_user(segment, prefix)
            record.append((verified - synced, self.ttnn.synchronized - verified, retained.validations - validated,
                           metrics.get('replay_fence'), metrics.get('validated_this_round')))
            if window is not None:
                window(block)
        return record

    def test_flag_off_every_steady_round_fences_three_times_and_validates_three_times(self):
        block = self.build()
        self.assertFalse(block.round_fences)
        rounds = self.serve(block, 4)
        # round 1 runs its trace plainly (F1 and the first-round fence), then F8 at the last commit
        self.assertEqual(rounds[0], (2, 1, 1, None, None))
        # F1, the replay's post-trace fence (F3) | F8 ; replay validates twice + the first commit
        self.assertEqual(rounds[1:], [(2, 1, 3, None, None)] * 3)
        self.assertFalse(any(verify_prestage.FENCES_ENGAGED_MARKER in line for line in self.lines))

    def test_flag_on_without_a_drafts_fence_the_replay_pays_f8_and_f3_goes(self):
        os.environ['QWEN_FAST_ROUND_FENCES'] = '1'
        block = self.build()
        self.assertTrue(block.round_fences and block.fixture.retained.round_fences)
        self.assertIn('%s users=4' % verify_prestage.FENCES_ENGAGED_MARKER, self.lines)
        rounds = self.serve(block, 4)
        # round 1 runs its trace plainly: nothing validated it, so its first commit validates
        self.assertEqual(rounds[0], (2, 0, 1, 'first', False))
        # F1 and the fence at replay | no F8 ; the replay validates once, the first commit skips
        self.assertEqual(rounds[1:], [(2, 0, 1, 'replay', True)] * 3)

    def test_flag_on_the_drafts_fence_arms_the_replay(self):
        os.environ['QWEN_FAST_ROUND_FENCES'] = '1'
        block = self.build()

        def drafts(block):
            waiting = verify_prestage.WhileWaiting(block, [])
            run_while_waiting(waiting)
            self.ttnn.synchronize_device(self.model.mesh_device)  # the coordinator's F9
            note_fenced(waiting)

        rounds = self.serve(block, 4, window=drafts)
        self.assertEqual(rounds[1:], [(1, 0, 1, 'f9', True)] * 3, 'only F1 is left at the verify')

    def test_the_block_passes_validated_this_round_only_under_the_flag(self):
        block = self.build()
        calls = []
        original = block.fixture.retained.commit_user
        block.fixture.retained.commit_user = lambda *args, **options: calls.append(sorted(options)) or original(
            *args, **options)
        self.serve(block, 2)
        self.assertEqual(calls[0], ['dma', 'publication', 'synchronize'])
        block.close()
        os.environ['QWEN_FAST_ROUND_FENCES'] = '1'
        fenced = self.build()
        calls.clear()
        original = fenced.fixture.retained.commit_user
        fenced.fixture.retained.commit_user = lambda *args, **options: calls.append(
            (sorted(options), options['validated_this_round'])) or original(*args, **options)
        self.serve(fenced, 2)
        self.assertEqual(calls[0], (['dma', 'publication', 'synchronize', 'validated_this_round'], False))
        self.assertEqual(calls[4], (['dma', 'publication', 'synchronize', 'validated_this_round'], True))

    def test_a_window_before_every_commit_is_decided_arms_nothing(self):
        os.environ['QWEN_FAST_ROUND_FENCES'] = '1'
        block = self.build()
        block.verify(self.four())  # verified, nothing committed yet
        waiting = verify_prestage.WhileWaiting(block, [])
        run_while_waiting(waiting)
        note_fenced(waiting)
        self.assertFalse(block.fixture.retained.replay_ready)
        with self.assertRaisesRegex(ValueError, 'every segment of the last round must be committed'):
            block.verify(self.four())

    def test_fence_token_and_note_round_fence_are_inert_without_the_flag(self):
        block = self.build()
        self.assertFalse(block.note_round_fence(0))

    def test_the_step_logs_one_fence_line_per_round(self):
        os.environ.update(QWEN_FAST_ROUND_FENCES='1', QWEN_FAST_PACKED_AUDIT='1')
        block = self.build()
        from test_serving_packed_step import FakeRequest, entry as step_entry

        stepped = []
        owners = []
        for index, name in enumerate('ABCD'):
            owner = FakeRequest(name, self.POSITIONS[index], stepped, carry=self.pool.slots[index].verifier.carry)
            owner.engine.pages = torch.full((1, tpv.PAGE_WIDTH), self.PAGES[index], dtype=torch.int32)
            owner.engine.widths = (1, 2, 4)
            owners.append(owner)
        for number in range(3):
            for index, owner in enumerate(owners):
                owner.session.pending, owner.session.phase = None, 'idle'
                owner.propose(list(range(16 * index, 16 * index + 16)), accept=index * 3, rows=16)
            serving_packed_step.packed_device_step([step_entry(owner) for owner in owners], cancelled=lambda: False,
                                                   block=block)
        fences = [line for line in self.lines if line.startswith(verify_prestage.FENCES_MARKER)]
        self.assertEqual(len(fences), 3)
        self.assertTrue(fences[0].startswith('[PACKED-FENCES] round=1 fence=first validated=0 '))
        self.assertTrue(fences[1].startswith('[PACKED-FENCES] round=2 fence=replay validated=1 '))
        self.assertIn(' path=off prestage_ms=0.00 ', fences[1])
        import lever_n_m3native_gate as gate

        parsed = [match.groups() for match in gate.FENCES_LINE.finditer(chr(10).join(fences))]
        self.assertEqual([line[:3] for line in parsed], [('1', 'first', '0'), ('2', 'replay', '1'), ('3', 'replay', '1')])


class BlockParentTests(tpv.FourUserFixture):
    """Flag off, the packed block over gdn_records' real retained class, against the PARENT copies
    of both modules: the same fences, validates, traces and copies, round for round."""

    def run_rounds(self, verifier_module, records_module, rounds=4):
        verifier_engine.note_prefill()
        tpv.FakeModelBatch.instances, tpv.FakeFeatures.instances = [], []
        ttnn = tpv.FakeTTNN()
        self.ttnn = ttnn
        tpv.FakeModelBatch.ttnn = ttnn
        self.helpers = tpv.helpers(ttnn)
        self.pool = tpv.pool(ttnn, self.helpers, users=4, packed={(4, 16): tpv.packed_tables(ttnn, users=4)})
        self.traces = count(1)

        class Counting(records_module.RetainedGDNBlock):
            def __init__(self, rows, operations, mesh):
                super().__init__(rows, operations)
                self.mesh, self.validations = mesh, 0

            def validate_bindings(self):
                self.validations += 1

            def bound_mesh(self):
                return self.mesh

            def validate_segment(self, segment):
                if self.segments is None and self.records:
                    self.segments = tuple(tuple(span) for span in self.records[0][1]['segments'])
                return records_module.RetainedGDNBlock.validate_segment(self, segment)

        with clean_environment(QWEN_FAST_FAST_COMMIT='1', QWEN_FAST_PIPELINED_COMMITS='1'), \
                patch.object(verifier_module, 'ModelBatch', fenced_model_batch(Counting)), \
                patch.object(verifier_module, 'PreparedTargetFeatures', tpv.FakeFeatures), \
                patch.object(verifier_module, 'prepare', packed_verifier.prepare), \
                patch.object(verifier_module, 'capture_operation', packed_verifier.capture_operation), \
                patch.object(verifier_module, 'sample_rows', packed_verifier.sample_rows):
            block = verifier_module.PackedVerifierEngine(ttnn, self.model, self.helpers, 'sampler', pool=self.pool,
                                                         shared_weights=self.weights, shape=self.shape(),
                                                         feature_taps=tpv.TAPS)
            record = []
            for number in range(rounds):
                before = ttnn.synchronized
                predictions, metrics = block.verify(self.four())
                for segment, prefix in zip(metrics['segments'], (9, 16, 0, 4)):
                    block.commit_user(segment, prefix)
                record.append((predictions, metrics['segments'], metrics['staged_buffers'], sorted(metrics),
                               ttnn.synchronized - before, block.fixture.retained.validations,
                               block.fixture.retained.replay_epoch))
            record.append((list(ttnn.executed), list(ttnn.execute_blocking), len(ttnn.host_copies),
                           [host.value.tolist() for host in ttnn.hosts]))
            block.close()
            return record

    def test_the_rounds_are_the_parents(self):
        parent_verifier, parent_records = parent_module('packed_verifier.py'), parent_module('gdn_records.py')
        if parent_verifier is None or parent_records is None:
            self.skipTest('no git history for %s' % PARENT)
        today = self.run_rounds(packed_verifier, gdn_records)
        before = self.run_rounds(parent_verifier, parent_records)
        self.assertEqual(today, before)


class CoordinatorTests(unittest.TestCase):
    """dflash_packed_proposal_coordinator.prepare(while_waiting=): the callable runs after every pair
    is enqueued and before the one fence, `fenced` right after it, only when there is a fence; its
    failure drops the snapshot and never the round."""

    def setUp(self):
        from test_dflash_packed_proposal_coordinator import FakeTrace

        FakeTrace.instances = []
        patcher = patch('dflash_proposal_trace.PreparedPackedDFlashProposal', FakeTrace)
        patcher.start()
        self.addCleanup(patcher.stop)
        environment = clean_environment()
        environment.start()
        self.addCleanup(environment.stop)
        self.order = []
        self.operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: self.order.append('fence')))
        self.mesh = object()

    def bridges(self, count=4):
        from test_dflash_packed_proposal_coordinator import make_bridge, make_device

        bridges = []
        for index in range(count):
            device = make_device(self.operations, self.mesh, slot=index)
            device.prepare_device.side_effect = lambda seed, index=index: self.order.append(('single', index)) or True
            bridges.append(make_bridge('r%d' % index, device, seed=100 + index))
        return bridges

    def window(self, fail=None):
        def call():
            self.order.append('while_waiting')
            if fail is not None:
                raise fail

        waiting = Mock(side_effect=call)
        waiting.fenced = Mock(side_effect=lambda: self.order.append('fenced'))
        waiting.drop = Mock(side_effect=lambda failure: self.order.append(('drop', type(failure).__name__)))
        return waiting

    def test_the_window_runs_after_both_pairs_and_before_the_fence(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator
        from test_dflash_packed_proposal_coordinator import FakeTrace

        original = FakeTrace.prepare_device
        FakeTrace.prepare_device = lambda trace, a, b: self.order.append(('pair', a, b)) or original(trace, a, b)
        self.addCleanup(setattr, FakeTrace, 'prepare_device', original)
        waiting = self.window()
        PackedProposalCoordinator().prepare(self.bridges(), while_waiting=waiting)
        self.assertEqual(self.order, [('pair', 100, 101), ('pair', 102, 103), 'while_waiting', 'fence', 'fenced'])
        waiting.drop.assert_not_called()

    def test_a_failing_window_drops_its_snapshot_and_the_round_goes_on(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        waiting = self.window(fail=RuntimeError('copy'))
        prepared = PackedProposalCoordinator().prepare(self.bridges(), while_waiting=waiting)
        self.assertEqual(len(prepared), 4)
        self.assertEqual(self.order[-4:], ['while_waiting', ('drop', 'RuntimeError'), 'fence', 'fenced'])

    def test_no_fence_no_window(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        waiting = self.window()
        PackedProposalCoordinator().prepare([], while_waiting=waiting)
        self.assertEqual(self.order, [])

    def test_a_failing_fenced_only_costs_the_arming(self):
        waiting = self.window()
        waiting.fenced.side_effect = RuntimeError('arm')
        note_fenced(waiting)  # never raises
        run_while_waiting(SimpleNamespace())  # a callable-less object: TypeError goes nowhere, no drop

    def test_without_the_argument_the_prepare_is_the_parents_call_for_call(self):
        parent = parent_module('dflash_packed_proposal_coordinator.py')
        if parent is None:
            self.skipTest('no git history for %s' % PARENT)
        import dflash_packed_proposal_coordinator as today

        def run(module):
            self.order.clear()
            coordinator = module.PackedProposalCoordinator()
            results = []
            for count in (4, 3, 4):
                bridges = self.bridges(count)
                prepared = coordinator.prepare(bridges)
                results.append([getattr(device.pool_slot, 'index', None) for device in prepared])
            return results, list(self.order)

        self.assertEqual(run(today), run(parent))


class ShippingTests(unittest.TestCase):
    def test_the_suite_runs_in_the_cpu_workflow(self):
        from pathlib import Path

        workflow = (Path(__file__).resolve().parent.parent.parent / '.github' / 'workflows'
                    / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        self.assertRegex(workflow, r'python -B -m unittest [^\n]*\btest_round_fences\b')


if __name__ == '__main__':
    unittest.main()
