"""Round-fence plan H2 - the drafts inside the step, the GDN commits after the pairs (early_draft.py;
QWEN_FAST_EARLY_DRAFT and QWEN_FAST_GDN_AFTER_PAIRS, both default off).

What is pinned here, on CPU:
  - the flags, and the arm's and the gate's refusals of GDN_AFTER_PAIRS without its parent or without the
    round fences;
  - the hook: inside execute_model the packed step, then this very hook's _drafts (so the drafts are
    today's body, call for call), then the flush; take_draft_token_ids hands over the cache when nothing
    changed, discards the early tickets and drafts again when something did, re-raises a draft failure,
    and a cache nobody took is dropped at the next step; a failing step still flushes; a prefill step is
    untouched;
  - the coordinator: after_reads runs after the fence and every pair's readback and before the selection,
    only when the batched pairs cover every prepared device;
  - the block, over gdn_records' real retained class: an armed round decides every commit and enqueues no
    trace, the flush enqueues them in decision order and re-owes the fence (an F9 before it cannot arm the
    replay), the next replay pays it; the traces, their order and blocking mode, the copies and the fences
    are an unarmed round's; verify() flushes a held round first (site=verify); a failed block drops;
  - the gate's H2 report and the H1a f9 waiver, the [PACKED] sequence comparison (acceptance_report), the
    image copy lists and the CPU workflow;
  - with every flag off, each module H2 touches is its PARENT (4864077a) call for call.
"""

from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import acceptance_report
import early_draft
import packed_verifier
import serving_packed_step
import serving_worker_hook
import test_packed_verifier as tpv
import verify_prestage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
# H2's parent: the gate-only commit after H1b (c8bd6377) and the admission stagger (d0302c1b); every module
# H2 touches is that tree's there.
PARENT = '4864077a'
EARLY = {'QWEN_FAST_EARLY_DRAFT': '1'}
GDN = {'QWEN_FAST_EARLY_DRAFT': '1', 'QWEN_FAST_GDN_AFTER_PAIRS': '1'}


def clean_environment(**flags):
    environ = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
    environ.update(flags)
    return patch.dict(os.environ, environ, clear=True)


def parent_module(relative):
    from test_padded_block import parent_module as load

    return load(relative, PARENT)


def captured_log():
    """early_draft.log_line captured (its lines, as the server log would hold them)."""
    lines = []
    return lines, patch.object(early_draft, 'log_line', side_effect=lines.append)


# ------------------------------------------------------------------------------------------------------
# Flags
# ------------------------------------------------------------------------------------------------------

class FlagTests(unittest.TestCase):
    def test_both_flags_are_zero_or_one_and_off_by_default(self):
        for reader, flag in ((early_draft.enabled, early_draft.EARLY_DRAFT_FLAG),
                             (early_draft.gdn_after_pairs_enabled, early_draft.GDN_AFTER_PAIRS_FLAG)):
            with self.subTest(flag=flag):
                self.assertFalse(reader({}))
                self.assertFalse(reader({flag: '0'}))
                for value in ('true', '2', ''):
                    with self.assertRaises(ValueError):
                        reader(dict(GDN, **{flag: value}))
        self.assertTrue(early_draft.enabled(EARLY))
        self.assertTrue(early_draft.gdn_after_pairs_enabled(GDN))

    def test_gdn_after_pairs_is_inert_without_the_early_draft(self):
        self.assertFalse(early_draft.gdn_after_pairs_enabled({early_draft.GDN_AFTER_PAIRS_FLAG: '1'}))
        self.assertTrue(early_draft.requested({early_draft.GDN_AFTER_PAIRS_FLAG: '1'}))
        self.assertFalse(early_draft.requested({early_draft.EARLY_DRAFT_FLAG: '0'}))

    def test_the_callers_import_the_module_only_under_a_flag(self):
        self.assertFalse(serving_worker_hook.early_draft_requested({}))
        self.assertFalse(serving_worker_hook.early_draft_requested({'QWEN_FAST_EARLY_DRAFT': '0'}))
        self.assertTrue(serving_worker_hook.early_draft_requested(EARLY))
        with self.assertRaises(ValueError):
            serving_worker_hook.early_draft_requested({'QWEN_FAST_EARLY_DRAFT': 'yes'})
        self.assertFalse(packed_verifier.gdn_after_pairs_requested({}))
        self.assertTrue(packed_verifier.gdn_after_pairs_requested({'QWEN_FAST_GDN_AFTER_PAIRS': '1'}))


# ------------------------------------------------------------------------------------------------------
# The hook
# ------------------------------------------------------------------------------------------------------

class Harness:
    """A FastWorkerHook (of `module`) over two fake bridges and a fake packed step, its packed decode
    patched to record itself: every event lands in `order`."""

    def __init__(self, test, environ, *, arm=True, module=serving_worker_hook):
        from test_serving_worker_hook import WorkerHookTests

        case = WorkerHookTests('test_committed_block_bypasses_baseline_forward_and_sampler')
        self.worker, bridge, _, _ = case.fixture()
        self.order, self.fail_draft, self.fail_step, self.on_step = [], None, None, None
        self.packed_step = SimpleNamespace(
            proposal_rows=Mock(return_value=16),
            arm_deferred_commits=Mock(side_effect=lambda: self.order.append('arm') or arm),
            flush_deferred_commits=Mock(side_effect=lambda site: self.order.append(('flush', site)) or 0))
        self.environ = dict(environ, QWEN_FAST_PIPELINED_PROPOSALS='1', QWEN_FAST_PACKED_PROPOSAL='1')
        with self.environment():
            self.hook = module.FastWorkerHook(self.worker, bridge, cancelled=lambda: False,
                                              packed_step=self.packed_step)
        self.original = self.hook.bridges
        self.bridges = {name: self.bridge(name) for name in 'ab'}
        self.hook.bridges = self.bridges
        self.hook._packed_coordinator = SimpleNamespace(prepare=Mock(side_effect=self.prepare), close=Mock())
        self.scheduled = SimpleNamespace(scheduled_new_reqs=[], scheduled_cached_reqs=SimpleNamespace(req_ids=['a', 'b']),
                                         total_num_scheduled_tokens=32, num_scheduled_tokens={'a': 16, 'b': 16})
        test.addCleanup(self.close)

    def bridge(self, name):
        session = SimpleNamespace(request_id=name, pending=None, phase='idle', position=4096, emitted=[1],
                                  finished=False, seed=7)
        runtime = SimpleNamespace(discard_proposal=Mock(side_effect=lambda: self.order.append(('discard', name))))
        request = SimpleNamespace(session=session, runtime=runtime, closed=False, cancelled=False)
        bridge = SimpleNamespace(request=request, failed=False, state=None, close=Mock())

        def drafts(packed_rows=None):
            self.order.append(('drafts', name, packed_rows))
            if self.fail_draft is not None:
                raise self.fail_draft
            if session.pending is None:
                session.pending, session.phase = ('ticket', name, len(self.order)), 'pending'
            return SimpleNamespace(req_ids=[name], draft_token_ids=[[ord(name), session.position]])

        bridge.drafts = Mock(side_effect=drafts)
        return bridge

    def prepare(self, bridges, **options):
        self.order.append(('prepare', [bridge.request.session.request_id for bridge in bridges], sorted(options)))
        after = options.get('after_reads')
        if after is not None:
            after()
        return []

    @contextmanager
    def environment(self):
        outputs = ModuleType('vllm.v1.outputs')
        outputs.DraftTokenIds = SimpleNamespace
        with clean_environment(**self.environ), patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
            yield

    def execute(self, scheduled=None):
        def step(bridges, scheduled, cancelled, packed_step):
            self.order.append('step')
            if self.fail_step is not None:
                raise self.fail_step
            if self.on_step is not None:
                self.on_step(bridges)
            return 'output'

        with self.environment(), patch('serving_packed_bridge.execute_packed_decode', side_effect=step) as decode:
            result = self.worker.execute_model(self.scheduled if scheduled is None else scheduled)
        self.decode = decode
        return result

    def take(self):
        with self.environment():
            return self.worker.take_draft_token_ids()

    def close(self):
        self.hook.bridges = self.original
        self.hook.close()


def summary(result):
    return None if result is None else (list(result.req_ids), [list(tokens) for tokens in result.draft_token_ids])


class HookTests(unittest.TestCase):
    def test_the_step_drafts_inside_execute_model_and_take_hands_over_the_cache(self):
        harness = Harness(self, GDN)
        lines, logging = captured_log()
        with logging:
            self.assertEqual(harness.execute(), 'output')
            self.assertEqual(harness.order, ['arm', 'step', ('prepare', ['a', 'b'], ['after_reads']), ('flush', 'window'),
                                             ('drafts', 'a', 16), ('drafts', 'b', 16), ('flush', 'end')])
            cached = harness.hook._early_draft.cache
            harness.order.clear()
            self.assertIs(harness.take(), cached)
        self.assertEqual(harness.order, [], 'nothing drafted again')
        self.assertEqual(summary(cached), (['a', 'b'], [[97, 4096], [98, 4096]]))
        self.assertEqual(lines[0], '%s gdn_after_pairs=1' % early_draft.ENGAGED_MARKER)
        self.assertRegex(lines[1], r'^\[PACKED-EARLY-DRAFT\] round=1 path=reuse live=2 draft_ms=[0-9.]+ reason=-$')
        self.assertEqual(harness.hook._early_draft.counts, dict(reuse=1, redo=0, failed=0, untaken=0))

    def test_the_early_drafts_are_todays_drafts_call_for_call(self):
        """The same prepare (bar after_reads), the same drafts calls and the same DraftTokenIds as the flag-off
        take_draft_token_ids, which drafts in the same body a moment later."""
        def events(order):
            return [event for event in order if isinstance(event, tuple) and event[0] in ('prepare', 'drafts')]

        today = Harness(self, {})
        self.assertEqual(today.execute(), 'output')
        self.assertEqual(today.order, ['step'])
        reference = summary(today.take())
        for environ in (EARLY, GDN):
            with self.subTest(environ=environ):
                early = Harness(self, environ)
                early.execute()
                result = summary(early.take())
                self.assertEqual(result, reference)
                drafted = [event if event[0] != 'prepare' else event[:2] for event in events(early.order)]
                self.assertEqual(drafted, [event if event[0] != 'prepare' else event[:2] for event in events(today.order)])
        self.assertEqual(events(today.order)[0], ('prepare', ['a', 'b'], []))
        early = Harness(self, EARLY)
        early.execute()
        self.assertEqual(events(early.order)[0], ('prepare', ['a', 'b'], []), 'no after_reads without GDN_AFTER_PAIRS')
        self.assertNotIn('arm', early.order)

    def test_a_budget_the_step_used_up_narrows_the_early_round_as_it_narrows_todays(self):
        """Finish by max_tokens: the step's apply_committed_output extends the runner state the budget is read
        from (real_remaining_budget) before execute_model returns, so the early draft sees the budget
        take_draft_token_ids would have seen and drafts the round at the engines' widths, as today does."""
        def run(environ):
            harness = Harness(self, environ)
            for bridge in harness.bridges.values():
                bridge.state = SimpleNamespace(sampling_params=SimpleNamespace(max_tokens=40),
                                               output_token_ids=[0] * 10)

            def apply_committed_output(bridges):
                bridges['a'].state.output_token_ids.extend([1] * 16)  # 14 left: under a 16-row round
                bridges['b'].state.output_token_ids.extend([1] * 4)

            harness.on_step = apply_committed_output
            harness.execute()
            result = summary(harness.take())
            return result, [event if event[0] != 'prepare' else event[:2] for event in harness.order
                            if isinstance(event, tuple) and event[0] in ('prepare', 'drafts')]

        today = run({})
        self.assertEqual(today[1][1:], [('drafts', 'a', None), ('drafts', 'b', None)])
        for environ in (EARLY, GDN):
            with self.subTest(environ=environ):
                self.assertEqual(run(environ), today)

    def test_the_flush_composes_with_the_window_and_pairs_packed_only(self):
        harness = Harness(self, dict(GDN, QWEN_FAST_PRESTAGE='1', QWEN_FAST_ROUND_FENCES='1',
                                     QWEN_FAST_PAIRS_PACKED_ONLY='1'))
        harness.packed_step.while_waiting = Mock(return_value='window')
        harness.execute()
        prepare = [event for event in harness.order if isinstance(event, tuple) and event[0] == 'prepare']
        self.assertEqual(prepare, [('prepare', ['a', 'b'], ['after_reads', 'packed_round', 'while_waiting'])])
        harness.packed_step.while_waiting.assert_called_once()

    def test_a_change_after_the_draft_discards_the_early_tickets_and_drafts_again(self):
        harness = Harness(self, GDN)
        lines, logging = captured_log()
        with logging:
            harness.execute()
            harness.bridges['a'].request.session.position = 4100
            harness.order.clear()
            result = harness.take()
        self.assertEqual(harness.order[:2], [('discard', 'a'), ('discard', 'b')])
        self.assertEqual(harness.order[2:], [('prepare', ['a', 'b'], []), ('drafts', 'a', 16), ('drafts', 'b', 16)],
                         "today's body; no flush option outside the step")
        self.assertEqual(summary(result), (['a', 'b'], [[97, 4100], [98, 4096]]))
        self.assertRegex(lines[-1], r'path=redo live=2 draft_ms=[0-9.]+ reason=request=a:position$')
        self.assertEqual(harness.hook._early_draft.counts['redo'], 1)

    def test_a_draft_failure_is_kept_and_raised_where_todays_would_be(self):
        harness = Harness(self, GDN)
        harness.fail_draft = RuntimeError('pair trace')
        lines, logging = captured_log()
        with logging:
            self.assertEqual(harness.execute(), 'output', "the step's outputs still reach vLLM")
            self.assertEqual(harness.order[-1], ('flush', 'end'))
            with self.assertRaisesRegex(RuntimeError, 'pair trace'):
                harness.take()
        self.assertRegex(lines[-1], r'path=failed live=0 .* reason=RuntimeError$')

    def test_a_cache_nobody_took_is_dropped_at_the_next_step(self):
        harness = Harness(self, GDN)
        lines, logging = captured_log()
        with logging:
            harness.execute()
            harness.order.clear()
            harness.execute()
        self.assertEqual(harness.order[:2], [('flush', 'reconcile'), 'arm'])
        self.assertTrue(any('path=untaken' in line for line in lines))
        self.assertEqual(harness.hook._early_draft.counts['untaken'], 1)

    def test_a_failing_step_still_flushes_and_drafts_nothing(self):
        harness = Harness(self, GDN)
        harness.fail_step = RuntimeError('verify')
        with self.assertRaisesRegex(RuntimeError, 'verify'):
            harness.execute()
        self.assertEqual(harness.order, ['arm', 'step', ('flush', 'end')])
        self.assertIs(harness.hook._early_draft.cache, early_draft.MISSING)
        harness.order.clear()
        harness.take()
        self.assertEqual(harness.order[0], ('prepare', ['a', 'b'], []), "no cache: today's body")

    def test_a_flush_failure_fails_every_bridge(self):
        harness = Harness(self, GDN)
        harness.packed_step.flush_deferred_commits.side_effect = lambda site: (_ for _ in ()).throw(RuntimeError(site))
        with self.assertRaisesRegex(RuntimeError, 'end'):
            harness.execute()
        self.assertTrue(all(bridge.failed for bridge in harness.bridges.values()))

    def test_a_prefill_step_is_untouched(self):
        harness = Harness(self, GDN)
        new = SimpleNamespace(scheduled_new_reqs=[object()], total_num_scheduled_tokens=2048)
        harness.hook.original_execute = Mock(return_value='stock')
        self.assertEqual(harness.execute(new), 'stock')
        self.assertEqual(harness.order, [])
        self.assertFalse(hasattr(harness.hook, '_early_draft'))

    def test_the_draft_key_and_its_difference(self):
        harness = Harness(self, GDN)
        before = early_draft.draft_key(harness.hook)
        self.assertEqual(early_draft.draft_key(harness.hook), before)
        for field, change in (('phase', lambda s: setattr(s, 'phase', 'pending')),
                              ('pending', lambda s: setattr(s, 'pending', 'ticket')),
                              ('emitted', lambda s: s.emitted.append(2)),
                              ('finished', lambda s: setattr(s, 'finished', True))):
            with self.subTest(field=field):
                session = harness.bridges['b'].request.session
                saved = dict(vars(session), emitted=list(session.emitted))
                change(session)
                self.assertEqual(early_draft.key_difference(before, early_draft.draft_key(harness.hook)),
                                 'request=b:%s' % field)
                vars(session).update(saved)
        harness.bridges['a'].failed = True
        self.assertEqual(early_draft.key_difference(before, early_draft.draft_key(harness.hook)), 'request=a:failed')
        harness.bridges['a'].failed = False
        harness.hook.bridges = dict(harness.bridges, c=harness.bridge('c'))
        self.assertEqual(early_draft.key_difference(before, early_draft.draft_key(harness.hook)), 'bridges')
        harness.hook.bridges = harness.bridges

    def test_discard_ticket_is_the_stale_ticket_reset(self):
        runtime = SimpleNamespace(discard_proposal=Mock())
        request = SimpleNamespace(session=SimpleNamespace(pending='t', phase='pending'), runtime=runtime)
        self.assertTrue(early_draft.discard_ticket(request))
        self.assertEqual((request.session.pending, request.session.phase), (None, 'idle'))
        runtime.discard_proposal.assert_called_once_with()
        self.assertFalse(early_draft.discard_ticket(request))


# ------------------------------------------------------------------------------------------------------
# The coordinator
# ------------------------------------------------------------------------------------------------------

class CoordinatorTests(unittest.TestCase):
    """after_reads inside select_round: after the fence and every pair's readback, before the selection."""

    def setUp(self):
        from test_dflash_round_b1 import CollectingTrace
        import test_dflash_packed_proposal_coordinator as fixtures

        self.fixtures, self.trace = fixtures, CollectingTrace
        fixtures.FakeTrace.instances = []
        CollectingTrace.events = []
        patcher = patch('dflash_proposal_trace.PreparedPackedDFlashProposal', CollectingTrace)
        patcher.start()
        self.addCleanup(patcher.stop)
        environment = clean_environment()
        environment.start()
        self.addCleanup(environment.stop)
        self.operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: CollectingTrace.events.append(('fence',))))
        self.mesh = object()
        self.books = (object(), object())

    def bridges(self, count=4):
        out = []
        for index in range(count):
            device = self.fixtures.make_device(self.operations, self.mesh, slot=index)
            device.predecessors, device.successors = self.books
            out.append(self.fixtures.make_bridge('r%d' % index, device, seed=100 + index))
        return out

    def run_round(self, module, bridges, *, b1=True, after=True):
        from test_dflash_round_b1 import round_b1

        after_reads = Mock(side_effect=lambda: self.trace.events.append(('flush',))) if after else None

        def select(parts, seeds, counts, predecessors, successors):
            self.trace.events.append(('select', tuple(seeds)))
            return tuple((seed,) for seed in seeds)

        options = {} if after_reads is None else dict(after_reads=after_reads)
        with round_b1(b1), patch('dflash_packed_proposal.select_packed_batched', side_effect=select):
            prepared = module.PackedProposalCoordinator().prepare(bridges, **options)
        return prepared, after_reads

    def test_the_flush_follows_every_readback_and_precedes_the_selection(self):
        import dflash_packed_proposal_coordinator as coordinator

        prepared, after_reads = self.run_round(coordinator, self.bridges())
        after_reads.assert_called_once_with()
        self.assertEqual([event[0] for event in self.trace.events], ['fence', 'collect', 'collect', 'flush', 'select'])
        self.assertEqual([trace.adopted for trace in self.fixtures.FakeTrace.instances], [((100,), (101,)), ((102,), (103,))])

    def test_no_flush_here_while_a_single_is_left_to_read_or_without_the_batched_selection(self):
        import dflash_packed_proposal_coordinator as coordinator

        prepared, after_reads = self.run_round(coordinator, self.bridges(3))
        self.assertEqual(len(prepared), 3)
        after_reads.assert_not_called()
        self.trace.events.clear()
        prepared, after_reads = self.run_round(coordinator, self.bridges(), b1=False)
        after_reads.assert_not_called()
        prepared, after_reads = self.run_round(coordinator, [])
        after_reads.assert_not_called()

    def test_reads_covered(self):
        from dflash_packed_proposal_coordinator import reads_covered

        a, b, c = object(), object(), object()
        trace = SimpleNamespace(device_a=a, device_b=b)
        self.assertTrue(reads_covered([([0, 1], trace)], [a, b]))
        self.assertFalse(reads_covered([([0, 1], trace)], [a, b, c]))

    def test_without_the_argument_the_batched_round_is_the_parents_call_for_call(self):
        parent = parent_module('dflash_packed_proposal_coordinator.py')
        if parent is None:
            self.skipTest('no git history for %s' % PARENT)
        import dflash_packed_proposal_coordinator as coordinator

        def run(module):
            self.trace.events.clear()
            self.fixtures.FakeTrace.instances = []
            results = []
            for count in (4, 3, 4):
                prepared, _ = self.run_round(module, self.bridges(count), after=False)
                results.append(len(prepared))
            traces = self.fixtures.FakeTrace.instances
            events = [(event[0], traces.index(event[1])) if event[0] == 'collect' else event for event in self.trace.events]
            return results, events, [trace.adopted for trace in traces]

        self.assertEqual(run(coordinator), run(parent))


# ------------------------------------------------------------------------------------------------------
# The block, over gdn_records' real retained class
# ------------------------------------------------------------------------------------------------------

class BlockTests(tpv.FourUserFixture):
    PREFIXES = (9, 16, 0, 4)

    def setUp(self):
        super().setUp()
        from test_round_fences import CountingRetained, fenced_model_batch

        self.environment = clean_environment(QWEN_FAST_FAST_COMMIT='1', QWEN_FAST_PIPELINED_COMMITS='1',
                                             QWEN_FAST_ROUND_FENCES='1', **GDN)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.lines = []
        for patcher in (patch.object(packed_verifier, 'ModelBatch', fenced_model_batch(CountingRetained)),
                        patch.object(packed_verifier, 'diagnostic', side_effect=self.lines.append)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def serve(self, block, rounds, *, armed=True, flush=True, window=None):
        """`rounds` rounds: (arm,) verify, every commit, (window,) (flush). Per round: the traces enqueued by the
        commits, those by the flush, the fence the verify's replay was armed by."""
        record = []
        for number in range(rounds):
            if armed:
                self.assertTrue(block.arm_deferred_commits())
            predictions, metrics = block.verify(self.four())
            before = len(self.ttnn.executed)
            for segment, prefix in zip(metrics['segments'], self.PREFIXES):
                block.commit_user(segment, prefix)
            committed = self.ttnn.executed[before:]
            if window is not None:
                window(block)
            flushed_from = len(self.ttnn.executed)
            if flush:
                block.flush_commits('window')
            record.append((list(committed), self.ttnn.executed[flushed_from:], metrics.get('replay_fence'), predictions))
        return record

    def test_engaged_at_attach_and_refused_without_the_round_fences(self):
        block = self.build()
        self.assertTrue(block.gdn_after_pairs)
        self.assertIn('%s users=4 pipelined=1' % early_draft.GDN_ENGAGED_MARKER, self.lines)
        self.assertEqual(block.describe()['gdn_after_pairs'], True)
        block.close()
        del os.environ['QWEN_FAST_ROUND_FENCES']
        refused = self.build()
        self.assertFalse(refused.gdn_after_pairs)
        self.assertIn('%s users=4 reason=round-fences-off' % early_draft.GDN_REFUSED_MARKER, self.lines)
        self.assertFalse(refused.arm_deferred_commits())
        refused.close()
        del os.environ['QWEN_FAST_EARLY_DRAFT']
        os.environ['QWEN_FAST_ROUND_FENCES'] = '1'
        inert = self.build()
        self.assertFalse(inert.gdn_after_pairs or inert.arm_deferred_commits())

    def test_an_armed_round_defers_its_commits_and_the_flush_enqueues_them_in_decision_order(self):
        block = self.build()
        rounds = self.serve(block, 1, flush=False)
        self.assertEqual(rounds[0][0], [], 'no commit trace enqueued at the decisions')
        self.assertEqual(block.deferred_commits, [(2, 9), (0, 16), (1, 4)])
        retained = block.fixture.retained
        self.assertTrue(retained.fence_owed)
        executed = len(self.ttnn.executed)
        self.assertEqual(block.flush_commits('window'), 3)
        self.assertEqual(self.ttnn.executed[executed:], [block.commits[2][9], block.commits[0][16], block.commits[1][4]])
        self.assertEqual(self.ttnn.execute_blocking[executed:], [False] * 3)
        self.assertEqual((retained.fence_owed, retained.replay_ready, block.deferred_commits, block.deferring),
                         (True, False, [], False))
        self.assertRegex(self.lines[-1], r'^\[PACKED-GDN-AFTER-PAIRS\] round=1 commits=3 site=window enqueue_ms=[0-9.]+ '
                                         r'segments=2:9,0:16,1:4$')
        self.assertEqual(block.flush_commits('end'), 0, 'once')

    def test_an_f9_before_the_flush_cannot_arm_the_replay_and_the_next_replay_pays_the_fence(self):
        from dflash_packed_proposal_coordinator import note_fenced, run_while_waiting

        block = self.build()
        retained = block.fixture.retained

        def drafts(block):
            waiting = verify_prestage.WhileWaiting(block, [])
            run_while_waiting(waiting)
            self.ttnn.synchronize_device(self.model.mesh_device)  # the coordinator's F9
            note_fenced(waiting)
            self.assertTrue(retained.replay_ready, 'F9 armed the decisions')

        rounds = self.serve(block, 3, window=drafts)
        self.assertEqual([record[2] for record in rounds], ['first', 'replay', 'replay'])
        self.assertTrue(all(record[0] == [] and len(record[1]) == 3 for record in rounds))

    def test_the_rounds_enqueue_what_an_unarmed_rounds_do(self):
        """The same traces in the same order and blocking mode, the same copies, fences and predictions: only
        the moment the commit traces are enqueued moves."""
        def run(armed):
            block = self.build()
            executed, blocking, synchronized = len(self.ttnn.executed), len(self.ttnn.execute_blocking), self.ttnn.synchronized
            copies = len(self.ttnn.host_copies)
            rounds = self.serve(block, 4, armed=armed, flush=armed)
            # Each block captures traces of its own: named by what they are.
            names = {id(block.trace): 'verify'}
            names.update({id(trace): (segment, prefix) for segment, commits in enumerate(block.commits)
                          for prefix, trace in commits.items()})
            result = ([names.get(id(trace), trace) for trace in self.ttnn.executed[executed:]],
                      self.ttnn.execute_blocking[blocking:],
                      self.ttnn.synchronized - synchronized, len(self.ttnn.host_copies) - copies,
                      [(record[2], record[3]) for record in rounds])
            block.close()
            return result

        today = run(False)
        self.assertEqual(today[0][:4], ['verify', (2, 9), (0, 16), (1, 4)])
        self.assertEqual(run(True), today)

    def test_an_unarmed_round_is_todays(self):
        block = self.build()
        rounds = self.serve(block, 2, armed=False, flush=False)
        self.assertTrue(all(len(record[0]) == 3 for record in rounds), 'the commits enqueue their own traces')
        self.assertFalse(any(line.startswith(early_draft.GDN_MARKER) for line in self.lines))

    def test_verify_flushes_a_held_round_first(self):
        block = self.build()
        self.serve(block, 1, flush=False)
        held = [block.commits[2][9], block.commits[0][16], block.commits[1][4]]
        executed = len(self.ttnn.executed)
        block.verify(self.four())
        self.assertEqual(self.ttnn.executed[executed:executed + 3], held, 'ahead of the next trace')
        self.assertTrue(any(line.startswith('%s round=1 commits=3 site=verify ' % early_draft.GDN_MARKER)
                            for line in self.lines))

    def test_a_failed_block_drops_what_it_held(self):
        block = self.build()
        self.serve(block, 1, flush=False)
        block.phase = 'failed'
        executed = len(self.ttnn.executed)
        self.assertEqual(block.flush_commits('end'), 0)
        self.assertEqual(len(self.ttnn.executed), executed)
        self.assertRegex(self.lines[-1], r'commits=0 site=end enqueue_ms=0.00 segments=2:9,0:16,1:4 dropped=3 '
                                         r'reason=block-failed$')

    def test_a_failing_flush_poisons_the_retained_block(self):
        block = self.build()
        self.serve(block, 1, flush=False)
        with patch.object(self.ttnn, 'execute_trace', side_effect=RuntimeError('trace')):
            with self.assertRaisesRegex(RuntimeError, 'trace'):
                block.flush_commits('window')
        self.assertTrue(block.fixture.retained.poisoned)
        self.assertEqual(block.phase, 'failed')

    def test_the_commit_line_says_deferred(self):
        os.environ['QWEN_FAST_PACKED_AUDIT'] = '1'
        block = self.build()
        self.serve(block, 1)
        self.assertTrue(any(line.startswith('[PACKED-COMMIT] round=1 mode=deferred ') for line in self.lines))

    def test_a_round_that_never_verified_is_disarmed_by_the_flush(self):
        block = self.build()
        self.assertTrue(block.arm_deferred_commits())
        self.assertEqual(block.flush_commits('end'), 0)
        self.assertFalse(block.defer_armed)
        rounds = self.serve(block, 1, armed=False, flush=False)
        self.assertEqual(len(rounds[0][0]), 3)


class RetainedTests(unittest.TestCase):
    def test_deferred_publications_reowe_the_fence_under_round_fences_only(self):
        import test_gdn_records as tgr
        from test_round_fences import decide_all

        with clean_environment(QWEN_FAST_FAST_COMMIT='1'):
            block = tgr.packed_block(segments=tgr.M3_SEGMENTS)
        with self.assertRaisesRegex(ValueError, 'round fences'):
            block.note_deferred_publications()
        block.use_round_fences()
        decide_all(block)
        serial = block.commit_serial
        self.assertTrue(block.note_round_fence(serial))
        self.assertTrue(block.replay_ready)
        block.note_deferred_publications()
        self.assertEqual((block.commit_serial, block.replay_ready, block.fence_owed), (serial + 1, False, True))
        self.assertFalse(block.note_round_fence(serial), 'a token from before cannot arm it')
        block.replay(Mock(return_value=None))
        self.assertEqual(block.replay_fence, 'replay')


class StepTests(unittest.TestCase):
    def test_the_step_arms_and_flushes_every_block_that_can(self):
        blocks = [SimpleNamespace(arm_deferred_commits=Mock(return_value=True), flush_commits=Mock(return_value=3)),
                  SimpleNamespace(arm_deferred_commits=Mock(return_value=False), flush_commits=Mock(return_value=0)),
                  SimpleNamespace()]
        step = serving_packed_step.PackedStep(blocks)
        self.assertTrue(step.arm_deferred_commits())
        self.assertEqual(step.flush_deferred_commits('end'), 3)
        blocks[0].flush_commits.assert_called_once_with('end')
        self.assertFalse(serving_packed_step.PackedStep([SimpleNamespace()]).arm_deferred_commits())


# ------------------------------------------------------------------------------------------------------
# The gate, the arm, the comparison
# ------------------------------------------------------------------------------------------------------

def early_line(round_number, path='reuse', live=4, reason='-'):
    lines, logging = captured_log()
    state = early_draft.EarlyDraft()
    state.rounds, state.draft_ms = round_number, 33.2
    with logging:
        state.note(path, live, reason)
    return lines[0]


def gdn_line(round_number, site='window', dropped=False):
    if dropped:
        return ('%s round=%d commits=0 site=%s enqueue_ms=0.00 segments=0:9,1:4 dropped=2 reason=block-failed'
                % (early_draft.GDN_MARKER, round_number, site))
    return '%s round=%d commits=3 site=%s enqueue_ms=0.21 segments=0:9,1:16,3:4' % (early_draft.GDN_MARKER,
                                                                                  round_number, site)


class GateTests(unittest.TestCase):
    ON = dict(GDN, QWEN_FAST_ROUND_FENCES='1', QWEN_FAST_PACKED_AUDIT='1', QWEN_FAST_PACKED_PROPOSAL='1',
              QWEN_FAST_PIPELINED_PROPOSALS='1', QWEN_FAST_ROUND_B1='1')

    def log(self, rounds=12, redo=(), failed=(), site='window', refused=False, dropped=(), untaken=()):
        import lever_n_m3native_gate as gate

        lines = ['%s gdn_after_pairs=1' % early_draft.ENGAGED_MARKER, '[PINDIAG] round fences engaged users=4',
                 gate.ROUND_B1_MARKER,
                 '%s users=4 %s' % ((early_draft.GDN_REFUSED_MARKER, 'reason=round-fences-off') if refused
                                    else (early_draft.GDN_ENGAGED_MARKER, 'pipelined=1'))]
        for number in range(1, rounds + 1):
            lines.append(gdn_line(number, site, dropped=number in dropped))
            lines.append('[PACKED-FENCES] round=%d fence=%s validated=1 replay_ms=0.40 commit_sync_ms=0.03 path=off '
                         'prestage_ms=0.00 diff_ms=0.00 write_ms=0.00' % (number, 'first' if number == 1 else 'replay'))
            path = ('redo' if number in redo else 'failed' if number in failed else 'untaken' if number in untaken
                    else 'reuse')
            lines.append(early_line(number, path, 0 if path in ('failed', 'untaken') else 4,
                                    'request=a:position' if path == 'redo' else '-'))
        return chr(10).join(lines)

    def test_a_clean_arm_passes_and_is_summarised(self):
        import lever_n_m3native_gate as gate

        report = gate.flag_marker_report(self.ON, 4, self.log())
        self.assertEqual(report['missing'], [])
        summary = report['round_fence_h2']
        self.assertEqual((summary['drafts'], summary['four_live'], summary['four_live_reuse'], summary['reuse_share_four_live']),
                         (12, 12, 12, 1.0))
        self.assertEqual((summary['flushes'], summary['flushed_commits'], summary['flush_sites'], summary['late']),
                         (12, 36, {'window': 12}, 0))
        self.assertEqual(summary['draft_ms_median_four_live'], 33.2)
        self.assertEqual(report['round_fence_h1a']['fence_kinds'], {'first': 1, 'replay': 11})

    def test_the_problems(self):
        import lever_n_m3native_gate as gate

        def missing(log, environ=None):
            return gate.flag_marker_report(environ or self.ON, 4, log)['missing']

        empty = missing('')
        for marker in (gate.EARLY_ENGAGED_MARKER, gate.EARLY_MARKER, gate.GDN_ENGAGED_MARKER, gate.GDN_MARKER):
            self.assertTrue(any(marker in line for line in empty), marker)
        self.assertTrue(any('reused in 12 of 13' in line for line in missing(self.log(rounds=13, redo=(4,)))))
        self.assertFalse(any('reused in' in line for line in missing(self.log(rounds=20, redo=(4,)))), '19/20 passes')
        self.assertTrue(any('early draft(s) raised' in line for line in missing(self.log(failed=(3,)))))
        self.assertTrue(any('never taken by vLLM' in line for line in missing(self.log(untaken=(5,)))))
        self.assertTrue(any('outside the step' in line for line in missing(self.log(site='reconcile'))))
        self.assertTrue(any("no flush after the pairs' readback" in line for line in missing(self.log(site='end'))))
        self.assertFalse(any("readback" in line for line in missing(self.log(site='end'),
                                                                     dict(self.ON, QWEN_FAST_ROUND_B1='0'))))
        self.assertTrue(any('refused it' in line for line in missing(self.log(refused=True))))
        self.assertTrue(any('dropped by a failed block' in line for line in missing(self.log(dropped=(2,)))))
        lone = missing('', {'QWEN_FAST_GDN_AFTER_PAIRS': '1'})
        self.assertTrue(any('does nothing' in line for line in lone))

    def test_the_h1a_f9_check_is_waived_only_under_gdn_after_pairs(self):
        import lever_n_m3native_gate as gate

        without = dict(self.ON)
        del without['QWEN_FAST_GDN_AFTER_PAIRS']
        self.assertTrue(any("no replay armed by the drafts' fence" in line
                            for line in gate.flag_marker_report(without, 4, self.log())['missing']))
        self.assertFalse(any("no replay armed" in line for line in gate.flag_marker_report(self.ON, 4, self.log())['missing']))

    def test_the_gate_reads_the_lines_the_modules_write(self):
        import lever_n_m3native_gate as gate

        self.assertEqual((gate.EARLY_ENGAGED_MARKER, gate.GDN_ENGAGED_MARKER, gate.GDN_REFUSED_MARKER),
                         (early_draft.ENGAGED_MARKER, early_draft.GDN_ENGAGED_MARKER, early_draft.GDN_REFUSED_MARKER))
        self.assertEqual((gate.EARLY_MARKER, gate.GDN_MARKER), (early_draft.MARKER + ' round=', early_draft.GDN_MARKER + ' round='))
        self.assertEqual(gate.GDN_IN_STEP_SITES, early_draft.IN_STEP_SITES)
        for path in ('reuse', 'redo', 'failed', 'untaken'):
            self.assertEqual(gate.EARLY_LINE.search(early_line(7, path, 4, 'x')).groups()[:3], ('7', path, '4'))
        # the block's own lines, from a real flush
        result = unittest.TestResult()
        case = BlockTests('test_an_armed_round_defers_its_commits_and_the_flush_enqueues_them_in_decision_order')
        case.run(result)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        line = '[PACKED-GDN-AFTER-PAIRS] round=1 commits=3 site=window enqueue_ms=0.12 segments=2:9,0:16,1:4'
        self.assertEqual(gate.GDN_LINE.search(line).groups()[:3], ('1', '3', 'window'))
        self.assertEqual(gate.GDN_LINE.search(gdn_line(2, dropped=True)).groups()[5:], ('2', 'block-failed'))


class ArmTests(unittest.TestCase):
    ARM = HERE / 'lever_n_m3native_run_arm.sh'
    START = '# Round-fence plan H2 (early_draft.py; every flag default off).'
    BASE = dict(M3NATIVE_PACKED_PROPOSAL='1', M3NATIVE_PIPELINED_PROPOSALS='1')

    def text(self):
        return self.ARM.read_text(encoding='utf-8')

    def validate(self, **environ):
        bash = shutil.which('bash')
        if bash is None:
            self.skipTest('no bash')
        text = self.text()
        start = text.index(self.START)
        end = text.index(chr(10) + 'fi' + chr(10), text.index('if [ -n "${M3NATIVE_EARLY_DRAFT:-}" ]; then', start)) + 4
        script = 'set -euo pipefail' + chr(10) + 'users="${USERS_UNDER_TEST}"' + chr(10) + text[start:end] + 'echo VALID' + chr(10)
        try:
            return subprocess.run([bash, '-c', script], capture_output=True, text=True, timeout=60,
                                  env=dict(PATH=os.environ.get('PATH', ''), USERS_UNDER_TEST=environ.pop('users', '4'),
                                           **environ))
        except OSError as error:
            self.skipTest('bash unusable: %s' % error)

    def test_the_arm_refuses_what_the_gate_would(self):
        for environ in ({}, dict(self.BASE, M3NATIVE_EARLY_DRAFT='1'),
                        dict(self.BASE, M3NATIVE_EARLY_DRAFT='1', M3NATIVE_GDN_AFTER_PAIRS='1', M3NATIVE_ROUND_FENCES='1')):
            with self.subTest(accepted=environ):
                result = self.validate(**environ)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('VALID', result.stdout)
        for environ, message in (
                (dict(M3NATIVE_EARLY_DRAFT='yes'), 'must be 1 or unset'),
                (dict(M3NATIVE_GDN_AFTER_PAIRS='1', M3NATIVE_ROUND_FENCES='1'), 'without M3NATIVE_EARLY_DRAFT=1'),
                (dict(self.BASE, M3NATIVE_EARLY_DRAFT='1', M3NATIVE_GDN_AFTER_PAIRS='1'), 'needs M3NATIVE_ROUND_FENCES=1'),
                (dict(self.BASE, M3NATIVE_EARLY_DRAFT='1', users='1'), 'serves the packed block only'),
                (dict(M3NATIVE_EARLY_DRAFT='1', M3NATIVE_PACKED_PROPOSAL='1'), 'needs M3NATIVE_PACKED_PROPOSAL=1')):
            with self.subTest(refused=environ):
                result = self.validate(**environ)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_both_switches_cross_before_the_entrypoint_after_h1b(self):
        text = self.text()
        entry = text.index('--entrypoint python3')
        for name in ('EARLY_DRAFT', 'GDN_AFTER_PAIRS'):
            line = '${M3NATIVE_%s:+-e QWEN_FAST_%s=1}' % (name, name)
            with self.subTest(name=name):
                self.assertEqual(text.count(line), 1)
                self.assertLess(text.index(line), entry)
                self.assertLess(text.index('${M3NATIVE_FUSED_COMMIT_AUDIT:+-e QWEN_FAST_FUSED_COMMIT_AUDIT=1}'),
                                text.index(line))

    def test_the_arm_parses(self):
        bash = shutil.which('bash')
        if bash is None:
            self.skipTest('no bash')
        result = subprocess.run([bash, '-n', str(self.ARM)], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(b'\r\n', self.ARM.read_bytes(), 'LF line endings')


class PackedCompareTests(unittest.TestCase):
    ROUNDS = {0: [(131072, 5, 5, '[12305, 198, 464]'), (131077, 3, 3, '[1, 2]')],
              1: [(131072, 8, 8, '[506, 279]')], 2: [(131072, 9, 9, '[369, 264]')], 3: [(131072, 9, 9, '[506, 11]')]}

    def run_log(self, order, tag, rounds=None):
        rounds = rounds or self.ROUNDS
        streams = [dict(request_id='cmpl-%s%d' % (tag, user)) for user in range(4)]
        lines = []
        for index in range(2):
            for segment, user in enumerate(order):
                if index < len(rounds[user]):
                    position, prefix, emitted, predictions = rounds[user][index]
                    lines.append('[PACKED] request=cmpl-%s%d-0-ab segment=%d position=%d prefix=%d emitted=%d '
                                 'predictions=%s' % (tag, user, segment, position, prefix, emitted, predictions))
        return chr(10).join(lines), streams

    def test_admission_order_changes_every_line_but_no_users_sequence(self):
        a = acceptance_report.packed_fingerprints(*self.run_log([0, 1, 2, 3], 'a'))
        b = acceptance_report.packed_fingerprints(*self.run_log([2, 1, 3, 0], 'b'))
        self.assertEqual((a['admission_order'], b['admission_order']), ([0, 1, 2, 3], [2, 1, 3, 0]))
        self.assertEqual(a['users'], b['users'])
        self.assertEqual((a['lines'], a['users']['0']['rounds']), (5, 2))
        compared = acceptance_report.compare_packed(*self.run_log([0, 1, 2, 3], 'a'), *self.run_log([2, 1, 3, 0], 'b'))
        self.assertEqual((compared['identical'], compared['comparable']), (False, False))

    def test_one_changed_prediction_is_found_at_its_round(self):
        changed = {**self.ROUNDS, 0: [self.ROUNDS[0][0], (131077, 3, 3, '[1, 3]')]}
        compared = acceptance_report.compare_packed(*self.run_log([0, 1, 2, 3], 'a'),
                                                    *self.run_log([0, 1, 2, 3], 'b', changed))
        self.assertFalse(compared['identical'])
        self.assertTrue(compared['comparable'])
        differing = [entry for entry in compared['users'] if not entry['identical']]
        self.assertEqual([entry['user'] for entry in differing], [0])
        self.assertEqual(differing[0]['first_difference'], dict(round=1, a=(131077, 3, 3, '[1, 2]'),
                                                                b=(131077, 3, 3, '[1, 3]')))
        same = acceptance_report.compare_packed(*self.run_log([0, 1, 2, 3], 'a'), *self.run_log([0, 1, 2, 3], 'c'))
        self.assertTrue(same['identical'])
        self.assertIn('identical=True', acceptance_report.compare_line(same))

    def test_a_real_log_equals_itself_and_its_fingerprint_is_stable(self):
        from test_acceptance_report import fixture

        log, streams = fixture('v185')
        compared = acceptance_report.compare_packed(log, streams, log, streams)
        self.assertTrue(compared['identical'])
        prints = acceptance_report.packed_fingerprints(log, streams)
        self.assertEqual(prints['lines'], 104)
        self.assertEqual(sorted(prints['admission_order']), [0, 1, 2, 3])
        self.assertEqual(prints['unattributed'], 0)
        json.dumps(prints)

    def test_the_offline_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for tag, rounds in (('a', None), ('b', None), ('c', {**self.ROUNDS, 3: [(131072, 9, 9, '[506, 12]')]})):
                log, streams = self.run_log([0, 1, 2, 3], tag, rounds)
                (root / ('%s.log' % tag)).write_text(log, encoding='utf-8')
                (root / ('%s.json' % tag)).write_text(json.dumps(dict(streams=streams)), encoding='utf-8')
            for tag, code in (('b', 0), ('c', 3)):
                with self.subTest(tag=tag), patch('sys.stdout', new_callable=io.StringIO) as out:
                    self.assertEqual(acceptance_report.main([str(root / ('%s.log' % tag)), str(root / ('%s.json' % tag)),
                                                             '--packed-reference', str(root / 'a.log'),
                                                             str(root / 'a.json'), '--json', str(root / 'out.json')]),
                                     code)
                    self.assertIn('[PACKED-COMPARE] identical=%s' % (code == 0), out.getvalue())
                    self.assertIn('packed_compare', json.loads((root / 'out.json').read_text(encoding='utf-8')))

    def test_the_gate_report_carries_the_fingerprints(self):
        import lever_n_m3native_gate as gate

        with tempfile.TemporaryDirectory() as directory:
            log, streams = self.run_log([0, 1, 2, 3], 'a')
            path = Path(directory) / 'server.log'
            path.write_text(log, encoding='utf-8')
            report = dict(streams=streams, gate_passed=True)
            with patch('sys.stdout', new_callable=io.StringIO):
                gate.add_run_diagnostics(report, path, sequential=False)
        self.assertEqual(report['packed_fingerprints']['admission_order'], [0, 1, 2, 3])
        self.assertIs(report['gate_passed'], True)


class ShippingTests(unittest.TestCase):
    def test_the_module_reaches_the_image_in_both_lists(self):
        from test_serving_image_copy_closure import context_modules, dockerfile_modules, dockerfile_text

        for name in ('early_draft.py', 'serving_worker_hook.py', 'packed_verifier.py', 'gdn_records.py',
                     'serving_packed_step.py', 'dflash_packed_proposal_coordinator.py'):
            with self.subTest(module=name):
                self.assertIn(name, dockerfile_modules(dockerfile_text()))
                self.assertIn(name, context_modules())

    def test_the_suite_runs_in_the_cpu_workflow(self):
        import re

        workflow = (ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        self.assertRegex(workflow, r'python -B -m unittest [^\n]*\btest_early_draft\b')
        self.assertTrue(re.search(r'\btest_early_draft\b', workflow))

    def test_the_new_files_are_lf(self):
        for path in (HERE / 'early_draft.py', HERE / 'test_early_draft.py'):
            with self.subTest(path=path.name):
                self.assertNotIn(b'\r\n', path.read_bytes())


# ------------------------------------------------------------------------------------------------------
# Flag off: the PARENT, call for call
# ------------------------------------------------------------------------------------------------------

class ParentTests(unittest.TestCase):
    """With every flag off, each module H2 touches against its PARENT copy: H1b's parent comparisons (which
    carry H1a's and M2's) re-pointed at PARENT, gdn_records' decision sequence, the coordinator, the
    acceptance report, and the hook's packed decode."""

    def run_repointed(self, case, patches):
        if parent_module('packed_verifier.py') is None:
            self.skipTest('no git history for %s' % PARENT)
        result = unittest.TestResult()
        with patches:
            case.run(result)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        self.assertEqual(result.skipped, [])

    def h1b(self, cls, method):
        import test_fused_commit

        self.run_repointed(getattr(test_fused_commit, cls)(method), patch.object(test_fused_commit, 'PARENT', PARENT))

    def test_h1bs_parent_comparisons(self):
        for method in ('test_the_steps_answers_and_rounds_are_the_parents', 'test_the_real_blocks_round_is_the_parents',
                       'test_the_verifier_rounds_are_the_parents_call_for_call',
                       'test_the_packed_blocks_fenced_rounds_are_the_parents',
                       'test_the_hooks_drafts_and_pass_throughs_are_the_parents', 'test_the_gates_report_is_the_parents',
                       'test_the_pair_trace_is_the_parents_call_for_call', 'test_the_window_is_the_parents',
                       'test_the_commit_is_the_parents'):
            with self.subTest(method=method):
                self.h1b('ParentTests', method)

    def test_the_pinned_sources_are_untouched(self):
        self.h1b('ShippingTests', 'test_the_pinned_sources_are_untouched')

    def test_gdn_records_decisions_and_the_bare_coordinator_are_the_parents(self):
        import test_round_fences

        for cls, method in (('RetainedFlagOffTests', 'test_the_sequence_is_the_parents_call_for_call'),
                            ('CoordinatorTests', 'test_without_the_argument_the_prepare_is_the_parents_call_for_call')):
            with self.subTest(method=method):
                self.run_repointed(getattr(test_round_fences, cls)(method), patch.object(test_round_fences, 'PARENT', PARENT))

    def test_the_acceptance_report_is_the_parents(self):
        import test_padded_block

        original = test_padded_block.parent_module
        self.run_repointed(test_padded_block.ParentTests('test_the_acceptance_report_of_a_flag_off_run_is_the_parents'),
                           patch.object(test_padded_block, 'parent_module',
                                        lambda relative, commit=None: original(relative, PARENT)))

    def test_the_hooks_packed_decode_is_the_parents(self):
        parent = parent_module('serving_worker_hook.py')
        if parent is None:
            self.skipTest('no git history for %s' % PARENT)

        def run(module):
            harness = Harness(self, {}, module=module)
            first = harness.execute()
            drafts = summary(harness.take())
            second = harness.execute()
            calls = [(call.args[0] is harness.bridges, call.args[1] is harness.scheduled, sorted(call.kwargs))
                     for call in harness.decode.call_args_list]
            return first, second, drafts, harness.order, calls, hasattr(harness.hook, '_early_draft')

        self.assertEqual(run(serving_worker_hook), run(parent))

    def test_the_untouched_modules_are_the_parents(self):
        result = subprocess.run(['git', 'diff', '--name-only', PARENT, '--', 'serving_runtime.py', 'fused_commit.py',
                                 'verify_prestage.py', 'dflash_proposal_trace.py', 'serving_packed_bridge.py',
                                 'serving_lifecycle.py', 'dflash_traced_publish.py', 'dflash_device.py'],
                                capture_output=True, cwd=str(HERE), timeout=60)
        if result.returncode != 0:
            self.skipTest('no git history for %s' % PARENT)
        self.assertEqual(result.stdout.decode().strip(), '')


if __name__ == '__main__':
    unittest.main()
