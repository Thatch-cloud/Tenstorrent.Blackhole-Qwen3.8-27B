import ast
import __future__
from collections import deque
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from serving_worker_hook import FastWorkerHook
from test_serving_runner_bridge import RunnerBridgeTests


class Worker:
    is_driver_worker = True

    def execute_model(self, scheduled):
        return self.model_runner.execute_model(scheduled)

    def sample_tokens(self, grammar_output):
        return self.model_runner.sample_tokens(grammar_output)


class WorkerHookTests(unittest.TestCase):
    def fixture(self):
        bridge, events, scheduled = RunnerBridgeTests().fixture()
        bridge.runner._pending_samples = deque()
        bridge.runner.execute_model = Mock(name='baseline_forward')
        bridge.runner.sample_tokens = Mock(name='baseline_sampler')
        worker = Worker()
        worker.model_runner = bridge.runner
        return worker, bridge, events, scheduled

    @unittest.skipUnless(os.environ.get('QWEN_PLUGIN_SOURCE'), 'Pinned plugin source required')
    def test_actual_upstream_worker_delegation(self):
        source = Path(os.environ['QWEN_PLUGIN_SOURCE']) / 'src/vllm_tt_plugin/worker.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
            and any(isinstance(method, ast.FunctionDef) and method.name == 'execute_model'
                for method in node.body)]
        self.assertEqual(len(classes), 1)
        methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef)
            and node.name in ('execute_model', 'sample_tokens')]
        self.assertEqual(len(methods), 2)
        namespace = {}
        for method in methods:
            self.assertEqual(method.decorator_list, [])
        exec(compile(ast.Module(body=methods, type_ignores=[]), '<pinned-worker>', 'exec',
            flags=__future__.annotations.compiler_flag), namespace)
        worker, bridge, _, scheduled = self.fixture()
        actual_worker_type = type('PinnedWorkerDelegation', (), {
            'is_driver_worker': True, 'execute_model': namespace['execute_model'],
            'sample_tokens': namespace['sample_tokens']})
        worker = actual_worker_type()
        worker.model_runner = bridge.runner
        hook = FastWorkerHook(worker, bridge, cancelled=lambda: False)
        from vllm.v1.outputs import ModelRunnerOutput
        result = worker.execute_model(scheduled)
        self.assertIsInstance(result, ModelRunnerOutput)
        self.assertEqual(result.sampled_token_ids, [list(range(11, 27))])
        with self.assertRaises(RuntimeError):
            worker.sample_tokens(None)
        hook.close()

    def test_committed_block_bypasses_baseline_forward_and_sampler(self):
        worker, bridge, events, scheduled = self.fixture()
        original_forward = bridge.runner.execute_model
        original_sampler = bridge.runner.sample_tokens
        hook = FastWorkerHook(worker, bridge, cancelled=lambda: False)
        outputs = ModuleType('vllm.v1.outputs')
        outputs.ModelRunnerOutput = SimpleNamespace
        outputs.DraftTokenIds = SimpleNamespace
        with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
            self.assertEqual(worker.take_draft_token_ids().draft_token_ids, [list(range(11, 26))])
            result = worker.execute_model(scheduled)
        self.assertEqual(result.sampled_token_ids, [list(range(11, 27))])
        self.assertEqual(len(bridge.runner._pending_samples), 0)
        original_forward.assert_not_called()
        original_sampler.assert_not_called()
        # The hook's own decode returns committed output and never defers, which is
        # what original_sampler.assert_not_called() above shows. A sampler call that
        # DOES arrive belongs to another request's prefill - a second user joining -
        # so it reaches the runner rather than raising. The gate on whether a prefill
        # is actually pending lives in the lifecycle, which is the only party that
        # knows; this test asserts the delegation, not a refusal.
        worker.sample_tokens(None)
        original_sampler.assert_called_once_with(None)
        hook.close()
        hook.close()
        self.assertIs(bridge.runner.execute_model, original_forward)
        self.assertIs(bridge.runner.sample_tokens, original_sampler)
        self.assertFalse(hasattr(worker, 'take_draft_token_ids'))
        self.assertTrue(bridge.request.closed)

    def test_drafting_asks_the_packed_step_for_the_rounds_width_and_hands_it_to_every_bridge(self):
        """Beside the 64-row block the engines capture only the sequential widths, so the
        width of a round is decided here, over every live request, before any proposal."""
        worker, bridge, events, scheduled = self.fixture()
        policy = Mock(return_value=16)
        hook = FastWorkerHook(worker, bridge, cancelled=lambda: False, packed_step=SimpleNamespace(proposal_rows=policy))
        original = hook.bridges
        bridges = {name: SimpleNamespace(request=SimpleNamespace(session=SimpleNamespace(request_id=name, pending=None)),
                                         drafts=Mock(return_value=SimpleNamespace(req_ids=[name], draft_token_ids=[[1]])))
                   for name in 'ab'}
        hook.bridges = bridges
        outputs = ModuleType('vllm.v1.outputs')
        outputs.DraftTokenIds = SimpleNamespace
        try:
            with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
                result = worker.take_draft_token_ids()
            self.assertEqual((result.req_ids, result.draft_token_ids), (['a', 'b'], [[1], [1]]))
            policy.assert_called_once_with([bridges['a'].request, bridges['b'].request])
            for name in 'ab':
                bridges[name].drafts.assert_called_once_with(packed_rows=16)
            # None from the policy: each bridge drafts exactly as before
            policy.return_value = None
            for name in 'ab':
                bridges[name].drafts.reset_mock()
            with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
                worker.take_draft_token_ids()
            for name in 'ab':
                bridges[name].drafts.assert_called_once_with()
            # a step without the policy (the sequential step): likewise
            hook.packed_step = lambda entries, *, cancelled: []
            for name in 'ab':
                bridges[name].drafts.reset_mock()
            with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
                worker.take_draft_token_ids()
            for name in 'ab':
                bridges[name].drafts.assert_called_once_with()
        finally:
            hook.bridges = original
            hook.close()

    def test_a_stale_pending_ticket_narrower_than_the_rounds_width_is_discarded_and_redrafted(self):
        """A request can go several ticks between drafting a ticket and being stepped on
        it - a tick that turns out to be some other request's prefill, or bookkeeping for
        a partner finishing - so it can still hold a ticket narrower than the width every
        OTHER live request drafts once proposal_rows decides the round is the block's:
        left alone that mix is exactly what packed_device_step refuses (run 35535533720).
        Discarded here, before drafting, the request's own drafts() redrafts fresh at the
        round's width, same as if nothing had been pending."""
        worker, bridge, events, scheduled = self.fixture()
        policy = Mock(return_value=16)
        hook = FastWorkerHook(worker, bridge, cancelled=lambda: False, packed_step=SimpleNamespace(proposal_rows=policy))
        original = hook.bridges
        stale_ticket = SimpleNamespace(tokens=(0, 0, 0, 0))
        stale_session = SimpleNamespace(request_id='a', pending=stale_ticket, phase='pending')
        stale_runtime = SimpleNamespace(discard_proposal=Mock())
        fresh_session = SimpleNamespace(request_id='b', pending=None, phase='idle')
        bridges = {
            'a': SimpleNamespace(request=SimpleNamespace(session=stale_session, runtime=stale_runtime),
                                 drafts=Mock(return_value=SimpleNamespace(req_ids=['a'], draft_token_ids=[[1]]))),
            'b': SimpleNamespace(request=SimpleNamespace(session=fresh_session, runtime=SimpleNamespace()),
                                 drafts=Mock(return_value=SimpleNamespace(req_ids=['b'], draft_token_ids=[[2]]))),
        }
        hook.bridges = bridges
        outputs = ModuleType('vllm.v1.outputs')
        outputs.DraftTokenIds = SimpleNamespace
        try:
            with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
                result = worker.take_draft_token_ids()
            self.assertEqual((result.req_ids, result.draft_token_ids), (['a', 'b'], [[1], [2]]))
            stale_runtime.discard_proposal.assert_called_once_with()
            self.assertEqual((stale_session.pending, stale_session.phase), (None, 'idle'))
            for name in 'ab':
                bridges[name].drafts.assert_called_once_with(packed_rows=16)
            # a ticket already at the round's width is untouched: no discard, no reset,
            # and the same object is handed on to the next drafts() call
            for name in 'ab':
                bridges[name].drafts.reset_mock()
            matched_ticket = SimpleNamespace(tokens=(0,) * 16)
            stale_session.pending, stale_session.phase = matched_ticket, 'pending'
            stale_runtime.discard_proposal.reset_mock()
            with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
                worker.take_draft_token_ids()
            stale_runtime.discard_proposal.assert_not_called()
            self.assertIs(stale_session.pending, matched_ticket)
            bridges['a'].drafts.assert_called_once_with(packed_rows=16)
            # the policy answers None for THIS round - too few live requests for the
            # block's own group (a partner just finished, run 35564623068) or a
            # survivor's remaining budget narrower than a block round both land here -
            # while still being a configured policy: a ticket pending at some other
            # width is exactly as stale as a mismatch against a real packed_rows
            # number, since the round has no shared width to hold it against either
            # way, and is discarded and redrafted fresh at the engine's own native
            # width rather than riding into a round with no capture anywhere to fall
            # back to (packed_device_step's refuse_round).
            policy.return_value = None
            stale_session.pending, stale_session.phase = stale_ticket, 'pending'
            for name in 'ab':
                bridges[name].drafts.reset_mock()
            with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
                worker.take_draft_token_ids()
            stale_runtime.discard_proposal.assert_called_once_with()
            self.assertEqual((stale_session.pending, stale_session.phase), (None, 'idle'))
            bridges['a'].drafts.assert_called_once_with()
            # no policy AT ALL (a packed_step with no proposal_rows - the plain
            # sequential default): nothing is ever discarded here, exactly as before -
            # each engine's own capture stays the deciding word
            hook.packed_step = lambda entries, *, cancelled: []
            stale_session.pending, stale_session.phase = stale_ticket, 'pending'
            stale_runtime.discard_proposal.reset_mock()
            for name in 'ab':
                bridges[name].drafts.reset_mock()
            with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
                worker.take_draft_token_ids()
            stale_runtime.discard_proposal.assert_not_called()
            self.assertIs(stale_session.pending, stale_ticket)
            bridges['a'].drafts.assert_called_once_with()
        finally:
            hook.bridges = original
            hook.close()

    def test_a_live_requests_real_max_tokens_budget_narrower_than_the_round_forces_a_native_redraft(self):
        """proposal_rows only ever sees session.finished - an EOS token or the session's
        own fixed 256-slot capture ceiling (GreedySession.max_new_tokens) - never a
        request's own, usually much shorter, max_tokens: that budget is vLLM's, enforced
        by excluding the request from the NEXT schedule, external to this engine's own
        bookkeeping. So a live request can be one round away from that exclusion while
        session.finished still reads False - run 35567165791, the round right after run
        35564623068's discard fix landed: the finishing request's own draft still ran a
        full proposal in the very tick that turned out to draft its LAST round, and its
        partners' fresh block-width tickets rode into a round the scheduler admitted one
        entry short, exactly like the bug the discard fix closed. Real remaining budget
        (state.sampling_params.max_tokens - len(state.output_token_ids), the same state
        apply_committed_output keeps in sync with the scheduler's own count) catches this
        a round earlier: narrower than the round's own width forces the WHOLE round to
        redraft at native width, exactly as a policy answering None outright already does."""
        worker, bridge, events, scheduled = self.fixture()
        policy = Mock(return_value=16)
        hook = FastWorkerHook(worker, bridge, cancelled=lambda: False, packed_step=SimpleNamespace(proposal_rows=policy))
        original = hook.bridges

        def make_bridge(name, *, max_tokens=None, output_token_ids=None):
            session = SimpleNamespace(request_id=name, pending=None, phase='idle', finished=False)
            state = None
            if max_tokens is not None:
                state = SimpleNamespace(sampling_params=SimpleNamespace(max_tokens=max_tokens),
                                        output_token_ids=list(output_token_ids or []))
            return SimpleNamespace(request=SimpleNamespace(session=session, runtime=SimpleNamespace()), state=state,
                                   drafts=Mock(return_value=SimpleNamespace(req_ids=[name], draft_token_ids=[[1]])))

        # a: 48-token budget, 47 already emitted - one round short of the block's 16 rows.
        # b: the same budget, plenty left.
        bridges = {'a': make_bridge('a', max_tokens=48, output_token_ids=[0] * 47),
                   'b': make_bridge('b', max_tokens=48, output_token_ids=[0] * 10)}
        hook.bridges = bridges
        outputs = ModuleType('vllm.v1.outputs')
        outputs.DraftTokenIds = SimpleNamespace
        try:
            with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
                worker.take_draft_token_ids()
            policy.assert_called_once_with([bridges['a'].request, bridges['b'].request])
            for name in 'ab':
                bridges[name].drafts.assert_called_once_with()
            # the same real shortfall, but the session itself already calls it
            # finished: the real FastRunnerBridge.drafts() short-circuits that one on
            # its own (this fixture's bare Mock does not), so the veto here skips it -
            # 'b' is the only one it has to judge, and 'b' has plenty of budget, so
            # the policy's width stands for both
            for name in 'ab':
                bridges[name].drafts.reset_mock()
            bridges['a'].request.session.finished = True
            with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
                worker.take_draft_token_ids()
            for name in 'ab':
                bridges[name].drafts.assert_called_once_with(packed_rows=16)
            bridges['a'].request.session.finished = False
            # real budget catches up: plenty of room for both now, so the policy's
            # width stands, unchanged from today's behaviour
            for name in 'ab':
                bridges[name].drafts.reset_mock()
            bridges['a'].state.output_token_ids = [0] * 10
            with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
                worker.take_draft_token_ids()
            for name in 'ab':
                bridges[name].drafts.assert_called_once_with(packed_rows=16)
            # no state at all (the existing fixtures never model one): nothing here
            # can second-guess the policy, so its width stands exactly as before
            for name in 'ab':
                bridges[name].drafts.reset_mock()
            bridges['a'].state = None
            with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
                worker.take_draft_token_ids()
            for name in 'ab':
                bridges[name].drafts.assert_called_once_with(packed_rows=16)
        finally:
            hook.bridges = original
            hook.close()

    def make_pipelined_bridges(self, names, *, operations, mesh, order):
        """Four bridges over a shared fake mesh, each with a fake dflash2
        DFlashDevice reachable at request.runtime.drafter - prepare_device()
        records ('prepare', name, seed) into the shared `order` list and returns
        True (successfully prewarmed); operations.synchronize_device records
        ('sync', mesh); each bridge's drafts() records ('drafts', name, kwargs)."""
        bridges = {}
        for index, name in enumerate(names):
            session = SimpleNamespace(request_id=name, seed=100 + index, pending=None, finished=False)
            device = SimpleNamespace(operations=operations, mesh=mesh,
                proposal_capture=SimpleNamespace(discard_pending=Mock(side_effect=lambda name=name: order.append(('discard', name)))),
                prepare_device=Mock(side_effect=lambda seed, name=name: order.append(('prepare', name, seed)) or True))
            request = SimpleNamespace(session=session, runtime=SimpleNamespace(drafter=device), closed=False, cancelled=False)
            bridges[name] = SimpleNamespace(request=request, failed=False,
                drafts=Mock(side_effect=lambda name=name, **kwargs: order.append(('drafts', name, kwargs)) or
                            SimpleNamespace(req_ids=[name], draft_token_ids=[[1]])))
        return bridges

    def test_pipelined_proposals_are_off_by_default(self):
        """QWEN_FAST_PIPELINED_PROPOSALS unset: no bridge is prewarmed, no shared
        fence happens, and every bridge's own drafts() runs exactly as it always
        has - the existing single-phase loop, untouched."""
        worker, bridge, events, scheduled = self.fixture()
        hook = FastWorkerHook(worker, bridge, cancelled=lambda: False)
        original = hook.bridges
        order = []
        operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: order.append(('sync', mesh))))
        bridges = self.make_pipelined_bridges('abcd', operations=operations, mesh=object(), order=order)
        hook.bridges = bridges
        outputs = ModuleType('vllm.v1.outputs')
        outputs.DraftTokenIds = SimpleNamespace
        try:
            self.assertNotIn('QWEN_FAST_PIPELINED_PROPOSALS', os.environ)
            with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
                result = worker.take_draft_token_ids()
            self.assertEqual((result.req_ids, result.draft_token_ids), (['a', 'b', 'c', 'd'], [[1], [1], [1], [1]]))
            for name in 'abcd':
                bridges[name].request.runtime.drafter.prepare_device.assert_not_called()
                bridges[name].drafts.assert_called_once_with()
            operations.synchronize_device.assert_not_called()
            self.assertEqual(order, [('drafts', name, {}) for name in 'abcd'])
        finally:
            hook.bridges = original
            hook.close()

    def test_pipelined_proposals_prewarm_every_eligible_bridge_then_fence_once(self):
        """QWEN_FAST_PIPELINED_PROPOSALS=1: every bridge's device work is enqueued
        (prepare_device) before any of them is read back, ONE synchronize_device
        fences the shared mesh, and only then does phase B - drafts() for every
        bridge, in the same original order, completely unchanged - run."""
        worker, bridge, events, scheduled = self.fixture()
        hook = FastWorkerHook(worker, bridge, cancelled=lambda: False)
        original = hook.bridges
        order = []
        mesh = object()
        operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: order.append(('sync', mesh))))
        bridges = self.make_pipelined_bridges('abcd', operations=operations, mesh=mesh, order=order)
        hook.bridges = bridges
        outputs = ModuleType('vllm.v1.outputs')
        outputs.DraftTokenIds = SimpleNamespace
        try:
            with patch.dict(os.environ, {'QWEN_FAST_PIPELINED_PROPOSALS': '1'}), \
                    patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
                result = worker.take_draft_token_ids()
            self.assertEqual((result.req_ids, result.draft_token_ids), (['a', 'b', 'c', 'd'], [[1], [1], [1], [1]]))
            self.assertEqual(order, [
                ('prepare', 'a', 100), ('prepare', 'b', 101), ('prepare', 'c', 102), ('prepare', 'd', 103),
                ('sync', mesh),
                ('drafts', 'a', {}), ('drafts', 'b', {}), ('drafts', 'c', {}), ('drafts', 'd', {}),
            ])
            operations.synchronize_device.assert_called_once_with(mesh)
            for name in 'abcd':
                bridges[name].request.runtime.drafter.proposal_capture.discard_pending.assert_not_called()
        finally:
            hook.bridges = original
            hook.close()

    def test_a_bridge_with_no_pipeline_eligible_drafter_falls_back_to_its_own_blocking_drafts(self):
        """A dspark bridge (no `.drafter`), one with no captured trace at all
        (`.drafter` present but no `prepare_device`), and one that already has a
        pending ticket are all left for phase B exactly as they run today; only
        the genuinely eligible bridges are prepared and fenced."""
        worker, bridge, events, scheduled = self.fixture()
        hook = FastWorkerHook(worker, bridge, cancelled=lambda: False)
        original = hook.bridges
        order = []
        mesh = object()
        operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: order.append(('sync', mesh))))
        bridges = self.make_pipelined_bridges('ad', operations=operations, mesh=mesh, order=order)
        # b: dspark - no drafter attribute on its runtime at all.
        session_b = SimpleNamespace(request_id='b', seed=200, pending=None, finished=False)
        bridges['b'] = SimpleNamespace(request=SimpleNamespace(session=session_b, runtime=SimpleNamespace(),
            closed=False, cancelled=False), failed=False,
            drafts=Mock(side_effect=lambda **kwargs: order.append(('drafts', 'b', kwargs)) or
                        SimpleNamespace(req_ids=['b'], draft_token_ids=[[1]])))
        # c: a pending ticket already - drafts() will not even call prepare() for it.
        session_c = SimpleNamespace(request_id='c', seed=201, pending='ticket', finished=False)
        bridges['c'] = SimpleNamespace(request=SimpleNamespace(session=session_c, runtime=SimpleNamespace(),
            closed=False, cancelled=False), failed=False,
            drafts=Mock(side_effect=lambda **kwargs: order.append(('drafts', 'c', kwargs)) or
                        SimpleNamespace(req_ids=['c'], draft_token_ids=[[1]])))
        hook.bridges = {name: bridges[name] for name in 'abcd'}
        outputs = ModuleType('vllm.v1.outputs')
        outputs.DraftTokenIds = SimpleNamespace
        try:
            with patch.dict(os.environ, {'QWEN_FAST_PIPELINED_PROPOSALS': '1'}), \
                    patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
                worker.take_draft_token_ids()
            prepares = [entry for entry in order if entry[0] == 'prepare']
            self.assertEqual([name for _, name, _ in prepares], ['a', 'd'])
            operations.synchronize_device.assert_called_once_with(mesh)
            drafts = [entry for entry in order if entry[0] == 'drafts']
            self.assertEqual([name for _, name, _ in drafts], ['a', 'b', 'c', 'd'], 'every bridge still drafts, in order')
        finally:
            hook.bridges = original
            hook.close()

    def test_a_phase_a_failure_fences_and_releases_every_already_prepared_bridge_before_reraising(self):
        """A device raising while being prepared must fail the round loudly, exactly
        as that bridge's own drafts() raising would - but only after every OTHER
        already-prepared bridge's enqueued work is fenced (one synchronize_device)
        and released (discard_pending), and before ANY bridge's readback (phase B
        - drafts() - must never run once phase A itself has failed)."""
        worker, bridge, events, scheduled = self.fixture()
        hook = FastWorkerHook(worker, bridge, cancelled=lambda: False)
        original = hook.bridges
        order = []
        mesh = object()
        operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: order.append(('sync', mesh))))
        bridges = self.make_pipelined_bridges('abcd', operations=operations, mesh=mesh, order=order)
        failure = RuntimeError('device fault preparing c')
        bridges['c'].request.runtime.drafter.prepare_device = Mock(side_effect=failure)
        hook.bridges = bridges
        outputs = ModuleType('vllm.v1.outputs')
        outputs.DraftTokenIds = SimpleNamespace
        try:
            with patch.dict(os.environ, {'QWEN_FAST_PIPELINED_PROPOSALS': '1'}), \
                    patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
                with self.assertRaises(RuntimeError) as failed:
                    worker.take_draft_token_ids()
            self.assertIs(failed.exception, failure)
            self.assertEqual([entry[:2] for entry in order],
                [('prepare', 'a'), ('prepare', 'b'), ('sync', mesh), ('discard', 'a'), ('discard', 'b')])
            operations.synchronize_device.assert_called_once_with(mesh)
            for name in 'abcd':
                bridges[name].drafts.assert_not_called()
            bridges['d'].request.runtime.drafter.prepare_device.assert_not_called()
        finally:
            hook.bridges = original
            hook.close()

    def test_pipelined_proposals_wrap_phase_a_in_phase_begin_and_end_lines(self):
        """QWEN_FAST_PIPELINED_PROPOSALS=1 with QWEN_FAST_PHASE_LOG=1: phase A (every
        eligible bridge's prewarm plus the one shared fence) gets its own
        '[PHASE] prepare_proposals <ids> begin/end' lines, the same shape phase B's own
        'propose' lines already have - so a slow round's phase-B total no longer looks
        like the whole round when phase A was where a chunk of the time actually went."""
        import serving_worker_hook

        worker, bridge, events, scheduled = self.fixture()
        hook = FastWorkerHook(worker, bridge, cancelled=lambda: False)
        original = hook.bridges
        order = []
        mesh = object()
        operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: order.append(('sync', mesh))))
        bridges = self.make_pipelined_bridges('ab', operations=operations, mesh=mesh, order=order)
        hook.bridges = bridges
        outputs = ModuleType('vllm.v1.outputs')
        outputs.DraftTokenIds = SimpleNamespace
        lines = []
        stub = ModuleType('loguru')
        stub.logger = SimpleNamespace(info=lambda template, *values: lines.append(template.format(*values)))
        try:
            with patch.dict(os.environ, {'QWEN_FAST_PIPELINED_PROPOSALS': '1'}), \
                    patch.dict('sys.modules', {'vllm.v1.outputs': outputs, 'loguru': stub}), \
                    patch.object(serving_worker_hook, 'PHASE_LOG', True):
                worker.take_draft_token_ids()
            # phase B's own 'propose' begin/end lines (one pair per bridge) are unchanged
            # and still fire; isolate phase A's new pair among them.
            phase_a = [line for line in lines if 'prepare_proposals' in line]
            self.assertEqual(len(phase_a), 2)
            self.assertEqual(phase_a[0], '[PHASE] prepare_proposals a,b begin')
            self.assertTrue(phase_a[1].startswith('[PHASE] prepare_proposals a,b end '))
            self.assertEqual(len(lines), 6, 'phase As pair plus two propose pairs, one per bridge')
            # phase A (both prepares, then the one fence) still runs entirely before
            # phase B (both drafts() calls) - the wrap changes nothing about that order
            self.assertEqual([entry[0] for entry in order], ['prepare', 'prepare', 'sync', 'drafts', 'drafts'])
        finally:
            hook.bridges = original
            hook.close()

    def test_the_default_non_pipelined_path_is_unaffected_by_phase_logging(self):
        """QWEN_FAST_PIPELINED_PROPOSALS unset: prepare_pipelined_drafts is never called at
        all (the existing off-by-default behaviour), so there is no phase A to wrap and
        QWEN_FAST_PHASE_LOG logs nothing for it - only phase B's own 'propose' lines, as
        before this change."""
        import serving_worker_hook

        worker, bridge, events, scheduled = self.fixture()
        hook = FastWorkerHook(worker, bridge, cancelled=lambda: False)
        original = hook.bridges
        order = []
        operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: order.append(('sync', mesh))))
        bridges = self.make_pipelined_bridges('ab', operations=operations, mesh=object(), order=order)
        hook.bridges = bridges
        outputs = ModuleType('vllm.v1.outputs')
        outputs.DraftTokenIds = SimpleNamespace
        lines = []
        stub = ModuleType('loguru')
        stub.logger = SimpleNamespace(info=lambda template, *values: lines.append(template.format(*values)))
        try:
            self.assertNotIn('QWEN_FAST_PIPELINED_PROPOSALS', os.environ)
            with patch.dict('sys.modules', {'vllm.v1.outputs': outputs, 'loguru': stub}), \
                    patch.object(serving_worker_hook, 'PHASE_LOG', True):
                worker.take_draft_token_ids()
            self.assertFalse(any(line.startswith('[PHASE] prepare_proposals') for line in lines))
            for name in 'ab':
                bridges[name].request.runtime.drafter.prepare_device.assert_not_called()
            operations.synchronize_device.assert_not_called()
        finally:
            hook.bridges = original
            hook.close()

    def test_packed_proposal_unset_never_creates_a_coordinator_and_matches_the_pipelined_default(self):
        """QWEN_FAST_PACKED_PROPOSAL unset (default): behaviour is byte-identical to
        test_pipelined_proposals_prewarm_every_eligible_bridge_then_fence_once above -
        same prepare/sync/drafts order, same result - and the hook never gains a
        _packed_coordinator attribute at all."""
        worker, bridge, events, scheduled = self.fixture()
        hook = FastWorkerHook(worker, bridge, cancelled=lambda: False)
        original = hook.bridges
        order = []
        mesh = object()
        operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: order.append(('sync', mesh))))
        bridges = self.make_pipelined_bridges('abcd', operations=operations, mesh=mesh, order=order)
        hook.bridges = bridges
        outputs = ModuleType('vllm.v1.outputs')
        outputs.DraftTokenIds = SimpleNamespace
        try:
            self.assertNotIn('QWEN_FAST_PACKED_PROPOSAL', os.environ)
            with patch.dict(os.environ, {'QWEN_FAST_PIPELINED_PROPOSALS': '1'}), \
                    patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
                result = worker.take_draft_token_ids()
            self.assertEqual((result.req_ids, result.draft_token_ids), (['a', 'b', 'c', 'd'], [[1], [1], [1], [1]]))
            self.assertEqual(order, [
                ('prepare', 'a', 100), ('prepare', 'b', 101), ('prepare', 'c', 102), ('prepare', 'd', 103),
                ('sync', mesh),
                ('drafts', 'a', {}), ('drafts', 'b', {}), ('drafts', 'c', {}), ('drafts', 'd', {}),
            ])
            self.assertFalse(hasattr(hook, '_packed_coordinator'))
        finally:
            hook.bridges = original
            hook.close()

    def test_packed_proposal_enabled_with_no_pooled_slots_falls_back_to_the_same_per_bridge_prepares(self):
        """QWEN_FAST_PACKED_PROPOSAL=1 but no bridge's drafter exposes a pool_slot
        (exactly make_pipelined_bridges' own fixture): PackedProposalCoordinator finds
        no full FOUR_AS_TWO_PAIRS pair to build and every bridge still runs its own
        single-user prepare_device(), in the same order, with the same one shared
        fence - the flag changes WHICH function decides that, not the result."""
        worker, bridge, events, scheduled = self.fixture()
        hook = FastWorkerHook(worker, bridge, cancelled=lambda: False)
        original = hook.bridges
        order = []
        mesh = object()
        operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: order.append(('sync', mesh))))
        bridges = self.make_pipelined_bridges('abcd', operations=operations, mesh=mesh, order=order)
        hook.bridges = bridges
        outputs = ModuleType('vllm.v1.outputs')
        outputs.DraftTokenIds = SimpleNamespace
        try:
            with patch.dict(os.environ, {'QWEN_FAST_PIPELINED_PROPOSALS': '1', 'QWEN_FAST_PACKED_PROPOSAL': '1'}), \
                    patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
                result = worker.take_draft_token_ids()
            self.assertEqual((result.req_ids, result.draft_token_ids), (['a', 'b', 'c', 'd'], [[1], [1], [1], [1]]))
            self.assertEqual(order, [
                ('prepare', 'a', 100), ('prepare', 'b', 101), ('prepare', 'c', 102), ('prepare', 'd', 103),
                ('sync', mesh),
                ('drafts', 'a', {}), ('drafts', 'b', {}), ('drafts', 'c', {}), ('drafts', 'd', {}),
            ])
            self.assertIsNotNone(getattr(hook, '_packed_coordinator', None))
        finally:
            hook.bridges = original
            hook.close()

    def test_packed_coordinator_is_closed_with_the_hook(self):
        worker, bridge, events, scheduled = self.fixture()
        hook = FastWorkerHook(worker, bridge, cancelled=lambda: False)
        original = hook.bridges
        order = []
        operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: order.append(('sync', mesh))))
        bridges = self.make_pipelined_bridges('ab', operations=operations, mesh=object(), order=order)
        hook.bridges = bridges
        outputs = ModuleType('vllm.v1.outputs')
        outputs.DraftTokenIds = SimpleNamespace
        with patch.dict(os.environ, {'QWEN_FAST_PIPELINED_PROPOSALS': '1', 'QWEN_FAST_PACKED_PROPOSAL': '1'}), \
                patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
            worker.take_draft_token_ids()
        hook.bridges = original
        coordinator = hook._packed_coordinator
        coordinator.close = Mock(wraps=coordinator.close)
        hook.close()
        coordinator.close.assert_called_once()

    def test_queued_sampler_or_second_owner_rejected(self):
        worker, bridge, _, _ = self.fixture()
        bridge.runner._pending_samples.append(object())
        with self.assertRaises(ValueError):
            FastWorkerHook(worker, bridge, cancelled=lambda: False)
        bridge.runner._pending_samples.clear()
        hook = FastWorkerHook(worker, bridge, cancelled=lambda: False)
        with self.assertRaises(ValueError):
            FastWorkerHook(worker, bridge, cancelled=lambda: False)
        hook.close()

    def test_cleanup_failure_does_not_restore_unsafe_baseline(self):
        worker, bridge, _, _ = self.fixture()
        hook = FastWorkerHook(worker, bridge, cancelled=lambda: False)
        bridge.close = Mock(side_effect=RuntimeError('trace cleanup failed'))
        with self.assertRaises(RuntimeError):
            hook.close()
        self.assertFalse(hook.closed)
        self.assertIs(worker.model_runner._qwen_fast_hook, hook)


class PhaseLogTests(unittest.TestCase):
    def test_phase_runs_the_call_and_returns_its_result_when_logging_is_off(self):
        import serving_worker_hook

        with patch.object(serving_worker_hook, 'PHASE_LOG', False):
            self.assertEqual(serving_worker_hook.phase('propose', 'r', lambda: 'drafts'), 'drafts')

    def test_phase_logs_begin_and_end_around_the_call(self):
        import sys
        from types import ModuleType
        import serving_worker_hook

        lines = []
        stub = ModuleType('loguru')
        stub.logger = SimpleNamespace(info=lambda template, *values: lines.append(template.format(*values)))
        with patch.dict(sys.modules, {'loguru': stub}), patch.object(serving_worker_hook, 'PHASE_LOG', True):
            self.assertEqual(serving_worker_hook.phase('propose', 'r', lambda: 'drafts'), 'drafts')
        self.assertEqual(lines[0], '[PHASE] propose r begin')
        self.assertTrue(lines[1].startswith('[PHASE] propose r end '))
