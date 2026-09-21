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
