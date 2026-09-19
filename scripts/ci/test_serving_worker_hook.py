import ast
import __future__
from collections import deque
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
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
