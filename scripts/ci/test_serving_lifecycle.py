from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from serving_lifecycle import FastServingLifecycle
from test_serving_fast_policy import FastPolicyTests
from test_serving_request_factory import RequestFactoryTests
from test_serving_worker_hook import WorkerHookTests


class LifecycleTests(unittest.TestCase):
    def fixture(self):
        worker, bridge, _, decode = WorkerHookTests().fixture()
        worker.model_runner.execute_model.return_value = None
        worker.model_runner.sample_tokens.return_value = SimpleNamespace(req_ids=['request'], sampled_token_ids=[[10]])
        _, _, _, arguments = RequestFactoryTests().fixture()
        new = SimpleNamespace(req_id='request', prompt_token_ids=[1] * 4096,
            num_computed_tokens=0, mm_features=[], prompt_embeds=None, lora_request=None,
            sampling_params=arguments['state'].sampling_params)
        scheduled = SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=[new],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]), scheduled_spec_decode_tokens={},
            num_scheduled_tokens={'request': 4096}, total_num_scheduled_tokens=4096)
        capture = SimpleNamespace(capture=Mock(side_effect=lambda: nullcontext()), close=Mock())

        def factory(state, features):
            self.assertIs(state, bridge.state)
            self.assertIs(features, capture)
            features.close()
            return bridge

        build = Mock(side_effect=factory)
        lifecycle = FastServingLifecycle(worker, config=FastPolicyTests().fixture(),
            capture_factory=Mock(return_value=capture), bridge_factory=build,
            eos_ids=(99,), cancelled=lambda: False)
        return lifecycle, worker, bridge, capture, build, scheduled, decode

    def test_prefill_to_committed_decode_to_finished_cleanup(self):
        lifecycle, worker, bridge, capture, build, prefill, decode = self.fixture()
        self.assertIsNone(worker.execute_model(prefill))
        build.assert_not_called()
        seed = worker.sample_tokens(None)
        self.assertEqual(seed.sampled_token_ids, [[10]])
        build.assert_called_once()
        outputs = ModuleType('vllm.v1.outputs')
        outputs.ModelRunnerOutput = outputs.DraftTokenIds = SimpleNamespace
        with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
            self.assertEqual(worker.take_draft_token_ids().draft_token_ids, [list(range(11, 26))])
            result = worker.execute_model(decode)
        self.assertEqual(result.sampled_token_ids, [list(range(11, 27))])
        finished = SimpleNamespace(finished_req_ids={'request'}, total_num_scheduled_tokens=0)
        worker.execute_model(finished)
        self.assertTrue(bridge.request.closed)
        self.assertIsNone(lifecycle.hook)
        self.assertIsNone(lifecycle.request_id)
        self.assertIsNone(worker.take_draft_token_ids())
        capture.close.assert_called_once()
        lifecycle.close()
        self.assertFalse(hasattr(worker, '_qwen_fast_lifecycle'))

    def test_terminal_seed_skips_factory_and_releases_features(self):
        lifecycle, worker, bridge, capture, build, prefill, _ = self.fixture()
        bridge.state.output_token_ids[:] = [99]
        worker.model_runner.sample_tokens.return_value.sampled_token_ids = [[99]]
        worker.execute_model(prefill)
        worker.sample_tokens(None)
        build.assert_not_called()
        capture.close.assert_called_once()
        self.assertIsNone(lifecycle.hook)
        lifecycle.close()

    def test_partial_prefill_rejected_without_device_execution(self):
        lifecycle, worker, _, capture, build, prefill, _ = self.fixture()
        prefill.total_num_scheduled_tokens = 2048
        with self.assertRaises(ValueError):
            worker.execute_model(prefill)
        lifecycle.original_execute.__self__.model_runner.execute_model.assert_not_called()
        capture.capture.assert_not_called()
        build.assert_not_called()
        self.assertTrue(lifecycle.failed)
        lifecycle.close()

    def test_factory_failure_poisons_lifecycle_and_keeps_features_for_cleanup(self):
        lifecycle, worker, _, capture, build, prefill, _ = self.fixture()
        build.side_effect = RuntimeError('trace setup failed')
        worker.execute_model(prefill)
        with self.assertRaises(RuntimeError):
            worker.sample_tokens(None)
        with self.assertRaises(ValueError):
            worker.take_draft_token_ids()
        lifecycle.close()
        capture.close.assert_called_once()
