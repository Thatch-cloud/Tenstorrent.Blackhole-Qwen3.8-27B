from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_serving_vllm_contract
import test_serving_vllm_state
from serving_runner_bridge import FastRunnerBridge


class RunnerBridgeTests(unittest.TestCase):
    def fixture(self):
        request, events, scheduled = test_serving_vllm_contract.SchedulerContractTests().fixture()
        runner, state = test_serving_vllm_state.RunnerStateTests().fixture()
        state.block_ids = ([4],)
        runner.non_dp_async_scheduling = False
        runner.tt_data_parallel_size = 1
        runner._update_states = Mock(side_effect=lambda output: events.append(('scheduler_update',)))
        binding = SimpleNamespace(engine=request.engine,
            refresh=Mock(side_effect=lambda *args, **kwargs: events.append(('pages_bound',))))
        return FastRunnerBridge(runner, request, binding), events, scheduled

    def test_composes_admission_page_refresh_commit_and_host_accounting(self):
        bridge, events, scheduled = self.fixture()
        outputs = ModuleType('vllm.v1.outputs')
        outputs.ModelRunnerOutput = SimpleNamespace
        with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
            output = bridge.execute_decode(scheduled, cancelled=lambda: False)
        self.assertEqual(output.sampled_token_ids, [list(range(11, 27))])
        self.assertEqual(events, [('scheduler_update',), ('pages_bound',), ('verify', 16),
            ('target', 16), ('history', 16)])
        self.assertEqual(bridge.state.output_token_ids, list(range(10, 27)))
        self.assertEqual(bridge.state.num_computed_tokens, 18)

    def test_capacity_rejected_before_page_refresh_and_verification(self):
        bridge, _, scheduled = self.fixture()
        bridge.runner.model_config.max_model_len = 4
        with self.assertRaises(ValueError):
            bridge.execute_decode(scheduled, cancelled=lambda: False)
        bridge.page_binding.refresh.assert_not_called()
        bridge.request.engine.verify.assert_not_called()
        self.assertTrue(bridge.failed)

    def test_page_upload_failure_prevents_device_verification(self):
        bridge, _, scheduled = self.fixture()
        bridge.page_binding.refresh.side_effect = RuntimeError('page upload failed')
        with self.assertRaises(RuntimeError):
            bridge.execute_decode(scheduled, cancelled=lambda: False)
        bridge.request.engine.verify.assert_not_called()
        self.assertTrue(bridge.failed)

    def test_preempted_schedule_is_not_executed(self):
        bridge, _, scheduled = self.fixture()
        scheduled.preempted_req_ids = {'request'}
        with self.assertRaises(ValueError):
            bridge.execute_decode(scheduled, cancelled=lambda: False)
        bridge.runner._update_states.assert_not_called()
        bridge.close()
        self.assertTrue(bridge.request.closed)
