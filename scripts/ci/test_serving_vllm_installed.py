import unittest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

import test_serving_vllm_contract
from serving_vllm_contract import draft_token_ids, execute_scheduled, model_runner_output


class InstalledVllmContractTests(unittest.TestCase):
    def test_canary_trace_capacity_reaches_pinned_worker_device_parameters(self):
        import ast
        from importlib.util import find_spec
        from pathlib import Path
        from types import SimpleNamespace
        from serving_canary_runner import additional_config
        from vllm_tt_plugin.config import get_tt_config

        config = get_tt_config(SimpleNamespace(additional_config=additional_config('/target')))
        source = Path(find_spec('vllm_tt_plugin').origin).with_name('worker.py').read_text()
        function = next(node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == 'device_params_from_tt_config')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<installed-device-params>', 'exec'), namespace)
        self.assertEqual(config['trace_mode'], 'decode_only')
        parameters = namespace['device_params_from_tt_config'](config, config['trace_mode'])
        self.assertEqual(parameters, dict(trace_region_size=1073741824, l1_small_size=24576))

    def test_target_generation_eos_is_not_rejected_as_custom_stopping(self):
        from vllm.sampling_params import SamplingParams
        from serving_fast_policy import validate_request_sampling

        parameters = SamplingParams(temperature=0, max_tokens=256)
        eos_ids = (248046, 248044)
        parameters.update_from_generation_config({'eos_token_id': list(eos_ids)}, 248046)
        self.assertEqual(parameters.stop_token_ids, [248044])
        validate_request_sampling(parameters, prompt_tokens=4096, eos_ids=eos_ids)
        parameters.stop_token_ids.append(13)
        with self.assertRaises(ValueError):
            validate_request_sampling(parameters, prompt_tokens=4096, eos_ids=eos_ids)

    def fixture(self):
        request, events, reference = test_serving_vllm_contract.SchedulerContractTests().fixture()
        scheduled = SchedulerOutput.make_empty()
        scheduled.scheduled_cached_reqs.req_ids = ['request']
        scheduled.scheduled_cached_reqs.num_computed_tokens = [2]
        scheduled.num_scheduled_tokens = reference.num_scheduled_tokens
        scheduled.total_num_scheduled_tokens = reference.total_num_scheduled_tokens
        scheduled.scheduled_spec_decode_tokens = reference.scheduled_spec_decode_tokens
        return request, events, scheduled

    def test_real_vllm_draft_and_multi_token_output_types(self):
        request, _, scheduled = self.fixture()
        drafts = draft_token_ids(request)
        self.assertIsInstance(drafts, DraftTokenIds)
        self.assertEqual(drafts.req_ids, ['request'])
        self.assertEqual(drafts.draft_token_ids, [list(range(11, 26))])
        output = execute_scheduled(request, scheduled, cancelled=lambda: False)
        result = model_runner_output(output)
        self.assertIsInstance(result, ModelRunnerOutput)
        self.assertEqual(result.sampled_token_ids, [list(range(11, 27))])
        self.assertEqual(result.req_id_to_index, {'request': 0})

    def test_real_scheduler_output_rejects_rewritten_drafts(self):
        request, _, scheduled = self.fixture()
        scheduled.scheduled_spec_decode_tokens['request'][0] = 99
        with self.assertRaises(ValueError):
            execute_scheduled(request, scheduled, cancelled=lambda: False)
        request.engine.verify.assert_not_called()

    def test_real_output_cancellation_exposes_no_draft_tokens(self):
        request, _, scheduled = self.fixture()
        polls = iter((False, True))
        output = execute_scheduled(request, scheduled, cancelled=lambda: next(polls))
        self.assertEqual(model_runner_output(output).sampled_token_ids, [[]])
        self.assertEqual(request.session.position, 2)
