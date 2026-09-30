from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import test_serving_fast_request
from serving_vllm_contract import admit_scheduler_output, draft_token_ids, execute_scheduled, model_runner_output


class SchedulerContractTests(unittest.TestCase):
    def fixture(self):
        request, events = test_serving_fast_request.FastServingTests().fixture()
        ticket = request.prepare('request')
        scheduled = SimpleNamespace(scheduled_new_reqs=[], finished_req_ids=set(), preempted_req_ids=set(),
            has_structured_output_requests=False, scheduled_encoder_inputs={},
            scheduled_cached_reqs=SimpleNamespace(req_ids=['request'], resumed_req_ids=set(),
                num_computed_tokens=[ticket.position]),
            num_scheduled_tokens={'request': len(ticket.tokens)}, total_num_scheduled_tokens=len(ticket.tokens),
            scheduled_spec_decode_tokens={'request': list(ticket.tokens[1:])})
        return request, events, scheduled

    def test_prepared_tokens_are_verified_once_after_admission(self):
        request, events, scheduled = self.fixture()
        self.assertEqual(events, [])
        ticket = admit_scheduler_output(request, scheduled)
        output = execute_scheduled(request, scheduled, cancelled=lambda: False)
        request.engine.verify.assert_called_once_with(ticket)
        self.assertEqual(output.token_ids, tuple(range(11, 27)))
        with self.assertRaises(ValueError):
            execute_scheduled(request, scheduled, cancelled=lambda: False)

    def test_stale_partial_and_mismatched_schedules_fail_before_verification(self):
        for field, value in (('num_scheduled_tokens', {'request': 8}), ('total_num_scheduled_tokens', 8),
                ('scheduled_spec_decode_tokens', {'request': [99] * 15}),
                ('preempted_req_ids', {'request'}), ('finished_req_ids', {'request'}),
                ('has_structured_output_requests', True), ('scheduled_new_reqs', [object()])):
            request, _, scheduled = self.fixture()
            setattr(scheduled, field, value)
            with self.assertRaises(ValueError):
                execute_scheduled(request, scheduled, cancelled=lambda: False)
            request.engine.verify.assert_not_called()
        request, _, scheduled = self.fixture()
        scheduled.scheduled_cached_reqs.num_computed_tokens = [1]
        with self.assertRaises(ValueError):
            admit_scheduler_output(request, scheduled)

    def test_a_refusal_names_what_it_judged(self):
        request, _, scheduled = self.fixture()
        scheduled.finished_req_ids = {'request'}
        scheduled.scheduled_cached_reqs.num_computed_tokens = [1]
        with self.assertRaises(ValueError) as caught:
            admit_scheduler_output(request, scheduled)
        self.assertIn("finished=['request']", str(caught.exception))
        self.assertIn('frontier=[1] ticket_position=', str(caught.exception))
        request.engine.verify.assert_not_called()

    def test_another_requests_completion_does_not_refuse_this_one(self):
        request, _, scheduled = self.fixture()
        scheduled.finished_req_ids = {'other'}
        self.assertIs(admit_scheduler_output(request, scheduled), request.session.pending)

    def test_vllm_output_keeps_variable_committed_length(self):
        request, _, scheduled = self.fixture()
        outputs = ModuleType('vllm.v1.outputs')
        outputs.DraftTokenIds = outputs.ModelRunnerOutput = SimpleNamespace
        with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
            advertised = draft_token_ids(request)
            self.assertEqual(advertised.draft_token_ids, [list(range(11, 26))])
            output = execute_scheduled(request, scheduled, cancelled=lambda: False)
            result = model_runner_output(output)
        self.assertEqual(result.sampled_token_ids, [list(range(11, 27))])
        self.assertEqual(result.req_id_to_index, {'request': 0})

    def test_preempted_prepared_request_can_close_without_publishing(self):
        request, events, _ = self.fixture()
        request.close('request')
        self.assertEqual(events, [('close_engine',), ('close_drafter',)])
        self.assertEqual(request.session.phase, 'closed')
