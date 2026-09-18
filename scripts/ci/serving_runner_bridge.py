"""Uninstalled synchronous TT runner bridge; construction needs a prepared fast request."""

from serving_vllm_contract import admit_scheduler_output, draft_token_ids, model_runner_output
from serving_vllm_state import apply_committed_output, validate_runner_reservation


class FastRunnerBridge:
    def __init__(self, runner, request, page_binding):
        if (runner.non_dp_async_scheduling or runner.tt_data_parallel_size != 1
                or page_binding.engine is not request.engine):
            raise ValueError('Synchronous TP2 single-request runner and matching page binding required')
        self.runner, self.request, self.page_binding = runner, request, page_binding
        self.state = runner.requests[request.session.request_id]
        self.failed = False

    def drafts(self):
        if self.failed or self.request.closed or self.request.cancelled:
            raise ValueError('Live fast request required')
        if self.request.session.finished:
            return None
        if self.request.session.pending is None:
            self.request.prepare(self.request.session.request_id)
        return draft_token_ids(self.request)

    def execute_decode(self, scheduled, *, cancelled):
        if self.failed:
            raise ValueError('Failed runner bridge cannot execute another block')
        ticket = admit_scheduler_output(self.request, scheduled)
        try:
            self.runner._update_states(scheduled)
            validate_runner_reservation(self.runner, self.state, ticket)
            if len(self.state.block_ids) != 1:
                raise ValueError('One explicit target KV page group required')
            self.page_binding.refresh(self.state.block_ids[0], position=ticket.position, rows=len(ticket.tokens))
            output = self.request.step(ticket.request_id, cancelled=cancelled)
            apply_committed_output(self.runner, self.state, output)
            return model_runner_output(output)
        except BaseException:
            self.failed = True
            raise

    def close(self):
        self.request.close(self.request.session.request_id)
