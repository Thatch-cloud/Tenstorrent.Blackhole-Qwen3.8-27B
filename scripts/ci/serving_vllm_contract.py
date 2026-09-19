"""Opt-in synchronous single-request scheduler contract for the pinned TT plugin."""


def prepared_ticket(request):
    ticket = request.session.pending
    if (request.closed or request.cancelled or request.busy or request.session.phase != 'pending'
            or request.engine.phase != 'idle' or ticket is None
            or ticket.request_id != request.session.request_id or ticket.position != request.session.position):
        raise ValueError('Live prepared request ticket required before scheduler admission')
    return ticket


def draft_token_ids(request):
    from vllm.v1.outputs import DraftTokenIds

    ticket = prepared_ticket(request)
    return DraftTokenIds(req_ids=[ticket.request_id], draft_token_ids=[list(ticket.tokens[1:])])


def admit_scheduler_output(request, scheduled):
    ticket = prepared_ticket(request)
    request_id = ticket.request_id
    cached = scheduled.scheduled_cached_reqs
    if (scheduled.scheduled_new_reqs or scheduled.finished_req_ids
            or getattr(scheduled, 'preempted_req_ids', None)
            or getattr(scheduled, 'has_structured_output_requests', False)
            or getattr(scheduled, 'scheduled_encoder_inputs', {})
            or list(cached.req_ids) != [request_id]
            or request_id in cached.resumed_req_ids
            or list(cached.num_computed_tokens) != [ticket.position]):
        raise ValueError('Single resident decode request at the exact scheduled frontier required')
    counts = scheduled.num_scheduled_tokens
    if (set(counts) != {request_id} or type(counts[request_id]) is not int
            or counts[request_id] != len(ticket.tokens)
            or type(scheduled.total_num_scheduled_tokens) is not int
            or scheduled.total_num_scheduled_tokens != len(ticket.tokens)):
        raise ValueError('Scheduler must reserve the complete prepared verifier block')
    proposals = scheduled.scheduled_spec_decode_tokens
    expected = {request_id: list(ticket.tokens[1:])} if len(ticket.tokens) > 1 else {}
    if proposals != expected:
        raise ValueError('Scheduled draft tokens differ from the prepared proposal')
    return ticket


def execute_scheduled(request, scheduled, *, cancelled):
    ticket = admit_scheduler_output(request, scheduled)
    return request.step(ticket.request_id, cancelled=cancelled)


def model_runner_output(output):
    from vllm.v1.outputs import ModelRunnerOutput

    return ModelRunnerOutput(req_ids=[output.request_id], req_id_to_index={output.request_id: 0},
        sampled_token_ids=[list(output.token_ids)], logprobs=None,
        prompt_logprobs_dict={output.request_id: None}, pooler_output=[])
