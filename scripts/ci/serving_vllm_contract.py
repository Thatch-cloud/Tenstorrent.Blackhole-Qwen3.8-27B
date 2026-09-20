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


def step_refusals(scheduled):
    """The step-level conditions no decode contract admits, each with its value.

    Run 35484349353 was refused on one of these five and the message named none
    of them, so the run's whole evidence was a line number. A refusal must say
    what it judged, as the prefill refusal in serving_lifecycle already does.
    """
    found = []
    new = [getattr(value, 'req_id', value) for value in (scheduled.scheduled_new_reqs or ())]
    if new:
        found.append('new=%r' % (new,))
    finished = scheduled.finished_req_ids
    if finished:
        found.append('finished=%r' % (sorted(finished, key=str),))
    preempted = getattr(scheduled, 'preempted_req_ids', None)
    if preempted:
        found.append('preempted=%r' % (sorted(preempted, key=str),))
    if getattr(scheduled, 'has_structured_output_requests', False):
        found.append('structured_output=True')
    encoder = getattr(scheduled, 'scheduled_encoder_inputs', {})
    if encoder:
        found.append('encoder_inputs=%r' % (encoder,))
    return found


def admit_scheduler_output(request, scheduled):
    ticket = prepared_ticket(request)
    request_id = ticket.request_id
    cached = scheduled.scheduled_cached_reqs
    found = step_refusals(scheduled)
    if list(cached.req_ids) != [request_id]:
        found.append('cached=%r expected=%r' % (list(cached.req_ids), [request_id]))
    if request_id in (getattr(cached, 'resumed_req_ids', None) or ()):
        found.append('resumed=%r' % (request_id,))
    if list(cached.num_computed_tokens) != [ticket.position]:
        found.append('frontier=%r ticket_position=%r' % (list(cached.num_computed_tokens), ticket.position))
    if found:
        raise ValueError('Single resident decode request at the exact scheduled frontier required: '
                         + '; '.join(found))
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
