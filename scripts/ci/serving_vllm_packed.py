"""Scheduler contract for a packed block of several resident decode requests.

`serving_vllm_contract.admit_scheduler_output` asserts

    list(cached.req_ids) != [request_id]

one resident decode. Probe 35436807668 measured TTScheduler scheduling every live
request's decode in ONE step - `cached=['B','A']`, 16 rows each, both proposal sets
present - so that contract cannot admit the step a second user produces.

Two things about that probe shape are load bearing here. The step carries BOTH
requests, so the contract is widened per request rather than relaxed; and it
carried them as ['B','A'], NOT in the order the requests were created. The pack
must therefore be assembled in the SCHEDULER's order, because the row segments of
the packed block are matched to the scheduler's rows. Assembling in registry order
would silently give each user the other's proposals.
"""

from serving_vllm_contract import prepared_ticket, step_refusals


def ordered_tickets(requests, scheduled):
    """One live ticket per scheduled request, in the scheduler's own order."""
    cached = scheduled.scheduled_cached_reqs
    order = list(cached.req_ids)
    by_id = {}
    for request in requests:
        ticket = prepared_ticket(request)
        if ticket.request_id in by_id:
            raise ValueError('Two live requests claim one request id')
        by_id[ticket.request_id] = (request, ticket)
    # The live set decides which finished ids concern this step: a partner that
    # completed was detached by the lifecycle before this step and is not here.
    found = step_refusals(scheduled, live=set(by_id))
    if not order:
        found.append('no cached request')
    elif len(set(order)) != len(order):
        found.append('duplicate order=%r' % (order,))
    if found:
        raise ValueError('Resident decode requests at the exact scheduled frontier required: '
                         + '; '.join(found))
    if set(by_id) != set(order):
        # Which side has the extra id decides what went wrong, and a bare refusal says
        # neither. Run 35699498963 hit this immediately after the first interleaving
        # state on hardware - a user decoding while another prefilled in chunks - and
        # the message left the cause unknowable from the artifact.
        raise ValueError('The scheduled requests and the prepared requests must be the same set: '
                         'scheduled_only=%r prepared_only=%r scheduled=%r prepared=%r'
                         % (sorted(set(order) - set(by_id)), sorted(set(by_id) - set(order)),
                            list(order), sorted(by_id)))
    resumed = set(getattr(cached, 'resumed_req_ids', ()) or ())
    positions = list(cached.num_computed_tokens)
    if len(positions) != len(order):
        raise ValueError('One computed-token frontier per scheduled request required')
    entries = []
    for request_id, position in zip(order, positions):
        request, ticket = by_id[request_id]
        if request_id in resumed or position != ticket.position:
            raise ValueError('Resident decode requests at the exact scheduled frontier required: '
                             'request=%r resumed=%r frontier=%r ticket_position=%r order=%r'
                             % (request_id, request_id in resumed, position, ticket.position, order))
        entries.append(dict(request_id=request_id, request=request, ticket=ticket))
    return entries


def admit_packed_scheduler_output(requests, scheduled):
    """Admit a step carrying every packed request's verifier block."""
    entries = ordered_tickets(requests, scheduled)
    counts = scheduled.num_scheduled_tokens
    expected_counts = {entry['request_id']: len(entry['ticket'].tokens) for entry in entries}
    total = sum(expected_counts.values())
    if (dict(counts) != expected_counts
            or type(scheduled.total_num_scheduled_tokens) is not int
            or scheduled.total_num_scheduled_tokens != total):
        raise ValueError('Scheduler must reserve the complete prepared verifier block for every request')
    expected_proposals = {entry['request_id']: list(entry['ticket'].tokens[1:])
                          for entry in entries if len(entry['ticket'].tokens) > 1}
    if dict(scheduled.scheduled_spec_decode_tokens) != expected_proposals:
        raise ValueError('Scheduled draft tokens differ from the prepared proposals')
    return entries


def packed_draft_token_ids(requests, order=None):
    """Draft tokens for every packed request, in the scheduler's order when given."""
    from vllm.v1.outputs import DraftTokenIds

    tickets = [prepared_ticket(request) for request in requests]
    by_id = {ticket.request_id: ticket for ticket in tickets}
    if len(by_id) != len(tickets):
        raise ValueError('Two live requests claim one request id')
    chosen = tickets if order is None else [by_id[request_id] for request_id in order]
    return DraftTokenIds(req_ids=[ticket.request_id for ticket in chosen],
                         draft_token_ids=[list(ticket.tokens[1:]) for ticket in chosen])


def packed_model_runner_output(outputs):
    """One ModelRunnerOutput covering every packed request, in the given order."""
    from vllm.v1.outputs import ModelRunnerOutput

    request_ids = [output.request_id for output in outputs]
    if not request_ids or len(set(request_ids)) != len(request_ids):
        raise ValueError('One committed output per packed request required')
    return ModelRunnerOutput(req_ids=request_ids,
        req_id_to_index={request_id: index for index, request_id in enumerate(request_ids)},
        sampled_token_ids=[list(output.token_ids) for output in outputs], logprobs=None,
        prompt_logprobs_dict={request_id: None for request_id in request_ids}, pooler_output=[])
