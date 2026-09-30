"""Matched complete requests; only the T16 register-resident rounded epilogue changes."""

import math

from dspark_request_variants import proposal_signature


SCHEDULE = ((False, True), (True, True), (False, False), (True, False), (True, False), (False, False))


def summarize(requests, summarize_requests, validate_audit, validate_route):
    if [(value.get('register_epilogue', {}).get('register_resident'), value.get('instrumented_timing'))
            for value in requests] != list(SCHEDULE):
        raise ValueError('Two fresh audits followed by complete unchanged/register_resident/register_resident/unchanged requests required')
    for value in requests:
        if (value.get('arm') != 'publication' or value.get('length') != 4096
                or any(value.get(name) is not True for name in ('exact', 'state_exact', 'inactive_exact'))
                or value.get('prompt_tokens') != requests[0].get('prompt_tokens')
                or value.get('emitted') != requests[0].get('emitted')
                or proposal_signature(value) != proposal_signature(requests[0])):
            raise ValueError('Same prompt, output, target state and proposal acceptance required')
        validate_route(value, 'publication')
        if value['instrumented_timing']:
            validate_audit(value)
    arms = {name: summarize_requests([value for value in requests
        if value['register_epilogue']['register_resident'] is enabled]) for name, enabled in (('unchanged', False), ('register_resident', True))}
    pairs = []
    for control, candidate in ((requests[2], requests[3]), (requests[5], requests[4])):
        baseline = control.get('committed_tokens_per_second')
        changed = candidate.get('committed_tokens_per_second')
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in (baseline, changed)):
            raise ValueError('Positive finite complete-cycle TG required')
        for request, rate in ((control, baseline), (candidate, changed)):
            if not math.isclose(rate, 1000 * request['committed_decode_tokens'] / request['decode_ms'], rel_tol=1e-9):
                raise ValueError('Per-request TG must match complete decode wall time')
        pairs.append(dict(unchanged_tg=baseline, register_resident_tg=changed, change_percent=100 * (changed / baseline - 1)))
    return dict(arms=arms, pairs=pairs,
        committed_tg_change_percent=100 * (arms['register_resident']['committed_tg'] / arms['unchanged']['committed_tg'] - 1),
        improvement_screen_passed=all(pair['change_percent'] > 2 for pair in pairs),
        performance_promoted=False, held_out_coding_quality=False, serving_defaults_changed=False,
        timing_boundary='Complete decode loop including drafting, verification, publication, readback and stalls')
