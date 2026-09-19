"""Complete-request precision comparison with independent per-arm proposal audits."""

import math

from dspark_request_variants import proposal_signature


SCHEDULE = (('hifi4', True), ('hifi2', True), ('hifi4', False),
    ('hifi2', False), ('hifi2', False), ('hifi4', False))


def summarize(requests, summarize_requests, validate_audit, validate_route):
    if [(value.get('drafter_precision'), value.get('instrumented_timing'))
            for value in requests] != list(SCHEDULE):
        raise ValueError('Two independent precision audits followed by complete ABBA requests required')
    audits = {value['drafter_precision']: value for value in requests[:2]}
    for value in requests:
        if (value.get('arm') != 'publication' or value.get('length') != 4096
                or any(value.get(name) is not True for name in ('exact', 'state_exact', 'inactive_exact'))
                or value.get('prompt_tokens') != requests[0].get('prompt_tokens')
                or value.get('emitted') != requests[0].get('emitted')):
            raise ValueError('Same full prompt and exact native target output/state required in both arms')
        validate_route(value, 'publication')
        audit = audits[value['drafter_precision']]
        if value['instrumented_timing']:
            validate_audit(value)
        elif proposal_signature(value) != proposal_signature(audit):
            raise ValueError('Timed proposals must reproduce their own precision arm audit')
        blocks = value.get('blocks', [])
        if (not blocks or any(type(block.get('accepted')) is not int
                or type(block.get('committed')) is not int or type(block.get('rows')) is not int
                or not 0 <= block['accepted'] < block['rows']
                or not 1 <= block['committed'] <= block['accepted'] + 1 for block in blocks)
                or sum(block['accepted'] for block in blocks) != value.get('accepted')
                or sum(block['committed'] for block in blocks) != value.get('committed_decode_tokens')):
            raise ValueError('Acceptance and committed-token counts must match complete block evidence')
    arms = {precision: summarize_requests([value for value in requests
        if value['drafter_precision'] == precision]) for precision in audits}
    pairs = []
    for control, candidate in ((requests[2], requests[3]), (requests[5], requests[4])):
        rates = []
        for request in (control, candidate):
            rate = request.get('committed_tokens_per_second')
            if (type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0
                    or not math.isclose(rate, 1000 * request['committed_decode_tokens'] / request['decode_ms'],
                        rel_tol=1e-9)):
                raise ValueError('TG must include the complete committed decode loop')
            rates.append(rate)
        pairs.append(dict(hifi4_tg=rates[0], hifi2_tg=rates[1], change_percent=100 * (rates[1] / rates[0] - 1)))
    return dict(arms=arms, pairs=pairs,
        committed_tg_change_percent=100 * (arms['hifi2']['committed_tg'] / arms['hifi4']['committed_tg'] - 1),
        improvement_screen_passed=all(pair['change_percent'] > 2 for pair in pairs),
        performance_promoted=False, held_out_coding_quality=False, serving_defaults_changed=False,
        timing_boundary='Complete decode loop including drafting, verification, publication, readback and stalls')
