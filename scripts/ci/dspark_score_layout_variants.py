"""Matched complete DSpark requests changing only Markov score layout."""

from dspark_request_variants import proposal_signature


POLICIES = {
    'control': dict(proposal_trace=True, commit_only_gdn=True, native_attention=True, target_attention_t16=True, score_layout=False),
    'scores': dict(proposal_trace=True, commit_only_gdn=True, native_attention=True, target_attention_t16=True, score_layout=True),
}
SCHEDULE = (('control', True), ('scores', True), ('control', False),
    ('scores', False), ('scores', False), ('control', False))


def validate_route(value, arm):
    from pathlib import Path
    from dspark_target_attention_variants import validate_route as attention_route
    from dspark_score_layout_hardware_gate import qualify, validate_hardware

    if arm not in POLICIES:
        raise ValueError('Unknown score-layout arm')
    attention_route(value, 'parallel')
    audit = value.get('score_layout')
    if arm == 'control':
        if audit is not None:
            raise ValueError('Native control must not execute candidate score layout')
        return
    if (not isinstance(audit, dict) or audit.get('restored') is not True
            or type(audit.get('calls')) is not int or audit['calls'] < 2
            or audit.get('qualification') != qualify(Path(__file__).parent)):
        raise ValueError('Used, restored and independently qualified score-layout scope required')
    if audit.get('hardware_audit_sha256') != validate_hardware(audit.get('hardware_audit'), Path(__file__).parent):
        raise ValueError('Complete learned hardware audit digest required')


def summarize_variants(requests):
    from dspark_request_experiment import summarize

    if [(value.get('arm'), value.get('instrumented_timing')) for value in requests] != list(SCHEDULE):
        raise ValueError('Two audits followed by complete A/B/B/A timed requests required')
    signatures = {}
    for value in requests:
        arm = value['arm']
        draft = value.get('dspark', {})
        if (draft.get('native_attention') is not POLICIES[arm]['native_attention']
                or draft.get('proposal_trace') is not True or value.get('commit_only_gdn') is not True
                or value['prompt_tokens'] != requests[0]['prompt_tokens']
                or value['emitted'] != requests[0]['emitted']
                or any(value.get(name) is not True for name in ('exact', 'state_exact', 'inactive_exact'))):
            raise ValueError('Matched exact target outputs/state and declared attention backend required')
        validate_route(value, arm)
        signature = proposal_signature(value)
        if arm in signatures and signatures[arm] != signature:
            raise ValueError('Each backend must reproduce its audited proposals and acceptance')
        signatures[arm] = signature
        if value['instrumented_timing']:
            blocks = [block for block in value['blocks'] if block['rows'] > 1]
            checks = draft.get('proposal_checks', [])
            if ([check.get('position') for check in checks] != [value['length'], *(block['position'] for block in blocks)]
                    or any(check.get('exact') is not True or check.get('tensors') != 6 for check in checks)):
                raise ValueError('Complete eager versus trace proposal audits required')
            if value.get('gdn_verify_checks') != [dict(position=block['position'], rows=block['rows'], unchanged=True)
                    for block in blocks]:
                raise ValueError('Every verifier must preserve GDN until publication')
    if signatures['control'] != signatures['scores']:
        raise ValueError('Score-layout change must preserve proposals and acceptance across arms')
    arms = {arm: summarize([value for value in requests if value['arm'] == arm]) for arm in POLICIES}
    return dict(arms=arms, order=[arm for arm, audit in SCHEDULE],
        committed_tg_change_percent=100 * (arms['scores']['committed_tg'] / arms['control']['committed_tg'] - 1),
        timing_boundary='Complete decode loops including copies, verification, publication, readback and stalls',
        held_out_coding_quality=False, serving_qualified=False)
