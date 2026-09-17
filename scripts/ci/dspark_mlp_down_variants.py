"""Matched folded-attention requests changing only the target T16 down projection."""

import math

from dspark_request_variants import proposal_signature


POLICIES = {
    'control': dict(proposal_trace=True, commit_only_gdn=True, native_attention=True, target_attention_t16=True),
    'down': dict(proposal_trace=True, commit_only_gdn=True, native_attention=True, target_attention_t16=True),
}
SCHEDULE = (('control', True), ('down', True), ('control', False),
    ('down', False), ('down', False), ('control', False))


def validate_route(value, arm):
    from dspark_target_attention_variants import validate_route as attention_route

    attention_route(value, 'parallel')
    audit = value.get('down_mlp')
    equal = value.get('down_mlp_equal_footprint', False)
    if type(equal) is not bool:
        raise ValueError('Explicit equal-footprint diagnostic selection required')
    if arm == 'control' and not equal:
        if audit is not None:
            raise ValueError('Native MLP control cannot contain candidate execution')
        return
    if arm not in ('control', 'down') or not isinstance(audit, dict):
        raise ValueError('Explicit down-only candidate audit required')
    if (audit.get('restored') is not True or audit.get('native_bindings_unchanged') is not True
            or audit.get('layers') != 64 or audit.get('rows') != 16
            or not isinstance(audit.get('hits'), list) or len(audit['hits']) != 64
            or any(type(count) is not int or (count < 1 if arm == 'down' else count != 0) for count in audit['hits'])):
        raise ValueError('Every target layer must execute and restore the qualified T16 MLP')
    setup = audit.get('setup_ms')
    if type(setup) not in (int, float) or not math.isfinite(setup) or setup <= 0:
        raise ValueError('Actual candidate weight preparation cost required')
    if equal:
        bindings = audit.get('prepared_bindings')
        if (audit.get('enabled') is not (arm == 'down') or audit.get('prepared_bindings_unchanged') is not True
                or not isinstance(bindings, list) or len(bindings) != 64
                or any(not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(type(address) is not int or address < 0 for address in pair) for pair in bindings)):
            raise ValueError('Both arms require stable prepared weight bindings and explicit execution routes')


def summarize_variants(requests):
    from dspark_request_experiment import summarize

    if [(value.get('arm'), value.get('instrumented_timing')) for value in requests] != list(SCHEDULE):
        raise ValueError('Two audits followed by complete A/B/B/A timed requests required')
    footprint = requests[0].get('down_mlp_equal_footprint', False)
    if any(value.get('down_mlp_equal_footprint', False) is not footprint for value in requests):
        raise ValueError('One memory-footprint policy required throughout the comparison')
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
    if signatures['control'] != signatures['down']:
        raise ValueError('MLP change must preserve proposals and acceptance across arms')
    arms = {arm: summarize([value for value in requests if value['arm'] == arm]) for arm in POLICIES}
    result = dict(arms=arms, order=[arm for arm, audit in SCHEDULE],
        committed_tg_change_percent=100 * (arms['down']['committed_tg'] / arms['control']['committed_tg'] - 1),
        timing_boundary='Complete decode loops including copies, verification, publication, readback and stalls',
        held_out_coding_quality=False, serving_qualified=False)
    if footprint:
        bindings = [value['down_mlp']['prepared_bindings'] for value in requests if not value['instrumented_timing']]
        result.update(equal_resident_weight_footprint=True,
            timed_weight_addresses_match=all(value == bindings[0] for value in bindings),
            scope='Memory-footprint diagnostic; retains extra weights in both arms, not native-footprint promotion')
    return result
