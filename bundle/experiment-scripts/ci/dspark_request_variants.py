"""Matched eager, captured-proposal and captured/commit-only DSpark request screens."""

POLICIES = {
    'eager': dict(proposal_trace=False, commit_only_gdn=False),
    'trace': dict(proposal_trace=True, commit_only_gdn=False),
    'trace_commit': dict(proposal_trace=True, commit_only_gdn=True),
}
SCHEDULE = tuple((arm, True) for arm in POLICIES) + tuple(
    (arm, False) for arm in (*POLICIES, *reversed(POLICIES)))


def proposal_signature(request):
    return [(block['position'], block['rows'], tuple(block['input_tokens']), block['accepted'], block['committed'])
        for block in request['blocks']]


def summarize_variants(requests):
    from dspark_request_experiment import summarize

    if [(value.get('arm'), value.get('instrumented_timing')) for value in requests] != list(SCHEDULE):
        raise ValueError('Three audited arms followed by complete A/B/C/C/B/A timed requests required')
    for value in requests:
        expected = POLICIES[value['arm']]
        if (value.get('dspark', {}).get('proposal_trace') is not expected['proposal_trace']
                or value.get('commit_only_gdn') is not expected['commit_only_gdn']
                or value['prompt_tokens'] != requests[0]['prompt_tokens'] or value['emitted'] != requests[0]['emitted']):
            raise ValueError('Matched context, exact target output and declared single-change arms required')
        if value['instrumented_timing']:
            blocks = value['blocks']
            if expected['proposal_trace']:
                checks = value['dspark'].get('proposal_checks', [])
                positions = [value['length'], *(block['position'] for block in blocks if block['rows'] > 1)]
                if ([check.get('position') for check in checks] != positions
                        or any(check.get('exact') is not True or check.get('tensors') != 6 for check in checks)):
                    raise ValueError('Every complete proposal replay must match its fixed-layout eager outputs on both chips')
            if expected['commit_only_gdn']:
                checks = value.get('gdn_verify_checks', [])
                expected_checks = [dict(position=block['position'], rows=block['rows'], unchanged=True)
                    for block in blocks if block['rows'] > 1]
                if checks != expected_checks:
                    raise ValueError('Every multirow verifier must preserve native GDN state before publication')
    signatures = {arm: proposal_signature(next(value for value in requests if value['arm'] == arm)) for arm in POLICIES}
    if (any(proposal_signature(value) != signatures[value['arm']] for value in requests)
            or signatures['trace'] != signatures['trace_commit']):
        raise ValueError('Each arm must reproduce its audited proposals; commit-only GDN must not change drafting or acceptance')
    arms = {arm: summarize([value for value in requests if value['arm'] == arm]) for arm in POLICIES}
    comparisons = {}
    for control, candidate in (('eager', 'trace'), ('trace', 'trace_commit'), ('eager', 'trace_commit')):
        comparisons[candidate + '_versus_' + control] = dict(
            committed_tg_change_percent=100 * (arms[candidate]['committed_tg'] / arms[control]['committed_tg'] - 1),
            setup_inclusive_change_percent=100 * (arms[candidate]['mean_setup_inclusive_ms']
                / arms[control]['mean_setup_inclusive_ms'] - 1))
    return dict(arms=arms, comparisons=comparisons, order=[arm for arm, audit in SCHEDULE],
        timing_boundary='Complete decode loops including proposal input copies, verification, publication and every stall',
        serving_qualified=False, held_out_coding_quality=False)
