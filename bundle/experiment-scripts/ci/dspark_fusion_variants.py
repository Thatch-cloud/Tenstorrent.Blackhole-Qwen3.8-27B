"""Matched combined requests changing only the target T16 gate/up projection."""

from dspark_score_layout_variants import POLICIES as SCORES
from dspark_score_layout_variants import summarize_variants as summarize_scores
from dspark_score_layout_variants import validate_route as score_route
from fused_t16_admission import REPORT_SHA256


POLICIES = {'control': dict(SCORES['scores']), 'fusion': dict(SCORES['scores'], fused_t16_mlp=True)}
SCHEDULE = (('control', True), ('fusion', True), ('control', False), ('fusion', False), ('fusion', False), ('control', False))


def validate_route(value, arm):
    if arm not in POLICIES:
        raise ValueError('Unknown fusion request arm')
    score_route(value, 'scores')
    audit = value.get('fused_t16_mlp')
    if arm == 'control':
        if audit is not None:
            raise ValueError('Native MLP control cannot contain fusion')
        return
    if (not isinstance(audit, dict) or audit.get('restored') is not True
            or audit.get('native_bindings_unchanged') is not True
            or audit.get('passed_simulator') != REPORT_SHA256
            or audit.get('rows') != 16 or audit.get('layers') != 64
            or audit.get('extra_weight_allocations') != 0):
        raise ValueError('Qualified restored target-only fusion required')
    hits = audit.get('hits')
    if (not isinstance(hits, list) or len(hits) != 64
            or any(type(count) is not int or count < 1 for count in hits) or len(set(hits)) != 1):
        raise ValueError('All 64 layers must execute the same T16 fusion route')
    weights = audit.get('weight_audit', {})
    checks = weights.get('checks', [])
    if weights.get('passed') is not True or len(checks) != 256:
        raise ValueError('Complete target packed-weight audit required')
    observed = {(check['layer'], check['offset'], check['chip']) for check in checks
        if check.get('exact') is True and check.get('pages') == 43520}
    expected = {(layer, offset, chip) for layer in range(64) for offset in (0, 1) for chip in (0, 1)}
    if observed != expected or not any(block['rows'] == 16 for block in value['blocks']):
        raise ValueError('Exact complete weight matrix and actual T16 verification required')


def summarize_variants(requests):
    return summarize_scores(requests, policies=POLICIES, schedule=SCHEDULE, route=validate_route, candidate='fusion')
