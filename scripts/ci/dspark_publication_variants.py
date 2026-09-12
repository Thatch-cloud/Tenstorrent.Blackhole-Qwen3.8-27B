"""Matched combined fusion requests changing only captured history projection."""

from dspark_fusion_variants import POLICIES as FUSION
from dspark_fusion_variants import validate_route as fusion_route
from dspark_score_layout_variants import summarize_variants as summarize_scores


POLICIES = {'control': dict(FUSION['fusion']),
    'publication': dict(FUSION['fusion'], captured_publication=True)}
SCHEDULE = (('control', True), ('publication', True), ('control', False),
    ('publication', False), ('publication', False), ('control', False))


def validate_route(value, arm):
    if arm not in POLICIES:
        raise ValueError('Unknown captured publication arm')
    fusion_route(value, 'fusion')
    evidence = value.get('captured_publication')
    if arm == 'control':
        if evidence is not None:
            raise ValueError('Control must use eager history publication')
        return
    if not isinstance(evidence, dict) or evidence.get('enabled') is not True:
        raise ValueError('Candidate must execute captured history publication')
    audit = value.get('instrumented_timing')
    if type(audit) is not bool:
        raise ValueError('Explicit request timing policy required')
    checks = evidence.get('checks')
    count = 1 + len(value['blocks']) if audit else 0
    if not isinstance(checks, list) or len(checks) != count or any(
            check.get('exact') is not True or check.get('tensors') != 20 for check in checks):
        raise ValueError('Every audited publication must match all twenty eager K/V shards')


def summarize_variants(requests):
    return summarize_scores(requests, policies=POLICIES, schedule=SCHEDULE,
        route=validate_route, candidate='publication')
