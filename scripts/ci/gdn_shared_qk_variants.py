"""Matched complete requests including shared normalization preparation costs."""

from dspark_publication_variants import POLICIES as PUBLICATION, SCHEDULE, validate_route as publication_route
from dspark_score_layout_variants import summarize_variants as summarize_scores
from gdn_shared_qk_gate import REPORT_SHA256


POLICIES = {'control': dict(PUBLICATION['publication']),
    'publication': dict(PUBLICATION['publication'], gdn_shared_qk=True)}


def validate_route(value, arm):
    if arm not in POLICIES:
        raise ValueError('Unknown shared-Q/K arm')
    publication_route(value, 'publication')
    evidence = value.get('gdn_shared_qk')
    if arm == 'control':
        if evidence is not None:
            raise ValueError('Native control must not install shared Q/K')
        return
    if (not isinstance(evidence, dict) or evidence.get('restored') is not True
            or evidence.get('released') is not True
            or evidence.get('admission', {}).get('report_sha256') != REPORT_SHA256):
        raise ValueError('Qualified, restored and released shared-Q/K scope required')
    loads = evidence.get('loads')
    if (not isinstance(loads, list) or len(loads) < 48 or len(loads) % 48
            or any(item != dict(rows=16, programs=3, retained_preparation_buffers=2) for item in loads)):
        raise ValueError('Complete T16 three-stage pipeline construction required')


def summarize_variants(requests):
    return summarize_scores(requests, policies=POLICIES, schedule=SCHEDULE,
        route=validate_route, candidate='publication')
