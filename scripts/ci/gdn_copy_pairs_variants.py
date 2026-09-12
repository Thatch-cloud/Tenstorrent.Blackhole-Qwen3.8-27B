"""Matched combined requests changing only T16 recurrence copy handoffs."""

from dspark_publication_variants import POLICIES as PUBLICATION, SCHEDULE, validate_route as publication_route
from dspark_score_layout_variants import summarize_variants as summarize_scores
from gdn_copy_pairs_gate import REPORT_SHA256


POLICIES = {'control': dict(PUBLICATION['publication']),
    'publication': dict(PUBLICATION['publication'], gdn_copy_pairs=True)}


def validate_route(value, arm):
    if arm not in POLICIES:
        raise ValueError('Unknown copy-pair arm')
    publication_route(value, 'publication')
    evidence = value.get('gdn_copy_pairs')
    if arm == 'control':
        if evidence is not None:
            raise ValueError('Native control must not install the copy-pair scope')
        return
    if (not isinstance(evidence, dict) or evidence.get('restored') is not True
            or evidence.get('admission', {}).get('report_sha256') != REPORT_SHA256):
        raise ValueError('Qualified and restored copy-pair scope required')
    loads = evidence.get('loads')
    if (not isinstance(loads, list) or len(loads) < 48 or len(loads) % 48
            or any(item.get('rows') != 16 or item.get('control_sha256') == item.get('candidate_sha256')
                   for item in loads)):
        raise ValueError('Complete T16 recurrence program construction required')


def summarize_variants(requests):
    return summarize_scores(requests, policies=POLICIES, schedule=SCHEDULE,
        route=validate_route, candidate='publication')
