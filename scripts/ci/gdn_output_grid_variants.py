"""Matched complete requests changing only the GDN partial-output grid."""

from dspark_publication_variants import POLICIES as PUBLICATION, SCHEDULE, validate_route as publication_route
from dspark_score_layout_variants import summarize_variants as summarize_scores
from gdn_output_grid_gate import REPORT_SHA256


POLICIES = {'control': dict(PUBLICATION['publication']),
    'publication': dict(PUBLICATION['publication'], gdn_output_grid=True)}


def validate_route(value, arm):
    if arm not in POLICIES:
        raise ValueError('Unknown GDN output grid arm')
    publication_route(value, 'publication')
    evidence = value.get('gdn_output_grid')
    if arm == 'control':
        if evidence is not None:
            raise ValueError('Control must retain native output grid')
        return
    if (not isinstance(evidence, dict) or evidence.get('restored') is not True
            or evidence.get('admission', {}).get('report_sha256') != REPORT_SHA256):
        raise ValueError('Admitted and restored GDN output grid required')
    hits = evidence.get('hits')
    if not isinstance(hits, list) or len(hits) != 48 or any(type(count) is not int or count < 1 for count in hits):
        raise ValueError('All 48 GDN output projections must use the candidate')


def summarize_variants(requests):
    return summarize_scores(requests, policies=POLICIES, schedule=SCHEDULE,
        route=validate_route, candidate='publication')
