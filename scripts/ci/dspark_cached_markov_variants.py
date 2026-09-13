"""Matched combined requests changing only request-owned Markov bias caching."""

from dspark_cached_markov_gate import RUN, REPORTS
from dspark_score_layout_variants import summarize_variants as summarize_scores
from gdn_shared_qk_variants import POLICIES as SHARED, SCHEDULE, validate_route as shared_route


POLICIES = {'control': dict(SHARED['publication']),
    'publication': dict(SHARED['publication'], bias_cache=True)}


def validate_route(value, arm):
    if arm not in POLICIES:
        raise ValueError('Unknown bias cache comparison arm')
    shared_route(value, 'publication')
    evidence = value.get('score_layout', {}).get('bias_cache')
    if arm == 'control':
        if evidence is not None:
            raise ValueError('Control must not install the bias cache')
        return
    if (not isinstance(evidence, dict) or evidence.get('released') is not True
            or evidence.get('reset_epoch') != 2
            or evidence.get('admission', {}).get('simulator_run') != RUN
            or evidence.get('admission', {}).get('reports') != REPORTS):
        raise ValueError('Qualified cold-reset and released request bias cache required')


def summarize_variants(requests):
    return summarize_scores(requests, policies=POLICIES, schedule=SCHEDULE,
        route=validate_route, candidate='publication')
