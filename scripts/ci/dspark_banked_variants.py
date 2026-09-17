"""Matched complete requests changing only captured history bank binding."""

from pathlib import Path

from dspark_score_layout_variants import POLICIES as SCORE_POLICIES
from dspark_score_layout_variants import summarize_variants as summarize_scores
from dspark_score_layout_variants import validate_route as score_route


EVIDENCE = Path(__file__).with_name('dspark-banked-trace-simulator.json')
POLICIES = {
    'control': dict(SCORE_POLICIES['scores']),
    'banked': dict(SCORE_POLICIES['scores'], banked_proposal=True, banked_proposal_evidence=EVIDENCE),
}
SCHEDULE = (('control', True), ('banked', True), ('control', False),
    ('banked', False), ('banked', False), ('control', False))


def validate_route(value, arm):
    from dspark_banked_gate import qualify

    if arm not in POLICIES:
        raise ValueError('Unknown banked request arm')
    score_route(value, 'scores')
    bank = value.get('dspark', {}).get('banked_proposal')
    if arm == 'control':
        if bank is not None:
            raise ValueError('Control must retain copied proposal history')
        return
    if not isinstance(bank, dict) or bank.get('evidence') != qualify(EVIDENCE, Path(__file__).parent):
        raise ValueError('Current simulator evidence required for banked request')
    counts = bank.get('replay_counts')
    blocks = [block for block in value['blocks'] if block['rows'] > 1]
    if (not isinstance(counts, list) or len(counts) != 2
            or any(type(count) is not int or count < 1 for count in counts)
            or sum(counts) != 1 + len(blocks)):
        raise ValueError('Both banks must replay and account for warmup plus every proposal')


def summarize_variants(requests):
    return summarize_scores(requests, policies=POLICIES, schedule=SCHEDULE,
        route=validate_route, candidate='banked')
