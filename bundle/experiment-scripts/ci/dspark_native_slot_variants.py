"""Matched full requests changing only target T16 native-state reads and paired publication."""

from pathlib import Path

from dspark_score_layout_variants import POLICIES as SCORES
from dspark_score_layout_variants import summarize_variants as summarize_scores
from dspark_score_layout_variants import validate_route as score_route
from gdn_native_slot_gate import qualify


POLICIES = {'control': dict(SCORES['scores']), 'direct': dict(SCORES['scores'], native_slot_gdn=True)}
SCHEDULE = (('control', True), ('direct', True), ('control', False), ('direct', False), ('direct', False), ('control', False))


def validate_route(value, arm):
    if arm not in POLICIES:
        raise ValueError('Unknown direct-state request arm')
    score_route(value, 'scores')
    evidence = value.get('native_slot_gdn')
    if arm == 'control':
        if evidence is not None:
            raise ValueError('Control must retain existing GDN snapshot path')
        return
    if (not isinstance(evidence, dict) or evidence.get('restored') is not True
            or evidence.get('qualification') != qualify(Path(__file__).parent) or evidence.get('rows') != 16
            or evidence.get('publication_prefixes') != list(range(17))):
        raise ValueError('Qualified restored T16 state and every paired publication prefix required')
    counts = evidence.get('calls_by_layer')
    if (not isinstance(counts, list) or len(counts) != 48
            or any(type(count) is not int or count < 1 for count in counts) or len(set(counts)) != 1):
        raise ValueError('Every target GDN layer must execute the same native-slot capture route')
    if not any(block['rows'] == 16 for block in value['blocks']):
        raise ValueError('Actual T16 verification required')


def summarize_variants(requests):
    return summarize_scores(requests, policies=POLICIES, schedule=SCHEDULE, route=validate_route, candidate='direct')
