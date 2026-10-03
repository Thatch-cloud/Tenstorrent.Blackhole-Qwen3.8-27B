"""Simulator-first admission for hardware correctness, not permission to claim speed."""

import json
import hashlib
from pathlib import Path

from dspark_markov_score_layout_gate import validate
from dspark_score_layout_gate import qualify as qualify_layout


def qualify(source_root):
    root = Path(source_root)
    stem = root / 'dspark-markov-score-layout-small-simulator'
    chain = validate(json.loads(stem.with_suffix('.json').read_text()), root, 64,
        stem.with_suffix('.exit-status').read_text())
    return dict(score_layout=qualify_layout(root), small_feedback=chain,
        scope='Admits hardware correctness only; full learned feedback and request audits remain mandatory',
        hardware_correctness_required=True, timing_qualified=False)


def validate_hardware(report, source_root):
    if (not isinstance(report, dict) or report.get('passed') is not True
            or report.get('released_cleanly') is not True or report.get('vocabulary') != 248320
            or report.get('proposals') != 15 or report.get('admission') != qualify(source_root)):
        raise ValueError('Complete learned-weight hardware correctness audit required')
    eager = [dict(pattern=pattern, repetition=None, step=step, chip=chip, token_exact=True, scores_exact=True)
        for pattern in range(2) for step in range(15) for chip in range(2)]
    replay = [dict(pattern=pattern, repetition=repetition, step=step, chip=chip, token_exact=True, scores_exact=True)
        for repetition, pattern in enumerate((0, 1, 0)) for step in range(15) for chip in range(2)]
    stages = [(pattern, 'eager') for pattern in range(2)] + [
        (pattern, f'replay_{repetition}') for repetition, pattern in enumerate((0, 1, 0))]
    inputs = [dict(pattern=pattern, stage=stage, operand=operand, chip=chip, exact=True)
        for pattern, stage in stages for operand in range(2) for chip in range(2)]
    if any(report.get(name) != expected for name, expected in (
            ('eager_checks', eager), ('replay_checks', replay), ('input_checks', inputs))):
        raise ValueError('Every learned hardware comparison must match exactly')
    hashes = report.get('weight_hashes_before')
    if (not isinstance(hashes, list) or len(hashes) != 2 or any(not isinstance(pair, list)
            or len(pair) != 2 or any(not isinstance(value, str) or len(value) != 64
                or any(character not in '0123456789abcdef' for character in value) for value in pair) for pair in hashes)
            or report.get('weight_hashes_after') != hashes):
        raise ValueError('Both complete learned weight replicas must remain unchanged')
    return hashlib.sha256(json.dumps(report, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
