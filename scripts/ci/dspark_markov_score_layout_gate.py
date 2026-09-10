"""Independent complete feedback matrices; synthetic evidence does not qualify coding quality."""

import hashlib
import json
from pathlib import Path

from dspark_score_layout_gate import qualify as qualify_layout


SOURCES = ('dspark-markov-score-layout-probe.py', 'dspark_markov_device.py',
    'dspark_markov_score_layout.py', 'dspark_score_layout.py', 'dspark_score_layout_io.cpp',
    'dspark_score_layout_compute.cpp', 'dspark_score_layout_gate.py',
    'attention_batch.py', 'feature_projection.py', 'gdn_multitoken_conv.py')


def validate(report, source_root, vocabulary, exit_status):
    root = Path(source_root)
    if vocabulary not in (64, 248320) or exit_status.strip() != '0':
        raise ValueError('Successful process exit and supported vocabulary required')
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('stage') != 'complete' or report.get('backend') != 'simulator'
            or report.get('error') or report.get('vocabulary') != vocabulary or report.get('proposals') != 15):
        raise ValueError('Complete fifteen-step simulator feedback evidence required')
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}
    if report.get('sources') != hashes or report.get('sources_after') != hashes:
        raise ValueError('Both source snapshots must match every current source')
    if report.get('score_layout_gate') != qualify_layout(root):
        raise ValueError('Current independently qualified score-layout evidence required')
    eager = [dict(pattern=pattern, repetition=None, step=step, chip=chip, token_exact=True, scores_exact=True)
        for pattern in range(3) for step in range(15) for chip in range(2)]
    replay = [dict(pattern=pattern, repetition=repetition, step=step, chip=chip, token_exact=True, scores_exact=True)
        for repetition, pattern in enumerate((0, 1, 2, 0)) for step in range(15) for chip in range(2)]
    input_stages = [(pattern, 'eager') for pattern in range(3)] + [
        (pattern, f'replay_{repetition}') for repetition, pattern in enumerate((0, 1, 2, 0))]
    inputs = [dict(stage=stage, pattern=pattern, operand=operand, chip=chip, exact=True)
        for pattern, stage in input_stages for operand in range(2) for chip in range(2)]
    weights = [dict(stage=stage, operand=operand, chip=chip, exact=True)
        for stage in ('before', 'after') for operand in range(2) for chip in range(2)]
    for name, expected in (('eager_checks', eager), ('replay_checks', replay),
            ('input_checks', inputs), ('weight_checks', weights)):
        if report.get(name) != expected:
            raise ValueError(f'Complete exact {name} matrix required')
    return dict(vocabulary=vocabulary, proposals=15, eager_checks=len(eager), replay_checks=len(replay),
        input_checks=len(inputs), weight_checks=len(weights))


def qualify(source_root):
    root = Path(source_root)
    results = []
    for vocabulary, suffix in ((64, '-small'), (248320, '')):
        stem = root / f'dspark-markov-score-layout{suffix}-simulator'
        results.append(validate(json.loads(stem.with_suffix('.json').read_text()), root,
            vocabulary, stem.with_suffix('.exit-status').read_text()))
    return dict(scope='Synthetic complete feedback only; learned weights and hardware require separate audits',
        results=results)
