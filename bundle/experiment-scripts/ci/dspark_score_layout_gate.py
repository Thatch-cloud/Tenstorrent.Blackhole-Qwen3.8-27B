"""Validate complete score-layout evidence, not whole-drafter correctness or speed."""

import hashlib
import json
from pathlib import Path


SOURCES = (
    'dspark-score-layout-probe.py', 'dspark_score_layout.py',
    'dspark_score_layout_io.cpp', 'dspark_score_layout_compute.cpp',
    'attention_batch.py', 'gdn_multitoken_conv.py',
)


def validate(report, source_root, vocabulary, exit_status):
    if vocabulary not in (64, 248320) or exit_status.strip() != '0':
        raise ValueError('Supported vocabulary and successful process exit required')
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'simulator' or report.get('stage') != 'complete' or report.get('error')
            or report.get('vocabulary') != vocabulary or report.get('rows') != [0, 6, 14]):
        raise ValueError('Completed clean simulator score-layout report required')
    hashes = {name: hashlib.sha256((Path(source_root) / name).read_bytes()).hexdigest() for name in SOURCES}
    if report.get('sources') != hashes or report.get('sources_after') != hashes:
        raise ValueError('Evidence must match every current source before and after execution')
    workers = (1, 110) if vocabulary == 64 else (110,)
    expected_eager = [dict(pattern=pattern, step=step, workers=limit, chip=chip,
        exact=True, native_token_exact=True, cpu_exact=True)
        for pattern in (0, 1) for step in (0, 6, 14) for limit in workers for chip in (0, 1)]
    expected_replay = [dict(repetition=repetition, pattern=pattern, step=step,
        chip=chip, exact=True, output_poison_replaced=True)
        for repetition, pattern in enumerate((0, 1, 0)) for step in (0, 6, 14) for chip in (0, 1)]
    if report.get('eager_checks') != expected_eager or report.get('replay_checks') != expected_replay:
        raise ValueError('Exact complete two-chip eager and changed-input replay matrices required')
    return dict(vocabulary=vocabulary, eager_checks=len(expected_eager), replay_checks=len(expected_replay))


def qualify(source_root):
    root = Path(source_root)
    results = []
    for vocabulary, suffix in ((64, '-small'), (248320, '')):
        stem = root / f'dspark-score-layout{suffix}-simulator'
        results.append(validate(json.loads(stem.with_suffix('.json').read_text()), root,
            vocabulary, stem.with_suffix('.exit-status').read_text()))
    return dict(scope='score-layout only; complete feedback chain and hardware remain unqualified', results=results)
