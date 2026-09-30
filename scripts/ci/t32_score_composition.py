"""Source-bound component composition audit; does not enable T32 hardware execution."""

import argparse
import hashlib
import json
from pathlib import Path

from dspark_score_layout_gate import qualify as qualify_layout
from t32_proposal_gate import REPORT_SHA256 as PROPOSAL_SHA256, source_closure


SCORE_SHA256 = '499f7e7664cf111ff8fa40516eb49818ce4a1f5bbdc122e21e6d5a47e07c005f'
RUNTIME = '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'
DELTAS = {
    'attention_mask_replay.py': (
        '7841495a15ee090aae7b78edc118ba0de2967bb3ad72b843d53251a091435749',
        '3e431742e35a2b94b4a02a60fa334a93a44a471eaefcacd25e52fbafdf03361f'),
    'dspark_t32_prepared.py': (
        '4c0c376cb2043f9ec4b13de88faeb754ccf522705857e6d28baef987df4e5b12',
        'c4e749ccde202f0f9cc157fd4c217e5d9a2c080f6313eecaeaa2815b2793853c'),
    'dspark_t32_score_layout.py': (None,
        '61056dbd40c69c6b1ca8c398b02fac43cfb25cbf254203ffbb2a86c343f407a2'),
}


def load(path, expected):
    path = Path(path)
    raw = path.read_bytes()
    if (hashlib.sha256(raw).hexdigest() != expected
            or path.with_suffix('.exit-status').read_text().strip() != '0'
            or (path.parent / 'simulator-runtime.txt').read_text().strip() != RUNTIME):
        raise ValueError('Pinned successful simulator artifact required')
    report = json.loads(raw)
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != (None if expected == PROPOSAL_SHA256 else 'simulator')
            or report.get('stage') != 'complete'
            or report.get('sources') != report.get('sources_after')):
        raise ValueError('Complete unchanged-source simulator report required')
    return report


def validate_dependencies(previous, current):
    changed = {name for name, checksum in current.items() if previous.get(name) != checksum}
    if changed != set(DELTAS):
        raise ValueError('Exactly the reviewed three-file T32 source delta required')
    for name, (before, after) in DELTAS.items():
        if previous.get(name) != before or current.get(name) != after:
            raise ValueError('Unreviewed T32 dependency delta: ' + name)


def validate_score(report):
    if (report.get('vocabulary') != 64 or report.get('proposals') != 31
            or report.get('score_layout') != 'fused' or report.get('target_integrated') is not False
            or report.get('native_arithmetic_reference') is not True
            or report.get('native_sources') != report.get('native_sources_after')):
        raise ValueError('Native-referenced 31-query fused feedback evidence required')
    eager = report.get('eager_checks', [])
    replay = report.get('replay_checks', [])
    if (len(eager) != 186 or len(replay) != 248
            or {(entry.get('pattern'), entry.get('step'), entry.get('chip')) for entry in eager}
                != {(pattern, step, chip) for pattern in range(3) for step in range(31) for chip in (0, 1)}
            or {(entry.get('repetition'), entry.get('step'), entry.get('chip')) for entry in replay}
                != {(repetition, step, chip) for repetition in range(4) for step in range(31) for chip in (0, 1)}
            or any(entry.get('token_exact') is not True or entry.get('full_vocabulary_exact') is not True for entry in eager)
            or any(entry.get('token_and_scores_exact') is not True or entry.get('bindings_stable') is not True for entry in replay)):
        raise ValueError('Every exact query on both chips and changed-input replay required')
    for field, count, flag in (('input_checks', 28, 'exact'), ('weight_checks', 8, 'exact'),
            ('stale_controls', 2, 'missing_update_detected')):
        records = report.get(field, [])
        if len(records) != count or any(record.get(flag) is not True for record in records):
            raise ValueError('Complete feedback controls required: ' + field)


def audit(directory, proposal_path, score_path):
    directory = Path(directory)
    proposal, score = load(proposal_path, PROPOSAL_SHA256), load(score_path, SCORE_SHA256)
    checks = proposal.get('checks', [])
    if (proposal.get('context') != 4096 or proposal.get('capacity') != 4384
            or proposal.get('proposals') != 31 or proposal.get('learned_layers') != 5
            or [check.get('anchor') for check in checks] != [20, 10]
            or any(check.get('exact') is not True or len(check.get('tokens', [])) != 31 for check in checks)
            or checks[0]['tokens'] == checks[1]['tokens']):
        raise ValueError('Complete retained learned proposal replay required')
    validate_score(score)
    current = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in source_closure(directory)}
    if len(current) != 68:
        raise ValueError('Complete reviewed 68-file proposal dependency closure required')
    validate_dependencies(proposal['sources'], current)
    for name, checksum in score['sources'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Fused feedback source changed: ' + name)
    layout = qualify_layout(directory)
    return dict(component_composition_audited=True, proposal_report_sha256=PROPOSAL_SHA256,
        score_report_sha256=SCORE_SHA256, score_layout=layout, dependency_count=len(current),
        sources=current, source_delta={name: dict(before=before, after=after) for name, (before, after) in DELTAS.items()},
        requires_disabled_context_ladder_sim=True, full_proposal_candidate_qualified=False,
        hardware_qualified=False, performance_qualified=False, serving_qualified=False,
        next_gate='Explicit full-model hardware correctness experiment; retain native-score control and complete target-state audit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--score', type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.directory, arguments.proposal, arguments.score), indent=2))
