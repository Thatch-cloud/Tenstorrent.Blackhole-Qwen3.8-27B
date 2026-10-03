"""Validate the entire weight-free draft-tail matrix; never grant runtime speed acceptance."""

import hashlib
import json
from pathlib import Path


SOURCES = ('draft-tail-probe.py', 'frozen_draft_tail.py', 'dspark_native_cached_layer.py',
    'dspark_full_attention.py', 'dspark_projection.py', 'attention_batch.py')
RUNTIME = '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'
HARDWARE_SHA256 = 'cb31fac3a1767be5d6c278d4cbf758745c1b0382091a3615449d5ef75e208bb0'


def qualify_hardware(directory, report_path):
    from frozen_draft_tail_hardware import validate_hardware
    raw = Path(report_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != HARDWARE_SHA256:
        raise ValueError('Pinned full-size hardware report required')
    report = json.loads(raw)
    validate_hardware(report)
    for name in SOURCES:
        if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != report['sources'][name]:
            raise ValueError('Hardware-qualified source changed: ' + name)
    return dict(report_sha256=HARDWARE_SHA256, passed=True, serving_qualified=False)


def validate(report):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'simulator' or report.get('performance_qualified') is not False
            or report.get('model_integrated') is not False or set(report.get('sources', {})) != set(SOURCES)):
        raise ValueError('Closed exact simulator-only draft-tail evidence required')
    expected = set()
    for position in (64, 96):
        for proposals in (7, 15):
            for chip in (0, 1):
                for name in ('reference_eager', 'candidate_eager'):
                    expected.add((position, proposals, chip, name, 0))
                for name in ('reference_replay', 'candidate_replay', 'history_unchanged', 'queries_unchanged'):
                    for seed in (1, 0, 2):
                        expected.add((position, proposals, chip, name, seed))
    seen = set()
    for check in report.get('checks', []):
        if (check.get('exact') is not True or any(type(check.get(field)) is not int
                for field in ('position', 'proposals', 'chip', 'seed'))):
            raise ValueError('Explicit exact check identities required')
        key = tuple(check.get(field) for field in ('position', 'proposals', 'chip', 'name', 'seed'))
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicate draft-tail check')
        seen.add(key)
    if seen != expected:
        raise ValueError('All 112 eager, replay and unchanged-input checks required')
    return report


def qualify(directory, evidence, expected_sha256):
    directory, evidence = Path(directory), Path(evidence)
    if (type(expected_sha256) is not str or len(expected_sha256) != 64
            or any(character not in '0123456789abcdef' for character in expected_sha256)):
        raise ValueError('Explicit pinned report digest required')
    raw = (evidence / 'draft-tail.json').read_bytes()
    if (hashlib.sha256(raw).hexdigest() != expected_sha256
            or (evidence / 'draft-tail.exit-status').read_text().strip() != '0'
            or (evidence / 'simulator-runtime.txt').read_text().strip() != RUNTIME):
        raise ValueError('Pinned clean simulator report and runtime required')
    report = validate(json.loads(raw))
    for name in SOURCES:
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != report['sources'][name]:
            raise ValueError('Simulator-qualified source changed: ' + name)
    return report
