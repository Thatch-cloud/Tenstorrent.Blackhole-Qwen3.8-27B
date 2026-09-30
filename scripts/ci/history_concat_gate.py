"""Admit a complete pinned two-chip concat-lifetime simulator matrix."""

import hashlib
import json
from pathlib import Path


SOURCES = ('history-concat-probe.py', 'history_concat_lifetime.py', 'dspark_history.py', 'gdn_multitoken_conv.py')
RUNTIME = '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'


def validate(report):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'simulator' or report.get('error')
            or report.get('performance_qualified') is not False or report.get('model_integrated') is not False
            or set(report.get('sources', {})) != set(SOURCES)
            or report['sources'] != report.get('sources_after')):
        raise ValueError('Complete unchanged-source concat simulator qualification required')
    expected = {(arm, count, seed, tensor, chip) for arm in ('original', 'candidate')
        for count in (65, 128) for seed in (0, 1) for tensor in ('output', 'borrowed') for chip in (0, 1)}
    seen = set()
    for check in report.get('checks', []):
        identity = tuple(check.get(key) for key in ('arm', 'count', 'seed', 'tensor', 'chip'))
        if (check.get('exact') is not True or identity not in expected or identity in seen
                or any(type(check.get(key)) is not int for key in ('count', 'seed', 'chip'))):
            raise ValueError('Unique exact original/candidate results on both chips required')
        seen.add(identity)
    if seen != expected:
        raise ValueError('All 32 output and borrowed-storage checks required')
    return report


def qualify(directory, evidence, expected_sha256):
    directory, evidence = Path(directory), Path(evidence)
    raw = (evidence / 'history-concat.json').read_bytes()
    if (hashlib.sha256(raw).hexdigest() != expected_sha256
            or (evidence / 'history-concat.exit-status').read_text().strip() != '0'
            or (evidence / 'simulator-runtime.txt').read_text().strip() != RUNTIME):
        raise ValueError('Pinned clean concat report and simulator runtime required')
    report = validate(json.loads(raw))
    for name, checksum in report['sources'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Qualified concat source changed: ' + name)
    return report
