"""Pinned fixed-history native attention prerequisite, not full-model qualification."""

import json
from pathlib import Path

from dspark_hardware_gate import digest


REPORT = 'dspark-native-fixed-attention-simulator.json'
SHA256 = '7948ee1939a145a4452e1faa3ea5da6aeb20be1e6aa10a48a58c25b4ce3b809a'
CHECKS = {'eager_checks': (4, 'passed'), 'replay_checks': (4, 'passed'),
    'input_checks': (48, 'exact'), 'layout_checks': (16, 'passed'),
    'fixture_controls': (8, 'detected'), 'stale_controls': (2, 'detected')}


def qualify(directory):
    directory = Path(directory)
    path = directory / REPORT
    if digest(path) != SHA256 or path.with_suffix('.exit-status').read_text().strip() != '0':
        raise ValueError('Pinned successful native fixed-history simulator report required')
    report = json.loads(path.read_text())
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'simulator' or report.get('stage') != 'complete'
            or report.get('positions') != [4096, 4109] or report.get('capacity') != 4384
            or report.get('proposal_rows') != 15 or report.get('key_chunk_size') != 64
            or report.get('numerical_tolerances') != dict(rtol=.01, atol=.01)):
        raise ValueError('Complete fixed-history numerical geometry required')
    for name, (count, field) in CHECKS.items():
        records = report.get(name, [])
        if len(records) != count or any(record.get(field) is not True for record in records):
            raise ValueError('Incomplete native fixed-history checks: ' + name)
    if any(record.get('replay_exact') is not True for record in report['replay_checks']):
        raise ValueError('Bit-exact changing-input native replay required')
    sources = report.get('sources', {})
    if not sources or sources != report.get('sources_after'):
        raise ValueError('Unchanged complete probe sources required')
    for name, expected in sources.items():
        if digest(directory / name) != expected:
            raise ValueError('Native fixed-history source changed: ' + name)
    if not report.get('native_sources') or report['native_sources'] != report.get('native_sources_after'):
        raise ValueError('Unchanged owned native simulator sources required')
    return {REPORT: SHA256}
