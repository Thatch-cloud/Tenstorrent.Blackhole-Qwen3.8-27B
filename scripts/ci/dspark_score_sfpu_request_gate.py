"""Numerical admission for SFPU request experiments, not serving acceptance."""

import hashlib
import json
from pathlib import Path


RUN_ID = 34817498912
REPORT_SHA256 = '04d6333660a92e9eeb717e39915f0fb5f6ac954b6a855818e303890c472858e9'


def qualify(directory, report_path):
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact completed 64K SFPU hardware report required')
    report = json.loads(payload)
    expected = dict(passed=True, closed_cleanly=True, backend='hardware', stage='complete',
        candidate='sfpu-score-centering', capacity=66560, positions=[65536, 66545],
        numerical_tolerances=dict(rtol=.01, atol=.01), performance_qualified=False)
    if any(report.get(name) != value for name, value in expected.items()):
        raise ValueError('Complete original-tolerance 64K numerical fixture required')
    counts = dict(eager_checks=4, replay_checks=4, input_checks=48,
        layout_checks=16, fixture_controls=8, stale_controls=2)
    for name, count in counts.items():
        flag = 'exact' if name == 'input_checks' else 'detected' if name.endswith('controls') else 'passed'
        records = report.get(name, [])
        if len(records) != count or any(record.get(flag) is not True for record in records):
            raise ValueError('Complete passing 64K checks required: ' + name)
    if any(record.get('replay_exact') is not True for record in report['replay_checks']):
        raise ValueError('Exact replay required')
    for before, after in (('sources', 'sources_after'), ('native_sources', 'native_sources_after')):
        if report[before] != report[after]:
            raise ValueError('Stable numerical-gate source evidence required')
    for dependencies in (report['sources'], report['candidate_sources'],
            report['factory_build']['factory_inputs']['builders']):
        for name, checksum in dependencies.items():
            if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != checksum:
                raise ValueError('Numerically qualified dependency changed: ' + name)
    return dict(run_id=RUN_ID, report_sha256=REPORT_SHA256, numerical_qualified=True,
        full_request_qualified=False, performance_qualified=False, serving_qualified=False)
