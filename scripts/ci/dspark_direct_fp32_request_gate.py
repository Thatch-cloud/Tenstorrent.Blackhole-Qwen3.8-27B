"""64K direct FP32 staging numerical evidence required before combined requests."""

import hashlib
import json
from pathlib import Path


RUN_ID = 34834720985
REPORT_SHA256 = '32979ee94bc643fa939f1d1c65297a5114c4d6a57ac77140af891e44af716c89'


def qualify(directory, report_path):
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact completed direct staging hardware numerical evidence required')
    report = json.loads(payload)
    expected = dict(passed=True, closed_cleanly=True, backend='hardware', stage='complete',
        candidate='direct-fp32-staging', capacity=66560, positions=[65536, 66545],
        numerical_tolerances=dict(rtol=.01, atol=.01), performance_qualified=False)
    if any(report.get(name) != value for name, value in expected.items()):
        raise ValueError('Original-tolerance full 64K numerical gate required')
    counts = dict(eager_checks=4, replay_checks=4, input_checks=48,
        layout_checks=16, fixture_controls=8, stale_controls=2)
    for name, count in counts.items():
        flag = 'exact' if name == 'input_checks' else 'detected' if name.endswith('controls') else 'passed'
        if len(report.get(name, [])) != count or any(value.get(flag) is not True for value in report[name]):
            raise ValueError('Missing or failed direct staging checks: ' + name)
    if any(value.get('replay_exact') is not True for value in report['replay_checks']):
        raise ValueError('Exact repeated execution required')
    if report['sources'] != report['sources_after'] or report['native_sources'] != report['native_sources_after']:
        raise ValueError('Stable numerical-gate sources required')
    for dependencies in (report['sources'], report['candidate_sources'],
            report['factory_build']['factory_inputs']['builders']):
        for name, checksum in dependencies.items():
            if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != checksum:
                raise ValueError('Numerically qualified mask dependency changed: ' + name)
    return dict(run_id=RUN_ID, report_sha256=REPORT_SHA256, numerical_qualified=True,
        full_request_qualified=False, performance_qualified=False)
