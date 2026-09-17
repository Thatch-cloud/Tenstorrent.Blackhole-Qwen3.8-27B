"""Pinned simulator admission for the online-sum hardware experiment."""

import hashlib
import json
from pathlib import Path


RUN_ID = 34821692776
REPORT_SHA256 = 'da08e6905d97a078bf09d7f09ba378c41b6ba240b09533e5c05c100a15aa3842'


def qualify(directory, report_path):
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact completed sum-update simulator evidence required')
    report = json.loads(payload)
    expected = dict(passed=True, closed_cleanly=True, backend='simulator', context=128,
        capacity=384, stage='complete', candidate='sfpu-online-sum-update', performance_qualified=False)
    if any(report.get(name) != value for name, value in expected.items()):
        raise ValueError('Complete sum-update simulator checks required')
    if report['sources'] != report['sources_after'] or report['native_sources'] != report['native_sources_after']:
        raise ValueError('Stable simulator sources required')
    for dependencies in (report['sources'], report['candidate_sources'],
            report['factory_build']['builders'], report['factory_build']['ladder_builders']):
        for name, checksum in dependencies.items():
            if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != checksum:
                raise ValueError('Simulator-qualified sum dependency changed: ' + name)
    return dict(run_id=RUN_ID, simulator_qualified=True, hardware_qualified=False,
        performance_qualified=False, report_sha256=REPORT_SHA256)
