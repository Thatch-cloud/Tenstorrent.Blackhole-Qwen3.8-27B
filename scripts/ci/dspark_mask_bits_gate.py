"""Pinned simulator admission for the mask fast-path hardware experiment."""

import hashlib
import json
from pathlib import Path


RUN_ID = 34825080088
REPORT_SHA256 = 'f8b035bf4fa7f95173695f6cb7b221f0274dcb3bb3e5cefd39f2c928e456229b'


def qualify(directory, report_path):
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact completed mask fast-path simulator evidence required')
    report = json.loads(payload)
    expected = dict(passed=True, closed_cleanly=True, backend='simulator', context=128,
        capacity=384, stage='complete', candidate='bitwise-negative-infinity-mask', performance_qualified=False)
    if any(report.get(name) != value for name, value in expected.items()):
        raise ValueError('Complete mask fast-path simulator checks required')
    if report['sources'] != report['sources_after'] or report['native_sources'] != report['native_sources_after']:
        raise ValueError('Stable simulator sources required')
    for dependencies in (report['sources'], report['candidate_sources'],
            report['factory_build']['builders'], report['factory_build']['ladder_builders']):
        for name, checksum in dependencies.items():
            if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != checksum:
                raise ValueError('Simulator-qualified sum dependency changed: ' + name)
    return dict(run_id=RUN_ID, simulator_qualified=True, hardware_qualified=False,
        performance_qualified=False, report_sha256=REPORT_SHA256)
