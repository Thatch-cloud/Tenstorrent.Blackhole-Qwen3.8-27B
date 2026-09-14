"""Pinned small simulator evidence admits only a hardware diagnostic candidate."""

import hashlib
import json
from pathlib import Path


REPORT_SHA256 = 'c2e5b57c28bbb1f726e8ba9f007cf9d38b0eb113737cd1c5a8098e606a39133a'
RUN_ID = 34810827485


def qualify(directory, report_path):
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact completed score simulator evidence required')
    report = json.loads(payload)
    expected = dict(passed=True, closed_cleanly=True, backend='simulator', context=128,
        capacity=384, candidate='bitwise-infinity-checks', performance_qualified=False)
    if any(report.get(name) != value for name, value in expected.items()):
        raise ValueError('Completed small candidate simulation required')
    if report['sources'] != report['sources_after'] or report['native_sources'] != report['native_sources_after']:
        raise ValueError('Stable simulator source evidence required')
    for dependencies in (report['sources'], report['candidate_sources'],
            report['factory_build']['builders'], report['factory_build']['ladder_builders']):
        for name, checksum in dependencies.items():
            if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != checksum:
                raise ValueError('Simulator-qualified dependency changed: ' + name)
    return dict(run_id=RUN_ID, report_sha256=REPORT_SHA256,
        simulator_qualified=True, hardware_qualified=False, performance_qualified=False)
