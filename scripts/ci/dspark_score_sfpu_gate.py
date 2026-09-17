"""Exact small simulator evidence for the isolated SFPU hardware diagnostic."""

import hashlib
import json
from pathlib import Path


REPORT_SHA256 = 'd70c9ba5e3d0cdad87901746f3121ab4a17b60915302c5172fce7f6a74909d42'
RUN_ID = 34815143244


def qualify(directory, report_path):
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact completed SFPU simulator evidence required')
    report = json.loads(payload)
    expected = dict(passed=True, closed_cleanly=True, backend='simulator', context=128,
        capacity=384, candidate='sfpu-score-centering', performance_qualified=False)
    if any(report.get(name) != value for name, value in expected.items()):
        raise ValueError('Completed small SFPU candidate simulation required')
    if report['sources'] != report['sources_after'] or report['native_sources'] != report['native_sources_after']:
        raise ValueError('Stable SFPU simulator source evidence required')
    for dependencies in (report['sources'], report['candidate_sources'],
            report['factory_build']['builders'], report['factory_build']['ladder_builders']):
        for name, checksum in dependencies.items():
            if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != checksum:
                raise ValueError('Simulator-qualified dependency changed: ' + name)
    return dict(run_id=RUN_ID, report_sha256=REPORT_SHA256,
        simulator_qualified=True, hardware_qualified=False, performance_qualified=False)
