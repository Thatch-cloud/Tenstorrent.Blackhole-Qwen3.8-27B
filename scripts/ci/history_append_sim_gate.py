"""Pin small-fixture simulator replay evidence before the 64K hardware probe."""

import hashlib
import json
from pathlib import Path


REPORT_SHA256 = '25db87cc6706729736699ed4a30c40c10be0f1cdfef9a0558c9dc356be47cb43'


def qualify(directory, report_path):
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Pinned simulator replay evidence required')
    report = json.loads(payload)
    if (report.get('backend') != 'simulator' or not all(report.get(key) is True
            for key in ('passed', 'closed_cleanly', 'trace_qualified'))
            or len(report.get('checks', ())) != 84
            or not all(check.get('exact') is True for check in report['checks'])):
        raise ValueError('Complete exact simulator replay matrix required')
    for name in ('history_append_plan.py', 'history_append_dma.py', 'history_append_dma.cpp'):
        if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != report['sources'][name]:
            raise ValueError('Simulator-qualified writer changed: ' + name)
    return report
