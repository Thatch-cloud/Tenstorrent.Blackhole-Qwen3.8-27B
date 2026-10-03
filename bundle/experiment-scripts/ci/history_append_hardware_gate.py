"""Immutable physical 64K addressing and replay admission, not model acceptance."""

import hashlib
import json
from pathlib import Path


REPORT_SHA256 = '7a962f5b7cc519d61cead689a225c68995ce0a600951b65c4fe0c8f2a24301cd'


def qualify(directory, report_path):
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Pinned physical history replay evidence required')
    report = json.loads(payload)
    if (report.get('backend') != 'hardware' or report.get('capacity') != 66560
            or report.get('initial_position') != 65535
            or not all(report.get(key) is True for key in ('passed', 'closed_cleanly', 'trace_qualified'))
            or len(report.get('checks', ())) != 84
            or not all(check.get('exact') is True for check in report['checks'])):
        raise ValueError('Complete physical 64K bank matrix required')
    for name, expected in report['sources'].items():
        if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Hardware-qualified source changed: ' + name)
    return report
