"""Tiny folded-verifier simulator admission, not 64K request qualification."""

import hashlib
import json
from pathlib import Path


RUN_ID = 34828115400
REPORT_SHA256 = '1917a37bc6699159581cbb9cc953ab75f1e09592faa1f1edfa3937b35da9e57e'


def qualify(directory, report_path):
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact completed tiny T16 verifier simulation required')
    report = json.loads(payload)
    if (report.get('passed') is not True or report.get('closed') is not True
            or report.get('backend') != 'simulator'
            or report.get('stale_controls') != 2 or report.get('mask_poison_controls') != 8
            or report.get('sources') != report.get('sources_after')):
        raise ValueError('Complete unchanged verifier simulator evidence required')
    for field, count in (('checks', 8), ('mask_checks', 16), ('source_checks', 4), ('unpoisoned_replay', 2)):
        values = report.get(field, [])
        if len(values) != count or any(value.get('exact') is not True for value in values):
            raise ValueError('All verifier replay, mask and KV controls must pass')
    if any(value.get('capacity') != 512 for field in ('checks', 'mask_checks', 'source_checks')
            for value in report[field]):
        raise ValueError('Explicit tiny verifier capacity required')
    for name, checksum in report['sources'].items():
        if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Simulator-qualified verifier source changed: ' + name)
    return dict(run_id=RUN_ID, report_sha256=REPORT_SHA256, simulator_qualified=True,
        hardware_qualified=False, performance_qualified=False)
