"""Immutable admission for the reviewed compact feedback simulator result."""

import hashlib
import json
from pathlib import Path

from compact_score_report import validate


REPORT_SHA256 = '22243fad7110052d740d9931fa4ff115b09008cddce64c65a95a529e2883b820'


def qualify(directory, evidence):
    evidence = Path(evidence)
    raw = (evidence / 'gdn-output-grid.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Reviewed compact feedback simulator report required')
    report = json.loads(raw)
    validate(report, directory)
    if (evidence / 'gdn-output-grid.exit-status').read_text().strip() != '0':
        raise ValueError('Successful simulator exit required')
    if (evidence / 'simulator-runtime.txt').read_text().strip() != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9':
        raise ValueError('Pinned simulator runtime required')
    if json.loads((evidence / 'container-cleanup.json').read_text()) != dict(
            stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0):
        raise ValueError('Clean simulator teardown required')
    return dict(report_sha256=REPORT_SHA256, report=report, hardware_qualified=False)
