"""Pinned L1 simulator admission for direct causal convolution windows."""

import hashlib
import json
from pathlib import Path

from gdn_direct_window_report import validate


REPORT_SHA256 = 'f1837c9621c53cdb69a8c9047e867acc843b65b3762ca99c5e48a3dcf5345835'


def qualify(directory, evidence, native_root='/opt/tt-metal'):
    evidence = Path(evidence)
    raw = (evidence / 'gdn-output-grid.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Reviewed L1 direct-window simulator report required')
    report = json.loads(raw)
    validate(report, directory, native_root)
    if (evidence / 'gdn-output-grid.exit-status').read_text().strip() != '0':
        raise ValueError('Successful simulator exit required')
    if (evidence / 'simulator-runtime.txt').read_text().strip() != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9':
        raise ValueError('Pinned simulator runtime required')
    if json.loads((evidence / 'container-cleanup.json').read_text()) != dict(
            stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0):
        raise ValueError('Clean simulator teardown required')
    return dict(report_sha256=REPORT_SHA256, report=report, passed=True, hardware_qualified=False)
