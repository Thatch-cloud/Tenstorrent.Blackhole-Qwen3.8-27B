"""Exact simulator evidence and source admission for overlapping window writes."""

import hashlib
import json
from pathlib import Path


REPORT_SHA256 = '92b90c94fefaf8171fc82acec8714e53ba97d709e797775b71574dd6bd377428'
SOURCES = ('gdn-output-grid-probe.py', 'window-write-candidate/gdn_conv_windows.py',
    'window-write-candidate/gdn_conv_windows.cpp', 'gdn_conv_windows.py', 'gdn_conv_windows.cpp',
    'attention_batch.py', 'feature_projection.py', 'gdn_multitoken_conv.py')


def validate(report):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'simulator' or report.get('stage') != 'complete'
            or report.get('logical_width') != 8240
            or report.get('hardware_qualified') is not False or report.get('timing_qualified') is not False
            or report.get('sources') != report.get('sources_after')
            or set(report.get('sources', {})) != set(SOURCES)):
        raise ValueError('Complete unchanged-source simulator-only evidence required')
    modes = [('eager', 0), ('replay', 0), ('replay', 1), ('replay', 2)]
    expected = [dict(seed=seed, mode=mode, arm=arm, slot=slot, chip=chip, exact=True)
        for mode, seed in modes for arm in ('control', 'overlap') for slot in range(4) for chip in range(2)]
    immutable = [dict(seed=seed, mode=mode, operand=operand, chip=chip, exact=True)
        for mode, seed in modes for operand in range(9) for chip in range(2)]
    if report.get('checks') != expected or report.get('immutable_checks') != immutable:
        raise ValueError('All exact windows and immutable inputs on both chips required')


def qualify(directory, evidence):
    directory, evidence = Path(directory), Path(evidence)
    payload = (evidence / 'gdn-output-grid.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Reviewed simulator report required')
    report = json.loads(payload)
    validate(report)
    if (evidence / 'gdn-output-grid.exit-status').read_text().strip() != '0':
        raise ValueError('Successful simulator exit required')
    if (evidence / 'simulator-runtime.txt').read_text().strip() != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9':
        raise ValueError('Pinned simulator runtime required')
    if json.loads((evidence / 'container-cleanup.json').read_text()) != dict(
            stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0):
        raise ValueError('Successful simulator cleanup required')
    for name in SOURCES:
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != report['sources'][name]:
            raise ValueError('Simulator-qualified source changed: ' + name)
    return dict(report_sha256=REPORT_SHA256, source_hashes=report['sources'], passed=True,
        logical_width=8240, hardware_qualified=False, performance_qualified=False)
