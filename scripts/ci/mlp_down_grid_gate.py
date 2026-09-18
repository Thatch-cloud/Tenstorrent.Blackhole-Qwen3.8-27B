"""Admit the exact reviewed T16 MLP-down simulator evidence; not throughput."""

import hashlib
import json
from pathlib import Path


REPORT_SHA256 = 'fb677f9c1dde8c1badd3f7a6f81378b8e6fa7fde06fd62e102d07a3a94d87404'
NATIVE_SOURCE = 'models/demos/blackhole/qwen36/tt/tp_common.py'
LOCAL_SOURCES = {'gdn-output-grid-probe.py': 'mlp-down-grid-probe.py',
    'mlp_down_grid.py': 'mlp_down_grid.py', 'attention_batch.py': 'attention_batch.py'}


def validate_report(report, *, rows=16):
    if type(rows) is not int or rows not in (16, 32):
        raise ValueError('Explicit supported MLP-down report width required')
    expected = dict(passed=True, closed_cleanly=True, backend='simulator', stage='complete',
        hardware_qualified=False, timing_qualified=False, projection='mlp_down', rows=rows,
        k=8704, n=5120, grids=dict(control=[11, 3], candidate=[11, 8]),
        output_placement='L1', weights='bfloat8_b')
    if (any(report.get(key) != value or type(report.get(key)) is not type(value)
            for key, value in expected.items()) or not report.get('sources')
            or report['sources'] != report.get('sources_after')):
        raise ValueError('Exact complete native MLP-down simulator policy required')
    labels = ('eager_0', 'eager_1', 'eager_2', 'replay_1', 'replay_2', 'replay_0')
    checks = [dict(label=label, chip=chip, exact=True) for label in labels for chip in (0, 1)]
    immutable = [dict(label=label, operand='activation', chip=chip, exact=True)
        for label in labels for chip in (0, 1)]
    immutable += [dict(label='complete', operand='weight', chip=chip, exact=True) for chip in (0, 1)]
    negative = [dict(label=label, poisoned=True) for label in labels[3:]]
    if (report.get('checks') != checks or report.get('immutable_checks') != immutable
            or report.get('negative_controls') != negative):
        raise ValueError('All numerical, input integrity and poisoned-replay controls required')


def qualify(evidence, directory, runtime):
    evidence, directory, runtime = Path(evidence), Path(directory), Path(runtime)
    raw = (evidence / 'gdn-output-grid.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact reviewed MLP-down report required')
    report = json.loads(raw)
    validate_report(report)
    if ((evidence / 'gdn-output-grid.exit-status').read_text().strip() != '0'
            or (evidence / 'simulator-runtime.txt').read_text().strip() != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'
            or json.loads((evidence / 'container-cleanup.json').read_bytes()) != dict(
                stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0)):
        raise ValueError('Clean pinned-runtime simulator execution required')
    paths = {'/experiment-scripts/ci/' + measured: directory / local
        for measured, local in LOCAL_SOURCES.items()}
    paths['/opt/tt-metal/' + NATIVE_SOURCE] = runtime / NATIVE_SOURCE
    if set(paths) != set(report['sources']):
        raise ValueError('Exact measured source closure required')
    for key, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != report['sources'][key]:
            raise ValueError('Simulator-qualified source changed: ' + key)
    return dict(report_sha256=REPORT_SHA256, checked_sources=list(paths),
        hardware_qualified=False, performance_qualified=False)
