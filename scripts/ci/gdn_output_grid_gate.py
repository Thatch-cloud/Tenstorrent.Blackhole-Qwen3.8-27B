"""Admit only the retained local simulator comparison, not hardware performance."""

import hashlib
import json
from pathlib import Path


REPORT_SHA256 = 'c8017ca9258dd32eb88f9833956bfe133b28487eb75e2487363c3ca86c9271ba'
LOCAL_SOURCES = ('gdn-output-grid-probe.py', 'output_projection_grid.py', 'attention_batch.py')
NATIVE_SOURCE = 'models/demos/blackhole/qwen36/tt/tp_common.py'


def qualify(report_path, directory, runtime):
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact retained GDN output-grid report required')
    report = json.loads(payload)
    expected = [dict(label=label, chip=chip, exact=True)
        for label in ('eager_0', 'eager_1', 'eager_2', 'replay_1', 'replay_2', 'replay_0')
        for chip in (0, 1)]
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('native_configuration_unchanged') is not True
            or report.get('grids') != dict(control=[11, 3], candidate=[11, 6]) or report.get('backend') != 'simulator'
            or report.get('checks') != expected or report.get('sources') != report.get('sources_after')
            or report.get('hardware_qualified') is not False or report.get('timing_qualified') is not False):
        raise ValueError('Complete stable simulator evidence required')
    paths = {f'/experiment-scripts/ci/{name}': Path(directory) / name for name in LOCAL_SOURCES}
    paths['/opt/tt-metal/' + NATIVE_SOURCE] = Path(runtime) / NATIVE_SOURCE
    if set(paths) != set(report['sources']):
        raise ValueError('Exact source closure required')
    for name, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != report['sources'][name]:
            raise ValueError('Simulator-qualified source changed: ' + name)
    return dict(report_sha256=REPORT_SHA256, checked_sources=list(paths),
        hardware_qualified=False, scope='Synthetic local GDN partial output only; no collective or full-model qualification')
