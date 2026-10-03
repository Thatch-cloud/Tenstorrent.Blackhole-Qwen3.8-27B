"""Source-bound admission of the retained norm-bridge simulator matrix."""

import hashlib
import json
from pathlib import Path

from gdn_shared_qk_gate import LOCAL_SOURCES


REPORT_SHA256 = '120eca72f4fde17cf79453fbfd535d393534828f1ef8ee272925125d9cf849f8'
DEPENDENCIES = tuple(name for name in LOCAL_SOURCES if not name.endswith('-probe.py')) + (
    'frozen_gdn_norm_prefetch.py', 'gdn_shared_qk_pipeline.py', 'gdn_vsplit_norm_batch.py')


def validate_report(report):
    if (any(report.get(name) is not True for name in ('passed', 'closed_cleanly',
            'norm_unchanged', 'state_math_unchanged', 'shared_qk_preparation', 'norm_bridge_prefetch'))
            or report.get('backend') != 'simulator' or report.get('rows') != 16
            or report.get('norm_staging_bytes') != 8192
            or report.get('sources') != report.get('sources_after')):
        raise ValueError('Complete unchanged-source T16 norm qualification required')
    for field, operands in (('checks', 3), ('immutable_checks', 6)):
        expected = [dict(mode=mode, operand=operand, chip=chip, exact=True)
            for mode in ('eager', 'replay_1', 'replay_2', 'replay_0')
            for operand in range(operands) for chip in (0, 1)]
        if report.get(field) != expected:
            raise ValueError('Complete exact replay matrix required: ' + field)


def qualify(report_path, directory, runtime):
    raw = Path(report_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact retained norm-prefetch report required')
    report = json.loads(raw)
    validate_report(report)
    paths = {f'/experiment-scripts/ci/{name}': Path(directory) / name for name in DEPENDENCIES}
    for name in report['sources']:
        if name.startswith('/opt/tt-metal/'):
            paths[name] = Path(runtime) / name.removeprefix('/opt/tt-metal/')
    if not any(name.startswith('/opt/tt-metal/') for name in paths):
        raise ValueError('Native source evidence required')
    for name, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != report['sources'].get(name):
            raise ValueError('Qualified norm dependency changed: ' + name)
    return dict(report_sha256=REPORT_SHA256, checked_sources=list(paths),
        norm_staging_bytes=8192, hardware_qualified=False, performance_qualified=False)
