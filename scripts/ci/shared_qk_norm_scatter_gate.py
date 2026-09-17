"""Source-bound admission of shared-Q/K plus direct-scatter normalization."""

import hashlib
import json
from pathlib import Path

from gdn_multitoken import HASHES, KERNEL_ROOT
from gdn_shared_qk_gate import API_SOURCES, LOCAL_SOURCES


REPORT_SHA256 = '33f2a5dc51918aa34dbda412e07d7595105ef1edd5010e4de20a4ede1dfe9602'
DEPENDENCIES = tuple(name for name in LOCAL_SOURCES if not name.endswith('-probe.py')) + (
    'shared_qk_norm_scatter.py', 'gdn_norm_scatter.py')


def validate_report(report):
    flags = ('passed', 'closed_cleanly', 'norm_unchanged', 'state_math_unchanged',
        'shared_qk_preparation', 'norm_bridge_scatter')
    if (any(report.get(name) is not True for name in flags)
            or report.get('stage') != 'complete' or report.get('backend') != 'simulator'
            or report.get('rows') != 16 or report.get('norm_direct_read_bytes') != 8192
            or report.get('hardware_qualified') is not False
            or report.get('timing_qualified') is not False or report.get('numerical_failures')
            or not report.get('sources') or report.get('sources') != report.get('sources_after')):
        raise ValueError('Complete unchanged-source shared-Q/K scatter qualification required')
    for field, operands in (('checks', 3), ('immutable_checks', 6)):
        expected = [dict(mode=mode, operand=operand, chip=chip, exact=True)
            for mode in ('eager', 'replay_1', 'replay_2', 'replay_0')
            for operand in range(operands) for chip in (0, 1)]
        if report.get(field) != expected:
            raise ValueError('Complete exact two-chip matrix required: ' + field)


def qualify(report_path, directory, runtime):
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact retained shared-Q/K scatter report required')
    report = json.loads(payload)
    validate_report(report)
    paths = {f'/experiment-scripts/ci/{name}': Path(directory) / name for name in DEPENDENCIES}
    native = [KERNEL_ROOT + '/' + name for name in HASHES]
    native += ['tt_metal/hw/inc/api/compute/' + name for name in API_SOURCES]
    paths.update({f'/opt/tt-metal/{name}': Path(runtime) / name for name in native})
    for name, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != report['sources'].get(name):
            raise ValueError('Qualified shared-Q/K scatter dependency changed: ' + name)
    return dict(report_sha256=REPORT_SHA256, checked_sources=list(paths),
        hardware_qualified=False, performance_qualified=False)
