"""Admit only the retained exact shared-Q/K pipeline simulator result, not performance."""

import hashlib
import json
from pathlib import Path

from gdn_multitoken import HASHES, KERNEL_ROOT


REPORT_SHA256 = 'a155b893786f68ac36b4f040454892df683a0da0c3aa0b22b898697b22241ffe'
LOCAL_SOURCES = ('gdn-shared-recurrence-probe.py', 'gdn_shared_qk_pipeline.py',
    'gdn_shared_qk_program.py', 'gdn_shared_qk_compute.py', 'gdn_shared_qk_dataflow.py',
    'gdn_shared_qk_recurrence.py', 'gdn_multitoken.py',
    'gdn_vsplit.py', 'gdn_vsplit_prepared.py', 'gdn_vsplit_norm_batch.py',
    'gdn_vsplit_prefetch.py', 'attention_batch.py')
API_SOURCES = ('eltwise_binary_sfpu.h', 'eltwise_binary.h', 'tile_move_copy.h', 'bcast.h')


def qualify(report_path, directory, runtime):
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact retained shared-Q/K recurrence simulator report required')
    report = json.loads(payload)
    def matrix(operands):
        return [dict(mode=mode, operand=operand, chip=chip, exact=True)
            for mode in ('eager', 'replay_1', 'replay_2', 'replay_0')
            for operand in range(operands) for chip in (0, 1)]
    if (any(report.get(key) is not True for key in ('passed', 'closed_cleanly',
            'norm_unchanged', 'state_math_unchanged', 'shared_qk_preparation')) or report.get('backend') != 'simulator'
            or report.get('rows') != 16 or report.get('checks') != matrix(3)
            or report.get('immutable_checks') != matrix(6) or report.get('numerical_failures')
            or report.get('hardware_qualified') is not False or report.get('timing_qualified') is not False
            or report.get('sources') != report.get('sources_after')):
        raise ValueError('Complete unchanged-source two-chip simulator matrix required')
    paths = {f'/experiment-scripts/ci/{name}': Path(directory) / name for name in LOCAL_SOURCES}
    native = [KERNEL_ROOT + '/' + name for name in HASHES]
    native += ['tt_metal/hw/inc/api/compute/' + name for name in API_SOURCES]
    paths.update({f'/opt/tt-metal/{name}': Path(runtime) / name for name in native})
    for name, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != report['sources'].get(name):
            raise ValueError('Simulator-qualified numerical dependency changed: ' + name)
    return dict(report_sha256=REPORT_SHA256, checked_sources=list(paths), hardware_qualified=False)
