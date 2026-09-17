"""Retained synthetic T16 recurrence evidence; not full-request qualification."""

import hashlib
import json
from pathlib import Path

from gdn_multitoken import HASHES, KERNEL_ROOT


REPORT_SHA256 = '14d9fa6ae9b2e5ec0674d5b846f382c141e38ee73fa423ddb82a8ac73258bd99'
LOCAL_SOURCES = ('gdn-copy-pairs-probe.py', 'gdn_copy_pairs.py', 'gdn_multitoken.py',
    'gdn_vsplit.py', 'gdn_vsplit_prepared.py', 'gdn_vsplit_norm_batch.py',
    'gdn_vsplit_prefetch.py', 'attention_batch.py')


def qualify(report_path, directory, runtime):
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact retained paired-copy simulator report required')
    report = json.loads(payload)
    def matrix(operands):
        return [dict(mode=mode, operand=operand, chip=chip, exact=True)
            for mode in ('eager', 'replay_1', 'replay_2', 'replay_0')
            for operand in range(operands) for chip in (0, 1)]
    if (any(report.get(key) is not True for key in ('passed', 'closed_cleanly',
            'norm_unchanged', 'dataflow_unchanged')) or report.get('backend') != 'simulator'
            or report.get('rows') != 16 or report.get('checks') != matrix(3)
            or report.get('immutable_checks') != matrix(6)
            or report.get('hardware_qualified') is not False or report.get('timing_qualified') is not False
            or report.get('sources') != report.get('sources_after')):
        raise ValueError('Complete unchanged-source two-chip simulator matrix required')
    paths = {f'/experiment-scripts/ci/{name}': Path(directory) / name for name in LOCAL_SOURCES}
    paths.update({f'/opt/tt-metal/{KERNEL_ROOT}/{name}': Path(runtime) / KERNEL_ROOT / name for name in HASHES})
    for name, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != report['sources'].get(name):
            raise ValueError('Simulator-qualified numerical dependency changed: ' + name)
    return dict(report_sha256=REPORT_SHA256, checked_sources=list(paths), hardware_qualified=False)
