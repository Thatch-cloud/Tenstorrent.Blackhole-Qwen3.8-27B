"""Admit the exact full-recurrence input-overlap simulator result and generated kernel."""

import hashlib
import json
from pathlib import Path

from gdn_input_overlap import reader as transform
from gdn_shared_qk_gate import LOCAL_SOURCES, API_SOURCES
from gdn_multitoken import HASHES, KERNEL_ROOT


REPORT_SHA256 = '2dcd515b0236ebf35143406ee50b5c46a0567e617325324bf4307fdf84aac05b'
CONTROL_SHA256 = '3a670f7e24dd53b8edcbaaac2934a9fc95cee1b46a9af8ccb5d0e90ad2f4df22'
CANDIDATE_SHA256 = '71b58bd93eab0b5e11dfbd5fe72b49684105a80caf83c8535df51cf4e16609b8'
KERNEL = dict(control_sha256=CONTROL_SHA256, candidate_sha256=CANDIDATE_SHA256,
    extra_cb_bytes=0)
DEPENDENCIES = tuple(name for name in LOCAL_SOURCES if not name.endswith('-probe.py')) + (
    'gdn_input_overlap.py',)


def validate_report(report):
    if (any(report.get(key) is not True for key in ('passed', 'closed_cleanly', 'norm_unchanged',
            'state_math_unchanged', 'shared_qk_preparation', 'input_overlap'))
            or report.get('stage') != 'complete' or report.get('backend') != 'simulator'
            or report.get('rows') != 16 or not report.get('sources')
            or report['sources'] != report.get('sources_after') or report.get('numerical_failures')
            or report.get('hardware_qualified') is not False or report.get('timing_qualified') is not False
            or report.get('generated_kernels') != [KERNEL]):
        raise ValueError('Complete unchanged-source input-overlap recurrence evidence required')
    for field, operands in (('checks', 3), ('immutable_checks', 6)):
        expected = [dict(mode=mode, operand=operand, chip=chip, exact=True)
            for mode in ('eager', 'replay_1', 'replay_2', 'replay_0')
            for operand in range(operands) for chip in (0, 1)]
        if report.get(field) != expected:
            raise ValueError('Complete exact eager/replay matrix required: ' + field)


def qualify(evidence, directory, runtime):
    evidence, directory, runtime = Path(evidence), Path(directory), Path(runtime)
    raw = (evidence / 'gdn-shared-recurrence.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact reviewed input-overlap simulator artifact required')
    if ((evidence / 'gdn-shared-recurrence.exit-status').read_text().strip() != '0'
            or (evidence / 'simulator-runtime.txt').read_text().strip() != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'
            or json.loads((evidence / 'container-cleanup.json').read_bytes()) != dict(
                stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0)):
        raise ValueError('Clean simulator execution on pinned runtime required')
    report = json.loads(raw)
    validate_report(report)
    paths = {f'/experiment-scripts/ci/{name}': directory / name for name in DEPENDENCIES}
    native = [KERNEL_ROOT + '/' + name for name in HASHES]
    native += ['tt_metal/hw/inc/api/compute/' + name for name in API_SOURCES]
    paths.update({f'/opt/tt-metal/{name}': runtime / name for name in native})
    for name, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != report['sources'].get(name):
            raise ValueError('Dual-state dependency differs from simulation: ' + name)
    from gdn_shared_qk_recurrence import load_kernels
    before = load_kernels(runtime)['recurrence']['reader']
    after = transform(before)
    if (hashlib.sha256(before.encode()).hexdigest() != CONTROL_SHA256
            or hashlib.sha256(after.encode()).hexdigest() != CANDIDATE_SHA256):
        raise ValueError('Generated recurrence differs from simulated control/candidate pair')
    return dict(report_sha256=REPORT_SHA256, kernel=dict(KERNEL), checked_sources=list(paths),
        hardware_qualified=False, performance_qualified=False)
