"""Admit the exact full-recurrence dual-state simulator result and generated kernel."""

import hashlib
import json
from pathlib import Path

from gdn_dual_state_copy import transform
from gdn_shared_qk_gate import LOCAL_SOURCES, API_SOURCES
from gdn_multitoken import HASHES, KERNEL_ROOT


REPORT_SHA256 = '30910463b67866b0ccad62784cedf9e82d88cc9b6f074368d71ea983f09c5f9e'
CONTROL_SHA256 = 'ce404cf287f9962243dc65f30d9736f45b0c95cfff503c6c20d68f8981a41153'
CANDIDATE_SHA256 = 'abf6a1d2ce0fd9f8380656a09d7963fcd7ac8ed9017f6dff49bcbcccb3cd7a68'
KERNEL = dict(control_sha256=CONTROL_SHA256, candidate_sha256=CANDIDATE_SHA256,
    state_tiles=4, output_cb=18, feedback_cb=30)
DEPENDENCIES = tuple(name for name in LOCAL_SOURCES if not name.endswith('-probe.py')) + (
    'gdn_dual_state_copy.py', 'gdn_copy_pairs.py')


def validate_report(report):
    if (any(report.get(key) is not True for key in ('passed', 'closed_cleanly', 'norm_unchanged',
            'state_math_unchanged', 'shared_qk_preparation', 'dual_state_copy'))
            or report.get('stage') != 'complete' or report.get('backend') != 'simulator'
            or report.get('rows') != 16 or not report.get('sources')
            or report['sources'] != report.get('sources_after') or report.get('numerical_failures')
            or report.get('hardware_qualified') is not False or report.get('timing_qualified') is not False
            or report.get('generated_kernels') != [KERNEL]):
        raise ValueError('Complete unchanged-source dual-state recurrence evidence required')
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
        raise ValueError('Exact reviewed dual-state simulator artifact required')
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
    before = load_kernels(runtime)['recurrence']['compute']
    after = transform(before)
    if (hashlib.sha256(before.encode()).hexdigest() != CONTROL_SHA256
            or hashlib.sha256(after.encode()).hexdigest() != CANDIDATE_SHA256):
        raise ValueError('Generated recurrence differs from simulated control/candidate pair')
    return dict(report_sha256=REPORT_SHA256, kernel=dict(KERNEL), checked_sources=list(paths),
        hardware_qualified=False, performance_qualified=False)
