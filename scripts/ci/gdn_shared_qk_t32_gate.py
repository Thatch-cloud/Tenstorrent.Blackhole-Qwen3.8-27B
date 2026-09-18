"""Source-bound T32 recurrence/scatter simulator admission, not hardware acceptance."""

import hashlib
import json
from pathlib import Path

from gdn_multitoken import HASHES, KERNEL_ROOT
from gdn_shared_qk_gate import API_SOURCES, LOCAL_SOURCES


REPORT_SHA256 = 'c88ceb06a8983b156c90161354f7f9f7da918b86e4329e03ad54ebf909bb66b3'
DEPENDENCIES = tuple(name for name in LOCAL_SOURCES if not name.endswith('-probe.py')) + (
    'gdn_shared_qk_t32_program.py', 'gdn_shared_qk_t32_pipeline.py',
    'gdn_shared_qk_t32_adapter.py', 'shared_qk_norm_t32_scatter.py', 'gdn_norm_scatter.py')


def validate_report(report):
    expected = dict(passed=True, closed_cleanly=True, stage='complete', backend='simulator',
        rows=32, hardware_qualified=False, timing_qualified=False, norm_unchanged=True,
        state_math_unchanged=True, shared_qk_preparation=True, norm_bridge_scatter=True,
        norm_direct_read_bytes=8192)
    if (any(type(report.get(key)) is not type(value) or report.get(key) != value
            for key, value in expected.items()) or report.get('error') or report.get('numerical_failures')
            or not report.get('sources') or report['sources'] != report.get('sources_after')):
        raise ValueError('Complete unchanged-source T32 recurrence/scatter evidence required')
    for field, operands in (('checks', 3), ('immutable_checks', 6)):
        matrix = [dict(mode=mode, operand=operand, chip=chip, exact=True)
            for mode in ('eager', 'replay_1', 'replay_2', 'replay_0')
            for operand in range(operands) for chip in (0, 1)]
        if report.get(field) != matrix:
            raise ValueError('Exact complete T32 two-chip matrix required: ' + field)
    stale = [dict(seed=seed, chip=chip, detected=True) for seed in (1, 2, 0) for chip in (0, 1)]
    if report.get('stale_controls') != stale:
        raise ValueError('Changed-input recurrent-state controls required')


def qualify(evidence, directory, runtime):
    evidence, directory, runtime = Path(evidence), Path(directory), Path(runtime)
    raw = (evidence / 'gdn-shared-recurrence.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact retained T32 recurrence/scatter report required')
    report = json.loads(raw)
    validate_report(report)
    if ((evidence / 'gdn-shared-recurrence.exit-status').read_text().strip() != '0'
            or (evidence / 'simulator-runtime.txt').read_text().strip() != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'
            or json.loads((evidence / 'container-cleanup.json').read_text()) != dict(
                stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0)):
        raise ValueError('Clean pinned-runtime T32 simulator execution required')
    paths = {'/experiment-scripts/ci/' + name: directory / name for name in DEPENDENCIES}
    native = [KERNEL_ROOT + '/' + name for name in HASHES]
    native += ['tt_metal/hw/inc/api/compute/' + name for name in API_SOURCES]
    paths.update({'/opt/tt-metal/' + name: runtime / name for name in native})
    for name, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != report['sources'].get(name):
            raise ValueError('T32 simulator-qualified dependency changed: ' + name)
    return dict(report_sha256=REPORT_SHA256, checked_sources=list(paths), rows=32,
        simulator_qualified=True, hardware_qualified=False, performance_qualified=False)
