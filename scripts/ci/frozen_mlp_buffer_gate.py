"""Source-bound T16 buffer admission; does not qualify hardware performance."""

import hashlib
import json
from pathlib import Path


REPORT_SHA256 = 'a37834f16393996a9fe2e1c0142c21e37cbde2100722dce9cdfe827b775db7dc'
CONTROL_SHA256 = '4c215d945e723ef3d8c2be1757e7c170bf21eebb01e204359243d766c6c7cf93'
CANDIDATE_SHA256 = 'ded6f11da9b2f501d925e6809467538499549fed799a97624550072faee92d46'


def qualify(directory, report_path):
    directory = Path(directory)
    raw = Path(report_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact retained buffer simulation required')
    report = json.loads(raw)
    if (report.get('passed') is not True or report.get('backend') != 'simulator'
            or report.get('buffer_candidate_sha256') != CANDIDATE_SHA256
            or report.get('checks') != [dict(rows=16, chip=chip, exact=True) for chip in (0, 1)]):
        raise ValueError('Exact two-chip T16 evidence required')
    for name, expected in (('fused_1d.py', CONTROL_SHA256), ('frozen_mlp_buffer_candidate.py', CANDIDATE_SHA256)):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Projection source differs: ' + name)
    from fused_t16_admission import qualify_simulator
    control = qualify_simulator(directory)
    expected = next(kernel for kernel in control['kernels'] if kernel['token_rows'] == 16)
    candidate = dict(expected, input_buffer_blocks=4, weight_buffer_blocks=4)
    if report.get('kernels') != [candidate]:
        raise ValueError('Only FIFO depth may differ from admitted target math')
    if hashlib.sha256((directory / 'fusion_trace.py').read_bytes()).hexdigest() != report['trace_source_sha256']:
        raise ValueError('Replay validator source changed')
    return dict(report_sha256=REPORT_SHA256, kernels=[candidate], passed=True,
        hardware_qualified=False, performance_qualified=False)
