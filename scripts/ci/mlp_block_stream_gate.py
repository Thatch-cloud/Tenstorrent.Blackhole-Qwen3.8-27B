"""Bind reviewed simulator evidence to an explicitly staged block-stream kernel."""

import copy
import hashlib
import json
from pathlib import Path

from mlp_block_stream import reader_source
from mlp_block_stream_projection import adapt_projection
from mlp_register_epilogue_gate import COMPUTE, SIM_PACKER, TYPECAST


REPORT_SHA256 = '548805b4bc3b64c1441f59c6a7f887ec35df40308f9c10cdbaa57cd4aa62a138'
CANDIDATE_SHA256 = '00f29b83acb3eb93929b623e426c9312c158c8b8a9b9db028e7011e4f51db6d7'


def validate_report(report, *, rows=16):
    if type(rows) is not int or rows not in (16, 32):
        raise ValueError('Explicit supported block-stream width required')
    if (report.get('passed') is not True or report.get('backend') != 'simulator'
            or report.get('timings') != [] or 'error' in report
            or report.get('checks') != [dict(rows=rows, chip=chip, exact=True) for chip in range(2)]):
        raise ValueError('Exact requested-width simulator eager coverage required')
    traces = report.get('trace_replays', [])
    if len(traces) != 1:
        raise ValueError('One complete requested-width replay matrix required')
    trace = traces[0]
    checks = [dict(arm=arm, repetition=repetition, pattern=pattern, chip=chip, exact=True)
        for repetition, pattern in enumerate((0, 1, 0)) for arm in ('control', 'fused') for chip in range(2)]
    negatives = [dict(arm=arm, chip=chip, stale_input_detected=True)
        for arm in ('control', 'fused') for chip in range(2)]
    if (trace.get('passed') is not True or trace.get('rows') != rows or trace.get('timings') != []
            or trace.get('checks') != checks or trace.get('negative_controls') != negatives):
        raise ValueError('Exact changing-input and stale-input checks required')
    before = report.get('stream_bytes_before', [])
    if (len(before) != 2 or any(not isinstance(value, str) or len(value) != 64 for value in before)
            or before != report.get('stream_bytes_after')):
        raise ValueError('Both stream byte identities must remain unchanged')


def qualify(directory, evidence, register_admission):
    directory, evidence = Path(directory), Path(evidence)
    raw = (evidence / 'fused-batch.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact reviewed block-stream replay report required')
    report = json.loads(raw)
    validate_report(report)
    if ((evidence / 'fused-batch.exit-status').read_text().strip() != '0'
            or (evidence / 'simulator-runtime.txt').read_text().strip() != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'
            or json.loads((evidence / 'container-cleanup.json').read_text()) !=
                dict(stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0)):
        raise ValueError('Pinned simulator runtime and clean teardown required')
    staged = json.loads((evidence / 'block-stream-candidate.json').read_text())
    for name in ('mlp_block_stream.py', 'mlp_block_stream.cpp', 'mlp_block_stream_projection.py',
            'mlp_block_stream_probe.py', 'mlp_register_epilogue.py', 'mlp_rounding_policy.py'):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != staged['after'].get(name):
            raise ValueError('Executed block-stream source changed: ' + name)
    register = directory / 'mlp-register-epilogue-candidate'
    candidate = adapt_projection((register / 'fused_1d.py').read_text())
    if hashlib.sha256(candidate.encode()).hexdigest() != CANDIDATE_SHA256:
        raise ValueError('Candidate differs from the executed projection')
    kernel, = report['kernels']
    baseline, = register_admission['kernels']
    if (register_admission.get('passed') is not True
            or baseline['fused_compute_sha256'] != COMPUTE
            or kernel['fused_compute_sha256'] != COMPUTE
            or kernel['rounding_runtime']['typecast_header_sha256'] != TYPECAST
            or report['packer_header_sha256'] != SIM_PACKER):
        raise ValueError('Previously admitted register arithmetic required')
    expected = copy.deepcopy(baseline)
    expected.update(native_weight_reader_sha256=baseline['reader_sha256']['fused_1d_weights.cpp'],
        block_stream_sources=report['stream_binding']['pack_sources'], weight_transport='block-major-raw-bf4',
        stream_page_bytes=27648, extra_weight_bytes_per_chip=50319360)
    generated = reader_source((register / 'fused_1d_weights.cpp').read_text())
    expected['reader_sha256']['fused_1d_weights.cpp'] = hashlib.sha256(generated.encode()).hexdigest()
    simulated = copy.deepcopy(expected)
    simulated['rounding_runtime']['packer_header_sha256'] = SIM_PACKER
    if kernel != simulated:
        raise ValueError('Complete block-stream kernel manifest differs from admitted register baseline')
    return dict(passed=True, report_sha256=REPORT_SHA256, kernels=[expected],
        hardware_qualified=False, performance_qualified=False)
