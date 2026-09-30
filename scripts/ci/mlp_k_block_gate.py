"""Bind combined K32 admission to reviewed simulator sources and arithmetic."""

import hashlib
import json
from pathlib import Path

from mlp_k_block import CHANGES, transform
from mlp_weight_pipeline_gate import validate_report


REPORT_SHA256 = 'd7594fda2ade8ff6252fcb0a7d0bb8c450ed49a4695c6ecd5d64b72dcc05517d'
CONTROL = {
    'fused_1d.py': '4c215d945e723ef3d8c2be1757e7c170bf21eebb01e204359243d766c6c7cf93',
    'fused_1d_input.cpp': '381739e8070b910c61395e5e82970200d96f1de8bff0de25f74d483a1d3258b9',
    'fused_1d_weights.cpp': 'c283b6b347045ee3b59363e4e3aa6b0b796e226a5ccbf6df9aa8930a570f46eb',
}
CANDIDATE = {
    'fused_1d.py': '6763135746ca51517894156187ef0a6d3461fdefbbeb1255004f9c8d1d7b1d02',
    'fused_1d_input.cpp': '0f21b8859f75d8f640b815fa7e6c34cb371040209bd5f8ea88267ab974bb6808',
    'fused_1d_weights.cpp': 'ffcaaeff95b07ffc3df4ed619bb9a069009394deee8136ed1b8efa0dc300ead5',
}


def checked_source(directory, name, expected):
    payload = (Path(directory) / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Reviewed source required: ' + name)
    return payload.decode()


def stage_candidate(directory):
    directory = Path(directory)
    payloads = {}
    for name in CHANGES:
        source = checked_source(directory, name, CONTROL[name])
        payloads[name] = transform(name, source).encode()
        if hashlib.sha256(payloads[name]).hexdigest() != CANDIDATE[name]:
            raise ValueError('Transformation differs from simulated source: ' + name)
    candidate = directory / 'mlp-k-block-candidate'
    candidate.mkdir()
    for name, payload in payloads.items():
        (candidate / name).write_bytes(payload)
    return candidate


def qualify(directory, evidence):
    from fused_t16_admission import qualify_simulator

    directory, evidence = Path(directory), Path(evidence)
    report = json.loads(checked_source(evidence, 'fused-batch.json', REPORT_SHA256))
    validate_report(report)
    if (evidence / 'fused-batch.exit-status').read_text().strip() != '0':
        raise ValueError('Clean simulator exit required')
    if (evidence / 'simulator-runtime.txt').read_text().strip() != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9':
        raise ValueError('Pinned simulator runtime required')
    if json.loads((evidence / 'container-cleanup.json').read_text()) != dict(
            stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0):
        raise ValueError('Clean simulator teardown required')
    control = qualify_simulator(directory)
    baseline = next(kernel for kernel in control['kernels'] if kernel['token_rows'] == 16)
    expected = dict(baseline, k_block=32, reader_sha256={
        name: CANDIDATE[name] for name in ('fused_1d_input.cpp', 'fused_1d_weights.cpp')})
    for name in CHANGES:
        checked_source(directory, name, CONTROL[name])
        checked_source(directory / 'mlp-k-block-candidate', name, CANDIDATE[name])
    if report.get('kernels') != [expected]:
        raise ValueError('Compute arithmetic or kernel geometry differs from qualified baseline')
    if report.get('buffer_candidate_sha256') != CANDIDATE['fused_1d.py']:
        raise ValueError('Simulated projection identity differs')
    checked_source(directory, 'fusion_trace.py', report['trace_source_sha256'])
    return dict(report_sha256=REPORT_SHA256, kernels=[expected], passed=True,
        hardware_qualified=False, performance_qualified=False)
