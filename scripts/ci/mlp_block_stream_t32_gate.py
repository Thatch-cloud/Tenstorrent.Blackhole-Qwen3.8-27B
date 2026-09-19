"""Admit T32 stream transport only from its own replay and regenerated source identity."""

import copy
import hashlib
import json
from pathlib import Path

from mlp_block_stream import reader_source
from mlp_block_stream_gate import validate_report
from mlp_block_stream_projection import adapt_projection
from mlp_block_stream_probe import adapt_probe
from mlp_block_stream_t32_stage import payloads
from mlp_register_epilogue_gate import COMPUTE, SIM_PACKER, TYPECAST


REPORT_SHA256 = '0fbdcbdd9c688a5a2e497b682ab3cd4fadcb40279fda952c8e24323a92856766'
CANDIDATE_SHA256 = '4e1127829c6aa567f5af3ea403d067923d3f545e983a0107a2a61f7ea81485f3'
STAGING_SHA256 = 'db087aeba8b9dfee329ab62dbca8143d409916262031018e50ecb77ca5075c41'


def candidate_sources(directory):
    directory = Path(directory)
    return payloads({
        'fused_1d.py': adapt_projection((directory / 'mlp-register-epilogue-candidate/fused_1d.py').read_text()),
        'fused-batch-probe.py': adapt_probe((directory / 'fused-batch-probe.py').read_text()),
        'mlp_block_stream_projection.py': (directory / 'mlp_block_stream_projection.py').read_text()})


def qualify(directory, evidence, register_admission):
    directory, evidence = Path(directory), Path(evidence)
    raw = (evidence / 'fused-batch.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact retained T32 block-stream report required')
    report = json.loads(raw)
    validate_report(report, rows=32)
    if ((evidence / 'fused-batch.exit-status').read_text().strip() != '0'
            or (evidence / 'simulator-runtime.txt').read_text().strip() != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'
            or json.loads((evidence / 'container-cleanup.json').read_text()) != dict(
                stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0)):
        raise ValueError('Pinned simulator runtime and clean T32 teardown required')
    staged_raw = (evidence / 'block-stream-t32-candidate.json').read_bytes()
    if hashlib.sha256(staged_raw).hexdigest() != STAGING_SHA256:
        raise ValueError('Exact retained T32 staging manifest required')
    staged = json.loads(staged_raw)
    baseline_raw = (evidence / 'block-stream-candidate.json').read_bytes()
    if hashlib.sha256(baseline_raw).hexdigest() != '20e5dc8d8e9b88e3ba34ab52a5fb9140b393f3c2020b72f47ff6f3cca72ab208':
        raise ValueError('Exact retained block-stream staging prerequisite required')
    baseline_stage = json.loads(baseline_raw)
    for name in ('mlp_block_stream.py', 'mlp_block_stream.cpp', 'mlp_block_stream_projection.py',
            'mlp_block_stream_probe.py', 'mlp_register_epilogue.py', 'mlp_rounding_policy.py'):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != baseline_stage['after'].get(name):
            raise ValueError('Executed T32 block-stream helper changed: ' + name)
    generated = candidate_sources(directory)
    for name, source in generated.items():
        if hashlib.sha256(source.encode()).hexdigest() != staged['sources'].get(name):
            raise ValueError('Regenerated T32 candidate differs from simulator: ' + name)
    if (hashlib.sha256(generated['fused_1d.py'].encode()).hexdigest() != CANDIDATE_SHA256
            or report.get('buffer_candidate_sha256') != CANDIDATE_SHA256
            or hashlib.sha256((directory / 'fusion_trace.py').read_bytes()).hexdigest() != report.get('trace_source_sha256')
            or report.get('packer_zero_graft') is not True or report.get('packer_header_sha256') != SIM_PACKER):
        raise ValueError('Executed T32 projection, trace and simulator packer required')
    kernel, = report['kernels']
    baseline, = register_admission['kernels']
    if (register_admission.get('passed') is not True or baseline['fused_compute_sha256'] != COMPUTE
            or kernel['fused_compute_sha256'] != COMPUTE
            or kernel['rounding_runtime']['typecast_header_sha256'] != TYPECAST):
        raise ValueError('Admitted unchanged register arithmetic required')
    expected = copy.deepcopy(baseline)
    expected.update(token_rows=32, native_weight_reader_sha256=baseline['reader_sha256']['fused_1d_weights.cpp'],
        block_stream_sources=report['stream_binding']['pack_sources'], weight_transport='block-major-raw-bf4',
        stream_page_bytes=27648, extra_weight_bytes_per_chip=50319360)
    reader = reader_source((directory / 'mlp-register-epilogue-candidate/fused_1d_weights.cpp').read_text())
    expected['reader_sha256']['fused_1d_weights.cpp'] = hashlib.sha256(reader.encode()).hexdigest()
    simulated = copy.deepcopy(expected)
    simulated['rounding_runtime']['packer_header_sha256'] = SIM_PACKER
    if kernel != simulated:
        raise ValueError('T32 kernel manifest differs from qualified arithmetic and transport')
    for name, digest in expected['block_stream_sources'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError('T32 raw weight packer changed: ' + name)
    return dict(passed=True, report_sha256=REPORT_SHA256, kernels=[expected],
        hardware_qualified=False, performance_qualified=False)
