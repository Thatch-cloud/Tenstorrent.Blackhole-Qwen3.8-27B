"""Admit the exact bulk-pipeline replay without weakening serial-reader qualification."""

import copy
import hashlib
import json
from pathlib import Path

from mlp_block_stream import reader_source
from mlp_block_stream_gate import qualify as qualify_serial, validate_report, CANDIDATE_SHA256
from mlp_block_stream_pipeline import transform
from mlp_block_stream_pipeline_stage import OVERLAY
from mlp_register_epilogue_gate import SIM_PACKER


REPORT_SHA256 = '1bb9581a17de463bd7614ddf602a44c0db444776bdc58b9f0609fb8177a4b411'
STAGING_SHA256 = 'a1845443b6c694b14bff7ac709ba229f222539b88fcbf9f5943623d0d5cd2b49'
READER_SHA256 = 'b197b129252cb81ff2e3afabf7e014e557b0ab5cba834c7b0be746dc2cd0279b'
HELPER_SHA256 = '0e6c0abcd2d7142834029924ab09d50d2365a109dae1750ac73e013c86a57d79'


def checksum(value):
    return hashlib.sha256(value).hexdigest()


def validate_weights(report):
    expected = [dict(chip=chip, pages=43520, mismatched_words=0, exact=True,
        workers=64, source_exact=True, projection=projection)
        for projection in ('gate', 'up') for chip in (0, 1)]
    if report.get('weight_checks') != expected:
        raise ValueError('All four native packed-weight comparisons required')


def source_record(directory):
    directory = Path(directory)
    original = (directory / 'mlp_block_stream.py').read_text()
    serial = reader_source((directory / 'fused_1d_weights.cpp').read_text())
    candidate = transform(serial)
    helper = (directory / 'mlp_block_stream_pipeline.py').read_bytes()
    if checksum(candidate.encode()) != READER_SHA256 or checksum(helper) != HELPER_SHA256:
        raise ValueError('Exact simulator-covered bulk reader and helper required')
    return dict(simulator_qualified=False, hardware_qualified=False, performance_qualified=False,
        arithmetic_changed=False, extra_buffer_bytes=0, before=checksum(original.encode()),
        after=checksum((original + OVERLAY).encode()), helper_sha256=checksum(helper),
        serial_reader_sha256=checksum(serial.encode()), candidate_reader_sha256=checksum(candidate.encode()))


def qualify(directory, serial_evidence, evidence, register_admission):
    directory, evidence = Path(directory), Path(evidence)
    serial = qualify_serial(directory, serial_evidence, register_admission)
    raw = (evidence / 'fused-batch.json').read_bytes()
    staged = (evidence / 'block-stream-pipeline-candidate.json').read_bytes()
    if checksum(raw) != REPORT_SHA256 or checksum(staged) != STAGING_SHA256:
        raise ValueError('Exact reviewed pipeline replay and staging reports required')
    report = json.loads(raw)
    validate_report(report)
    validate_weights(report)
    if ((evidence / 'fused-batch.exit-status').read_text().strip() != '0'
            or (evidence / 'simulator-runtime.txt').read_text().strip() != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'
            or json.loads((evidence / 'container-cleanup.json').read_text()) != dict(
                stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0)):
        raise ValueError('Pinned simulator runtime and complete clean teardown required')
    generated = source_record(directory)
    if json.loads(staged) != generated:
        raise ValueError('Pipeline sources differ from retained simulator staging')
    expected = copy.deepcopy(serial['kernels'][0])
    expected['reader_sha256']['fused_1d_weights.cpp'] = READER_SHA256
    expected['block_stream_sources']['mlp_block_stream.py'] = generated['after']
    simulated = copy.deepcopy(expected)
    simulated['rounding_runtime']['packer_header_sha256'] = SIM_PACKER
    if (report.get('kernels') != [simulated] or report.get('packer_zero_graft') is not True
            or report.get('packer_header_sha256') != SIM_PACKER
            or report.get('buffer_candidate_sha256') != CANDIDATE_SHA256
            or report.get('trace_source_sha256') != checksum((directory / 'fusion_trace.py').read_bytes())):
        raise ValueError('Exact pipeline reader, unchanged compute and trace required')
    binding = report.get('stream_binding', {})
    if (binding.get('compute_sha256') != expected['fused_compute_sha256']
            or binding.get('stream_reader_sha256') != READER_SHA256
            or binding.get('pack_sources') != expected['block_stream_sources']
            or binding.get('arithmetic_changed') is not False):
        raise ValueError('Executed stream binding must match the pipeline manifest')
    return dict(passed=True, report_sha256=REPORT_SHA256, kernels=[expected],
        hardware_qualified=False, performance_qualified=False)
