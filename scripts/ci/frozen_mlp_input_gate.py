"""Source-bound admission of the retained full-K activation-prefetch simulation."""

import hashlib
import json
from pathlib import Path


REPORT_SHA256 = '72ffca8dcf0af26d8f9743410142267aace4f065cb91c67c69ed868fb5efb5a3'
CANDIDATE_SHA256 = 'ac50a904fa4457bedce0504a694df2e43db781723fe96c6382005576cd1e518a'
READER_SHA256 = '41448b6c257f3b406fc38434032bab6ed5cb891fc984d27a61797f6f021022d4'
CONTROL_SHA256 = '4c215d945e723ef3d8c2be1757e7c170bf21eebb01e204359243d766c6c7cf93'


def validate_report(report):
    if (report.get('passed') is not True or report.get('backend') != 'simulator'
            or report.get('buffer_candidate_sha256') != CANDIDATE_SHA256
            or report.get('input_reader_candidate_sha256') != READER_SHA256
            or report.get('checks') != [dict(rows=16, chip=chip, exact=True) for chip in (0, 1)]):
        raise ValueError('Exact T16 activation-prefetch simulator evidence required')
    traces = report.get('trace_replays', [])
    if len(traces) != 1 or traces[0].get('rows') != 16 or traces[0].get('passed') is not True:
        raise ValueError('One complete T16 trace matrix required')
    expected = [dict(arm=arm, repetition=repetition, pattern=pattern, chip=chip, exact=True)
        for repetition, pattern in enumerate((0, 1, 0)) for arm in ('control', 'fused') for chip in (0, 1)]
    negative = [dict(arm=arm, chip=chip, stale_input_detected=True)
        for arm in ('control', 'fused') for chip in (0, 1)]
    if traces[0].get('checks') != expected or traces[0].get('negative_controls') != negative:
        raise ValueError('Complete changed-input replay and stale-input controls required')
    weights = report.get('weight_checks', [])
    if (len(weights) != 4 or {(item.get('projection'), item.get('chip')) for item in weights}
            != {(projection, chip) for projection in ('gate', 'up') for chip in (0, 1)}
            or any(item.get('pages') != 43520 or item.get('mismatched_words') != 0
                or item.get('exact') is not True or item.get('source_exact') is not True for item in weights)):
        raise ValueError('Complete packed-weight integrity required')


def qualify(directory, report_path):
    directory = Path(directory)
    raw = Path(report_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact retained input-prefetch simulation required')
    report = json.loads(raw)
    validate_report(report)
    candidate = directory / 'frozen-mlp-input-candidate'
    for path, expected in ((directory / 'fused_1d.py', CONTROL_SHA256),
            (candidate / 'fused_1d.py', CANDIDATE_SHA256),
            (candidate / 'fused_1d_input.cpp', READER_SHA256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Projection source differs: ' + str(path))
    from fused_t16_admission import qualify_simulator
    control = qualify_simulator(directory)
    expected = dict(next(kernel for kernel in control['kernels'] if kernel['token_rows'] == 16))
    expected.update(full_k_input_prefetch=True, input_buffer_tiles=160)
    expected['reader_sha256'] = dict(expected['reader_sha256'])
    expected['reader_sha256']['fused_1d_input.cpp'] = READER_SHA256
    if report.get('kernels') != [expected]:
        raise ValueError('Only input coordination and capacity may change')
    for name, checksum in expected['reader_sha256'].items():
        if hashlib.sha256((candidate / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Candidate reader source differs: ' + name)
    if hashlib.sha256((directory / 'fusion_trace.py').read_bytes()).hexdigest() != report['trace_source_sha256']:
        raise ValueError('Replay validator source changed')
    return dict(report_sha256=REPORT_SHA256, kernels=[expected], passed=True,
        hardware_qualified=False, performance_qualified=False)
