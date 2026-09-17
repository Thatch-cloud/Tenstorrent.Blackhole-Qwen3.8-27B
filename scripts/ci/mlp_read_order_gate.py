"""Exact simulator and source admission for a read-order-only T16 candidate."""

import hashlib
import json
from pathlib import Path

from mlp_weight_read_order import transform


REPORT_SHA256 = '3bac1b049aef7a12affd6463282aa773b199becfa1cddc605bb2f0074a41fbf8'


def stage_candidate(directory):
    directory = Path(directory)
    candidate = directory / 'mlp-read-order-candidate'
    candidate.mkdir()
    for name in ('fused_1d.py', 'fused_1d_input.cpp'):
        (candidate / name).write_bytes((directory / name).read_bytes())
    (candidate / 'fused_1d_weights.cpp').write_bytes(transform((directory / 'fused_1d_weights.cpp').read_text()).encode())
    return candidate


def validate_report(report):
    if (report.get('passed') is not True or report.get('backend') != 'simulator'
            or report.get('math_approx_mode') is not True or report.get('timings') != []
            or report.get('checks') != [dict(rows=16, chip=chip, exact=True) for chip in range(2)]):
        raise ValueError('Passed untimed native-exact T16 simulator evidence required')
    replays = report.get('trace_replays', [])
    expected = [dict(arm=arm, repetition=repetition, pattern=pattern, chip=chip, exact=True)
        for repetition, pattern in enumerate((0, 1, 0)) for arm in ('control', 'fused') for chip in range(2)]
    negative = [dict(arm=arm, chip=chip, stale_input_detected=True) for arm in ('control', 'fused') for chip in range(2)]
    if (len(replays) != 1 or replays[0].get('rows') != 16 or replays[0].get('passed') is not True
            or replays[0].get('checks') != expected or replays[0].get('negative_controls') != negative
            or replays[0].get('timings') != []):
        raise ValueError('Complete changed-input replay and stale-input controls required')
    weights = report.get('weight_checks', [])
    if ([(value.get('projection'), value.get('chip')) for value in weights]
            != [(projection, chip) for projection in ('gate', 'up') for chip in range(2)]
            or any(value.get('exact') is not True or value.get('source_exact') is not True
                or value.get('pages') != 43520 or value.get('workers') != 64
                or value.get('mismatched_words') != 0 for value in weights)):
        raise ValueError('Complete byte-exact packed-weight comparisons required')


def qualify(directory, evidence, expected_report_sha256=REPORT_SHA256):
    from fused_t16_admission import qualify_simulator

    directory, evidence = Path(directory), Path(evidence)
    raw = (evidence / 'fused-batch.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_report_sha256:
        raise ValueError('Exact reviewed read-order simulator report required')
    if (evidence / 'fused-batch.exit-status').read_text().strip() != '0':
        raise ValueError('Clean simulator exit required')
    if (evidence / 'simulator-runtime.txt').read_text().strip() != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9':
        raise ValueError('Pinned simulator runtime required')
    cleanup = json.loads((evidence / 'container-cleanup.json').read_text())
    if cleanup != dict(stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0):
        raise ValueError('Clean simulator container teardown required')
    report = json.loads(raw)
    validate_report(report)
    control = qualify_simulator(directory)
    baseline = next(kernel for kernel in control['kernels'] if kernel['token_rows'] == 16)
    expected = dict(baseline, reader_sha256=dict(baseline['reader_sha256']))
    candidate = directory / 'mlp-read-order-candidate'
    for name in ('fused_1d.py', 'fused_1d_input.cpp'):
        if (candidate / name).read_bytes() != (directory / name).read_bytes():
            raise ValueError('Only weight-reader issue order may differ: ' + name)
    original = (directory / 'fused_1d_weights.cpp').read_text()
    changed = transform(original).encode()
    if (candidate / 'fused_1d_weights.cpp').read_bytes() != changed:
        raise ValueError('Exact reviewed read permutation required')
    expected['reader_sha256']['fused_1d_weights.cpp'] = hashlib.sha256(changed).hexdigest()
    if report.get('kernels') != [expected]:
        raise ValueError('Arithmetic, grids, buffers or reader identity differs from simulator')
    for name, report_key in (('fused_1d.py', 'buffer_candidate_sha256'), ('fusion_trace.py', 'trace_source_sha256')):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != report.get(report_key):
            raise ValueError('Projection or replay helper differs from simulator')
    return dict(report_sha256=expected_report_sha256, kernels=[expected], passed=True,
        hardware_qualified=False, performance_qualified=False)
