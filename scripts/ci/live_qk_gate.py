"""Fail-closed source and comparison matrix gates for the live-query experiment."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


SOURCES = ('draft-live-qk-probe.py', 'draft_live_qk.py', 'draft_live_qk_compute.cpp',
    'draft_live_qk_io.cpp', 'draft_dot.py', 'draft_dot_compute.cpp', 'draft_dot_io.cpp',
    'draft_attention.py', 'draft_row_sum.py', 'draft_row_sum_compute.cpp', 'draft_row_sum_io.cpp', 'live_qk_gate.py')
NATIVE_SOURCES = ('tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary_bcast.h',
    'tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h',
    'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h')


def source_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCES}


def native_hashes(root):
    return {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() for name in NATIVE_SOURCES}


def require_matrix(rows, fields, expected, flags):
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Missing comparison rows')
    if any(any(type(row.get(field)) is not (str if field == 'component' else int) for field in fields) for row in rows):
        raise ValueError('Comparison indices must have exact types')
    actual = [tuple(row.get(field) for field in fields) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Missing or duplicated comparison matrix')
    if any(any(row.get(flag) is not True for flag in flags(row)) for row in rows):
        raise ValueError('Inexact comparison or failed invariant')


def qualify_correctness(report, context, sources, native):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('context') != context or context not in (31, 2048)
            or report.get('attention') is not True or report.get('sources') != sources
            or report.get('native_sources') != native or set(native) != set(NATIVE_SOURCES)):
        raise ValueError('Failed, incomplete or source-mismatched attention evidence')
    components = ('scores', 'probabilities', 'output')
    require_matrix(report.get('eager_checks'), ('pattern', 'chip', 'component'),
        {(pattern, chip, name) for pattern in range(2) for chip in range(2) for name in components},
        lambda row: ('exact_live', 'padding_zero') if row['component'] == 'scores' else ('exact_all_rows',))
    require_matrix(report.get('replay_checks'), ('repetition', 'pattern', 'arm', 'chip', 'component'),
        {(repetition, pattern, arm, chip, name) for repetition, pattern in enumerate((0, 1, 0))
            for arm in range(2) for chip in range(2) for name in components}, lambda row: ('exact',))
    require_matrix(report.get('negative_controls'), ('arm', 'chip'),
        {(arm, chip) for arm in range(2) for chip in range(2)}, lambda row: ('stale_detected',))


def qualify_simulator(report, context, sources, native):
    if report.get('backend') != 'simulator':
        raise ValueError('Simulator evidence required before hardware')
    qualify_correctness(report, context, sources, native)


def timing_summary(samples):
    require_matrix(samples, ('pattern', 'block', 'order', 'arm'),
        {(pattern, block, order, arm) for pattern in range(2) for block in range(3)
            for order, arm in enumerate((0, 1, 1, 0))}, lambda row: ('outputs_exact', 'inputs_unchanged', 'bindings_stable'))
    for sample in samples:
        duration = sample.get('ms')
        if (type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0
                or type(sample.get('replays')) is not int or sample['replays'] != 50):
            raise ValueError('Positive finite latency and exactly50 replays required')
    ratios = []
    for pattern in range(2):
        for block in range(3):
            values = [median(sample['ms'] for sample in samples
                if sample['pattern'] == pattern and sample['block'] == block and sample['arm'] == arm)
                for arm in (0, 1)]
            ratios.append(values[0] / values[1])
    return dict(control_ms=median(sample['ms'] for sample in samples if sample['arm'] == 0),
        candidate_ms=median(sample['ms'] for sample in samples if sample['arm'] == 1),
        block_speedups=ratios, eligible_for_learned_integration=all(ratio > 1.02 for ratio in ratios))


def qualify_hardware(report, sources):
    if report.get('backend') != 'hardware' or report.get('fixture_sha256') is not None:
        raise ValueError('Synthetic component hardware report required')
    qualify_correctness(report, report.get('context'), sources, report.get('native_sources', {}))
    simulator = json.loads(Path(__file__).with_name(f"live-qk-simulator-{report['context']}.json").read_text())
    qualify_simulator(simulator, report['context'], sources, report['native_sources'])
    if report.get('simulator_report_sha256') != hashlib.sha256(Path(__file__).with_name(
            f"live-qk-simulator-{report['context']}.json").read_bytes()).hexdigest():
        raise ValueError('Simulator report provenance mismatch')
    summary = timing_summary(report.get('timing_samples'))
    if report.get('timing_summary') != summary:
        raise ValueError('Timing summary differs from raw samples')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware-result', type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(qualify_hardware(json.loads(options.hardware_result.read_text()), source_hashes())))
