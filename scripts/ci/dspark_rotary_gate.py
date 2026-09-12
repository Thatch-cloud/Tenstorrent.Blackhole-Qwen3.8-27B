"""Independent reconciliation of the DSpark rotary simulator matrix; no full-model qualification."""

import argparse
import importlib.util
import json
import math
from pathlib import Path

from dspark_intake import FILES
from dspark_markov_gate import coordinates
from dspark_rotary_device import CASES, COMPOSED_POLICY, POLICY


def qualify(report, *, sources, native, cpu_report_sha256, exit_status, composed=False):
    if type(composed) is not bool:
        raise ValueError('Explicit boolean rotary variant required')
    policy = COMPOSED_POLICY if composed else POLICY
    if (exit_status.strip() != '0' or report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('error') or report.get('stage') != 'complete' or report.get('backend') != 'simulator'
            or report.get('target_integrated') is not False or report.get('eligible_for_hardware') is not False
            or report.get('accuracy_policy') != policy or report.get('config_sha256') != FILES['config.json'][1]
            or not cpu_report_sha256 or report.get('cpu_report_sha256') != cpu_report_sha256
            or not sources or not native or report.get('sources') != sources or report.get('sources_after') != sources
            or report.get('native_sources') != native or report.get('native_sources_after') != native):
        raise ValueError('Clean simulator result, unchanged current sources and pinned YaRN CPU prerequisite required')
    cases = report.get('cases')
    if (cases != [list(case) for case in CASES] or any(type(value) is not int for case in cases for value in case)):
        raise ValueError('All local query, short-key and 4K-key geometries required')
    coordinates(report.get('eager_checks'), ('case', 'pattern', 'chip'),
        {(case, pattern, chip) for case in range(3) for pattern in range(4) for chip in range(2)}, ('full_padded_close',))
    replay_order = (0, 1, 2, 3, 0)
    coordinates(report.get('replay_checks'), ('case', 'repetition', 'pattern', 'chip'),
        {(case, repetition, pattern, chip) for case in range(3) for repetition, pattern in enumerate(replay_order)
            for chip in range(2)}, ('exact', 'bindings_stable'))
    coordinates(report.get('input_checks'), ('case', 'phase', 'ordinal', 'tensor', 'chip'),
        {(case, phase, ordinal, tensor, chip) for case in range(3) for phase, count in (('eager', 4), ('replay', 5))
            for ordinal in range(count) for tensor in range(3) for chip in range(2)}, ('exact',))
    coordinates(report.get('dependency_controls'), ('case', 'control', 'chip'),
        {(case, control, chip) for case in range(3) for control in ('positions', 'padding') for chip in range(2)}, ())
    for entry in report['dependency_controls']:
        flag = 'detected' if entry['control'] == 'positions' else 'isolated'
        if entry.get(flag) is not True:
            raise ValueError('Position updates must affect valid outputs; poisoned padding must not')
    coordinates(report.get('stale_controls'), ('case', 'chip'), {(case, chip) for case in range(3) for chip in range(2)},
        ('missing_update_detected',))
    for entry in report['eager_checks']:
        error, valid_error, exact = entry.get('max_abs'), entry.get('valid_max_abs'), entry.get('cpu_bitwise_exact')
        if (any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in (error, valid_error))
                or valid_error > error or type(exact) is not bool or (exact and error != 0)):
            raise ValueError('Finite full/valid-row errors and truthful bitwise diagnostics required')
        if composed and exact is not True:
            raise ValueError('Composed rotary requires bitwise CPU equality on every padded output')
    counts = {name: len(report[name]) for name in
        ('eager_checks', 'replay_checks', 'input_checks', 'dependency_controls', 'stale_controls')}
    return dict(passed=True, checks=sum(counts.values()), counts=counts, accuracy_policy=policy,
        worst_cpu_error=max(entry['max_abs'] for entry in report['eager_checks']),
        worst_valid_cpu_error=max(entry['valid_max_abs'] for entry in report['eager_checks']),
        bitwise_cpu_checks=sum(entry['cpu_bitwise_exact'] for entry in report['eager_checks']),
        target_integrated=False, eligible_for_hardware=False,
        scope='Synthetic rotary heads only; no learned projections, attention, target verifier, quality or TG evidence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--exit-status', type=Path, required=True)
    parser.add_argument('--metal-root', type=Path, required=True)
    parser.add_argument('--composed', action='store_true')
    options = parser.parse_args()
    spec = importlib.util.spec_from_file_location('dspark_rotary_probe', Path(__file__).with_name('dspark-rotary-probe.py'))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    result = qualify(json.loads(options.report.read_text()), sources=probe.source_hashes(),
        native=probe.fingerprints(options.metal_root), cpu_report_sha256=probe.CPU_REPORT_SHA256,
        exit_status=options.exit_status.read_text(), composed=options.composed)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
