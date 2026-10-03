"""Reconcile every Markov simulator coordinate and immutable input/source prerequisite."""

import argparse
import importlib.util
import json
import math
from pathlib import Path

from dspark_markov_fixture import expected_manifest
from dspark_native_reference import POLICY


def coordinates(entries, fields, expected, flags):
    if not isinstance(entries, list):
        raise ValueError('Complete check list required')
    seen = set()
    expected_types = tuple(type(value) for value in next(iter(expected)))
    for entry in entries:
        if not isinstance(entry, dict) or any(entry.get(flag) is not True for flag in flags):
            raise ValueError('Every recorded numerical/replay/ownership check must pass')
        if any(field not in entry for field in fields):
            raise ValueError('Missing audit coordinate')
        if any(type(entry[field]) is not kind for field, kind in zip(fields, expected_types, strict=True)):
            raise ValueError('Audit coordinate types must match the declared matrix')
        key = tuple(entry[field] for field in fields)
        if key not in expected or key in seen:
            raise ValueError('Duplicate or unexpected audit coordinate')
        seen.add(key)
    if seen != expected:
        raise ValueError('Missing required audit coordinates')


def qualify_structure(report, *, sources, native, exit_status, policy, eager_flags):
    if (exit_status.strip() != '0' or report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('error') or report.get('backend') != 'simulator' or report.get('stage') != 'complete'
            or report.get('target_integrated') is not False or report.get('eligible_for_hardware') is not False
            or not sources or not native or report.get('sources') != sources or report.get('sources_after') != sources
            or report.get('native_sources') != native or report.get('native_sources_after') != native):
        raise ValueError('Complete clean simulator result, outer exit0 and current unchanged sources required')
    vocabulary, steps = report.get('vocabulary'), report.get('proposals')
    if type(vocabulary) is not int or type(steps) is not int or (vocabulary, steps) not in ((64, 3), (248320, 7)):
        raise ValueError('Qualified toy3 or learned full-vocabulary7 proposal geometry required')
    if report.get('accuracy_policy') != policy:
        raise ValueError('Explicit prototype arithmetic policy required')
    learned = vocabulary == 248320
    if report.get('fixture') != (expected_manifest() if learned else None):
        raise ValueError('Both hash-pinned full learned matrices required for the learned gate')
    patterns = 2 if learned else 3
    replay_order = [*range(patterns), 0]
    coordinates(report.get('eager_checks'), ('pattern', 'step', 'chip'),
        {(pattern, step, chip) for pattern in range(patterns) for step in range(steps) for chip in range(2)},
        eager_flags)
    coordinates(report.get('replay_checks'), ('repetition', 'pattern', 'step', 'chip'),
        {(repetition, pattern, step, chip) for repetition, pattern in enumerate(replay_order)
            for step in range(steps) for chip in range(2)}, ('token_and_scores_exact', 'bindings_stable'))
    coordinates(report.get('input_checks'), ('phase', 'ordinal', 'tensor', 'chip'),
        {(phase, ordinal, tensor, chip) for phase, count in (('eager', patterns), ('replay', len(replay_order)))
            for ordinal in range(count) for tensor in range(2) for chip in range(2)}, ('exact',))
    coordinates(report.get('weight_checks'), ('phase', 'tensor', 'chip'),
        {(phase, tensor, chip) for phase in ('before', 'after') for tensor in range(2) for chip in range(2)}, ('exact',))
    coordinates(report.get('stale_controls'), ('chip',), {(0,), (1,)}, ('missing_update_detected',))
    tokens = {}
    for entry in report['eager_checks']:
        token, error = entry.get('token'), entry.get('max_abs')
        if (type(token) is not int or not 0 <= token < vocabulary or type(error) not in (float, int)
                or not math.isfinite(error) or error < 0):
            raise ValueError('Finite score diagnostics and full-vocabulary IDs required')
        key = entry['pattern'], entry['step']
        if key in tokens and tokens[key] != token:
            raise ValueError('Replicated chips must agree on each proposal')
        tokens[key] = token
    counts = {name: len(report[name]) for name in
        ('eager_checks', 'replay_checks', 'input_checks', 'weight_checks', 'stale_controls')}
    return dict(passed=True, checks=sum(counts.values()), counts=counts, vocabulary=vocabulary, proposals=steps,
        learned_matrices=learned, full_vocabulary=True, device_feedback=True,
        worst_score_error=max(entry['max_abs'] for entry in report['eager_checks']),
        target_integrated=False, eligible_for_hardware=False,
        scope='Replicated selector only; synthetic base logits, no learned backbone, fabric, acceptance or TG evidence')


def qualify(report, *, sources, native, exit_status):
    if report.get('native_arithmetic_reference') or 'fp32_replacement_qualified' in report:
        raise ValueError('Native arithmetic reports cannot replace the original FP32 gate')
    return qualify_structure(report, sources=sources, native=native, exit_status=exit_status,
        policy='FP32 bias and sum; no SGLang BF16 bitwise claim', eager_flags=('token_exact', 'full_vocabulary_close'))


def qualify_native(report, *, sources, native, exit_status):
    result = qualify_structure(report, sources=sources, native=native, exit_status=exit_status,
        policy=POLICY, eager_flags=('token_exact', 'full_vocabulary_exact'))
    if (report.get('native_arithmetic_reference') is not True or report.get('fp32_replacement_qualified') is not False
            or report.get('fp32_diagnostic_tolerances') != dict(rtol=1e-4, atol=1e-4)):
        raise ValueError('Separate native proposal policy and original FP32 diagnostic tolerances required')
    vocabulary = result['vocabulary']
    by_coordinate = {(entry['pattern'], entry['step'], entry['chip']): entry for entry in report['eager_checks']}
    for entry in report['eager_checks']:
        if entry['max_abs'] != 0:
            raise ValueError('Native arithmetic requires exact full-vocabulary scores, not a widened tolerance')
        for field in ('previous', 'fp32_previous'):
            if type(entry.get(field)) is not int or not 0 <= entry[field] < vocabulary:
                raise ValueError('Both native and independent FP32 predecessor IDs required')
        for field in ('same_input_fp32', 'fp32_trajectory'):
            diagnostic = entry.get(field)
            if not isinstance(diagnostic, dict):
                raise ValueError('Complete independent and same-input FP32 diagnostics required')
            error, mismatched = diagnostic.get('max_abs'), diagnostic.get('mismatched')
            close, token = diagnostic.get('full_vocabulary_close'), diagnostic.get('token')
            if (type(error) not in (int, float) or not math.isfinite(error) or error < 0
                    or type(mismatched) is not int or not 0 <= mismatched <= vocabulary
                    or type(close) is not bool or close != (mismatched == 0)
                    or (error == 0 and mismatched != 0) or type(token) is not int or not 0 <= token < vocabulary):
                raise ValueError('Finite, consistent original-tolerance FP32 diagnostics and greedy IDs required')
        pattern, step, chip = (entry[field] for field in ('pattern', 'step', 'chip'))
        if step:
            previous = by_coordinate[pattern, step - 1, chip]
            expected_native, expected_fp32 = previous['token'], previous['fp32_trajectory']['token']
        else:
            anchors = (1596, vocabulary - 1) if result['learned_matrices'] else (2, vocabulary - 1, 0)
            expected_native = expected_fp32 = anchors[pattern]
        if entry['previous'] != expected_native or entry['fp32_previous'] != expected_fp32:
            raise ValueError('Each arithmetic policy must retain its own sequential predecessor trajectory')
        if entry['previous'] == entry['fp32_previous'] and entry['same_input_fp32'] != entry['fp32_trajectory']:
            raise ValueError('Identical predecessor inputs must have identical FP32 diagnostics')
        peer = by_coordinate[pattern, step, 1 - chip]
        if any(entry[field] != peer[field] for field in
                ('previous', 'fp32_previous', 'same_input_fp32', 'fp32_trajectory')):
            raise ValueError('Replicated chips must agree on retained FP32 diagnostics')
    entries = report['eager_checks']
    result.update(accuracy_policy=POLICY, fp32_replacement_qualified=False,
        worst_same_input_fp32_error=max(entry['same_input_fp32']['max_abs'] for entry in entries),
        fp32_score_checks_failed=sum(not entry['fp32_trajectory']['full_vocabulary_close'] for entry in entries),
        fp32_token_checks_differed=sum(entry['token'] != entry['fp32_trajectory']['token'] for entry in entries))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--exit-status', type=Path, required=True)
    parser.add_argument('--metal-root', type=Path, required=True)
    parser.add_argument('--native-reference', action='store_true', help='Qualify only the separate native proposal policy')
    options = parser.parse_args()
    spec = importlib.util.spec_from_file_location('dspark_markov_probe_source', Path(__file__).with_name('dspark-markov-probe.py'))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    qualifier = qualify_native if options.native_reference else qualify
    result = qualifier(json.loads(options.report.read_text()), sources=probe.source_hashes(),
        native=probe.fingerprints(options.metal_root), exit_status=options.exit_status.read_text())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
