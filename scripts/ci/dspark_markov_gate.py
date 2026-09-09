"""Reconcile every Markov simulator coordinate and immutable input/source prerequisite."""

import argparse
import importlib.util
import json
import math
from pathlib import Path

from dspark_markov_fixture import expected_manifest


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


def qualify(report, *, sources, native, exit_status):
    if (exit_status.strip() != '0' or report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('error') or report.get('backend') != 'simulator' or report.get('stage') != 'complete'
            or report.get('target_integrated') is not False or report.get('eligible_for_hardware') is not False
            or not sources or not native or report.get('sources') != sources or report.get('sources_after') != sources
            or report.get('native_sources') != native or report.get('native_sources_after') != native):
        raise ValueError('Complete clean simulator result, outer exit0 and current unchanged sources required')
    vocabulary, steps = report.get('vocabulary'), report.get('proposals')
    if type(vocabulary) is not int or type(steps) is not int or (vocabulary, steps) not in ((64, 3), (248320, 7)):
        raise ValueError('Qualified toy3 or learned full-vocabulary7 proposal geometry required')
    if report.get('accuracy_policy') != 'FP32 bias and sum; no SGLang BF16 bitwise claim':
        raise ValueError('Explicit prototype arithmetic policy required')
    learned = vocabulary == 248320
    if report.get('fixture') != (expected_manifest() if learned else None):
        raise ValueError('Both hash-pinned full learned matrices required for the learned gate')
    patterns = 2 if learned else 3
    replay_order = [*range(patterns), 0]
    coordinates(report.get('eager_checks'), ('pattern', 'step', 'chip'),
        {(pattern, step, chip) for pattern in range(patterns) for step in range(steps) for chip in range(2)},
        ('token_exact', 'full_vocabulary_close'))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--exit-status', type=Path, required=True)
    parser.add_argument('--metal-root', type=Path, required=True)
    options = parser.parse_args()
    spec = importlib.util.spec_from_file_location('dspark_markov_probe_source', Path(__file__).with_name('dspark-markov-probe.py'))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    result = qualify(json.loads(options.report.read_text()), sources=probe.source_hashes(),
        native=probe.fingerprints(options.metal_root), exit_status=options.exit_status.read_text())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
