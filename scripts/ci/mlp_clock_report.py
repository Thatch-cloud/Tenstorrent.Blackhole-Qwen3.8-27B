"""Validate complete bounded diagnostic coverage, never infer throughput."""

import argparse
import hashlib
import json
from pathlib import Path

from mlp_clock_samples import ZONES


def validate_numerics(report, *, backend):
    if (backend not in ('simulator', 'hardware') or report.get('backend') != backend
            or report.get('passed') is not True or report.get('math_approx_mode') is not True
            or report.get('missing_execution_rejected') is not True
            or 'committed_tg' not in report or report['committed_tg'] is not None
            or report.get('timings') != [] or report.get('error')):
        raise ValueError('Passed untimed diagnostic and missing-execution negative control required')
    if report.get('checks') != [dict(rows=16, chip=chip, exact=True) for chip in range(2)]:
        raise ValueError('Both T16 eager outputs must match native')
    weights = report.get('weight_checks', [])
    expected_weights = [(projection, chip) for projection in ('gate', 'up') for chip in range(2)]
    if (len(weights) != 4 or [(value.get('projection'), value.get('chip')) for value in weights] != expected_weights
            or any(value.get('exact') is not True or value.get('source_exact') is not True
                or value.get('pages') != 43520 or value.get('workers') != 64
                or value.get('mismatched_words') != 0 for value in weights)):
        raise ValueError('Complete byte-exact weight comparisons required')
    replays = report.get('trace_replays', [])
    checks = [dict(arm=arm, repetition=repetition, pattern=pattern, chip=chip, exact=True)
        for repetition, pattern in enumerate((0, 1, 0)) for arm in ('control', 'fused') for chip in range(2)]
    negative = [dict(arm=arm, chip=chip, stale_input_detected=True)
        for arm in ('control', 'fused') for chip in range(2)]
    if (len(replays) != 1 or replays[0].get('rows') != 16 or replays[0].get('passed') is not True
            or replays[0].get('checks') != checks or replays[0].get('negative_controls') != negative
            or replays[0].get('timings') != []):
        raise ValueError('All changed-input native and fused replay checks required')


def validate(report, *, backend):
    validate_numerics(report, backend=backend)
    expected_samples = []
    for role, workers in (('input', (0, 1)), ('weights', (0,))):
        for chip in range(2):
            for worker in workers:
                indices = (0, 4) if role == 'input' and worker == 1 else (0, 1, 2, 3)
                expected_samples.extend((role, chip, worker, ZONES[role][index][1]) for index in indices)
    captures = report.get('clock_samples', [])
    if [capture.get('label') for capture in captures] != ['eager', 'replay-0', 'replay-0', 'replay-1', 'replay-2']:
        raise ValueError('Eager and all four replay sample sets required')
    for capture in captures:
        samples = capture.get('samples', [])
        actual = [(sample.get('role'), sample.get('chip'), sample.get('worker'), sample.get('zone')) for sample in samples]
        if actual != expected_samples or capture.get('poisoned_before_execution') is not True:
            raise ValueError('Complete unique two-chip sample coverage required')
        previous = {}
        for sample in samples:
            start, end, duration = (sample.get(key) for key in ('start_cycle', 'end_cycle', 'duration_cycles'))
            if (any(type(value) is not int for value in (start, end, duration))
                    or not 0 <= start <= end < 2**64 or not 0 <= duration <= 100_000_000
                    or end - start != duration):
                raise ValueError('Bounded internally consistent raw cycle samples required')
            identity = sample['role'], sample['chip'], sample['worker']
            if start < previous.get(identity, 0):
                raise ValueError('Reader sample order changed')
            previous[identity] = end
    return dict(passed=True, backend=backend, capture_sets=5, samples=100,
        exact_eager_outputs=2, exact_replay_outputs=12, diagnostic_only=True, committed_tg=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--backend', choices=('simulator', 'hardware'), required=True)
    options = parser.parse_args()
    if (options.directory / 'fused-batch.exit-status').read_text().strip() != '0':
        raise ValueError('Clean process exit required')
    raw = (options.directory / 'fused-batch.json').read_bytes()
    result = validate(json.loads(raw), backend=options.backend)
    print(json.dumps(dict(result, report_sha256=hashlib.sha256(raw).hexdigest())))


if __name__ == '__main__':
    main()
