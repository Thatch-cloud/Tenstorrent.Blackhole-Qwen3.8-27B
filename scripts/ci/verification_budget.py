"""Optimistic bounds from captured verifier timings, not measured speculative throughput."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def verification_budget(report, *, rows=8, target_tps=200.0):
    if type(rows) is not int or rows < 1 or not math.isfinite(target_tps) or target_tps <= 0:
        raise ValueError('Positive row count and finite target throughput required')
    if report.get('passed') is not True:
        raise ValueError('Passed hardware report required')
    if report.get('instrumented_timing'):
        raise ValueError('Uninstrumented verifier timings required for throughput budgeting')
    results = []
    for timing in report.get('timings', []):
        if timing.get('rows') != rows:
            continue
        if timing.get('instrumented_timing') or timing.get('device_profile'):
            raise ValueError('Uninstrumented verifier timings required for throughput budgeting')
        samples = [block['batch_ms'] for block in timing['blocks']]
        if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
            raise ValueError('Positive finite verifier block timings required')
        median = statistics.median(samples)
        cycle_budget = 1000 * rows / target_tps
        results.append(dict(context=timing['length'], rows=rows, verifier_median_ms=median,
            verifier_range_ms=[min(samples), max(samples)], maximum_committed_per_cycle=rows,
            zero_draft_overhead_full_acceptance_ceiling_tps=1000 * rows / median,
            cycle_budget_ms=cycle_budget, remaining_draft_commit_budget_ms=cycle_budget - median,
            minimum_verifier_reduction_fraction=max(0.0, 1 - cycle_budget / median)))
    if not results:
        raise ValueError('Requested verifier row count not present')
    return dict(scope=__doc__, assumptions=['all proposals accepted', 'zero draft/selection/commit overhead',
        'one newly committed token per verification row', 'historical static verifier costs remain applicable'],
        source_timing_scope=report.get('timing_scope'), target_tps=target_tps, bounds=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=8)
    options = parser.parse_args()
    data = options.report.read_bytes()
    result = verification_budget(json.loads(data), rows=options.rows)
    result.update(source=str(options.report), source_sha256=hashlib.sha256(data).hexdigest())
    options.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
