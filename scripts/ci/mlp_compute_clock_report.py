"""Validate compute-clock coverage without treating processor intervals as throughput."""

import argparse
import hashlib
import json
from pathlib import Path

from mlp_clock_report import validate_numerics
from mlp_compute_clock import ZONES


def validate(report, *, backend):
    validate_numerics(report, backend=backend)
    expected = [(chip, processor, 0, name, block, subblock)
        for chip in range(2) for processor in range(3) for name, predicate, block, subblock in ZONES]
    captures = report.get('compute_clock_samples', [])
    if [record.get('label') for record in captures] != ['eager', 'replay-0', 'replay-0', 'replay-1', 'replay-2']:
        raise ValueError('Eager and every replay compute sample set required')
    for capture in captures:
        samples = capture.get('samples', [])
        actual = [tuple(sample.get(field) for field in ('chip', 'processor', 'worker', 'zone', 'block', 'subblock'))
            for sample in samples]
        if actual != expected or capture.get('poisoned_before_execution') is not True:
            raise ValueError('Complete unique chip/processor/interval coverage required')
        previous = {}
        for sample in samples:
            start, end, duration = (sample.get(field) for field in ('start_cycle', 'end_cycle', 'duration_cycles'))
            key = sample['chip'], sample['processor']
            if (any(type(value) is not int for value in (start, end, duration))
                    or start < previous.get(key, 0) or not 0 <= end - start <= 100_000_000
                    or duration != end - start):
                raise ValueError('Bounded ordered processor wall-clock intervals required')
            previous[key] = end
    return dict(passed=True, backend=backend, samples=len(captures) * len(expected),
        diagnostic_only=True, active_utilization_measured=False, committed_tg=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--backend', choices=('simulator', 'hardware'), required=True)
    options = parser.parse_args()
    raw = options.report.read_bytes()
    print(json.dumps(dict(validate(json.loads(raw), backend=options.backend),
        report_sha256=hashlib.sha256(raw).hexdigest())))
