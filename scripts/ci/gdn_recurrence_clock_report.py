"""Independent recurrence diagnostic coverage checks; no throughput qualification."""

import argparse
import hashlib
import json
from pathlib import Path

from gdn_recurrence_clock import ZONES


MODES = ('eager', 'replay_1', 'replay_2', 'replay_0')


def validate(report):
    required = ('passed', 'closed_cleanly', 'norm_unchanged', 'state_math_unchanged',
        'shared_qk_preparation', 'recurrence_clock', 'missing_execution_rejected',
        'missing_execution_rejected_after_replay')
    if (any(report.get(key) is not True for key in required)
            or report.get('stage') != 'complete' or report.get('backend') != 'simulator'
            or report.get('rows') != 16 or not report.get('sources')
            or report['sources'] != report.get('sources_after') or report.get('numerical_failures')
            or report.get('hardware_qualified') is not False or report.get('timing_qualified') is not False
            or report.get('committed_tg') is not None):
        raise ValueError('Clean unchanged-source simulator diagnostic required')
    for field, operands in (('checks', 3), ('immutable_checks', 6)):
        expected = [dict(mode=mode, operand=operand, chip=chip, exact=True)
            for mode in MODES for operand in range(operands) for chip in (0, 1)]
        if report.get(field) != expected:
            raise ValueError('Complete exact recurrence matrix required: ' + field)
    kernels = report.get('generated_kernels', [])
    if len(kernels) != 1 or kernels[0].get('token') != 8:
        raise ValueError('One middle-token instrumented recurrence required')
    for field in ('control_sha256', 'candidate_sha256'):
        value = kernels[0].get(field)
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in '0123456789abcdef' for character in value)):
            raise ValueError('Generated kernel fingerprints required')
    if kernels[0]['control_sha256'] == kernels[0]['candidate_sha256']:
        raise ValueError('Diagnostic kernel must differ from control')
    captures = report.get('recurrence_clock_samples', [])
    if [capture.get('label') for capture in captures] != list(MODES):
        raise ValueError('Every eager/replay sample set required')
    expected = [(chip, processor, 0, 8, name)
        for chip in range(2) for processor in range(3) for name, _ in ZONES]
    for capture in captures:
        samples = capture.get('samples', [])
        actual = [tuple(sample.get(field) for field in ('chip', 'processor', 'worker', 'token', 'zone'))
            for sample in samples]
        if actual != expected or capture.get('poisoned_before_execution') is not True:
            raise ValueError('Complete poisoned two-chip phase coverage required')
        previous = {}
        for sample in samples:
            start, end, duration = (sample.get(field) for field in ('start_cycle', 'end_cycle', 'duration_cycles'))
            key = sample['chip'], sample['processor']
            if (any(type(value) is not int for value in (start, end, duration))
                    or start < previous.get(key, 0) or not 0 <= end - start <= 100_000_000
                    or duration != end - start):
                raise ValueError('Ordered bounded phase intervals required')
            previous[key] = end
    return dict(passed=True, backend='simulator', samples=len(captures) * len(expected),
        diagnostic_only=True, source_admission_required=True,
        active_utilization_measured=False, committed_tg=None, performance_qualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    options = parser.parse_args()
    raw = options.report.read_bytes()
    print(json.dumps(dict(validate(json.loads(raw)), report_sha256=hashlib.sha256(raw).hexdigest())))
