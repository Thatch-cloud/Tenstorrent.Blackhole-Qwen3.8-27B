"""Validate execution evidence without mistaking finite approximation for model quality."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from dspark_projection_precision_stage import PROJECTIONS


def validate(report, *, projection='self_attn.q_proj.weight'):
    if projection not in PROJECTIONS:
        raise ValueError('Supported expected projection required')
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('stage') != 'complete' or report.get('backend') != 'simulator'
            or report.get('projection') != projection
            or report.get('component_execution_only') is not True
            or report.get('target_correctness_qualified') is not False or report.get('committed_tg') is not None):
        raise ValueError('Complete query-projection execution-only simulator report required')
    if not report.get('sources') or report['sources'] != report.get('sources_after'):
        raise ValueError('Stable source fingerprints required')
    differences = report.get('differences', [])
    if [(value.get('case'), value.get('chip')) for value in differences] != [(case, chip) for case in range(2) for chip in range(2)]:
        raise ValueError('Both inputs and chips must report approximation errors')
    for value in differences:
        if value.get('finite') is not True or type(value.get('exact')) is not bool:
            raise ValueError('Explicit finite and exactness observations required')
        for field in ('max_abs', 'rms_error', 'reference_rms'):
            number = value.get(field)
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise ValueError('Finite nonnegative numerical metrics required')
        if value['rms_error'] > value['max_abs'] or (value['exact'] and value['max_abs'] != 0):
            raise ValueError('Inconsistent approximation metrics')
    expected = [dict(ordinal=ordinal, case=case, chip=chip, exact=True, poison_replaced=True)
        for ordinal, case in enumerate((1, 0, 1)) for chip in range(2)]
    if report.get('replay_checks') != expected:
        raise ValueError('Complete poisoned changed-input replay required')
    integrity = [dict(case=case, mode=mode, exact=True)
        for mode, cases in (('eager', (0, 1)), ('replay', (1, 0, 1))) for case in cases]
    if report.get('integrity_checks') != integrity:
        raise ValueError('Every eager/replay must preserve inputs, weights and bindings')
    return dict(component_execution_passed=True, target_correctness_qualified=False,
        committed_tg=None, differences=differences, replay_checks=6)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--projection', choices=PROJECTIONS, default='self_attn.q_proj.weight')
    options = parser.parse_args()
    raw = options.report.read_bytes()
    print(json.dumps(dict(validate(json.loads(raw), projection=options.projection),
        report_sha256=hashlib.sha256(raw).hexdigest()), indent=2))
