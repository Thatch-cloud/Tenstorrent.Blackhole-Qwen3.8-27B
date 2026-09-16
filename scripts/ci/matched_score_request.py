"""Restore the winning score-layout path around the admitted combined 64K runtime."""

import json
import os
from pathlib import Path
import runpy
import sys

from dspark_hardware_gate import digest
from dspark_64k_score_timed import measurement_scope


def validate_score_execution(report, records):
    requests = report.get('request_checks', [])
    if (len(records) != 2 or len(requests) != 2
            or report.get('passed') is not True or report.get('full_request_passed') is not True
            or report.get('closed_cleanly') is not True):
        raise ValueError('Two complete clean fused-score requests required')
    for request, record in zip(requests, records, strict=True):
        if (record.get('restored') is not True or record.get('calls', 0) < 1
                or request.get('score_64k_reintegration') != record):
            raise ValueError('Fused score layout must execute and restore in each complete request')


def main():
    if (os.environ.get('QWEN_MATCHED_SCORE_LAYOUT') != '1'
            or os.environ.get('QWEN_SPLITK_WORKERS', '8') != '8'
            or os.environ.get('QWEN_HISTORY_WAIT_PROFILE', '0') != '0'):
        raise ValueError('Isolated eight-worker combined score restoration required')
    directory = Path(__file__).parent
    entry = directory / 'matched_combined_timed_request.py'
    if sys.argv[1:] == ['--source-preflight']:
        runpy.run_path(str(entry), run_name='__main__')
        return
    import full_dspark_request
    from dspark_prepared_proposal import TracedDSparkDevice
    from dspark_score_layout_scope import ScoreLayoutArm
    from dspark_score_layout_hardware_audit import audit

    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh combined score-restoration report required')
    names = (Path(__file__).name, 'dspark_64k_score_timed.py', 'dspark_score_layout_scope.py',
        'dspark_score_layout_hardware_audit.py')

    def sources():
        return {name: digest(directory / name) for name in names}

    before = sources()
    records, failure = [], None
    try:
        with measurement_scope(full_dspark_request, TracedDSparkDevice, ScoreLayoutArm, audit, records):
            runpy.run_path(str(entry), run_name='__main__')
        report = json.loads(output.read_text())
        validate_score_execution(report, records)
        if before != sources():
            raise ValueError('Score restoration sources changed')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        report = json.loads(output.read_text()) if output.exists() else {}
        report['matched_score'] = dict(records=records, failure=failure, sources=before,
            sources_after=sources(), score_path='fused', serving_qualified=False)
        if 'matched_timed' in report:
            report['matched_timed']['score_path'] = 'fused'
            report['matched_timed']['score_path_override'] = 'matched_score_request.py'
        if failure is not None:
            report.update(passed=False, full_request_passed=False, pp=None, committed_tg=None)
        output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
