"""Restore the winning score-layout path around the admitted combined 64K runtime."""

import json
import os
from pathlib import Path
import runpy
import sys

from dspark_hardware_gate import digest
from dspark_64k_score_timed import measurement_scope
from dspark_score_pair import paired_scope


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


def validate_pair_execution(report, records, pairs):
    records = json.loads(json.dumps(records))
    pairs = json.loads(json.dumps(pairs))
    requests = report.get('request_checks', [])
    if (len(records) != 1 or len(pairs) != 2 or len(requests) != 2
            or [pair['summary']['arm'] for pair in pairs] != ['control', 'score_layout']
            or any(pair['request'] != request for pair, request in zip(pairs, requests, strict=True))
            or requests[0].get('score_64k_reintegration') is not None
            or requests[1].get('score_64k_reintegration') != records[0]
            or records[0].get('restored') is not True or records[0].get('calls', 0) < 1
            or any(report.get(key) is not True for key in ('passed', 'full_request_passed', 'closed_cleanly'))):
        raise ValueError('One complete native control and one actually executed fused-score candidate required')


def main():
    if (os.environ.get('QWEN_MATCHED_SCORE_LAYOUT') != '1'
            or os.environ.get('QWEN_SPLITK_WORKERS', '8') != '8'
            or os.environ.get('QWEN_HISTORY_WAIT_PROFILE', '0') != '0'):
        raise ValueError('Isolated eight-worker combined score restoration required')
    directory = Path(__file__).parent
    paired = os.environ.get('QWEN_MATCHED_SCORE_PAIR', '0')
    if paired not in ('0', '1'):
        raise ValueError('Explicit paired selection required')
    paired = paired == '1'
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
        'dspark_score_layout_hardware_audit.py', 'dspark_score_pair.py')

    def sources():
        return {name: digest(directory / name) for name in names}

    before = sources()
    records, pairs, failure = [], [], None

    def candidate(module):
        return measurement_scope(module, TracedDSparkDevice, ScoreLayoutArm, audit, records)

    try:
        scope = paired_scope(full_dspark_request, candidate, pairs,
            lambda record: print(json.dumps(record), flush=True)) if paired else candidate(full_dspark_request)
        with scope:
            runpy.run_path(str(entry), run_name='__main__')
        report = json.loads(output.read_text())
        if paired:
            validate_pair_execution(report, records, pairs)
        else:
            validate_score_execution(report, records)
        if before != sources():
            raise ValueError('Score restoration sources changed')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        report = json.loads(output.read_text()) if output.exists() else {}
        report['matched_score'] = dict(records=records, failure=failure, sources=before,
            sources_after=sources(), score_path='paired' if paired else 'fused',
            pairs=pairs, repetitions_per_arm=1 if paired else 2, serving_qualified=False)
        if 'matched_timed' in report:
            report['matched_timed']['score_path'] = 'paired' if paired else 'fused'
            report['matched_timed']['score_path_override'] = 'matched_score_request.py'
        if paired:
            report.update(pp=None, committed_tg=None, performance_qualified=False, diagnostic_only=True)
            for arm in report.get('request_comparison', {}).get('arms', {}).values():
                arm.update(pp=None, committed_tg=None,
                    qualification_scope='Different arms; use matched_score.pairs, never pooled throughput')
            if 'request_summary' in report:
                report['request_summary'].update(pp=None, committed_tg=None)
        if failure is not None:
            report.update(passed=False, full_request_passed=False, pp=None, committed_tg=None)
        output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
