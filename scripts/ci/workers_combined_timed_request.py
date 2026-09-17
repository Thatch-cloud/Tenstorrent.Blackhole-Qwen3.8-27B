"""Measure complete requests with the audited sixteen-worker draft runtime."""

import json
import os
from pathlib import Path
import runpy
import sys

import dspark_splitk_attention
from dspark_hardware_gate import digest
from splitk_worker_scope import worker_scope
from workers_combined_gate import validate_calls
from workers_combined_timed import identity_scope


def main():
    if (os.environ.get('QWEN_SPLITK_WORKERS') != '16'
            or os.environ.get('QWEN_DSPARK_SFPU_TIMED') != '1'
            or os.environ.get('QWEN_DSPARK_SFPU_REQUEST_SCREEN') != '0'):
        raise ValueError('Explicit clean sixteen-worker timing required')
    directory = Path(__file__).parent
    entry = directory / 'matched_combined_timed_request.py'
    if sys.argv[1:] == ['--source-preflight']:
        with identity_scope():
            runpy.run_path(str(entry), run_name='__main__')
        return
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh sixteen-worker timing report required')
    names = (Path(__file__).name, 'workers_combined_gate.py', 'workers_combined_timed.py',
        'splitk_worker_scope.py')

    def sources():
        return {name: digest(directory / name) for name in names}

    before = sources()
    records, failure = [], None
    try:
        with identity_scope(), worker_scope(dspark_splitk_attention, records):
            runpy.run_path(str(entry), run_name='__main__')
        report = json.loads(output.read_text())
        scopes = report['splitk_timed']['scopes']
        if len(scopes) != 1 or before != sources():
            raise ValueError('One source-stable combined timing runtime required')
        validate_calls(records, scopes[0]['attention_calls'])
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        report = json.loads(output.read_text()) if output.exists() else {}
        report['workers_timed'] = dict(calls=records, sources=before, sources_after=sources(),
            failure=failure, serving_qualified=False)
        if failure is not None:
            report.update(passed=False, full_request_passed=False, pp=None, committed_tg=None)
        output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
