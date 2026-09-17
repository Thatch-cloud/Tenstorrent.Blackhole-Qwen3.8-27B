"""Fresh combined-model audit for the hardware-qualified sixteen-worker path."""

import json
import os
from pathlib import Path
import runpy
import sys

import dspark_splitk_attention
from dspark_hardware_gate import digest
from splitk_worker_scope import worker_scope
from splitk_workers_hardware_gate import qualify


def main():
    if (os.environ.get('QWEN_SPLITK_WORKERS') != '16'
            or os.environ.get('QWEN_DSPARK_SFPU_REQUEST_SCREEN') != '1'
            or os.environ.get('QWEN_DSPARK_SFPU_TIMED', '0') != '0'):
        raise ValueError('Fresh sixteen-worker combined correctness screen required')
    directory = Path(__file__).parent
    evidence = qualify(directory, directory / 'dspark-workers-64k-hardware.json')
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh worker-limit request report required')
    names = (Path(__file__).name, 'splitk_worker_scope.py', 'splitk_workers_hardware_gate.py')

    def sources():
        return {name: digest(directory / name) for name in names}

    before = sources()
    records, failure = [], None
    try:
        with worker_scope(dspark_splitk_attention, records):
            runpy.run_path(str(directory / 'matched_combined_request.py'), run_name='__main__')
        report = json.loads(output.read_text())
        scopes = report['splitk_combined']['scopes']
        if (len(scopes) != 1 or not records or len(records) != scopes[0]['attention_calls']
                or before != sources()):
            raise ValueError('Every combined draft attention execution must use the qualified worker override')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        report = json.loads(output.read_text()) if output.exists() else {}
        report['workers_combined'] = dict(hardware_admission=evidence, calls=records,
            sources=before, sources_after=sources(), failure=failure,
            full_request_qualified=False, performance_qualified=False, serving_qualified=False)
        if failure is not None:
            report.update(passed=False, correctness_screen_passed=False)
        output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
