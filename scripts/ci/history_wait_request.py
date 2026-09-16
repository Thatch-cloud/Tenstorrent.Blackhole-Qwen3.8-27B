"""Attribute two complete audited requests; diagnostic timings are not benchmarks."""

import json
import os
from pathlib import Path
import runpy
import sys

from captured_publication_profile import profile_publications
from dspark_hardware_gate import digest
from history_writer_profile import profile_writers
from history_input_profile import profile_inputs


def main():
    if (os.environ.get('QWEN_HISTORY_WAIT_PROFILE') != '1'
            or os.environ.get('QWEN_MATCHED_COMBINED') != '1'
            or os.environ.get('QWEN_DSPARK_SFPU_TIMED') != '1'):
        raise ValueError('Explicit matched full-request diagnostic required')
    directory = Path(__file__).parent
    workers = os.environ.get('QWEN_SPLITK_WORKERS', '8')
    if workers not in ('8', '16'):
        raise ValueError('Qualified worker configuration required')
    entry = directory / ('workers_combined_timed_request.py' if workers == '16'
        else 'matched_combined_timed_request.py')
    if sys.argv[1:] == ['--source-preflight']:
        runpy.run_path(str(entry), run_name='__main__')
        return
    import incremental_history_scope
    import torch
    from dspark_publication_scope import CapturedPublicationArm

    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh diagnostic report required')
    names = (Path(__file__).name, 'history_writer_profile.py', 'captured_publication_profile.py',
        'history_input_profile.py')

    def sources():
        return {name: digest(directory / name) for name in names}

    before = sources()
    publications, writers, inputs, failure = [], [], [], None
    try:
        with profile_publications(CapturedPublicationArm, publications, lambda record: None,
                samples_per_request=32, max_requests=2), profile_writers(incremental_history_scope, writers), \
                profile_inputs(CapturedPublicationArm, torch, inputs):
            runpy.run_path(str(entry), run_name='__main__')
        report = json.loads(output.read_text())
        requests = report['request_checks']
        blocks = sum(len(request['blocks']) for request in requests)
        if (len(requests) != 2 or len(publications) != blocks or len(writers) != blocks + 2
                or len(inputs) != blocks or any(not record['passed'] for record in publications + writers + inputs)
                or report.get('full_request_passed') is not True or before != sources()):
            raise ValueError('Every complete request publication and both warmups must be observed')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        report = json.loads(output.read_text()) if output.exists() else {}
        report['history_wait_profile'] = dict(publications=publications, writers=writers, inputs=inputs,
            failure=failure, sources=before, sources_after=sources(), added_device_fences=False,
            instrumented_pp=report.get('pp'), instrumented_tg=report.get('committed_tg'))
        report.update(pp=None, committed_tg=None, diagnostic_only=True, performance_qualified=False)
        if failure is not None:
            report.update(passed=False, full_request_passed=False)
        output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
