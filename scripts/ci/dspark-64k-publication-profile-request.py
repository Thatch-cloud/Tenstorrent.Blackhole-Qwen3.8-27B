"""Attribute one complete control; no candidate, TG acceptance or final weight audit."""

import json
import os
from pathlib import Path
import runpy
import sys

from captured_publication_profile import profile_publications
from control_profile_stop import ControlProfileComplete, stop_after_control


def main():
    entry = Path(__file__).with_name('dspark-64k-lazy-timed-request.py')
    if sys.argv[1:] == ['--source-preflight']:
        runpy.run_path(str(entry), run_name='__main__')
        return
    if os.environ.get('QWEN_PUBLICATION_PROFILE') != '1':
        raise ValueError('Explicit combined publication profiling required')
    from dspark_publication_scope import CapturedPublicationArm
    import full_dspark_request

    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh publication profile report required')
    records, requests, failure, completed = [], [], None, False
    try:
        with profile_publications(CapturedPublicationArm, records,
                lambda record: print(json.dumps(dict(stage='publication_profile', **record)), flush=True),
                samples_per_request=32, max_requests=1), stop_after_control(full_dspark_request, requests):
            runpy.run_path(str(entry), run_name='__main__')
        raise ValueError('Control diagnostic stop was not reached')
    except ControlProfileComplete:
        if (len(requests) != 1 or len(records) != len(requests[0]['blocks'])
                or not records or any(not record['passed'] for record in records)
                or not output.exists()):
            raise ValueError('Every control publication must be retained')
        report = json.loads(output.read_text())
        if report.get('closed_cleanly') is not True or report.get('checkpoint_closed') is not True:
            raise ValueError('Diagnostic must release devices and checkpoint cleanly')
        completed = True
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report['publication_profile'] = dict(records=records, requests=requests, failure=failure,
                diagnostic_complete=completed, scope=__doc__, performance_qualified=False)
            report.update(pp=None, committed_tg=None, performance_qualified=False,
                diagnostic_only=True, passed=False, full_request_passed=False)
            for record in report.get('score_pair', {}).get('records', []):
                summary = record['summary']
                summary['instrumented_committed_tg'] = summary.pop('committed_tg', None)
                summary['committed_tg'] = None
                summary['instrumented'] = True
            if failure is not None:
                report.update(passed=False, full_request_passed=False)
            output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
