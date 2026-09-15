"""Instrument the first three publications in each complete paired request."""

import json
import os
from pathlib import Path
import runpy
import sys

from captured_publication_profile import profile_publications


def main():
    entry = Path(__file__).with_name('dspark-64k-lazy-timed-request.py')
    if sys.argv[1:] == ['--source-preflight']:
        runpy.run_path(str(entry), run_name='__main__')
        return
    if os.environ.get('QWEN_PUBLICATION_PROFILE') != '1':
        raise ValueError('Explicit combined publication profiling required')
    from dspark_publication_scope import CapturedPublicationArm

    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh publication profile report required')
    records, failure = [], None
    try:
        with profile_publications(CapturedPublicationArm, records,
                lambda record: print(json.dumps(dict(stage='publication_profile', **record)), flush=True)):
            runpy.run_path(str(entry), run_name='__main__')
        if (len(records) != 6 or any(not record['passed'] for record in records)
                or [(record['request_ordinal'], record['sample']) for record in records]
                != [(ordinal, sample) for ordinal in range(2) for sample in range(3)]):
            raise ValueError('Three completed publication samples per paired request required')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report['publication_profile'] = dict(records=records, failure=failure,
                scope=__doc__, performance_qualified=False)
            report.update(pp=None, committed_tg=None, performance_qualified=False)
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
