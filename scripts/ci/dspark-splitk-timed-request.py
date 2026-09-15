"""Two clean full-response requests using the audited combined split-K runtime."""

import json
from pathlib import Path
import runpy
import sys

from dspark_splitk_combined_build import digest
from dspark_splitk_timed import entry_scope


def main():
    directory = Path(__file__).parent
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists() or '--preflight' in sys.argv:
        raise ValueError('Fresh post-build split-K timing output required')
    dependencies = (Path(__file__).name, 'dspark_splitk_timed.py', 'dspark_splitk_request_gate.py')
    sources = {name: digest(directory / name) for name in dependencies}
    records = []
    failure = None
    try:
        with entry_scope(directory, records):
            runpy.run_path(str(directory / 'dspark-target-hardware.py'), run_name='__main__')
        if sources != {name: digest(directory / name) for name in dependencies}:
            raise ValueError('Timing integration source changed')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report['splitk_timed'] = dict(scopes=records, sources=sources, serving_qualified=False)
            if failure is not None:
                report.update(passed=False, full_request_passed=False, splitk_timed_error=failure)
            output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
