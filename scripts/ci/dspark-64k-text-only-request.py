"""Complete combined text-only construction audit; no serving default changes."""

import json
import os
from pathlib import Path
import runpy
import sys

from dspark_splitk_combined_build import digest
from qwen_text_only_load import text_only_load


def main():
    required = ('QWEN_TEXT_ONLY_LOAD', 'QWEN_64K_SCORE_AUDIT', 'QWEN_DSPARK_SFPU_REQUEST_SCREEN')
    if (any(os.environ.get(name) != '1' for name in required)
            or os.environ.get('QWEN_DSPARK_SFPU_TIMED', '0') != '0'):
        raise ValueError('Explicit combined text-only correctness audit required')
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    directory = Path(__file__).parent
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh text-only audit report required')
    dependencies = (Path(__file__).name, 'qwen_text_only_load.py')
    sources = {name: digest(directory / name) for name in dependencies}
    records, failure = [], None
    try:
        with text_only_load(Qwen36Model, records):
            runpy.run_path(str(directory / 'dspark-64k-score-request.py'), run_name='__main__')
        if (len(records) != 1 or records[0].get('restored') is not True
                or sources != {name: digest(directory / name) for name in dependencies}):
            raise ValueError('One source-stable restored text-only construction required')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report['text_only_load'] = dict(records=records, sources=sources,
                performance_qualified=False, serving_qualified=False)
            if failure is not None:
                report.update(passed=False, correctness_screen_passed=False, text_only_load_error=failure)
            output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
