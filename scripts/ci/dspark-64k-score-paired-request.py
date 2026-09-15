"""One loaded model, matched complete control and score-layout candidate requests."""

import json
import os
from pathlib import Path
import runpy
import sys

from dspark_splitk_combined_build import digest
from dspark_64k_score_timed import identity_scope, measurement_scope
from dspark_score_pair import paired_scope


def main():
    directory = Path(__file__).parent
    if sys.argv[1:] == ['--source-preflight']:
        with identity_scope():
            runpy.run_path(str(directory / 'dspark-64k-shared-timed-request.py'), run_name='__main__')
        return
    if os.environ.get('QWEN_SCORE_PAIRED') != '1':
        raise ValueError('Explicit paired runtime diagnostic required')
    import full_dspark_request
    from dspark_prepared_proposal import TracedDSparkDevice
    from dspark_score_layout_scope import ScoreLayoutArm
    from dspark_score_layout_hardware_audit import audit

    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh paired report required')
    dependencies = (Path(__file__).name, 'dspark_score_pair.py', 'dspark_64k_score_timed.py')
    sources = {name: digest(directory / name) for name in dependencies}
    records, score_records, failure = [], [], None
    def candidate(module):
        return measurement_scope(module, TracedDSparkDevice, ScoreLayoutArm, audit, score_records)
    try:
        with identity_scope(), paired_scope(full_dspark_request, candidate, records,
                lambda record: print(json.dumps(record), flush=True)):
            runpy.run_path(str(directory / 'dspark-64k-shared-timed-request.py'), run_name='__main__')
        if len(score_records) != 1 or sources != {name: digest(directory / name) for name in dependencies}:
            raise ValueError('One source-stable candidate and one control required')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report['score_pair'] = dict(records=records, score_records=score_records, sources=sources,
                failure=failure, order=['control', 'score_layout'], repetitions_per_arm=1,
                performance_qualified=False, serving_qualified=False)
            report.update(pp=None, committed_tg=None, performance_qualified=False)
            for arm in report.get('request_comparison', {}).get('arms', {}).values():
                arm.update(pp=None, committed_tg=None,
                    qualification_scope='Different paired arms; use score_pair records, not pooled throughput')
            if failure is not None:
                report.update(passed=False, full_request_passed=False)
            output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
