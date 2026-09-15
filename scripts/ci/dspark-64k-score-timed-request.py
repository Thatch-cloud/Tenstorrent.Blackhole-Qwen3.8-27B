"""Two complete clean 64K score-layout requests after immutable combined admission."""

import json
from pathlib import Path
import runpy
import sys

from dspark_splitk_combined_build import digest
from dspark_64k_score_timed import identity_scope, measurement_scope


def main():
    directory = Path(__file__).parent
    if sys.argv[1:] == ['--source-preflight']:
        with identity_scope():
            runpy.run_path(str(directory / 'dspark-64k-shared-timed-request.py'), run_name='__main__')
        return
    import full_dspark_request
    from dspark_prepared_proposal import TracedDSparkDevice
    from dspark_score_layout_scope import ScoreLayoutArm
    from dspark_score_layout_hardware_audit import audit

    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh combined score-layout timing required')
    dependencies = (Path(__file__).name, 'dspark_64k_score_timed.py', 'dspark_64k_score_gate.py')
    sources = {name: digest(directory / name) for name in dependencies}
    records, failure = [], None
    try:
        with identity_scope(), measurement_scope(full_dspark_request, TracedDSparkDevice,
                ScoreLayoutArm, audit, records):
            runpy.run_path(str(directory / 'dspark-64k-shared-timed-request.py'), run_name='__main__')
        if len(records) != 2 or sources != {name: digest(directory / name) for name in dependencies}:
            raise ValueError('Two complete source-stable score-layout timing requests required')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report['score_layout_timed'] = dict(records=records, sources=sources, serving_qualified=False)
            if failure is not None:
                report.update(passed=False, full_request_passed=False, score_layout_timed_error=failure)
            output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
