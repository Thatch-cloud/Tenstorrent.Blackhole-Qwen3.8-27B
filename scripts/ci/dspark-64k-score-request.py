"""Complete 64K score-layout audit with split-K, fused MLP and shared Q/K."""

import json
from pathlib import Path
import runpy
import sys

from dspark_splitk_combined_build import digest
from dspark_64k_score_scope import score_scope


def main():
    import full_dspark_request
    from dspark_prepared_proposal import TracedDSparkDevice
    from dspark_score_layout_scope import ScoreLayoutArm
    from dspark_score_layout_hardware_audit import audit

    directory = Path(__file__).parent
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh combined score-layout audit required')
    dependencies = (Path(__file__).name, 'dspark_64k_score_scope.py', 'dspark_score_layout_scope.py',
        'dspark_score_layout_hardware_audit.py', 'dspark_score_layout_hardware_gate.py')
    sources = {name: digest(directory / name) for name in dependencies}
    records, failure = [], None
    try:
        with score_scope(full_dspark_request, TracedDSparkDevice, ScoreLayoutArm, audit, records):
            runpy.run_path(str(directory / 'dspark-64k-shared-qk-request.py'), run_name='__main__')
        if len(records) != 1 or sources != {name: digest(directory / name) for name in dependencies}:
            raise ValueError('One complete source-stable score-layout request required')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report['score_layout_64k'] = dict(records=records, sources=sources,
                performance_qualified=False, serving_qualified=False)
            if failure is not None:
                report.update(passed=False, correctness_screen_passed=False, score_layout_error=failure)
            output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
