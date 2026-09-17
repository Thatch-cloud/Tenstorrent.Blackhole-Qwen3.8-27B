"""Launch the clean combined MLP route without altering the audited control files."""

import json
from pathlib import Path
import runpy
import sys

from dspark_splitk_combined_build import digest
from dspark_64k_mlp_timed import identity_scope, measurement_scope


def main():
    directory = Path(__file__).parent
    if sys.argv[1:] == ['--source-preflight']:
        with identity_scope():
            runpy.run_path(str(directory / 'dspark-splitk-timed-request.py'), run_name='__main__')
        return
    import full_dspark_request
    from fused_t16_scope import FusedT16Arm
    from models.tt_transformers.tt.ccl import tt_all_reduce

    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh MLP timing report required')
    dependencies = (Path(__file__).name, 'dspark_64k_mlp_timed.py', 'dspark_64k_mlp_gate.py')
    sources = {name: digest(directory / name) for name in dependencies}
    failure = None
    try:
        with identity_scope(), measurement_scope(full_dspark_request, FusedT16Arm, tt_all_reduce):
            runpy.run_path(str(directory / 'dspark-splitk-timed-request.py'), run_name='__main__')
        report = json.loads(output.read_text())
        if (len(report.get('request_checks', [])) != 2
                or any('mlp_64k_reintegration' not in request for request in report['request_checks'])
                or sources != {name: digest(directory / name) for name in dependencies}):
            raise ValueError('Two complete source-stable MLP timing requests required')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report['mlp_timing_sources'] = sources
            if failure is not None:
                report.update(passed=False, full_request_passed=False, mlp_timing_error=failure)
            output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
