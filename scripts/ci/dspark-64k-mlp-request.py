"""Audit fused T16 MLP inside the complete qualified split-K request runtime."""

import json
from pathlib import Path
import runpy
import sys

from dspark_splitk_combined_build import digest


def main():
    import full_dspark_request
    from dspark_64k_mlp_scope import audit_scope
    from fused_t16_scope import FusedT16Arm
    from models.tt_transformers.tt.ccl import tt_all_reduce

    directory = Path(__file__).parent
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh MLP reintegration report required')
    dependencies = (Path(__file__).name, 'dspark_64k_mlp_scope.py', 'fused_t16_scope.py',
        'fused_t16_admission.py', 'fused_1d.py')
    sources = {name: digest(directory / name) for name in dependencies}
    failure = None
    try:
        with audit_scope(full_dspark_request, FusedT16Arm, tt_all_reduce):
            runpy.run_path(str(directory / 'dspark-splitk-combined-request.py'), run_name='__main__')
        report = json.loads(output.read_text())
        if (len(report.get('request_checks', [])) != 1
                or 'mlp_64k_reintegration' not in report['request_checks'][0]
                or sources != {name: digest(directory / name) for name in dependencies}):
            raise ValueError('Complete source-stable MLP reintegration evidence required')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report['mlp_reintegration_sources'] = sources
            if failure is not None:
                report.update(passed=False, correctness_screen_passed=False, mlp_reintegration_error=failure)
            output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
