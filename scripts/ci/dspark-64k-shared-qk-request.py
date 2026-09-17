"""Complete 64K split-K, fused-MLP and shared-Q/K correctness screen."""

import json
import os
from pathlib import Path
import runpy
import sys

from dspark_splitk_combined_build import digest
from dspark_64k_shared_qk import admission, shared_factory_scope


def main():
    import fused_t16_scope
    from gdn_shared_qk_scope import scoped_shared_qk

    directory = Path(__file__).parent
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh shared-Q/K combined audit required')
    evidence = admission(directory, os.environ['TT_METAL_HOME'])
    dependencies = (Path(__file__).name, 'dspark_64k_shared_qk.py', 'gdn_shared_qk_scope.py',
        'gdn_shared_qk_gate.py')
    sources = {name: digest(directory / name) for name in dependencies}
    records = []
    failure = None
    try:
        with shared_factory_scope(fused_t16_scope, scoped_shared_qk, evidence, records):
            runpy.run_path(str(directory / 'dspark-64k-mlp-request.py'), run_name='__main__')
        if len(records) != 1 or sources != {name: digest(directory / name) for name in dependencies}:
            raise ValueError('One complete source-stable shared-Q/K request required')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report['shared_qk_64k'] = dict(records=records, sources=sources,
                performance_qualified=False, serving_qualified=False)
            if failure is not None:
                report.update(passed=False, correctness_screen_passed=False, shared_qk_error=failure)
            output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
