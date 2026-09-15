"""Two clean EOS requests with split-K, fused MLP and shared-Q/K recurrence."""

import json
import os
from pathlib import Path
import runpy
import sys

from dspark_splitk_combined_build import digest
from dspark_64k_shared_qk import admission
from dspark_64k_shared_timed import identity_scope, factory_scope


def main():
    directory = Path(__file__).parent
    if sys.argv[1:] == ['--source-preflight']:
        with identity_scope():
            runpy.run_path(str(directory / 'dspark-64k-mlp-timed-request.py'), run_name='__main__')
        return
    import fused_t16_scope
    from gdn_shared_qk_scope import scoped_shared_qk

    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh shared-Q/K timing report required')
    evidence = admission(directory, os.environ['TT_METAL_HOME'])
    dependencies = (Path(__file__).name, 'dspark_64k_shared_timed.py', 'dspark_64k_shared_gate.py')
    sources = {name: digest(directory / name) for name in dependencies}
    records, failure = [], None
    try:
        with identity_scope(), factory_scope(fused_t16_scope, scoped_shared_qk, evidence, records):
            runpy.run_path(str(directory / 'dspark-64k-mlp-timed-request.py'), run_name='__main__')
        if len(records) != 2 or sources != {name: digest(directory / name) for name in dependencies}:
            raise ValueError('Two source-stable shared-Q/K timing requests required')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report['shared_qk_timed'] = dict(records=records, sources=sources, serving_qualified=False)
            if failure is not None:
                report.update(passed=False, full_request_passed=False, shared_qk_timed_error=failure)
            output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
