"""Fresh combined T16 split-K request audit, retaining existing target/state gates."""

import json
from pathlib import Path
import runpy
import sys

from dspark_splitk_combined_build import digest
from dspark_splitk_combined_runtime import entry_scope
from dspark_audit_observer import observe_audit


def main():
    directory = Path(__file__).parent
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists() or '--preflight' in sys.argv:
        raise ValueError('Fresh post-build combined request audit required')
    dependencies = (Path(__file__).name, 'dspark_splitk_combined_runtime.py',
        'dspark_splitk_combined_build.py', 'dspark_splitk_hardware_gate.py', 'dspark_audit_observer.py')
    sources = {name: digest(directory / name) for name in dependencies}
    records = []
    failure = None
    try:
        import dspark_prepared_proposal
        from dflash_request_runtime import DFlashRequestRuntime

        def emit(record):
            print(json.dumps(record), flush=True)

        with entry_scope(records), observe_audit(dspark_prepared_proposal, DFlashRequestRuntime, emit):
            runpy.run_path(str(directory / 'dspark-target-hardware.py'), run_name='__main__')
        if len(records) != 1 or records[0]['attention_calls'] == 0 or not records[0]['kernel_restored']:
            raise ValueError('One executed and cleanly restored combined request scope required')
        if sources != {name: digest(directory / name) for name in dependencies}:
            raise ValueError('Combined request integration source changed')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report.update(splitk_combined=dict(scopes=records, sources=sources,
                full_request_qualified=False, performance_qualified=False, serving_qualified=False))
            if failure is not None:
                report.update(passed=False, splitk_combined_error=failure)
            output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
