"""Full-context numerical/replay worker-limit check using the unchanged maxima binary."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import dspark_splitk_hardware_build as build
import matched_context_attention
from dspark_hardware_gate import digest
from dspark_splitk_maxima_hardware import hardware_identity
from splitk_worker_scope import worker_scope
from splitk_workers_gate import qualify


@contextmanager
def cache_only():
    original = build.find_entry

    def find(cache, inputs):
        identity, manifest = original(cache, inputs)
        if manifest is None:
            raise ValueError('Worker-only experiment requires the exact existing maxima binary cache')
        return identity, manifest

    with patch.object(build, 'find_entry', find):
        yield


def main():
    if (os.environ.get('QWEN_SPLITK_WORKERS') != '16'
            or os.environ.get('QWEN_MATCHED_CONTEXT') != '65536'):
        raise ValueError('Explicit full-64K sixteen-worker experiment required')
    directory = Path(__file__).parent
    admission = qualify(directory, directory / 'dspark-splitk-workers-simulator.json')
    if sys.argv[1:] == ['--build']:
        with cache_only(), hardware_identity():
            build.main()
        return
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh worker-limit hardware report required')
    names = (Path(__file__).name, 'splitk_worker_scope.py', 'splitk_workers_gate.py')
    before = {name: digest(directory / name) for name in names}
    records, failure = [], None
    try:
        with hardware_identity(), worker_scope(matched_context_attention, records):
            runpy.run_path(str(directory / 'matched-context-attention-probe.py'), run_name='__main__')
        if len(records) != 3 or before != {name: digest(directory / name) for name in names}:
            raise ValueError('All eager/captured executions must use the unchanged worker override')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report['worker_experiment'] = dict(simulator=admission, calls=records,
                sources=before, sources_after={name: digest(directory / name) for name in names}, failure=failure)
            if failure is not None:
                report['passed'] = False
            output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
