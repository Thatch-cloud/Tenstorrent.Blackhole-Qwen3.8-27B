"""Sixteen-worker maxima simulator screen; not a model throughput result."""

import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import dspark_score_bitwise
import dspark_splitk_attention
from dspark_hardware_gate import digest
from dspark_splitk_maxima_build import builder_identity
from splitk_worker_scope import worker_scope


def main():
    if os.environ.get('QWEN_SPLITK_WORKERS') != '16' or os.environ.get('QWEN_SPLITK_MAXIMA') != '1':
        raise ValueError('Explicit maxima worker-limit simulator experiment required')
    directory = Path(__file__).parent
    original = dspark_score_bitwise.candidate_entrypoint
    records = []
    with builder_identity(), worker_scope(dspark_splitk_attention, records), \
            patch.object(dspark_score_bitwise, 'candidate_entrypoint', lambda unused: original(Path(__file__).resolve())):
        runpy.run_path(str(directory / 'dspark-splitk-probe.py'), run_name='__main__')
    if len(records) != 3:
        raise ValueError('Two eager executions and one capture must use sixteen workers')
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    report = json.loads(output.read_text())
    report.update(candidate='splitk-maxima-worker-limit16', worker_calls=records, performance_qualified=False)
    report['diagnostic_override']['max_cores_per_head'] = 16
    report['diagnostic_override']['local_maximum_storage'] = 'float32'
    report['candidate_sources'].update({name: digest(directory / name) for name in (
        'dspark_splitk_maxima_factory.py', 'dspark_splitk_maxima_build.py', 'splitk_worker_scope.py', Path(__file__).name)})
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
