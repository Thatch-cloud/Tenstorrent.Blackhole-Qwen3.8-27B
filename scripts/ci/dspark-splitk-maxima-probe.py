"""Simulator-only maxima precision ablation; existing numerical gates unchanged."""

import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import dspark_score_bitwise
from dspark_hardware_gate import digest
from dspark_splitk_maxima_build import builder_identity


def main():
    directory = Path(__file__).parent
    original = dspark_score_bitwise.candidate_entrypoint
    with builder_identity(), patch.object(dspark_score_bitwise, 'candidate_entrypoint',
            lambda unused: original(Path(__file__).resolve())):
        runpy.run_path(str(directory / 'dspark-splitk-probe.py'), run_name='__main__')
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    report = json.loads(output.read_text())
    report['candidate'] = 'splitk-local-maxima-fp32-ablation'
    report['performance_qualified'] = False
    report['candidate_sources'].update({name: digest(directory / name) for name in (
        'dspark_splitk_maxima_factory.py', 'dspark_splitk_maxima_build.py', Path(__file__).name)})
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
