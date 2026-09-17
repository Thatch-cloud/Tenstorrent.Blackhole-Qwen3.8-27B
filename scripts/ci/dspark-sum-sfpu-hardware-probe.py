"""Full 64K numerical fixture for sum update using the cached combined library."""

import hashlib
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import dspark_score_bitwise
import dspark_score_sfpu_hardware
from dspark_sum_sfpu_hardware import hardware_scope


def main():
    directory = Path(__file__).parent
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    original_entrypoint = dspark_score_bitwise.candidate_entrypoint

    def entrypoint(unused_script):
        return original_entrypoint(Path(__file__).resolve())

    with patch.object(dspark_score_sfpu_hardware, 'hardware_scope', hardware_scope), \
            patch.object(dspark_score_bitwise, 'candidate_entrypoint', entrypoint):
        runpy.run_path(str(directory / 'dspark-score-sfpu-hardware-probe.py'), run_name='__main__')
    report = json.loads(output.read_text())
    report['candidate'] = 'sfpu-online-sum-update'
    report['candidate_sources'].update({name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ('dspark_sum_sfpu.py', 'dspark_sum_sfpu_gate.py',
            'dspark_sum_sfpu_hardware.py', Path(__file__).name)})
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
