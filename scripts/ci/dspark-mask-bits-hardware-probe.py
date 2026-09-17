"""Full 64K mask correctness with the unchanged combined sum-update library."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import dspark_score_bitwise
from dspark_mask_bits import mask_scope
from dspark_mask_bits_gate import qualify


def main():
    if os.environ.get('QWEN_DSPARK_MASK_BITS') != '1':
        raise ValueError('Explicit mask hardware experiment required')
    directory = Path(__file__).parent
    qualify(directory, directory / 'dspark-mask-bits.json')
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    original_entrypoint = dspark_score_bitwise.candidate_entrypoint

    def entrypoint(unused_script):
        return original_entrypoint(Path(__file__).resolve())

    with mask_scope(), patch.object(dspark_score_bitwise, 'candidate_entrypoint', entrypoint):
        runpy.run_path(str(directory / 'dspark-sum-sfpu-hardware-probe.py'), run_name='__main__')
    report = json.loads(output.read_text())
    report['candidate'] = 'bitwise-negative-infinity-mask'
    report['candidate_sources'].update({name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ('dspark_mask_bits.py', 'dspark_mask_bits_gate.py', Path(__file__).name)})
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
