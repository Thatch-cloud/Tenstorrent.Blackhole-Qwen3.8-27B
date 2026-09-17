"""Full 64K numerical comparison of simulator-qualified direct FP32 staging."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import dspark_score_bitwise
from dspark_attention_header_boundary import boundary_scope
from dspark_direct_fp32_stage import staging_scope
from dspark_center_fill_gate import qualify
from dspark_center_tile_fill import center_fill_scope
from dspark_normalization_direct_stage import normalization_stage_scope


def main():
    if (os.environ.get('QWEN_DSPARK_DIRECT_FP32_STAGE') != '1'
            or os.environ.get('QWEN_DSPARK_NORMALIZATION_DIRECT_STAGE') != '1'
            or os.environ.get('QWEN_DSPARK_CENTER_TILE_FILL') != '1'):
        raise ValueError('Explicit direct staging hardware experiment required')
    directory = Path(__file__).parent
    qualify(directory, directory / 'dspark-center-tile-fill.json')
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    original_entrypoint = dspark_score_bitwise.candidate_entrypoint

    def entrypoint(unused_script):
        return original_entrypoint(Path(__file__).resolve())

    with center_fill_scope(), normalization_stage_scope(), staging_scope(), boundary_scope(os.environ['TT_METAL_HOME']), \
            patch.object(dspark_score_bitwise, 'candidate_entrypoint', entrypoint):
        runpy.run_path(str(directory / 'dspark-mask-bits-hardware-probe.py'), run_name='__main__')
    report = json.loads(output.read_text())
    report['candidate'] = 'center-tile-fill'
    report['candidate_sources'].update({name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ('dspark_direct_fp32_stage.py', 'dspark_center_fill_gate.py', 'dspark_center_tile_fill.py', 'dspark_normalization_direct_stage.py',
            'dspark_attention_header_boundary.py', Path(__file__).name)})
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
