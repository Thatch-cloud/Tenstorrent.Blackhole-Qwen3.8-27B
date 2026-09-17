"""Direct normalization staging candidate under the combined draft and target simulator gates."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import subprocess

from dspark_attention_header_boundary import boundary_scope
from dspark_direct_fp32_stage import staging_scope
from dspark_normalization_direct_stage import normalization_stage_scope

from dspark_score_bitwise import bitwise_infinity_checks, candidate_entrypoint
from dspark_score_smoke_geometry import small_score_fixture
from dspark_score_sfpu import factory_scope, kernel_scope
from dspark_sum_sfpu import sum_scope
from dspark_mask_bits import mask_scope


def main():
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_LADDER_CONTEXT') != '128'
            or os.environ.get('QWEN_LADDER_SCORE_SMOKE') != '1'
            or os.environ.get('QWEN_SUM_SFPU') != '1'
            or os.environ.get('QWEN_MASK_BITS') != '1'
            or os.environ.get('QWEN_DIRECT_FP32_STAGE') != '1'
            or os.environ.get('QWEN_NORMALIZATION_DIRECT_STAGE') != '1'
            or '--hardware' in sys.argv or Path('/dev/tenstorrent').exists()):
        raise ValueError('Explicit small CPU-only sum-update simulator required')
    directory = Path(__file__).parent
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    with normalization_stage_scope(), staging_scope(diagnostic=True), boundary_scope(os.environ['TT_METAL_HOME']), mask_scope(), sum_scope(), small_score_fixture(), bitwise_infinity_checks(), factory_scope(), kernel_scope(), \
            candidate_entrypoint(Path(__file__).resolve()):
        runpy.run_path(str(directory / 'dspark-ladder-attention-probe.py'), run_name='__main__')
        if os.environ.get('QWEN_PRECISE_DRAFT_ACTIVE') != '1':
            raise ValueError('Target compilation must run while the draft header is installed')
        from native_draft_sdpa import KERNEL_DIRECTORY
        header = Path(os.environ['TT_METAL_HOME']) / KERNEL_DIRECTORY / 'compute_common.hpp'
        active_header = hashlib.sha256(header.read_bytes()).hexdigest()
        target_output = output.with_suffix('.target.json')
        subprocess.run([sys.executable, str(directory / 'target-t16-attention-64k-probe.py'),
            '--output', str(target_output)], check=True, timeout=110)
        target = json.loads(target_output.read_text())
        if (target.get('passed') is not True or target.get('closed') is not True
                or target.get('backend') != 'simulator'
                or hashlib.sha256(header.read_bytes()).hexdigest() != active_header):
            raise ValueError('Exact folded target gate under unchanged draft headers required')
    report = json.loads(output.read_text())
    report.update(candidate='normalization-direct-staging', performance_qualified=False,
        target_attention=target, shared_active_header_sha256=active_header,
        candidate_sources={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ('dspark_score_bitwise.py', 'dspark_score_smoke_geometry.py',
                'dspark_score_sfpu.py', 'dspark_score_sfpu_build.py', 'dspark_sum_sfpu.py',
                'dspark_sum_sfpu_build.py', 'dspark_mask_bits.py', 'dspark_attention_header_boundary.py', 'dspark_direct_fp32_stage.py', 'dspark_normalization_direct_stage.py', Path(__file__).name)})
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
