"""Simulator-only native split-K candidate with unchanged mixed attention gates."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import dspark_score_bitwise
import dspark_splitk_attention
from dspark_splitk_device_audit import audit_layout
from dspark_splitk_unfused_correction import unfused_correction_scope
from dspark_splitk_fp32_build import validate as validate_factory
from dspark_splitk_attention import splitk_scope
from dspark_splitk_layout import scheduling


def main():
    if os.environ.get('QWEN_SPLITK_ATTENTION') != '1':
        raise ValueError('Explicit split-K experiment required')
    factory = validate_factory(os.environ['TT_METAL_HOME'])
    directory = Path(__file__).resolve().parent
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    original_entrypoint = dspark_score_bitwise.candidate_entrypoint
    original_execute = dspark_splitk_attention.execute_folded
    audited = False

    def execute(*args, **kwargs):
        nonlocal audited
        kwargs.update(key_chunk_size=512, max_cores_per_head=1, stripe_keys=True, fp32_dest_acc=True)
        if not audited:
            print(json.dumps(dict(stage='splitk-diagnostic-config', key_chunk_size=512,
                max_cores_per_head=1, stripe_keys=True, fp32_dest_acc=True,
                performance_qualified=False)), flush=True)
            kwargs['audit'] = audit_layout
            audited = True
        with patch.dict(os.environ, {'QWEN_SPLITK_FP32_INTERMEDIATES': '1',
                'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1'}):
            return original_execute(*args, **kwargs)

    def entrypoint(unused_script):
        return original_entrypoint(Path(__file__).resolve())

    with unfused_correction_scope(), splitk_scope(), patch.object(dspark_score_bitwise, 'candidate_entrypoint', entrypoint), \
            patch.object(dspark_splitk_attention, 'execute_folded',
                wraps=execute) as execution:
        runpy.run_path(str(directory / 'dspark-center-tile-fill-probe.py'), run_name='__main__')
        if execution.call_count < 4:
            raise ValueError('Split-K adapter must execute every eager fixture, not the old candidate')
    report = json.loads(output.read_text())
    report.update(candidate='native-decode-fp32-reciprocal-diagnostic', performance_qualified=False,
        splitk_factory=factory,
        draft_attention_backend='scaled_dot_product_attention_decode',
        draft_math='native decode reduction; prefill scalar selectors do not apply',
        scheduling_model=scheduling(), splitk_execution_calls=execution.call_count,
        diagnostic_override=dict(key_chunk_size=512, max_cores_per_head=1, stripe_keys=True,
            fp32_dest_acc=True, score_storage='float32', statistics_storage='bfloat16',
            arithmetic_unpack='tf32', sum_input_storage='bfloat16', normalization_input_storage='float32',
            reciprocal_storage='float32',
            probability_rounding='unchanged-fp32-exponent-storage',
            purpose='retain precise exponents while investigating reduction accuracy'))
    report['candidate_sources'].update({name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ('dspark_splitk_attention.py', 'dspark_splitk_layout.py',
            'dspark_splitk_device_audit.py', 'dspark_splitk_unfused_correction.py',
            'dspark_splitk_fp32_mask.py', 'dspark_splitk_copy_formats.py', 'dspark_splitk_sum_input.py',
            'dspark_splitk_final_input.py', 'dspark_splitk_precise_exp.py', 'dspark_splitk_output_rounding.py',
            'dspark_splitk_reciprocal_storage.py',
            'dspark_splitk_fp32_factory.py', 'dspark_splitk_fp32_build.py', Path(__file__).name)})
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
