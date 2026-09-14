"""Simulator-only native split-K candidate with unchanged mixed attention gates."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import subprocess
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
    original_run = subprocess.run
    audited = False

    def execute(*args, **kwargs):
        nonlocal audited
        kwargs.update(key_chunk_size=256, max_cores_per_head=2, stripe_keys=False, fp32_dest_acc=True)
        if not audited:
            print(json.dumps(dict(stage='splitk-diagnostic-config', key_chunk_size=256,
                max_cores_per_head=2, stripe_keys=False, fp32_dest_acc=True,
                performance_qualified=False)), flush=True)
            kwargs['audit'] = audit_layout
            audited = True
        with patch.dict(os.environ, {'QWEN_SPLITK_FP32_INTERMEDIATES': '1',
                'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1'}):
            return original_execute(*args, **kwargs)

    def entrypoint(unused_script):
        return original_entrypoint(Path(__file__).resolve())

    def run(*args, **kwargs):
        command = args[0] if args else kwargs.get('args')
        if isinstance(command, (list, tuple)) and len(command) > 1 and Path(command[1]).name == 'target-t16-attention-64k-probe.py':
            environment = dict(kwargs.get('env', os.environ))
            for name in ('TT_METAL_DPRINT_CORES', 'TT_METAL_DPRINT_RISCVS',
                    'TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC', 'TT_METAL_DPRINT_FILE'):
                environment.pop(name, None)
            kwargs['env'] = environment
            print(json.dumps(dict(stage='splitk-target-gate', draft_diagnostics_inherited=False)), flush=True)
        return original_run(*args, **kwargs)

    with unfused_correction_scope(), splitk_scope(), patch.object(dspark_score_bitwise, 'candidate_entrypoint', entrypoint), \
            patch.object(subprocess, 'run', side_effect=run), \
            patch.object(dspark_splitk_attention, 'execute_folded',
                wraps=execute) as execution:
        runpy.run_path(str(directory / 'dspark-center-tile-fill-probe.py'), run_name='__main__')
        if execution.call_count != 3:
            raise ValueError(f'Split-K requires two eager calls and one capture call; got {execution.call_count}')
    report = json.loads(output.read_text())
    report.update(candidate='native-decode-fp32-sfpu-row-sum-diagnostic', performance_qualified=False,
        splitk_factory=factory,
        draft_attention_backend='scaled_dot_product_attention_decode',
        draft_math='native decode reduction; prefill scalar selectors do not apply',
        scheduling_model=scheduling(), splitk_execution_calls=execution.call_count,
        diagnostic_override=dict(key_chunk_size=256, max_cores_per_head=2, stripe_keys=False,
            fp32_dest_acc=True, score_storage='float32', statistics_storage='bfloat16',
            local_denominator_storage='float32', transfer_statistics_storage='bfloat16',
            arithmetic_unpack='tf32-with-explicit-fp32-sum-copy', sum_input_storage='float32', normalization_input_storage='float32',
            reciprocal_storage='float32',
            temporary_per_row_sum_audit=False,
            local_numerator_add='native-fpu',
            probability_rounding='unchanged-fp32-exponent-storage',
            purpose='validate two-worker merging with FP32 local denominators and original key order'))
    report['candidate_sources'].update({name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ('dspark_splitk_attention.py', 'dspark_splitk_layout.py',
            'dspark_splitk_device_audit.py', 'dspark_splitk_unfused_correction.py',
            'dspark_splitk_fp32_mask.py', 'dspark_splitk_copy_formats.py', 'dspark_splitk_sum_input.py',
            'dspark_splitk_final_input.py', 'dspark_splitk_precise_exp.py', 'dspark_splitk_output_rounding.py',
            'dspark_splitk_reciprocal_storage.py',
            'dspark_splitk_sum_audit.py',
            'dspark_splitk_merge_exp.py',
            'dspark_splitk_accumulator_add.py',
            'dspark_splitk_denominator.py',
            'dspark_splitk_fp32_factory.py', 'dspark_splitk_fp32_build.py', Path(__file__).name)})
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
