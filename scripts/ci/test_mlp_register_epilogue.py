from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from frozen_recipe_context import REVISION
from fused_1d import BF16_PRODUCT, fused_compute
from mlp_register_epilogue import CAST, END, FINAL, START, TAIL, activation_control, adapt_projection, transform


def fixture():
    source = '''void kernel_main() {
                            if (last_out) {
                                native_output();
                            } else {
                                tile_regs_commit();
                                preserve_partial_accumulation();
                            }
}
'''
    return fused_compute(source, pairs_per_worker=3)


class RegisterEpilogueTests(unittest.TestCase):
    def test_activation_diagnostic_keeps_original_packing_and_product(self):
        source = fixture()
        result = activation_control(source)
        self.assertEqual(result[result.index(TAIL):], source[source.index(TAIL):])
        restored = result.replace('                                silu_tile_init();\n'
            '                                silu_tile(0);\n', '').replace(
            '                                tile_regs_wait();',
            '                                apply_activation_from_pack<KernelActivation::SILU>(1);', 1)
        self.assertEqual(restored, source)
        self.assertNotIn('typecast_tile', result)
        with self.assertRaises(ValueError):
            activation_control(result)

    def test_rounds_both_operands_before_unchanged_product(self):
        candidate = transform(fixture())
        positions = [candidate.index(anchor) for anchor in ('silu_tile(0);',
            f'typecast_tile<{CAST}>(0);', f'typecast_tile<{CAST}>(1);', BF16_PRODUCT,
            'tile_regs_commit();')]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn('rounded_cb', candidate)
        self.assertNotIn('copy_tile(', candidate)
        self.assertNotIn('apply_activation_from_pack', candidate)
        self.assertEqual(candidate.count('pack_tile(0, out_dfb_id);'), 1)
        self.assertIn('tile_regs_wait();', candidate)
        self.assertIn('llk_pack_reconfig_l1_acc(0)', candidate)

    def test_only_final_output_branch_and_postloop_epilogue_change(self):
        control = fixture()
        candidate = transform(control)
        self.assertIn(control[:control.index(START)], candidate)
        self.assertIn(control[control.index(END):control.index(TAIL)], candidate)
        self.assertIn(FINAL, candidate)

    def test_wrong_width_or_changed_arithmetic_fails_closed(self):
        for source in (transform(fixture()), fixture().replace('rounded_cb, 6', 'rounded_cb, 14'),
                fixture().replace(BF16_PRODUCT, 'mul_binary_tile(0, 1, 0);'),
                fixture().replace('SILU>(1)', 'SILU>(2)'), fixture() + 'unexpected'):
            with self.assertRaises(ValueError):
                transform(source)

    def test_projection_retains_buffers_readers_and_requires_t16(self):
        source = subprocess.check_output(['git', 'show', f'{REVISION}:scripts/ci/fused_1d.py']).decode()
        candidate = adapt_projection(source)
        original_call = source[source.index('    def __call__'):]
        self.assertEqual(candidate[candidate.index('    def __call__'):], original_call)
        self.assertIn('token_rows != 16', candidate)
        self.assertIn('math_approx_mode is not True', candidate)
        self.assertIn('intermediates or pairs_per_worker != 3', candidate)
        diagnostic = adapt_projection(source, diagnose_activation=True)
        self.assertIn('import activation_control as register_epilogue', diagnostic)
        self.assertIn('register_epilogue=False, activation_diagnostic=True', diagnostic)
        self.assertIn('intermediate_rounding="native-pack"', diagnostic)
        self.assertEqual(diagnostic[diagnostic.index('    def __call__'):], original_call)
        with self.assertRaises(ValueError):
            adapt_projection(candidate)

    def test_retained_native_compute_preserves_k_loop(self):
        path = Path('D:/qwen-evidence/35092212895/sources/ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp')
        if not path.exists():
            self.skipTest('Retained runtime source unavailable')
        control = fused_compute(path.read_text(encoding='utf-8'), pairs_per_worker=3)
        candidate = transform(control)
        self.assertIn(control[control.index('void kernel_main()'):control.index(START)], candidate)
        self.assertIn(control[control.index(END):control.index(TAIL)], candidate)

    def test_fresh_staged_import_and_manifest_without_orchestrator_path(self):
        from mlp_register_epilogue_stage import stage
        names = ('fused_1d.py', 'fused-batch-probe.py', 'gdn_multitoken.py')
        sources = {name: subprocess.check_output(['git', 'show', f'{REVISION}:scripts/ci/{name}'])
            for name in names}
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            scripts = checkout / 'scripts/ci'
            scripts.mkdir(parents=True)
            for name, source in sources.items():
                (scripts / name).write_bytes(source)
            (scripts / 'simulator-suite.sh').write_text(
                'timeout -k 15 9000 python3 -u /experiment-scripts/ci/fused-batch-probe.py\n')
            manifest = checkout / 'candidate.json'
            with patch('mlp_register_epilogue_stage.subprocess.check_output',
                    side_effect=lambda args: sources[args[-1].split('/')[-1]]):
                stage(checkout, manifest)
            result = subprocess.run([sys.executable, '-B', '-c', 'import fused_1d'],
                cwd=scripts, env=dict(os.environ, PYTHONPATH=str(scripts)), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            recorded = json.loads(manifest.read_text())
            self.assertFalse(recorded['performance_qualified'])
            self.assertIn('mlp_register_epilogue.py', recorded['after'])
            probe = (scripts / 'fused-batch-probe.py').read_text()
            self.assertIn('trace_rows = (16,)', probe)
            self.assertIn('validate_replays', probe)
            with self.assertRaises(ValueError):
                stage(checkout, manifest)


if __name__ == '__main__':
    unittest.main()
