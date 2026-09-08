import os
from pathlib import Path
import subprocess
import sys
import unittest


class LearnedAttentionSuiteTests(unittest.TestCase):
    def test_selector_dot_rejects_unvalidated_hardware_configuration(self):
        environment = {name: value for name, value in os.environ.items()
            if name not in ('TT_METAL_SIMULATOR', 'TT_METAL_MOCK_CLUSTER_DESC_PATH', 'TT_METAL_SLOW_DISPATCH_MODE')}
        environment.update(QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1')
        for configuration in ([], ['--cache-tiles', '--workers', '110'],
                ['--cache-tiles', '--columns-per-task', '8']):
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('draft-dot-probe.py')),
                '--hardware', '--keys', '32', '--width', '256', '--selector-fixture', '/not-a-fixture',
                '--output', '/not-an-output', *configuration], env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Learned selector hardware requires validated', result.stderr)

    def test_selector_projection_rejects_hardware_before_fixture_load(self):
        environment = {name: value for name, value in os.environ.items()
            if name not in ('TT_METAL_SIMULATOR', 'TT_METAL_MOCK_CLUSTER_DESC_PATH', 'TT_METAL_SLOW_DISPATCH_MODE')}
        environment.update(QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1')
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('learned-attention-probe.py')),
            '--hardware', '--fixture', '/not-a-fixture', '--convolution-fixture', '/not-a-fixture',
            '--mlp-fixture', '/not-a-fixture', '--selector-fixture', '/not-a-fixture',
            '--output', '/not-an-output'], env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Hardware stack requires all five simulator-validated layers and selector projection', result.stderr)

    def test_multi_layer_stack_rejects_hardware_before_fixture_load(self):
        environment = {name: value for name, value in os.environ.items()
            if name not in ('TT_METAL_SIMULATOR', 'TT_METAL_MOCK_CLUSTER_DESC_PATH', 'TT_METAL_SLOW_DISPATCH_MODE')}
        environment.update(QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1')
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('learned-attention-probe.py')),
            '--hardware', '--fixture', '/not-a-fixture', '--convolution-fixture', '/not-a-fixture',
            '--mlp-fixture', '/not-a-fixture', '--stack-fixtures', '/not-a-fixture',
            '--output', '/not-an-output'], env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Hardware stack requires all five simulator-validated layers and selector projection', result.stderr)

    def test_complete_layer_requires_convolution_fixture(self):
        environment = {name: value for name, value in os.environ.items()
            if name not in ('TT_METAL_SIMULATOR', 'TT_METAL_MOCK_CLUSTER_DESC_PATH', 'TT_METAL_SLOW_DISPATCH_MODE')}
        environment.update(QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1')
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('learned-attention-probe.py')),
            '--hardware', '--fixture', '/not-a-fixture',
            '--mlp-fixture', '/not-a-fixture', '--output', '/not-an-output'],
            env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Complete draft layer requires convolution fixture', result.stderr)

    def test_unvalidated_integrated_attention_configuration_is_rejected(self):
        environment = {name: value for name, value in os.environ.items()
            if name not in ('TT_METAL_SIMULATOR', 'TT_METAL_MOCK_CLUSTER_DESC_PATH', 'TT_METAL_SLOW_DISPATCH_MODE')}
        environment.update(QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1')
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('learned-attention-probe.py')),
            '--hardware', '--fixture', '/not-a-fixture', '--convolution-fixture', '/not-a-fixture',
            '--output', '/not-an-output'], env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Integrated attention hardware requires the validated short cached fused path', result.stderr)

    def run_suite(self, fail_health=False, mode='learned-attention', stack=False):
        source = Path(__file__).with_name('baseline-suite.sh').read_text()
        start = source.index('if [ "${QWEN_RUN_MODE:-baseline}" = ' + mode + ' ]; then')
        end = source.index('\nfi\n', start) + len('\nfi\n')
        stub = '''set -euo pipefail
timeout() {
    printf '%s ' "$@"
    printf '\\n'
    if [[ "$*" == *device-readback.py && "$FAIL_HEALTH" == 1 ]]; then return 17; fi
}
'''
        guard = next(line for line in source.splitlines() if line.startswith('[[ "${QWEN_LEARNED_STACK'))
        return subprocess.run(['bash', '-c', stub + guard + '\n' + source[start:end]],
            env=dict(os.environ, QWEN_RUN_MODE=mode, FAIL_HEALTH=str(int(fail_health)), QWEN_LEARNED_STACK=str(int(stack))),
            capture_output=True, text=True)

    def test_health_precedes_precise_inspection_free_probe(self):
        result = self.run_suite()
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 17)
        self.assertIn('device-readback.py', lines[0])
        for argument in ('learned-attention-probe.py', '--hardware', '--fp32-rope',
                '--explicit-softmax', '--pairwise-softmax', '--pairwise-dots'):
            self.assertIn(argument, lines[1])
        self.assertNotIn('--inspect-attention', lines[1])
        self.assertIn('draft-row-sum-probe.py', lines[2])
        self.assertIn('--timing', lines[2])
        self.assertIn('--fused-row-sum', lines[3])
        self.assertNotIn('--pairwise-softmax', lines[3])
        self.assertNotIn('--inspect-attention', lines[3])
        for index, shape in enumerate(('--keys 32 --width 128', '--keys 2080 --width 128', '--keys 128 --width 2080'), start=4):
            self.assertIn('draft-dot-probe.py', lines[index])
            self.assertIn('--hardware --timing', lines[index])
            self.assertIn(shape, lines[index])
        self.assertIn('--fused-dots', lines[7])
        self.assertNotIn('--pairwise-dots', lines[7])
        for index, shape in enumerate(('--keys 32 --width 128', '--keys 2080 --width 128', '--keys 128 --width 2080'), start=8):
            self.assertIn('draft-dot-probe.py', lines[index])
            self.assertIn('--hardware --timing --cache-tiles', lines[index])
            self.assertIn(shape, lines[index])
        self.assertIn('--fused-dots --cache-dot-tiles', lines[11])
        self.assertNotIn('--pairwise-dots', lines[11])
        self.assertIn('--cache-tiles --workers 110 --keys 2080 --width 128', lines[12])
        self.assertIn('--workers 110 --keys 128 --width 2080 --columns-per-task 8', lines[13])
        for argument in ('3600', 'learned-attention-probe.py', '--context 2048', '--fp32-rope',
                '--explicit-softmax', '--fused-row-sum', '--fused-dots', '--cache-dot-tiles',
                '--wide-dot-placement', 'learned-attention-long-wide.json'):
            self.assertIn(argument, lines[14])
        for argument in ('900', 'learned-attention-probe.py', '--hardware', '--fp32-rope',
                '--explicit-softmax', '--fused-row-sum', '--fused-dots', '--cache-dot-tiles',
                '--convolution-fixture /experiment-convolution-fixture', 'learned-attention-integrated.json'):
            self.assertIn(argument, lines[15])
        for argument in ('1800', 'learned-attention-probe.py', '--hardware', '--fp32-rope',
                '--explicit-softmax', '--fused-row-sum', '--fused-dots', '--cache-dot-tiles',
                '--convolution-fixture /experiment-convolution-fixture', '--mlp-fixture /experiment-mlp-fixture',
                'learned-layer-complete.json'):
            self.assertIn(argument, lines[16])

    def test_failed_health_stops_probe(self):
        result = self.run_suite(True)
        self.assertEqual(result.returncode, 17)
        self.assertNotIn('learned-attention-probe.py', result.stdout)

    def test_five_layer_opt_in_is_separate_and_health_gated(self):
        result = self.run_suite(stack=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 5)
        self.assertIn('device-readback.py', lines[0])
        for argument in ('draft-mlp-replay-probe.py', '--hardware --timing',
                '--fixture /experiment-mlp-fixture', '--convolution-fixture /experiment-convolution-fixture',
                'learned-mlp-replay.json'):
            self.assertIn(argument, lines[1])
        for argument in ('draft-mlp-trace-probe.py', '--hardware --timing',
                '--fixture /experiment-mlp-fixture', '--convolution-fixture /experiment-convolution-fixture',
                'learned-mlp-trace.json'):
            self.assertIn(argument, lines[2])
        for argument in ('draft-dot-probe.py', '--hardware --timing --keys 32 --width 256 --cache-tiles',
                '--selector-fixture /experiment-selector-fixture', 'learned-selector-dot.json'):
            self.assertIn(argument, lines[3])
        for argument in ('--stack-layers 5', '--selector-fixture /experiment-selector-fixture',
                '--stack-fixtures /experiment-stack-fixture', '--cache-dot-tiles', 'learned-five-layers.json'):
            self.assertIn(argument, lines[4])
        failed = self.run_suite(fail_health=True, stack=True)
        self.assertEqual(failed.returncode, 17)
        self.assertNotIn('learned-attention-probe.py', failed.stdout)
        self.assertNotEqual(self.run_suite(mode='learned-mlp', stack=True).returncode, 0)

    def test_mlp_health_precedes_bounded_learned_probe(self):
        result = self.run_suite(mode='learned-mlp')
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn('device-readback.py', lines[0])
        for argument in ('600', 'fused-batch-probe.py', '--hardware --timing --device-weight-check',
                '--fixture /experiment-projection-fixture', 'fused-batch.json'):
            self.assertIn(argument, lines[1])
        for argument in ('1800', 'learned-mlp-probe.py', '--hardware',
                '--fixture /experiment-projection-fixture', 'learned-mlp.json'):
            self.assertIn(argument, lines[2])
        for argument in ('1800', 'learned-mlp-probe.py', '--hardware',
                '--convolution-fixture /experiment-convolution-fixture', 'learned-mlp-integrated.json'):
            self.assertIn(argument, lines[3])

    def test_mlp_failed_health_stops_probe(self):
        result = self.run_suite(True, 'learned-mlp')
        self.assertEqual(result.returncode, 17)
        self.assertNotIn('learned-mlp-probe.py', result.stdout)
        self.assertNotIn('fused-batch-probe.py', result.stdout)
