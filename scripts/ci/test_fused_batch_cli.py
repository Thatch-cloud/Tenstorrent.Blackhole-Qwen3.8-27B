import os
from pathlib import Path
import subprocess
import sys
import unittest


class FusedBatchCLITests(unittest.TestCase):
    def test_timing_and_device_weight_guards_precede_fixture_loading(self):
        for flags, environment, message in (
                (['--trace-t16'], {'TT_METAL_SIMULATOR': 'placeholder'},
                    'T16 trace coverage requires --trace-replay'),
                (['--hardware', '--trace-replay'],
                    {'QWEN_HARDWARE_TESTS': '1', 'QWEN_CARDS_ALLOCATED': '1'},
                    'Trace replay requires byte-exact weight checks'),
                (['--trace-replay'], {'TT_METAL_SIMULATOR': 'placeholder'},
                    'Trace replay requires byte-exact weight checks'),
                (['--timing'], {'TT_METAL_SIMULATOR': 'placeholder'}, 'Latency measurements require allocated hardware'),
                (['--hardware'], {'QWEN_HARDWARE_TESTS': '1', 'QWEN_CARDS_ALLOCATED': '1'},
                    'Hardware promotion requires byte-exact device weight checks')):
            clean = {name: value for name, value in os.environ.items()
                if name not in ('TT_METAL_SIMULATOR', 'TT_METAL_SLOW_DISPATCH_MODE', 'TT_METAL_MOCK_CLUSTER_DESC_PATH')}
            clean.update(environment)
            with self.subTest(flags=flags):
                result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('fused-batch-probe.py')),
                    '--fixture', '/missing', '--output', '/missing', *flags], env=clean, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
