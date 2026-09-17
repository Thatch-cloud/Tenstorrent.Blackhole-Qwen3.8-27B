import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dspark_markov_gate import qualify


class DSparkDiagnosticTests(unittest.TestCase):
    def test_diagnostic_completion_is_not_selector_qualification(self):
        report = json.loads(Path(__file__).with_name('dspark-markov-rounding-diagnostic.json').read_text())
        self.assertTrue(report['completed'])
        self.assertFalse(report['accuracy_qualified'])
        self.assertTrue(all(entry['matmul_grouped_exact'] for entry in report['checks']))
        self.assertTrue(all(entry['default_add_vs_fp32_max_abs'] == 0 for entry in report['checks']))
        with self.assertRaises(ValueError):
            qualify(report, sources=report['sources'], native=report['native_sources'], exit_status='0')

    def test_original_learned_failure_remains_rejected(self):
        report = json.loads(Path(__file__).with_name('dspark-markov-simulator-learned-failed.json').read_text())
        self.assertFalse(report['passed'])
        self.assertTrue(report['closed_cleanly'])
        self.assertIn('53427 / 248320', report['error'])
        with self.assertRaises(ValueError):
            qualify(report, sources=report['sources'], native=report['native_sources'], exit_status='1')

    def test_diagnostic_refuses_hardware_before_loading_any_fixture(self):
        environment = {key: value for key, value in os.environ.items()
            if key not in ('TT_METAL_SIMULATOR', 'TT_METAL_SLOW_DISPATCH_MODE')}
        environment.update(QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'unexpected.json'
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('dspark-markov-rounding-probe.py')),
                '--output', str(output), '--fixture', str(Path(directory) / 'not-loaded')],
                env=environment, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Simulator required', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
