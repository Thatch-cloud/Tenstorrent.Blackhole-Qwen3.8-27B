from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class ContextProbeCLITests(unittest.TestCase):
    def test_experimental_capacities_require_simulator_and_preserve_default_guard(self):
        from attention_mask_replay import validate_ticket

        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(ValueError):
                validate_ticket(2048, 16, 2304)
        with patch.dict('os.environ', {'QWEN_CONTEXT_LADDER_SIM': '1', 'QWEN_SIM_ONLY': '1'}, clear=True):
            for context in (2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144):
                validate_ticket(context, 16, context + 256)
            with patch.dict('os.environ', {'QWEN_CARDS_ALLOCATED': '1'}), self.assertRaises(ValueError):
                validate_ticket(2048, 16, 2304)
    def run_probe(self, *arguments):
        return subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('target-t16-context-probe.py')),
            *arguments], capture_output=True, text=True, timeout=30)

    def test_rejects_hardware_before_runtime_import(self):
        result = self.run_probe('--hardware', '--context', '2048', '--output', 'unused.json')
        self.assertEqual(result.returncode, 2)
        self.assertIn('simulator-first', result.stderr)

    def test_rejects_unsupported_context(self):
        result = self.run_probe('--context', '1024', '--output', 'unused.json')
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid choice', result.stderr)

    def test_existing_evidence_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            output.write_text('retained evidence')
            result = self.run_probe('--context', '2048', '--output', str(output))
            self.assertEqual(result.returncode, 2)
            self.assertIn('Fresh ladder report required', result.stderr)
            self.assertEqual(output.read_text(), 'retained evidence')


if __name__ == '__main__':
    unittest.main()
