from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ContextProbeCLITests(unittest.TestCase):
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
