import os
from pathlib import Path
import subprocess
import sys
import unittest


class T32CommitProbeTests(unittest.TestCase):
    def run_probe(self, flags):
        root = Path(__file__).resolve().parents[2]
        environment = dict(os.environ)
        environment.pop('TT_METAL_SIMULATOR', None)
        return subprocess.run([sys.executable, '-B', str(root / 'optimisation/sim/gdn-multitoken.py'),
            '--source-root', '/unused', '--rows', '32', '--norm-gate', '--conv', '--model-adapter',
            *flags], env=environment, capture_output=True, text=True, timeout=10)

    def test_incomplete_adapter_stays_blocked(self):
        result = self.run_probe([])
        self.assertEqual(result.returncode, 2)
        self.assertIn('wider packed-history prerequisites', result.stderr)

    def test_complete_adapter_still_requires_simulator(self):
        result = self.run_probe(['--batch-conv', '--dma-windows', '--packed-checkpoints',
            '--continuation', '--compact-prologue', '--norm-batch-layer',
            '--defer-conv-publication', '--commit-only-gdn'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Simulator library and slow dispatch mandatory', result.stderr)


if __name__ == '__main__':
    unittest.main()
