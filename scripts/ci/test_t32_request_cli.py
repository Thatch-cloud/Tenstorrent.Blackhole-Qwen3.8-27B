import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class T32RequestCLITests(unittest.TestCase):
    def test_hardware_environment_is_rejected_before_loading_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'report.json'
            environment = dict(os.environ, QWEN_SIM_ONLY='1', QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1')
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('t32-full-request-probe.py')),
                '--checkpoint', 'missing', '--config', 'missing', '--target', 'missing', '--output', str(output)],
                env=environment, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('ValueError', result.stderr)
            self.assertNotIn('ModuleNotFoundError', result.stderr)
            self.assertFalse(output.exists())

    def test_ci_route_is_bounded_and_mounts_real_request_harness(self):
        directory = Path(__file__).parent
        runner = (directory / 'run-simulator.sh').read_text()
        suite = (directory / 'simulator-suite.sh').read_text()
        self.assertIn('docker cp speculative-decoding "$container:/speculative-decoding"', runner)
        self.assertIn('--memory 64g --cpus 16', runner)
        self.assertIn('timeout -k 15 9000 python3 -u /experiment-scripts/ci/t32-full-request-probe.py', suite)
        self.assertIn('t32-request.exit-status', suite)


if __name__ == '__main__':
    unittest.main()
