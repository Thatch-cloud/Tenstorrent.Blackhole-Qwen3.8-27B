import os
from pathlib import Path
import shutil
import subprocess
import unittest


class LadderSimLaunchTests(unittest.TestCase):
    def test_launcher_admits_weight_free_case_and_rejects_unknown(self):
        bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
        if not bash:
            self.skipTest('bash unavailable')
        runner = Path(__file__).with_name('run-simulator.sh').read_text()
        prefix = runner.split('mkdir -p experiment-results\n', 1)[0]
        self.assertNotIn('docker ', prefix)
        environment = {key: value for key, value in os.environ.items() if not key.startswith('QWEN_')}
        environment.update(QWEN_SIM_ONLY='1', QWEN_LEARNED_STACK='1')
        for case, expected in (('ladder-cache', 0), ('invalid-case', 2)):
            with self.subTest(case=case):
                result = subprocess.run(
                    [bash, '-c', prefix + '\nprintf "admitted\\n"\n'],
                    env=dict(environment, QWEN_SIM_CASE=case), capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertEqual('admitted' in result.stdout, expected == 0)
                if expected:
                    self.assertIn('Unsupported QWEN_SIM_CASE: invalid-case', result.stderr)
        self.assertTrue(any('ladder-cache' in line and "kinds=''" in line for line in runner.splitlines()))
        suite = Path(__file__).with_name('simulator-suite.sh').read_text()
        self.assertIn('ladder-cache-probe.py', suite)
        self.assertIn('ladder-cache.exit-status', suite)


if __name__ == '__main__':
    unittest.main()
