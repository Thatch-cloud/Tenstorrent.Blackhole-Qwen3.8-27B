import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from frozen_sim_phase import run_phase


class PhaseTests(unittest.TestCase):
    def test_success_failure_and_timeout_preserve_exit_code(self):
        for status in (0, 1, 124, 137):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'phase.json'
                with patch('frozen_sim_phase.subprocess.run',
                        return_value=subprocess.CompletedProcess([], status)) as execute:
                    self.assertEqual(run_phase('probe', 510, output, ['python3', 'probe.py']), status)
                execute.assert_called_once_with(
                    ['timeout', '-k', '15', '510', 'python3', 'probe.py'], check=False)
                report = json.loads(output.read_text())
                self.assertEqual(report['exit_code'], status)
                self.assertGreaterEqual(report['elapsed_seconds'], 0)
                with self.assertRaises(ValueError):
                    run_phase('probe', 510, output, ['python3'])


if __name__ == '__main__':
    unittest.main()
