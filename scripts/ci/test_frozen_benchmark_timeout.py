from pathlib import Path
import subprocess
import unittest

from frozen_benchmark_timeout import transform
from frozen_recipe_context import REVISION


class BenchmarkTimeoutTests(unittest.TestCase):
    def test_current_and_frozen_request_runner_preserve_setup_timeouts(self):
        current = Path(__file__).with_name('dspark-hardware-suite.sh').read_text()
        frozen = subprocess.check_output(['git', 'show',
            f'{REVISION}:scripts/ci/dspark-hardware-suite.sh']).decode()
        for source in (current, frozen):
            with self.subTest(frozen=source == frozen):
                changed = transform(source)
                self.assertEqual(changed.count('timeout '), source.count('timeout ') - 1)
                self.assertIn('runner=(python3 -u "/experiment-scripts/ci/$probe.py"', changed)
                self.assertIn('status=${PIPESTATUS[0]}', changed)
                with self.assertRaises(ValueError):
                    transform(changed)


if __name__ == '__main__':
    unittest.main()
