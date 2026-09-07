import os
from pathlib import Path
import subprocess
import unittest


class LearnedAttentionSuiteTests(unittest.TestCase):
    def run_suite(self, fail_health=False):
        source = Path(__file__).with_name('baseline-suite.sh').read_text()
        start = source.index('if [ "${QWEN_RUN_MODE:-baseline}" = learned-attention ]; then')
        end = source.index('\nfi\n', start) + len('\nfi\n')
        stub = '''set -euo pipefail
timeout() {
    printf '%s ' "$@"
    printf '\\n'
    if [[ "$*" == *device-readback.py && "$FAIL_HEALTH" == 1 ]]; then return 17; fi
}
'''
        return subprocess.run(['bash', '-c', stub + source[start:end]],
            env=dict(os.environ, QWEN_RUN_MODE='learned-attention', FAIL_HEALTH=str(int(fail_health))),
            capture_output=True, text=True)

    def test_health_precedes_precise_inspection_free_probe(self):
        result = self.run_suite()
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn('device-readback.py', lines[0])
        for argument in ('learned-attention-probe.py', '--hardware', '--fp32-rope',
                '--explicit-softmax', '--pairwise-softmax', '--pairwise-dots'):
            self.assertIn(argument, lines[1])
        self.assertNotIn('--inspect-attention', lines[1])

    def test_failed_health_stops_probe(self):
        result = self.run_suite(True)
        self.assertEqual(result.returncode, 17)
        self.assertNotIn('learned-attention-probe.py', result.stdout)
