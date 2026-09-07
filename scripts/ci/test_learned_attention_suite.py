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
        self.assertEqual(len(lines), 12)
        self.assertIn('device-readback.py', lines[0])
        for argument in ('learned-attention-probe.py', '--hardware', '--fp32-rope',
                '--explicit-softmax', '--pairwise-softmax', '--pairwise-dots'):
            self.assertIn(argument, lines[1])
        self.assertNotIn('--inspect-attention', lines[1])
        self.assertIn('draft-row-sum-probe.py', lines[2])
        self.assertIn('--timing', lines[2])
        self.assertIn('--fused-row-sum', lines[3])
        self.assertNotIn('--pairwise-softmax', lines[3])
        self.assertNotIn('--inspect-attention', lines[3])
        for index, shape in enumerate(('--keys 32 --width 128', '--keys 2080 --width 128', '--keys 128 --width 2080'), start=4):
            self.assertIn('draft-dot-probe.py', lines[index])
            self.assertIn('--hardware --timing', lines[index])
            self.assertIn(shape, lines[index])
        self.assertIn('--fused-dots', lines[7])
        self.assertNotIn('--pairwise-dots', lines[7])
        for index, shape in enumerate(('--keys 32 --width 128', '--keys 2080 --width 128', '--keys 128 --width 2080'), start=8):
            self.assertIn('draft-dot-probe.py', lines[index])
            self.assertIn('--hardware --timing --cache-tiles', lines[index])
            self.assertIn(shape, lines[index])
        self.assertIn('--fused-dots --cache-dot-tiles', lines[11])
        self.assertNotIn('--pairwise-dots', lines[11])

    def test_failed_health_stops_probe(self):
        result = self.run_suite(True)
        self.assertEqual(result.returncode, 17)
        self.assertNotIn('learned-attention-probe.py', result.stdout)
