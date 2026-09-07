import os
from pathlib import Path
import subprocess
import unittest


class AttentionReplaySuiteTests(unittest.TestCase):
    def run_suite(self, mode, fail_build=False):
        source = Path(__file__).with_name('baseline-suite.sh').read_text()
        start = source.index('if [[ "${QWEN_RUN_MODE:-baseline}" = full-attention-replay')
        end = source.index('\nif [ ', start)
        stub = '''set -euo pipefail
timeout() {
    printf '%s|' "${QWEN_SDPA_TREE_SCRATCH_ROUNDS:-unset}"
    printf '%s ' "$@"
    printf '\\n'
    if [[ "$*" == *sdpa-tree-build.sh && "$FAIL_BUILD" == 1 ]]; then return 17; fi
    return 0
}
'''
        environment = dict(os.environ, QWEN_RUN_MODE=mode, FAIL_BUILD=str(int(fail_build)))
        environment.pop('QWEN_SDPA_TREE_SCRATCH_ROUNDS', None)
        return subprocess.run(['bash', '-c', stub + source[start:end]], env=environment,
                              capture_output=True, text=True)

    def test_wide_replay_builds_before_health_and_model(self):
        result = self.run_suite('full-attention-tree-replay')
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn('sdpa-tree-build.sh', lines[0])
        self.assertTrue(lines[0].startswith('unset|'))
        self.assertIn('device-readback.py', lines[1])
        self.assertTrue(lines[1].startswith('1|'))
        self.assertTrue(lines[2].startswith('1|'))
        for argument in ('--attention-replay', '--attention-mask-once', '--replay-group-rows 8',
                         '--max-rows 32', '--replay-inputs', '--norm-batch', '--captured-commit'):
            self.assertIn(argument, lines[2])

    def test_failed_build_never_opens_devices(self):
        result = self.run_suite('full-attention-tree-replay', fail_build=True)
        self.assertEqual(result.returncode, 17)
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertNotIn('full-prefix.py', result.stdout)
        self.assertNotIn('device-readback.py', result.stdout)

    def test_existing_replay_modes_do_not_build_or_expand(self):
        for mode in ('full-attention-replay', 'full-attention-mask-once'):
            with self.subTest(mode=mode):
                result = self.run_suite(mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(result.stdout.splitlines()), 2)
                self.assertNotIn('sdpa-tree-build.sh', result.stdout)
                self.assertNotIn('--replay-group-rows', result.stdout)
                self.assertEqual('--attention-mask-once' in result.stdout,
                                 mode == 'full-attention-mask-once')
