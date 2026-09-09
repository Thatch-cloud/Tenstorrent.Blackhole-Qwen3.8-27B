import os
from pathlib import Path
import subprocess
import unittest


class TinyMlpSuiteTests(unittest.TestCase):
    def run_route(self, tiny=True, failure=''):
        source = Path(__file__).with_name('baseline-suite.sh').read_text()
        start = source.index('if [ "${QWEN_CCL_LAZY_BUILD:-0}" = 1 ]; then')
        end = source.index('if [ "${QWEN_RUN_MODE:-baseline}" = learned-mlp ]; then', start)
        model_start = source.index('if [ "${QWEN_RUN_MODE:-baseline}" = full-norm-engine ]; then')
        model_end = source.index('if [ "${QWEN_RUN_MODE:-baseline}" = full-norm-selection ]; then', model_start)
        stub = '''set -euo pipefail
bash() { printf 'build %s\\n' "$*"; if [[ "$FAILURE" == build ]]; then return 17; fi; }
tee() { cat; }
grep() { return 1; }
timeout() {
    printf 'run %s\\n' "$*"
    if [[ "$*" == *ccl-link-probe.py* && "$FAILURE" == link ]]; then return 18; fi
    if [[ "$*" == *tiny-mlp-hardware.py* && "$FAILURE" == mlp ]]; then return 19; fi
    return 0
}
'''
        environment = dict(os.environ, QWEN_CCL_LAZY_BUILD='1', QWEN_TINY_MLP=str(int(tiny)),
            QWEN_MTP_DRAFTS='0', QWEN_DFLASH_DRAFTS='0', QWEN_RUN_MODE='full-norm-engine', FAILURE=failure)
        return subprocess.run(['bash', '-c', stub + source[start:end] + source[model_start:model_end]],
            env=environment, capture_output=True, text=True)

    def test_tiny_route_continues_after_link_probe_to_actual_mlp(self):
        result = self.run_route()
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 4)
        for line, name in zip(lines, ('ccl-links-build.sh', 'ccl-link-probe.py', 'device-readback.py', 'tiny-mlp-hardware.py'), strict=True):
            self.assertIn(name, line)
        self.assertIn('--simulator-report /experiment-scripts/ci/tiny-mlp-simulator.json', lines[-1])
        self.assertIn('--output /experiment/results/tiny-mlp.json', lines[-1])
        self.assertNotIn('full-prefix.py', result.stdout)

    def test_standalone_link_route_retains_its_early_exit(self):
        result = self.run_route(tiny=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 2)
        self.assertNotIn('tiny-mlp-hardware.py', result.stdout)

    def test_each_failed_prerequisite_and_mlp_stops_the_suite(self):
        for failure, code, calls in (('build', 17, 1), ('link', 18, 2), ('mlp', 19, 4)):
            result = self.run_route(failure=failure)
            with self.subTest(failure=failure):
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(len(result.stdout.splitlines()), calls)

    def test_host_wrapper_requires_the_mlp_artifact_and_validator(self):
        source = Path(__file__).with_name('run-baseline.sh').read_text()
        start = source.rindex('if [ "$tiny_mlp" = 1 ]; then')
        block = source[start:]
        self.assertIn('docker cp "$test_id:/experiment/results/tiny-mlp.json" "$output/tiny-mlp.json"', block)
        self.assertIn('python3 scripts/ci/tiny_mlp_gate.py --hardware-result "$output/tiny-mlp.json"', block)
        self.assertNotIn('|| true', block)
