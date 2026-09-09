import os
from pathlib import Path
import subprocess
import unittest


class TensixMlpSuiteTests(unittest.TestCase):
    def run_route(self, enabled=True, failure=''):
        source = Path(__file__).with_name('baseline-suite.sh').read_text()
        start = source.index('if [ "${QWEN_CCL_LAZY_BUILD:-0}" = 1 ]; then')
        end = source.index('if [ "${QWEN_RUN_MODE:-baseline}" = learned-mlp ]; then', start)
        model_start = source.index('if [ "${QWEN_RUN_MODE:-baseline}" = full-norm-engine ]; then')
        model_end = source.index('if [ "${QWEN_RUN_MODE:-baseline}" = full-norm-selection ]; then', model_start)
        stub = '''set -euo pipefail
python3() { printf 'preflight %s\\n' "$*"; if [[ "$FAILURE" == preflight ]]; then return 16; fi; }
bash() { printf 'build %s\\n' "$*"; if [[ "$FAILURE" == build ]]; then return 17; fi; }
tee() { cat; }
grep() { return 1; }
timeout() {
    printf 'run %s\\n' "$*"
    if [[ "$*" == *ccl-link-probe.py* && "$FAILURE" == link ]]; then return 18; fi
    if [[ "$*" == *device-readback.py* && "$FAILURE" == readback ]]; then return 19; fi
    if [[ "$*" == *tensix-stream-mlp-hardware.py* && "$FAILURE" == mlp ]]; then return 20; fi
    return 0
}
'''
        environment = dict(os.environ, QWEN_CCL_LAZY_BUILD='1', QWEN_TENSIX_MLP=str(int(enabled)),
            QWEN_TINY_MLP='0', QWEN_MTP_DRAFTS='0', QWEN_DFLASH_DRAFTS='0',
            QWEN_RUN_MODE='full-norm-engine', FAILURE=failure)
        return subprocess.run(['bash', '-c', stub + source[start:end] + source[model_start:model_end]],
            env=environment, capture_output=True, text=True)

    def test_preflight_then_build_link_readback_and_actual_mlp(self):
        result = self.run_route()
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 5)
        for line, name in zip(lines, ('tensix-stream-mlp-hardware.py', 'ccl-links-build.sh', 'ccl-link-probe.py',
                'device-readback.py', 'tensix-stream-mlp-hardware.py'), strict=True):
            self.assertIn(name, line)
        self.assertIn('--preflight', lines[0])
        self.assertIn('--simulator-exit-status /experiment-scripts/ci/tensix-mlp-simulator.exit-status', lines[0])
        self.assertIn('--simulator-report /experiment-scripts/ci/tensix-mlp-simulator.json', lines[-1])
        self.assertIn('--output /experiment/results/tensix-mlp.json', lines[-1])
        self.assertNotIn('full-prefix.py', result.stdout)

    def test_link_probe_is_not_mislabeled_as_mlp_success(self):
        result = self.run_route(enabled=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 2)
        self.assertNotIn('tensix-stream-mlp-hardware.py', result.stdout)

    def test_each_failure_stops_before_the_next_stage(self):
        for failure, code, calls in (('preflight', 16, 1), ('build', 17, 2), ('link', 18, 3),
                ('readback', 19, 4), ('mlp', 20, 5)):
            result = self.run_route(failure=failure)
            with self.subTest(failure=failure):
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(len(result.stdout.splitlines()), calls)

    def test_host_requires_exact_artifact_and_independent_validator(self):
        source = Path(__file__).with_name('run-baseline.sh').read_text()
        block = source[source.rindex('if [ "$tensix_mlp" = 1 ]; then'):]
        self.assertIn('docker cp "$test_id:/experiment/results/tensix-mlp.json" "$output/tensix-mlp.json"', block)
        self.assertIn('python3 scripts/ci/tensix_mlp_hardware_gate.py --hardware-result "$output/tensix-mlp.json"', block)
        self.assertNotIn('|| true', block)
        self.assertIn('-e "QWEN_TENSIX_MLP=$tensix_mlp"', source)
        self.assertIn('if [ "$tensix_mlp" = 1 ]; then ccl_build=1; projection_links=4; fi', source)


if __name__ == '__main__':
    unittest.main()
