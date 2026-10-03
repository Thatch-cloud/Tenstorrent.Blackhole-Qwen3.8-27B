import os
from pathlib import Path
import subprocess
import unittest


class LiveQKSuiteTests(unittest.TestCase):
    def route(self, failure=''):
        source = Path(__file__).with_name('baseline-suite.sh').read_text()
        start = source.index('if [ "${QWEN_LIVE_QK:-0}" = 1 ]; then')
        end = source.index('if [ "${QWEN_MTP_DRAFTS:-0}" != 0 ]; then', start)
        block = source[start:end].replace('> /experiment/results/live-qk-native-hashes.json', '> /dev/null')
        stub = '''set -euo pipefail
python3() { if [[ "$FAILURE" == preflight ]]; then return 12; fi; }
timeout() {
    printf 'run %s\\n' "$*"
    if [[ "$FAILURE" == short && "$*" == *'--context 31 '* ]]; then return 13; fi
    if [[ "$FAILURE" == long && "$*" == *'--context 2048 '* ]]; then return 14; fi
}
'''
        return subprocess.run(['bash', '-c', stub + block + '\necho unexpected_model_path'],
            env=dict(os.environ, QWEN_LIVE_QK='1', QWEN_RUN_MODE='baseline', QWEN_CCL_LAZY_BUILD='0',
                PYTHONPATH='/opt/tt-metal/ttnn:/opt/tt-metal', FAILURE=failure),
            capture_output=True, text=True)

    def test_component_route_executes_both_contexts_without_model_or_rebuild(self):
        result = self.route()
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 2)
        for line, context in zip(lines, (31, 2048), strict=True):
            self.assertIn(f'--context {context}', line)
            self.assertIn(f'live-qk-simulator-{context}.json', line)
            self.assertIn(f'live-qk-{context}.json', line)
        self.assertNotIn('unexpected_model_path', result.stdout)

    def test_failed_preflight_or_context_cannot_pass(self):
        for failure, status, runs in (('preflight', 12, 0), ('short', 13, 1), ('long', 14, 2)):
            result = self.route(failure)
            self.assertEqual(result.returncode, status, result.stderr)
            self.assertEqual(len(result.stdout.splitlines()), runs)

    def test_host_requires_both_artifacts_and_independent_validation(self):
        source = Path(__file__).with_name('run-baseline.sh').read_text()
        start = source.rindex('if [ "$live_qk" = 1 ]; then')
        block = source[start:source.index('if [ "$tiny_mlp" = 1 ]; then', start)]
        self.assertIn('for context in 31 2048', block)
        self.assertIn('docker cp "$test_id:/experiment/results/live-qk-$context.json"', block)
        self.assertIn('python3 scripts/ci/live_qk_gate.py --hardware-result', block)
        self.assertNotIn('|| true', block)
        self.assertIn('-e "QWEN_LIVE_QK=$live_qk"', source)
        self.assertIn('if [ "$live_qk" = 1 ]; then descriptor=p150_x2_mesh_graph_descriptor.textproto; fi', source)
