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

    def test_support_staged_before_container_execution(self):
        bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
        if not bash:
            self.skipTest('bash unavailable')
        runner = Path(__file__).with_name('run-simulator.sh').read_text()
        staging = 'docker cp scripts' + runner.split('docker cp scripts', 1)[1].split('docker start -a', 1)[0]
        command = 'set -eu\ncontainer=fixture\ndocker() { printf "%s\\n" "$*"; }\n' + staging
        result = subprocess.run([bash, '-c', command],
            env=dict(os.environ, QWEN_SIM_CASE='ladder-cache', QWEN_CCL_LAZY_BUILD='0'),
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [
            'cp scripts fixture:/experiment-scripts',
            'cp optimisation/sim fixture:/simulator-support'])
        root = Path(__file__).resolve().parents[2]
        for name in ('run-native-fixed-attention.py', 'blackhole-packer-zero-flags.patch'):
            self.assertTrue((root / 'optimisation/sim' / name).is_file())

    def test_context_reaches_probe_and_references_precede_capture(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / '.github/workflows/qwen-ladder-cache-sim.yml').read_text()
        self.assertIn('context: [65536, 131072]', workflow)
        self.assertIn("QWEN_LADDER_CONTEXT: '${{ matrix.context }}'", workflow)
        runner = Path(__file__).with_name('run-simulator.sh').read_text()
        self.assertIn('-e "QWEN_LADDER_CONTEXT=${QWEN_LADDER_CONTEXT:-}"', runner)
        suite = Path(__file__).with_name('simulator-suite.sh').read_text()
        self.assertIn('--context "${QWEN_LADDER_CONTEXT:?}"', suite)
        probe = Path(__file__).with_name('ladder-cache-probe.py').read_text()
        compile(probe, 'ladder-cache-probe.py', 'exec')
        self.assertLess(probe.index('snapshot(serial)'), probe.index('with page_geometry(context)'))
        candidate = probe.split('with page_geometry(context)', 1)[1]
        self.assertNotIn('to_memory_config(', candidate)
        self.assertNotIn('paged_update_cache(', candidate)
        self.assertIn("len(report['checks']) != 10", candidate)


if __name__ == '__main__':
    unittest.main()
