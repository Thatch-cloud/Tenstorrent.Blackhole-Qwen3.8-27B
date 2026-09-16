import subprocess
import unittest
import os
import json
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch

import frozen_recipe_context
from frozen_recipe_context import REVISION, adapt_probe_sources, adapt_cache_launcher, geometry
from frozen_context_geometry import CONTEXTS, selected_geometry, factory_selector


class FrozenRecipeContextTests(unittest.TestCase):
    def test_cleanup_is_bounded_and_preserves_failure(self):
        sources = {name: subprocess.check_output(
            ['git', 'show', f'{REVISION}:scripts/ci/{name}'], text=True)
            for name in ('run-simulator.sh', 'simulator-suite.sh')}
        launcher = adapt_cache_launcher(sources)['run-simulator.sh']
        cleanup = 'cleanup() {' + launcher.split('cleanup() {', 1)[1].split('trap cleanup EXIT', 1)[0]
        bash = 'C:/Program Files/Git/bin/bash.exe' if sys.platform == 'win32' else 'bash'
        for initial, failed, expected in ((0, '', 0), (0, 'cp', 70), (17, 'cp', 17), (0, 'rm', 70)):
            with self.subTest(initial=initial, failed=failed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'experiment-results').mkdir()
                script = '''set -euo pipefail
container=owned-test-container
timeout() {
    printf '%s\\n' "$*" >> commands.txt
    if [ "$5" = "$FAIL_ACTION" ]; then return 124; fi
    return 0
}
''' + cleanup + '\ntrap cleanup EXIT\nexit ' + str(initial) + '\n'
                process = subprocess.run([bash], input=script, text=True, cwd=root,
                    env=dict(os.environ, FAIL_ACTION=failed), capture_output=True, timeout=10)
                self.assertEqual(process.returncode, expected, process.stderr)
                commands = (root / 'commands.txt').read_text().splitlines()
                self.assertEqual(commands[0], '-k 1 5 docker stop -t 2 owned-test-container')
                self.assertEqual(commands[-1], '-k 1 5 docker rm -f owned-test-container')
                self.assertEqual(len(commands), 4)
                result = json.loads((root / 'experiment-results/container-cleanup.json').read_text())
                self.assertEqual(result['copy_exit'], 124 if failed == 'cp' else 0)
                self.assertEqual(result['remove_exit'], 124 if failed == 'rm' else 0)

    def test_deployment_preserves_historical_hardware_runtime(self):
        names = ('dspark_attention_chunk_trial.py', 'dspark-native-8k-attention-probe.py',
            'dspark_stats_pack.py', 'dspark_fp32_intermediates.py', 'run-simulator.sh',
            'simulator-suite.sh', 'dspark_runtime_cache.py')
        originals = {name: subprocess.check_output(
            ['git', 'show', f'{REVISION}:scripts/ci/{name}']) for name in names}
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            scripts = checkout / 'scripts/ci'
            scripts.mkdir(parents=True)
            for name, source in originals.items():
                (scripts / name).write_bytes(source)
            manifest = checkout / 'deployment.json'

            def git(command):
                arguments = command[3:]
                if arguments == ['rev-parse', 'HEAD']:
                    return REVISION.encode() + b'\n'
                if arguments == ['status', '--porcelain', '--untracked-files=no']:
                    return b''
                if arguments[0] == 'show':
                    return originals[arguments[1].rsplit('/', 1)[1]]
                raise AssertionError(command)

            with patch('sys.argv', ['adapter', '--checkout', str(checkout), '--context', '65536',
                    '--manifest', str(manifest)]), \
                    patch.object(frozen_recipe_context.subprocess, 'check_output', side_effect=git):
                frozen_recipe_context.main()
            self.assertEqual((scripts / 'dspark_runtime_cache.py').read_bytes(),
                originals['dspark_runtime_cache.py'])
            report = json.loads(manifest.read_text())
            self.assertNotIn('dspark_runtime_cache.py', report['after'])
            self.assertIn('frozen_binary_cache.py', report['after'])
            self.assertFalse(report['performance_qualified'])
            self.assertEqual(set(report['after']), set(names) - {'dspark_runtime_cache.py'} |
                {'frozen_binary_cache.py', 'frozen_context_geometry.py',
                 'frozen_sim_build_cache.py', 'frozen_sim_phase.py', 'frozen_sim_assets.py',
                 'frozen_probe_evidence.py'})

    def test_cache_launcher_preserves_bounded_original_probe(self):
        names = ('run-simulator.sh', 'simulator-suite.sh')
        sources = {name: subprocess.check_output(
            ['git', 'show', f'{REVISION}:scripts/ci/{name}'], text=True) for name in names}
        result = adapt_cache_launcher(sources)
        self.assertIn('frozen_sim_build_cache.py', result['simulator-suite.sh'])
        self.assertIn('dst=/frozen-simulator-cache', result['run-simulator.sh'])
        self.assertIn('"/experiment-scripts/ci/$QWEN_SIM_CASE-probe.py"', result['simulator-suite.sh'])
        self.assertNotIn('--device', result['run-simulator.sh'])
        self.assertNotIn('--max-time 180', result['run-simulator.sh'])
        self.assertIn('timeout -k 5 60 python3', result['run-simulator.sh'])

    def test_single_environment_flag_selects_all_sizes_without_fallback(self):
        for context in CONTEXTS:
            with patch.dict(os.environ, {'QWEN_DSPARK_REQUEST_CONTEXT': str(context)}):
                self.assertEqual(selected_geometry(), geometry(context))
        for value in ('', '32k', '131000', '262000', '524288', '32768 '):
            with patch.dict(os.environ, {'QWEN_DSPARK_REQUEST_CONTEXT': value}):
                with self.assertRaises(ValueError):
                    selected_geometry()

    def test_geometry_preserves_output_headroom_and_chunk_size(self):
        for context in CONTEXTS:
            shape = geometry(context)
            self.assertEqual(shape['capacity'], context + 256)
            self.assertEqual(shape['storage_keys'], context + 320)
            self.assertEqual(shape['padded_keys'], context + 512)
            self.assertEqual(shape['key_chunk'], 256)
            self.assertEqual(shape['positions'][1] + 15, shape['capacity'])
            self.assertGreaterEqual(shape['target_sequence_capacity'], context + 256)
            self.assertEqual(shape['target_sequence_capacity'] & (shape['target_sequence_capacity'] - 1), 0)
            if context < 65536:
                self.assertEqual(shape['target_sequence_capacity'], 65536)
        with self.assertRaises(ValueError):
            geometry(524288)

    def test_historical_source_changes_geometry_not_math_or_poison(self):
        names = ('dspark_attention_chunk_trial.py', 'dspark-native-8k-attention-probe.py',
            'dspark_stats_pack.py', 'run-simulator.sh', 'dspark_fp32_intermediates.py')
        sources = {name: subprocess.check_output(
            ['git', 'show', f'{REVISION}:scripts/ci/{name}'], text=True) for name in names}
        baseline = adapt_probe_sources(sources, 8192)
        factories = []
        for context in CONTEXTS:
            adapted = adapt_probe_sources(sources, context)
            self.assertEqual(adapted, baseline)
            for name, source in adapted.items():
                if name.endswith('.py'):
                    compile(source, name, 'exec')
            original = sources[names[0]]
            candidate = adapted[names[0]]
            self.assertEqual(original[original.index('    kernel ='):],
                candidate[candidate.index('    kernel ='):])
            self.assertIn('key_padding, 8192.', candidate)
            self.assertIn('key_padding, -8192.', candidate)
            self.assertIn("get_compile_time_arg_val(3) == {selected_geometry()['padded_keys'] // 32}",
                adapted['dspark_stats_pack.py'])
            self.assertIn('get_compile_time_arg_val(8) == 8', adapted['dspark_stats_pack.py'])
            with patch.dict(os.environ, {'QWEN_DSPARK_REQUEST_CONTEXT': str(context)}):
                namespace = {}
                exec(adapted['dspark_fp32_intermediates.py'], namespace)
                factories.append(namespace['REPLACEMENT'])
                expected_tiles = geometry(context)['padded_keys'] // 32
                self.assertIn(f'Skt == {expected_tiles}', namespace['REPLACEMENT'])
                self.assertIn('stats_df = qwen_draft_fp32_intermediates ? tt::DataFormat::Float32',
                    namespace['REPLACEMENT'])
                if context == 8192:
                    historical = {}
                    exec(sources['dspark_fp32_intermediates.py'], historical)
                    self.assertEqual(namespace['REPLACEMENT'].replace(factory_selector(), 'Skt == 272'),
                        historical['REPLACEMENT'])
        self.assertEqual(len(set(factories)), 1)
        broken = dict(sources)
        broken[names[0]] += '\nPADDED_KEYS = 8704\n'
        with self.assertRaises(ValueError):
            adapt_probe_sources(broken, 32768)


if __name__ == '__main__':
    unittest.main()
