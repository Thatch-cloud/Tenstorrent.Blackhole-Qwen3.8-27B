import subprocess
import unittest
import os
import json
import hashlib
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch

import frozen_recipe_context
from frozen_recipe_context import (REVISION, adapt_probe_sources, adapt_cache_launcher,
    adapt_scalar_reciprocal, adapt_eager_only, geometry, COMBINED_RUNTIME_CONTEXTS,
    combined_runtime_directory)
from frozen_context_geometry import CONTEXTS, selected_geometry, factory_selector
from frozen_runtime_context import FILES


# frozen_wide_chunk_replay.py reads attention_replay.py straight off the
# --checkout directory (it is not otherwise staged by frozen_recipe_context.py
# at all) and only for context 65536; a real checkout always has the file,
# these synthetic ones below do not unless this is written in. Minimal, not
# the real file, but carries both anchors _patch_attention_replay needs.
MINIMAL_ATTENTION_REPLAY = '''"""Minimal fixture carrying the two frozen_wide_chunk_replay.py anchors."""


class ReplayAttentionReader:
    def __init__(self, operations, mesh, rows, capacity, pages_host, upload, *, max_group_rows=4,
                 short_context=False):
        grid = mesh.compute_with_storage_grid_size()
        config = operations.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
            exp_approx_mode=False, q_chunk_size=0, k_chunk_size=256)
'''


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
            'simulator-suite.sh', 'dspark_runtime_cache.py') + FILES
        originals = {name: subprocess.check_output(
            ['git', 'show', f'{REVISION}:scripts/ci/{name}']) for name in names}
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            scripts = checkout / 'scripts/ci'
            scripts.mkdir(parents=True)
            for name, source in originals.items():
                (scripts / name).write_bytes(source)
            (scripts / 'attention_replay.py').write_text(MINIMAL_ATTENTION_REPLAY)
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
            for name, checksum in report['after'].items():
                self.assertEqual(hashlib.sha256((scripts / name).read_bytes()).hexdigest(), checksum, name)
            self.assertNotIn('dspark_runtime_cache.py', report['after'])
            self.assertIn('frozen_binary_cache.py', report['after'])
            self.assertFalse(report['performance_qualified'])
            self.assertIn("max_seq_len=selected_geometry()['target_sequence_capacity']",
                (scripts / 'dspark-target-hardware.py').read_text())
            self.assertIn("geometry(context)['capacity']", (scripts / 'dspark_8k_scope.py').read_text())
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
            self.assertGreaterEqual(shape['target_page_count'] * shape['target_block_size'], context + 256)
            self.assertEqual(shape['target_cache_blocks'], shape['target_page_count'] + 8)
            if context < 65536:
                self.assertEqual((shape['target_page_count'], shape['target_cache_blocks']), (1024, 1032))
            else:
                self.assertEqual(shape['target_page_count'] * shape['target_block_size'], context + 256)
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
        reciprocal = adapt_scalar_reciprocal(baseline)
        self.assertNotIn('with scalar_reciprocal()', baseline['dspark-native-8k-attention-probe.py'])
        self.assertIn('with scalar_reciprocal(), scoped_stats_pack()', reciprocal['dspark-native-8k-attention-probe.py'])
        self.assertIn("reciprocal_variant='scalar-fp32'", reciprocal['dspark-native-8k-attention-probe.py'])
        compile(reciprocal['dspark-native-8k-attention-probe.py'], 'reciprocal-probe', 'exec')
        eager_only = adapt_eager_only(reciprocal)['dspark-native-8k-attention-probe.py']
        compile(eager_only, 'eager-only-probe', 'exec')
        self.assertIn("report['eager_complete'] = True", eager_only)
        self.assertIn("report['complete_probe_coverage'] = False\n        return", eager_only)
        self.assertLess(eager_only.index("report['eager_complete'] = True"), eager_only.index("progress('capture')"))
        for name in set(baseline) - {'dspark-native-8k-attention-probe.py'}:
            self.assertEqual(baseline[name], reciprocal[name])
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


class Rung65536CombinedRuntimeStagingTests(unittest.TestCase):
    """--combined-runtime at context 65536: the argparse guard, and where the staged
    tree lands on disk. See docs/t16-recipe-rung-65k.md."""

    def test_combined_runtime_contexts_are_32768_and_65536_only(self):
        self.assertEqual(COMBINED_RUNTIME_CONTEXTS, (32768, 65536))

    def test_staged_directory_is_unsuffixed_only_for_the_32768_default(self):
        checkout = Path('/checkout')
        self.assertEqual(combined_runtime_directory(checkout, 32768), checkout / 'scripts/ci')
        self.assertEqual(combined_runtime_directory(checkout, 65536), checkout / 'scripts/ci-65536')
        for context in (4096, 8192, 16384, 131072, 262144):
            with self.subTest(context=context), self.assertRaises(ValueError):
                combined_runtime_directory(checkout, context)

    def _run_main(self, checkout, manifest, *, context, combined_runtime):
        from frozen_runtime_context import FILES as RUNTIME_FILES
        from frozen_combined_adapters import FILES as COMBINED_FILES
        names = ('dspark_attention_chunk_trial.py', 'dspark-native-8k-attention-probe.py',
            'dspark_stats_pack.py', 'dspark_fp32_intermediates.py', 'run-simulator.sh',
            'simulator-suite.sh') + RUNTIME_FILES
        if combined_runtime:
            names += ('target-t16-attention-8k-probe.py', 'attention_mask_replay.py') + COMBINED_FILES
        originals = {name: subprocess.check_output(
            ['git', 'show', f'{REVISION}:scripts/ci/{name}']) for name in names}
        scripts = checkout / 'scripts/ci'
        scripts.mkdir(parents=True)
        for name, source in originals.items():
            (scripts / name).write_bytes(source)
        if context == 65536:
            (scripts / 'attention_replay.py').write_text(MINIMAL_ATTENTION_REPLAY)

        def git(command):
            arguments = command[3:]
            if arguments == ['rev-parse', 'HEAD']:
                return REVISION.encode() + b'\n'
            if arguments == ['status', '--porcelain', '--untracked-files=no']:
                return b''
            if arguments[0] == 'show':
                return originals[arguments[1].rsplit('/', 1)[1]]
            raise AssertionError(command)

        argv = ['adapter', '--checkout', str(checkout), '--context', str(context), '--manifest', str(manifest)]
        if combined_runtime:
            argv += ['--scalar-reciprocal', '--target-replay', '--combined-runtime']
        with patch('sys.argv', argv), \
                patch.object(frozen_recipe_context.subprocess, 'check_output', side_effect=git):
            frozen_recipe_context.main()
        return json.loads(manifest.read_text())

    def test_combined_runtime_65536_lands_in_its_own_directory_32768_stays_put(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            report = self._run_main(checkout, checkout / 'deployment-65536.json',
                context=65536, combined_runtime=True)
            self.assertEqual(report['staged_directory'], 'scripts/ci-65536')
            self.assertTrue((checkout / 'scripts/ci-65536/dspark_8k_admission.py').exists())
            # The 32768 path (the historical input checkout itself) must be untouched by a
            # 65536 combined-runtime run - both trees can coexist under one checkout.
            self.assertNotIn('enabled=request_context() == 65536',
                (checkout / 'scripts/ci/dspark_runtime_cache.py').read_text())
            self.assertIn('enabled=request_context() == 65536',
                (checkout / 'scripts/ci-65536/dspark_runtime_cache.py').read_text())
            for name, checksum in report['after'].items():
                self.assertEqual(hashlib.sha256(
                    (checkout / 'scripts/ci-65536' / name).read_bytes()).hexdigest(), checksum, name)

    def test_combined_runtime_32768_default_still_lands_at_the_historical_unsuffixed_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            report = self._run_main(checkout, checkout / 'deployment-32768.json',
                context=32768, combined_runtime=True)
            self.assertEqual(report['staged_directory'], 'scripts/ci')
            self.assertFalse((checkout / 'scripts/ci-32768').exists())
            self.assertIn('enabled=request_context() == 32768',
                (checkout / 'scripts/ci/dspark_runtime_cache.py').read_text())

    def test_non_combined_runtime_staging_is_unaffected_by_this_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            report = self._run_main(checkout, checkout / 'deployment-plain.json',
                context=65536, combined_runtime=False)
            self.assertEqual(report['staged_directory'], 'scripts/ci')
            self.assertFalse((checkout / 'scripts/ci-65536').exists())

    def test_combined_runtime_still_refuses_unsupported_contexts(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            (checkout / 'scripts/ci').mkdir(parents=True)
            manifest = checkout / 'deployment-rejected.json'
            argv = ['adapter', '--checkout', str(checkout), '--context', '131072',
                '--manifest', str(manifest), '--scalar-reciprocal', '--target-replay', '--combined-runtime']
            with patch('sys.argv', argv), self.assertRaises(SystemExit):
                frozen_recipe_context.main()
            self.assertFalse(manifest.exists())


if __name__ == '__main__':
    unittest.main()
