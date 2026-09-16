import subprocess
import unittest
import os
from unittest.mock import patch

from frozen_recipe_context import REVISION, adapt_probe_sources, adapt_cache_launcher, geometry
from frozen_context_geometry import CONTEXTS, selected_geometry, factory_selector


class FrozenRecipeContextTests(unittest.TestCase):
    def test_cache_launcher_preserves_bounded_original_probe(self):
        names = ('run-simulator.sh', 'simulator-suite.sh')
        sources = {name: subprocess.check_output(
            ['git', 'show', f'{REVISION}:scripts/ci/{name}'], text=True) for name in names}
        result = adapt_cache_launcher(sources)
        self.assertIn('frozen_sim_build_cache.py', result['simulator-suite.sh'])
        self.assertIn('dst=/frozen-simulator-cache', result['run-simulator.sh'])
        self.assertIn('"/experiment-scripts/ci/$QWEN_SIM_CASE-probe.py"', result['simulator-suite.sh'])
        self.assertNotIn('--device', result['run-simulator.sh'])

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
