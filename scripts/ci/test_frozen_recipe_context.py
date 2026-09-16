import subprocess
import unittest

from frozen_recipe_context import REVISION, adapt_probe_sources, geometry


class FrozenRecipeContextTests(unittest.TestCase):
    def test_geometry_preserves_output_headroom_and_chunk_size(self):
        for context in (8192, 32768, 65536):
            shape = geometry(context)
            self.assertEqual(shape['capacity'], context + 256)
            self.assertEqual(shape['storage_keys'], context + 320)
            self.assertEqual(shape['padded_keys'], context + 512)
            self.assertEqual(shape['key_chunk'], 256)
            self.assertEqual(shape['positions'][1] + 15, shape['capacity'])
        with self.assertRaises(ValueError):
            geometry(131072)

    def test_historical_source_changes_geometry_not_math_or_poison(self):
        names = ('dspark_attention_chunk_trial.py', 'dspark-native-8k-attention-probe.py')
        sources = {name: subprocess.check_output(
            ['git', 'show', f'{REVISION}:scripts/ci/{name}'], text=True) for name in names}
        self.assertEqual(adapt_probe_sources(sources, 8192), sources)
        for context in (32768, 65536):
            adapted = adapt_probe_sources(sources, context)
            for name, source in adapted.items():
                compile(source, name, 'exec')
            original = sources[names[0]]
            candidate = adapted[names[0]]
            self.assertEqual(original[original.index('    kernel ='):],
                candidate[candidate.index('    kernel ='):])
            self.assertIn('key_padding, 8192.', candidate)
            self.assertIn('key_padding, -8192.', candidate)
        broken = dict(sources)
        broken[names[0]] += '\nPADDED_KEYS = 8704\n'
        with self.assertRaises(ValueError):
            adapt_probe_sources(broken, 32768)


if __name__ == '__main__':
    unittest.main()
