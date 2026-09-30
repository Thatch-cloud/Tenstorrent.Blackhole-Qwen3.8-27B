"""Tests for frozen_wide_chunk_sum_update.py's parameterized port of
dspark_ladder_sum_update.scalar_sum_update()."""

import sys
import types
import unittest


class SumUpdateScopeTests(unittest.TestCase):

    def setUp(self):
        sys.modules.pop('frozen_wide_chunk_sum_update', None)
        import frozen_wide_chunk_sum_update
        self.mod = frozen_wide_chunk_sum_update

    def tearDown(self):
        sys.modules.pop('frozen_wide_chunk_sum_update', None)
        sys.modules.pop('native_draft_sdpa', None)

    def test_module_default_skt(self):
        self.assertEqual(self.mod.SKT, 2080)

    def test_scope_adds_exactly_one_substitution_gated_on_module_skt(self):
        fake = types.ModuleType('native_draft_sdpa')
        fake.replacements = lambda: {'compute_common.hpp': ()}
        sys.modules['native_draft_sdpa'] = fake

        with self.mod.sum_update_scope():
            result = fake.replacements()
        substitutions = result['compute_common.hpp']
        self.assertEqual(len(substitutions), 1)
        before, after = substitutions[0]
        self.assertEqual(before, self.mod.BEFORE)
        self.assertIn(f'get_compile_time_arg_val(3) == {self.mod.SKT}', after)
        self.assertIn('current[offset] = current[offset] + previous[offset] * correction', after)
        # The original bf16 call is preserved as the else-branch fallback.
        self.assertIn(self.mod.BEFORE, after)

    def test_scope_does_not_leak_after_exit(self):
        fake = types.ModuleType('native_draft_sdpa')
        original = lambda: {'compute_common.hpp': ()}
        fake.replacements = original
        sys.modules['native_draft_sdpa'] = fake
        with self.mod.sum_update_scope():
            pass
        self.assertIs(fake.replacements, original)

    def test_respects_patched_skt_at_staging(self):
        """Simulates what frozen_wide_chunk_normalization._patch_skt_constant
        does at staging time: a text substitution of the module source
        changes SKT before the file is ever imported. Here, verified at the
        import level by patching the constant directly and re-checking the
        generated text uses the new value - the actual staging patch is
        tested against real source text in test_frozen_wide_chunk_normalization.py."""
        from unittest.mock import patch
        fake = types.ModuleType('native_draft_sdpa')
        fake.replacements = lambda: {'compute_common.hpp': ()}
        sys.modules['native_draft_sdpa'] = fake
        with patch.object(self.mod, 'SKT', 2112):
            with self.mod.sum_update_scope():
                result = fake.replacements()
        _, after = result['compute_common.hpp'][0]
        self.assertIn('get_compile_time_arg_val(3) == 2112', after)
        self.assertNotIn('get_compile_time_arg_val(3) == 2080', after)


if __name__ == '__main__':
    unittest.main()
