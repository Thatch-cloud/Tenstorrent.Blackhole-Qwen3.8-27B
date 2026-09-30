"""Tests for frozen_wide_chunk_score_center.py's parameterized port of
dspark_ladder_score_center.scalar_score_center() - generated for both
accepted Skt values (unlike the ladder's own function, which refuses 2080)."""

import sys
import types
import unittest
from unittest.mock import patch


class ScoreCenterScopeTests(unittest.TestCase):

    def setUp(self):
        sys.modules.pop('frozen_wide_chunk_score_center', None)
        import frozen_wide_chunk_score_center
        self.mod = frozen_wide_chunk_score_center

    def tearDown(self):
        sys.modules.pop('frozen_wide_chunk_score_center', None)
        sys.modules.pop('native_draft_sdpa', None)

    def test_module_default_skt(self):
        self.assertEqual(self.mod.SKT, 2080)

    def test_helper_generation_has_no_key_tiles_restriction(self):
        """Unlike dspark_ladder_score_center.scalar_score_center(key_tiles=...),
        which raises for anything outside (40, 2112), this module's
        generators accept any Skt - confirmed here for both accepted values
        plus an arbitrary third, proving the restriction was not silently
        carried over."""
        for skt in (2080, 2112, 9999):
            with self.subTest(skt=skt):
                self.assertIn(f'get_compile_time_arg_val(3) == {skt}', self.mod._mask_after(skt))
                self.assertIn(f'get_compile_time_arg_val(3) == {skt}', self.mod._init_after(skt))
                self.assertIn(f'get_compile_time_arg_val(3) == {skt}', self.mod._subtract_after(skt))

    def test_scope_adds_five_substitutions_in_order(self):
        fake = types.ModuleType('native_draft_sdpa')
        fake.replacements = lambda: {'compute_common.hpp': ()}
        sys.modules['native_draft_sdpa'] = fake

        with self.mod.score_center_scope():
            result = fake.replacements()
        substitutions = result['compute_common.hpp']
        self.assertEqual(len(substitutions), 5)
        befores = [before for before, _ in substitutions]
        self.assertEqual(befores, ['#include <cstdint>', self.mod.HELPER_ANCHOR,
            self.mod.MASK, self.mod.INIT, self.mod.SUBTRACT])

    def test_helper_includes_debug_print_instrumentation_unconditionally(self):
        """Matches what dspark_ladder_score_center.scalar_score_center always
        does for either of its own two accepted key_tiles values - the
        `if key_tiles in (40, 2112):` guard around the DEVICE_PRINT wrapping
        there is unreachable-false given that function's own earlier
        validation, so this is not an optional feature being dropped."""
        self.assertIn('DEVICE_PRINT("QWEN_SCORE_ENTER', self.mod.HELPER)
        self.assertIn('DEVICE_PRINT("QWEN_SCORE_DONE', self.mod.HELPER)

    def test_scope_does_not_leak_after_exit(self):
        fake = types.ModuleType('native_draft_sdpa')
        original = lambda: {'compute_common.hpp': ()}
        fake.replacements = original
        sys.modules['native_draft_sdpa'] = fake
        with self.mod.score_center_scope():
            pass
        self.assertIs(fake.replacements, original)

    def test_respects_patched_skt_at_staging(self):
        fake = types.ModuleType('native_draft_sdpa')
        fake.replacements = lambda: {'compute_common.hpp': ()}
        sys.modules['native_draft_sdpa'] = fake
        with patch.object(self.mod, 'SKT', 2112):
            with self.mod.score_center_scope():
                result = fake.replacements()
        mask_after = result['compute_common.hpp'][2][1]
        self.assertIn('get_compile_time_arg_val(3) == 2112', mask_after)
        self.assertNotIn('get_compile_time_arg_val(3) == 2080', mask_after)


if __name__ == '__main__':
    unittest.main()
