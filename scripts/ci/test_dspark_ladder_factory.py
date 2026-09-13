import os
from pathlib import Path
import unittest

from dspark_fp32_intermediates import SOURCE
from dspark_ladder_factory import KEY_TILES, predicate, scoped_stats_pack, selector_assert, transform


class LadderFactoryTests(unittest.TestCase):
    def test_selectors_cover_only_explicit_ladder_and_existing_baseline(self):
        self.assertEqual(KEY_TILES, (40, 168, 272, 296, 1064, 2088))
        self.assertIn(predicate('get_compile_time_arg_val(3)'), selector_assert())
        self.assertIn('get_compile_time_arg_val(8) == 8', selector_assert())

    def test_scoped_pack_restores_after_failure(self):
        import dspark_stats_pack
        import native_draft_sdpa
        original = native_draft_sdpa.replacements
        assertion = dspark_stats_pack.SELECTOR_ASSERT
        with self.assertRaises(RuntimeError):
            with scoped_stats_pack():
                self.assertEqual(dspark_stats_pack.SELECTOR_ASSERT, selector_assert())
                raise RuntimeError('abort')
        self.assertIs(native_draft_sdpa.replacements, original)
        self.assertEqual(dspark_stats_pack.SELECTOR_ASSERT, assertion)

    @unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned native source required')
    def test_transform_retains_precision_guards_and_rejects_unknown_sources(self):
        original = (Path(os.environ['TT_NATIVE_TEST_ROOT']) / SOURCE).read_bytes()
        candidate = transform(original).decode()
        self.assertIn(predicate('Skt'), candidate)
        self.assertIn('fp32_dest_acc_en && !exp_approx_mode', candidate)
        self.assertIn('!is_causal && compute_use_provided_mask && !is_chunked', candidate)
        with self.assertRaises(ValueError):
            transform(original + b'\n')
