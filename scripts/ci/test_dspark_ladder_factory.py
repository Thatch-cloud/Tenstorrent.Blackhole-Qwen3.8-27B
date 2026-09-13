import os
import re
from pathlib import Path
import unittest

from dspark_fp32_intermediates import SOURCE
from dspark_ladder_factory import KEY_TILES, geometry_predicate, scoped_stats_pack, selector_assert, transform


class LadderFactoryTests(unittest.TestCase):
    def test_chunk_sizes_are_paired_not_cross_product_admission(self):
        expression = geometry_predicate('keys', 'chunk')
        pairs = {(int(keys), int(chunk)) for keys, chunk in
            re.findall(r'\(keys == (\d+) && chunk == (\d+)\)', expression)}
        expected = {(40, 8), (168, 8), (272, 8), (296, 8), (1072, 16), (2112, 32)}
        self.assertEqual(pairs, expected)
        self.assertEqual(expression.count('||'), len(expected) - 1)
        for keys, chunk in expected:
            for other_chunk in {8, 16, 32} - {chunk}:
                self.assertNotIn((keys, other_chunk), pairs)

    def test_selectors_cover_only_explicit_ladder_and_existing_baseline(self):
        self.assertEqual(KEY_TILES, (40, 168, 272, 296, 1072, 2112))
        self.assertIn(geometry_predicate('get_compile_time_arg_val(3)', 'get_compile_time_arg_val(8)'), selector_assert())
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
        self.assertIn(geometry_predicate('Skt', 'Sk_chunk_t'), candidate)
        self.assertIn('fp32_dest_acc_en && !exp_approx_mode', candidate)
        self.assertIn('!is_causal && compute_use_provided_mask && !is_chunked', candidate)
        self.assertIn('(qwen_draft_fp32_intermediates && Skt == 2112 && Sk_chunk_t == 32)', candidate)
        self.assertIn('? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;', candidate)
        self.assertIn('cb_ids.out_im_A = allocate_tile_cb(out_im_tiles, im_tile_size, im_df)', candidate)
        self.assertIn('cb_ids.out_im_B = allocate_tile_cb(out_im_tiles, im_tile_size, im_df)', candidate)
        self.assertIn('cb_ids.out = allocate_tile_cb(out0_t, out_tile_size, out_df)', candidate)
        with self.assertRaises(ValueError):
            transform(original + b'\n')
