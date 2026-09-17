import os
from pathlib import Path
import unittest

import dspark_score_sfpu
import native_draft_sdpa
from dspark_direct_fp32_stage import REPLACEMENT, staging_scope, transform


class StagingTests(unittest.TestCase):
    def test_only_staging_changes_and_restores(self):
        original = dspark_score_sfpu.HELPER
        with staging_scope():
            candidate = dspark_score_sfpu.HELPER
            self.assertIn(REPLACEMENT, candidate)
            self.assertNotIn('scratch[index] = source[index];', candidate)
            self.assertEqual(candidate.split('void qwen_prepare_center_scratch')[1],
                original.split('void qwen_prepare_center_scratch')[1])
        self.assertEqual(dspark_score_sfpu.HELPER, original)
        with self.assertRaises(ValueError):
            transform(candidate)

    @unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned native sources required')
    def test_composes_with_mixed_attention_boundary(self):
        from dspark_attention_header_boundary import SEPARATOR, SUFFIX, boundary_scope
        from dspark_ladder_factory import scoped_stats_pack
        from dspark_ladder_sum_update import scalar_sum_update
        import dspark_ladder_score_center
        from dspark_mask_bits import mask_scope
        from dspark_score_bitwise import bitwise_infinity_checks
        from dspark_score_smoke_geometry import small_score_fixture
        from dspark_sum_sfpu import sum_scope

        root = Path(os.environ['TT_NATIVE_TEST_ROOT'])
        sources = {name: (root / native_draft_sdpa.KERNEL_DIRECTORY / name).read_bytes()
            for name in native_draft_sdpa.SOURCE_HASHES}
        with staging_scope(), boundary_scope(root), mask_scope(), sum_scope(), small_score_fixture(), \
                bitwise_infinity_checks(), dspark_score_sfpu.kernel_scope(), scalar_sum_update(), \
                dspark_ladder_score_center.scalar_score_center(key_tiles=40), scoped_stats_pack():
            patched = native_draft_sdpa.patched_sources(sources)['compute_common.hpp']
        self.assertTrue(REPLACEMENT.encode() in patched, 'Direct staging must reach the generated kernel')
        self.assertTrue(b'qwen_stage_score_tile(in0_cb, QWEN_SCORE_SCRATCH_CB, true);' in patched)
        self.assertTrue(b'qwen_stage_score_tile(previous_cb, previous_scratch);' in patched)
        self.assertEqual(patched.split(SEPARATOR.encode())[1][:-len(SUFFIX)], sources['compute_common.hpp'])


if __name__ == '__main__':
    unittest.main()
