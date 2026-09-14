import os
from pathlib import Path
import unittest

import dspark_fp32_build
import dspark_ladder_build
import dspark_ladder_score_center
import native_draft_sdpa
from dspark_ladder_factory import scoped_stats_pack
from dspark_score_smoke_geometry import small_score_fixture
from dspark_score_sfpu import factory_transform, sfpu_score_center, factory_scope, kernel_scope


@unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned native sources required')
class SfpuScoreSourceTests(unittest.TestCase):
    def test_entry_scopes_apply_same_factory_and_kernel(self):
        root = Path(os.environ['TT_NATIVE_TEST_ROOT'])
        original = (root / dspark_fp32_build.SOURCE).read_bytes()
        sources = {name: (root / native_draft_sdpa.KERNEL_DIRECTORY / name).read_bytes()
            for name in native_draft_sdpa.SOURCE_HASHES}
        with small_score_fixture(), factory_scope(), kernel_scope():
            with dspark_ladder_build.factory_scope():
                candidate = dspark_fp32_build.transform(original)
            with dspark_ladder_score_center.scalar_score_center(key_tiles=40), scoped_stats_pack():
                patched = native_draft_sdpa.patched_sources(sources)['compute_common.hpp']
        self.assertIn(b'qwen_normalization_modes.at(cb_ids.qk_im)', candidate)
        self.assertIn(b'sfpu_sub_bcast_col(j, 1)', patched)
        self.assertNotIn(b'qwen_scalar_score_transform(in0_cb, in1_cb, rows * cols, cols, false)', patched)

    def test_factory_roundtrip_preserves_output_precision(self):
        root = Path(os.environ['TT_NATIVE_TEST_ROOT'])
        original = (root / dspark_fp32_build.SOURCE).read_bytes()
        with small_score_fixture(), dspark_ladder_build.factory_scope():
            baseline = dspark_fp32_build.transform(original)
        candidate = factory_transform(baseline)
        self.assertEqual(factory_transform(candidate, reverse=True), baseline)
        self.assertIn(b'qwen_normalization_modes.at(cb_ids.qk_im)', candidate)
        self.assertEqual([line for line in candidate.splitlines() if b'tt::DataFormat im_df =' in line],
            [line for line in baseline.splitlines() if b'tt::DataFormat im_df =' in line])

    def test_small_kernel_composition_removes_scalar_center_not_mask(self):
        root = Path(os.environ['TT_NATIVE_TEST_ROOT']) / native_draft_sdpa.KERNEL_DIRECTORY
        sources = {name: (root / name).read_bytes() for name in native_draft_sdpa.SOURCE_HASHES}
        with small_score_fixture(), dspark_ladder_score_center.scalar_score_center(key_tiles=40), \
                scoped_stats_pack(), sfpu_score_center(key_tiles=16):
            patched = native_draft_sdpa.patched_sources(sources)['compute_common.hpp']
        self.assertNotIn(b'qwen_scalar_score_transform(in0_cb, in1_cb, rows * cols, cols, false)', patched)
        self.assertIn(b'qwen_scalar_score_transform(cb_qk_im, cb_mask_in', patched)
        self.assertIn(b'sfpu_sub_bcast_col(j, 1)', patched)
        self.assertIn(b'dst_tiles = 1; granularity = cols;', patched)
        self.assertIn(b'CircularBuffer(get_compile_time_arg_val(42)).pop_front(1)', patched)
        self.assertIn(b'maximum == 0xff800000U ? 0U : maximum', patched)
