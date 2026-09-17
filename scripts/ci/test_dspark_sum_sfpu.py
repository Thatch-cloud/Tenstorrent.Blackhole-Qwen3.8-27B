import os
from pathlib import Path
import unittest

import dspark_fp32_build
import dspark_ladder_build
from dspark_ladder_factory import scoped_stats_pack
import dspark_ladder_score_center
from dspark_ladder_sum_update import scalar_sum_update
from dspark_score_sfpu import factory_scope, kernel_scope
from dspark_score_smoke_geometry import small_score_fixture
from dspark_sum_sfpu import sum_scope, factory_transform
import native_draft_sdpa


@unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned native source required')
class SumSourceTests(unittest.TestCase):
    def test_factory_and_kernel_composition(self):
        root = Path(os.environ['TT_NATIVE_TEST_ROOT'])
        original = (root / dspark_fp32_build.SOURCE).read_bytes()
        sources = {name: (root / native_draft_sdpa.KERNEL_DIRECTORY / name).read_bytes()
            for name in native_draft_sdpa.SOURCE_HASHES}
        with sum_scope(), small_score_fixture(), factory_scope(), kernel_scope():
            with dspark_ladder_build.factory_scope():
                factory = dspark_fp32_build.transform(original)
            with scalar_sum_update(), dspark_ladder_score_center.scalar_score_center(key_tiles=40), scoped_stats_pack():
                kernel = native_draft_sdpa.patched_sources(sources)['compute_common.hpp']
        restored = factory_transform(factory, reverse=True)
        self.assertEqual(factory_transform(restored), factory)
        self.assertIn(b'qwen_normalization_modes.at(qwen_sum_scratch_cb)', factory)
        self.assertIn(b'qwen_sum_update_sfpu(alias_prev_sum, alias_cur_sum, cb_exp_max_diff)', kernel)
        self.assertIn(b'add_binary_tile(1, 0, 0)', kernel)
        self.assertNotIn(b'current[offset] = current[offset] + previous[offset] * correction', kernel)
        self.assertLess(kernel.index(b'void qwen_stage_score_tile('), kernel.index(b'void qwen_sum_update_sfpu('))
        self.assertLess(kernel.index(b'void qwen_sum_update_sfpu('),
            kernel.index(b'qwen_sum_update_sfpu(alias_prev_sum'))


if __name__ == '__main__':
    unittest.main()
