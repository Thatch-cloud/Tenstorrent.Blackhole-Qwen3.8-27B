import os
from pathlib import Path
import unittest

import native_draft_sdpa
from dspark_fp32_intermediates import SOURCE
from dspark_ladder_factory import scoped_stats_pack, transform
from dspark_ladder_normalization import factory_transform, scratch_normalization
from dspark_ladder_scalar_reciprocal import scalar_reciprocal
from dspark_ladder_score_center import scalar_score_center
from dspark_ladder_stage_print import stage_snapshots
from dspark_ladder_sum_update import scalar_sum_update


@unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned native source required')
class NormalizationTests(unittest.TestCase):
    def test_factory_roundtrip_and_only_scratch_uses_direct_unpack(self):
        root = Path(os.environ['TT_NATIVE_TEST_ROOT'])
        original = transform((root / SOURCE).read_bytes())
        candidate = factory_transform(original)
        self.assertEqual(factory_transform(candidate, reverse=True), original)
        self.assertEqual(candidate.count(b'UnpackToDestFp32'), 1)
        self.assertIn(b'qwen_normalization_modes.at(cb_ids.recip_scratch)', candidate)
        self.assertIn(b'qwen_draft_fp32_intermediates && Skt == 2112', candidate)
        with self.assertRaises(ValueError):
            factory_transform(candidate)

    def test_composed_kernel_retains_original_path_and_pinned_scratch_argument(self):
        root = Path(os.environ['TT_NATIVE_TEST_ROOT']) / native_draft_sdpa.KERNEL_DIRECTORY
        sources = {name: (root / name).read_bytes() for name in native_draft_sdpa.SOURCE_HASHES}
        self.assertIn(b'constexpr uint32_t cb_arg_offset = 34;', sources['sdpa.cpp'])
        self.assertIn(b'cb_recip_scratch = get_compile_time_arg_val(cb_arg_offset + 8)', sources['sdpa.cpp'])
        with scalar_reciprocal(), scalar_sum_update(), scalar_score_center(), \
                stage_snapshots(row=3, column=0), scratch_normalization(), scoped_stats_pack():
            candidate = native_draft_sdpa.patched_sources(sources)['compute_common.hpp']
        self.assertEqual(candidate.count(b'qwen_normalize_scratch<vDHt>'), 1)
        self.assertIn(b'get_compile_time_arg_val(42), cb_out)', candidate)
        self.assertIn(b'mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>', candidate)
        self.assertIn(b'static_assert(Sq_chunk_t == 1)', candidate)
        self.assertLess(candidate.index(b'CircularBuffer(scratch_cb).wait_front(1)'),
            candidate.index(b'scratch[index] = source[index]'))
