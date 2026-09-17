import os
from pathlib import Path
import unittest

import native_draft_sdpa
from dspark_ladder_output_rounding import final_output_rounding
from dspark_ladder_scalar_reciprocal import scalar_reciprocal
from dspark_ladder_sum_update import scalar_sum_update
from dspark_ladder_score_center import scalar_score_center
from dspark_ladder_stage_print import stage_snapshots
from dspark_ladder_factory import scoped_stats_pack


class OutputRoundingTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned native source required')
    def test_full_composition_scopes_rounding_to_final_64k_output(self):
        directory = Path(os.environ['TT_NATIVE_TEST_ROOT']) / native_draft_sdpa.KERNEL_DIRECTORY
        sources = {name: (directory / name).read_bytes() for name in native_draft_sdpa.SOURCE_HASHES}
        with scalar_reciprocal(), scalar_sum_update(), stage_snapshots(row=3, column=0), \
                scalar_score_center(), final_output_rounding(), scoped_stats_pack():
            result = native_draft_sdpa.patched_sources(sources)['compute_common.hpp'].decode()
        self.assertEqual(result.count('bool round_output = false'), 1)
        self.assertEqual(result.count('if constexpr (round_output) typecast_tile<'), 1)
        self.assertEqual(result.count('if constexpr (round_output) typecast_tile_init<'), 1)
        self.assertIn('false, false, !QWEN_DRAFT_EXP_APPROX && get_compile_time_arg_val(3) == 2112>', result)
        self.assertIn('false, true>(\n                    alias_mm2_prev_out', result)
