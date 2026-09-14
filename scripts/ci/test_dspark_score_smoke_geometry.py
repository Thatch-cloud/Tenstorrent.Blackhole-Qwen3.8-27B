import unittest

import dspark_ladder_attention
import dspark_ladder_factory
import dspark_ladder_fixtures
import dspark_ladder_geometry
import dspark_ladder_score_center
import native_draft_sdpa
from dspark_score_smoke_geometry import small_score_fixture


class SmallScoreFixtureTests(unittest.TestCase):
    def test_small_fixture_matches_factory_and_score_selector(self):
        original = dspark_ladder_geometry.geometry
        with small_score_fixture():
            for module in (dspark_ladder_geometry, dspark_ladder_attention, dspark_ladder_fixtures):
                fixture = module.geometry(128)
                self.assertEqual(fixture['capacity'], 384)
                self.assertEqual(fixture['native_keys'], 512)
                self.assertEqual(fixture['probe_positions'], (128, 369))
                self.assertEqual(module.geometry(65536), original(65536))
            self.assertIn('(keys == 16 && chunk == 8)', dspark_ladder_factory.geometry_predicate('keys', 'chunk'))
            with dspark_ladder_score_center.scalar_score_center(key_tiles=40):
                sources = native_draft_sdpa.replacements()['compute_common.hpp']
                selected = [after for before, after in sources if before == '    sub_bcast_cols_init(in0_cb, in1_cb);']
                self.assertEqual(len(selected), 1)
                self.assertIn('== 16', selected[0])
                self.assertNotIn('== 40', selected[0])
        self.assertIs(dspark_ladder_geometry.geometry, original)
        self.assertEqual(original(128)['capacity'], 1152)
