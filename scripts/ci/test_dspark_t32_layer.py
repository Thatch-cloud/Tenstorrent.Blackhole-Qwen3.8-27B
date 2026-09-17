import unittest
from unittest.mock import patch

import dspark_t32_layer as candidate
from dspark_t32_inputs import fixed_mask, geometry
import test_dspark_cached_layer as baseline_tests
from test_dspark_history import require_tensor


class T32LayerTests(unittest.TestCase):
    def test_layer_keeps_history_rotary_and_projection_boundaries(self):
        fixture = baseline_tests.DSparkCachedLayerTests()
        with patch.object(baseline_tests, 'cached', candidate), \
                patch.object(baseline_tests, 'geometry', geometry), \
                patch.object(baseline_tests, 'full_mask', side_effect=lambda position, proposals:
                    fixed_mask(position, position, proposals)), \
                patch('dspark_t32_attention.require_tensor', side_effect=require_tensor):
            for position in (64, 4384):
                with self.subTest(position=position):
                    fixture.run_layer(position, 31)


if __name__ == '__main__':
    unittest.main()
