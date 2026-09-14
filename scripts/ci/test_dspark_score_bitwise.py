from unittest.mock import patch

import test_dspark_ladder_score_center as baseline
from dspark_score_bitwise import bitwise_infinity_checks, transform


class BitwiseScoreTests(baseline.ScoreCenterTests):
    def setUp(self):
        self.enterContext(patch.object(baseline, 'STUB', transform(baseline.STUB)))
        self.enterContext(bitwise_infinity_checks())

    def test_unknown_source_rejected(self):
        with self.assertRaises(ValueError):
            transform('')
