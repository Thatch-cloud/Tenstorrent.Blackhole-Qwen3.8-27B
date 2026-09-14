from unittest.mock import patch
from pathlib import Path

import test_dspark_ladder_score_center as baseline
from dspark_score_bitwise import bitwise_infinity_checks, transform


class BitwiseScoreTests(baseline.ScoreCenterTests):
    def setUp(self):
        self.enterContext(patch.object(baseline, 'STUB', transform(baseline.STUB)))
        self.enterContext(bitwise_infinity_checks())

    def test_unknown_source_rejected(self):
        with self.assertRaises(ValueError):
            transform('')

    def test_smoke_exits_before_full_ladder(self):
        directory = Path(__file__).parent
        suite = (directory / 'simulator-suite.sh').read_text()
        start = suite.index('if [ "${QWEN_SCORE_BITWISE:-0}" = 1 ]; then')
        end = suite.index('fi', start)
        branch = suite[start:end]
        self.assertIn('timeout -k 15 105', branch)
        self.assertIn('exit 0', branch)
        self.assertLess(end, suite.index('for context in 65536'))
        workflow = (directory.parents[1] / '.github/workflows/qwen-ttsim.yml').read_text()
        self.assertIn('timeout -k 15 465 bash scripts/ci/run-simulator.sh', workflow)
