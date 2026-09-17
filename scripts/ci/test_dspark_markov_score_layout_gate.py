import copy
import json
from pathlib import Path
import unittest

from dspark_markov_score_layout_gate import validate


class FeedbackEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent
        self.report = json.loads((self.root / 'dspark-markov-score-layout-small-simulator.json').read_text())

    def test_small_evidence_matches_current_sources(self):
        result = validate(self.report, self.root, 64, '0\n')
        self.assertEqual(result, dict(vocabulary=64, proposals=15, eager_checks=90,
            replay_checks=120, input_checks=28, weight_checks=8))

    def test_every_matrix_is_required(self):
        for name in ('eager_checks', 'replay_checks', 'input_checks', 'weight_checks'):
            report = copy.deepcopy(self.report)
            report[name].pop()
            with self.assertRaises(ValueError):
                validate(report, self.root, 64, '0')

    def test_stale_failed_or_wrong_scope_rejected(self):
        mutations = (
            lambda report: report['sources_after'].clear(),
            lambda report: report['eager_checks'][0].update(token_exact=False),
            lambda report: report['replay_checks'][0].update(scores_exact=False),
            lambda report: report.update(stage='replay_3'),
            lambda report: report.update(closed_cleanly=False),
            lambda report: report.update(score_layout_gate={}),
        )
        for mutation in mutations:
            report = copy.deepcopy(self.report)
            mutation(report)
            with self.assertRaises(ValueError):
                validate(report, self.root, 64, '0')
        for vocabulary, status in ((248320, '0'), (64, '1')):
            with self.assertRaises(ValueError):
                validate(self.report, self.root, vocabulary, status)


if __name__ == '__main__':
    unittest.main()
