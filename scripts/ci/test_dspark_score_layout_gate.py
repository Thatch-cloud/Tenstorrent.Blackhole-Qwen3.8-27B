import copy
import json
from pathlib import Path
import unittest

from dspark_score_layout_gate import validate


class ScoreEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent
        self.report = json.loads((self.root / 'dspark-score-layout-small-simulator.json').read_text())

    def test_retained_small_evidence(self):
        result = validate(self.report, self.root, 64, '0\n')
        self.assertEqual(result, dict(vocabulary=64, eager_checks=24, replay_checks=18))

    def test_reject_incomplete_or_mutated_evidence(self):
        mutations = (
            lambda report: report['eager_checks'].pop(),
            lambda report: report['replay_checks'].pop(),
            lambda report: report['eager_checks'][0].update(cpu_exact=False),
            lambda report: report['replay_checks'][0].update(output_poison_replaced=False),
            lambda report: report.update(closed_cleanly=False),
            lambda report: report['sources'].update({'dspark_score_layout.py': 'stale'}),
            lambda report: report['sources_after'].clear(),
            lambda report: report.update(backend='hardware'),
            lambda report: report.update(stage='replay_2_14'),
        )
        for mutation in mutations:
            report = copy.deepcopy(self.report)
            mutation(report)
            with self.assertRaises(ValueError):
                validate(report, self.root, 64, '0')

    def test_small_report_cannot_qualify_full_vocabulary_or_failed_process(self):
        for vocabulary, status in ((248320, '0'), (64, '1')):
            with self.assertRaises(ValueError):
                validate(self.report, self.root, vocabulary, status)


if __name__ == '__main__':
    unittest.main()
