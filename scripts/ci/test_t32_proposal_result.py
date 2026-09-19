import copy
import hashlib
from pathlib import Path
import unittest

from test_dspark_native_fixed_probe import load


class ProposalResultTests(unittest.TestCase):
    def setUp(self):
        self.validator = load('t32-proposal-result')
        sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.iterdir() if path.suffix in ('.py', '.cpp', '.sh')}
        self.report = dict(passed=True, closed_cleanly=True, stage='complete', score_layout='fused',
            learned_layers=5, proposals=31, full_request_qualified=False, sources=sources,
            sources_after=sources, attention={'fixture': True}, attention_after={'fixture': True},
            score_reference_checks=[dict(position=4096, tensors=6, exact=True,
                reference='native-score-eager') for ordinal in range(3)],
            replay_checks=[dict(position=4096, tensors=6, exact=True) for ordinal in range(2)],
            checks=[dict(anchor=anchor, exact=True, tokens=list(range(31))) for anchor in (20, 10)])

    def test_complete_fixture_is_not_a_performance_qualification(self):
        self.assertEqual(self.validator.validate(self.report),
            dict(passed=True, full_request_qualified=False, performance_qualified=False))

    def test_incomplete_and_mislabeled_results_fail(self):
        for field, value in (('closed_cleanly', False), ('score_layout', 'native'),
                ('score_reference_checks', []), ('replay_checks', []), ('sources_after', {}),
                ('checks', []), ('full_request_qualified', True)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validator.validate(dict(self.report, **{field: value}))
        report = copy.deepcopy(self.report)
        report['checks'][0]['tokens'][0] = True
        with self.assertRaises(ValueError):
            self.validator.validate(report)


if __name__ == '__main__':
    unittest.main()
