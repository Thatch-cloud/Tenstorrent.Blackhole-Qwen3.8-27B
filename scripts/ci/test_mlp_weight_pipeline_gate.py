import copy
import unittest

from mlp_weight_pipeline_gate import validate_report
from test_mlp_clock_report import fixture


class ReadOrderGateTests(unittest.TestCase):
    def test_complete_numerical_report(self):
        report = fixture()
        del report['clock_samples']
        del report['missing_execution_rejected']
        validate_report(report)

    def test_missing_or_corrupt_matrix_rejected(self):
        mutations = (
            lambda report: report.update(passed=False),
            lambda report: report.update(backend='hardware'),
            lambda report: report.update(math_approx_mode=False),
            lambda report: report['checks'].pop(),
            lambda report: report['trace_replays'][0]['checks'].pop(),
            lambda report: report['trace_replays'][0]['negative_controls'].pop(),
            lambda report: report['weight_checks'][0].update(mismatched_words=1),
            lambda report: report['weight_checks'][0].update(pages=1),
            lambda report: report['weight_checks'].append(copy.deepcopy(report['weight_checks'][0])),
        )
        for mutate in mutations:
            report = fixture()
            mutate(report)
            with self.assertRaises(ValueError):
                validate_report(report)
