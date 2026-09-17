import unittest

from dspark_projection_precision_report import validate


def fixture():
    return dict(passed=True, closed_cleanly=True, stage='complete', backend='simulator',
        projection='self_attn.q_proj.weight', component_execution_only=True,
        target_correctness_qualified=False, committed_tg=None,
        sources={'probe': 'a' * 64}, sources_after={'probe': 'a' * 64},
        differences=[dict(case=case, chip=chip, finite=True, exact=False, max_abs=.1,
            rms_error=.01, reference_rms=1.) for case in range(2) for chip in range(2)],
        replay_checks=[dict(ordinal=ordinal, case=case, chip=chip, exact=True, poison_replaced=True)
            for ordinal, case in enumerate((1, 0, 1)) for chip in range(2)],
        integrity_checks=[dict(case=case, mode=mode, exact=True)
            for mode, cases in (('eager', (0, 1)), ('replay', (1, 0, 1))) for case in cases])


class PrecisionReportTests(unittest.TestCase):
    def test_execution_pass_does_not_qualify_target_quality(self):
        result = validate(fixture())
        self.assertTrue(result['component_execution_passed'])
        self.assertFalse(result['target_correctness_qualified'])
        self.assertIsNone(result['committed_tg'])

    def test_partial_or_nonfinite_evidence_rejected(self):
        for mutate in (lambda report: report['replay_checks'].pop(),
                lambda report: report['integrity_checks'].pop(),
                lambda report: report['differences'][0].update(max_abs=float('nan')),
                lambda report: report['sources_after'].clear(),
                lambda report: report.update(target_correctness_qualified=True),
                lambda report: report.update(committed_tg=200)):
            report = fixture()
            mutate(report)
            with self.assertRaises(ValueError):
                validate(report)
