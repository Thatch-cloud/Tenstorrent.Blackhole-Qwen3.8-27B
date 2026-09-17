import unittest

from dspark_projection_precision_report import SOURCE_NAMES, validate, validate_set
from dspark_projection_precision_stage import PROJECTIONS


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
    def complete_set(self):
        sources = {name: 'a' * 64 for name in SOURCE_NAMES}
        reports = []
        for projection in PROJECTIONS:
            report = fixture()
            report.update(projection=projection, sources=dict(sources), sources_after=dict(sources),
                weight_sha256='b' * 64)
            reports.append(report)
        return reports, sources

    def test_complete_set_requires_exact_identity_without_quality_claim(self):
        reports, sources = self.complete_set()
        result = validate_set(list(reversed(reports)), expected_sources=sources, expected_weights={projection: 'b' * 64 for projection in PROJECTIONS})
        self.assertEqual(set(result['projections']), set(PROJECTIONS))
        self.assertFalse(result['target_correctness_qualified'])
        self.assertIsNone(result['committed_tg'])

    def test_set_rejects_missing_duplicate_changed_or_incomplete_evidence(self):
        for mutate in (lambda reports: reports.pop(),
                lambda reports: reports.append(reports[0]),
                lambda reports: reports[-1].update(projection=PROJECTIONS[0]),
                lambda reports: reports[-1]['sources'].update({'dspark_layer.py': 'c' * 64}),
                lambda reports: reports[-1].update(weight_sha256='c' * 64),
                lambda reports: reports[-1]['replay_checks'].pop()):
            reports, sources = self.complete_set()
            mutate(reports)
            with self.assertRaises(ValueError):
                validate_set(reports, expected_sources=sources, expected_weights={projection: 'b' * 64 for projection in PROJECTIONS})

    def test_set_rejects_invalid_external_identities(self):
        reports, sources = self.complete_set()
        for invalid in ('', '0' * 64, 'g' * 64, None):
            with self.assertRaises(ValueError):
                validate_set(reports, expected_sources=sources, expected_weights={projection: invalid for projection in PROJECTIONS})
        sources.pop(SOURCE_NAMES[0])
        with self.assertRaises(ValueError):
            validate_set(reports, expected_sources=sources, expected_weights={projection: 'b' * 64 for projection in PROJECTIONS})

    def test_projection_identity_cannot_be_substituted(self):
        report = fixture()
        report['projection'] = 'mlp.down_proj.weight'
        with self.assertRaises(ValueError):
            validate(report)
        self.assertTrue(validate(report, projection='mlp.down_proj.weight')['component_execution_passed'])

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
