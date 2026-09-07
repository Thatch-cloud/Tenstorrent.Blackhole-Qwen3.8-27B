import unittest

from verification_budget import verification_budget


class VerificationBudgetTests(unittest.TestCase):
    def report(self, samples=(63., 64., 65.)):
        return dict(passed=True, timings=[dict(length=4095, rows=8, blocks=[dict(batch_ms=value) for value in samples])])

    def test_verifier_only_ceiling_exposes_negative_draft_budget(self):
        result = verification_budget(self.report())['bounds'][0]
        self.assertEqual(result['zero_draft_overhead_full_acceptance_ceiling_tps'], 125)
        self.assertEqual(result['cycle_budget_ms'], 40)
        self.assertEqual(result['remaining_draft_commit_budget_ms'], -24)
        self.assertEqual(result['minimum_verifier_reduction_fraction'], .375)

    def test_positive_budget_is_not_a_throughput_measurement(self):
        result = verification_budget(self.report((20.,)))
        self.assertIn('not measured', result['scope'])
        self.assertEqual(result['bounds'][0]['remaining_draft_commit_budget_ms'], 20)

    def test_missing_invalid_or_failed_evidence_rejected(self):
        for report in (dict(passed=False), self.report(()), self.report((0.,)), self.report((float('nan'),))):
            with self.assertRaises(ValueError):
                verification_budget(report)
        with self.assertRaises(ValueError):
            verification_budget(self.report(), rows=16)
