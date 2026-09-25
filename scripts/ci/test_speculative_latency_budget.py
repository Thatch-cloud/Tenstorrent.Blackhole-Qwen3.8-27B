import unittest

from speculative_latency_budget import analyze


class LatencyBudgetTests(unittest.TestCase):
    def block(self, **updates):
        return dict(dict(rows=16, committed=12, draft_ms=25, verify_readback_ms=66,
            select_commit_ms=5, cycle_ms=100), **updates)

    def test_budget_includes_full_cycle_not_sum_of_selected_phases(self):
        result = analyze([self.block(), self.block(committed=8)])
        self.assertEqual(result['observed_block_tg'], 100)
        self.assertEqual(result['required_cycle_ms'], 50)
        self.assertEqual(result['required_cycle_reduction_percent'], 50)
        self.assertEqual(result['required_committed_at_current_latency'], 20)
        self.assertAlmostEqual(result['zero_draft_ceiling_tg'], 10000 / 75)

    def test_reject_invalid_or_mixed_measurements(self):
        for blocks in ([], [self.block(rows=1)], [self.block(committed=17)],
                [self.block(cycle_ms=0)], [self.block(draft_ms=float('nan'))],
                [self.block(), self.block(rows=32)], [self.block(verify_readback_ms=101)]):
            with self.assertRaises(ValueError):
                analyze(blocks)
