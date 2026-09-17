import copy
import unittest

from frozen_cycle_budget import summarize


class CycleBudgetTests(unittest.TestCase):
    def fixture(self):
        request = dict(arm='publication', exact=True, state_exact=True, inactive_exact=True,
            instrumented_timing=False, blocks=[dict(draft_ms=50, verify_readback_ms=74,
                select_commit_ms=22, cycle_ms=147, committed=11, blocking_trace_host_ms=73)])
        audit = copy.deepcopy(request)
        audit['instrumented_timing'] = True
        audit['blocks'][0]['cycle_ms'] = 14000
        return dict(passed=True, streams=1, ctx_tokens=32768, request_checks=[audit, request],
            request_comparison=dict(arms=dict(publication=dict(committed_tg=74))))

    def test_audit_excluded_and_nested_trace_not_added(self):
        result = summarize(self.fixture())['arms']['publication']
        self.assertEqual(result['blocks'], 1)
        self.assertEqual(result['mean']['cycle_ms'], 147)
        self.assertEqual(result['target_cycle_ms'], 55)
        self.assertEqual(result['request_tg'], 74)
        self.assertEqual(result['required_cycle_reduction_ms'], 92)
        self.assertAlmostEqual(result['hypothetical_zero_component_block_tg']['draft_ms'], 11000 / 97)
        self.assertAlmostEqual(result['hypothetical_zero_component_block_tg']['verify_readback_ms'], 11000 / 73)

    def test_invalid_component_budget_rejected(self):
        report = self.fixture()
        report['request_checks'][1]['blocks'][0]['draft_ms'] = 147
        with self.assertRaises(ValueError):
            summarize(report)

    def test_single_recipe_ladder_preserves_complete_request_tg(self):
        report = self.fixture()
        del report['request_comparison']
        report.update(fresh_context_audit=True, committed_tg=74,
            request_summary=dict(ctx=32768, committed_tg=74, measured_requests=1))
        result = summarize(report)['arms']['publication']
        self.assertEqual(result['request_tg'], 74)
        self.assertEqual(result['required_cycle_reduction_ms'], 92)
        for field, value in (('ctx', 8192), ('committed_tg', 200), ('measured_requests', 2)):
            changed = copy.deepcopy(report)
            changed['request_summary'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize(changed)

    def test_ladder_summary_cannot_be_used_for_multiple_arms(self):
        report = self.fixture()
        del report['request_comparison']
        report.update(fresh_context_audit=True, committed_tg=74,
            request_summary=dict(ctx=32768, committed_tg=74, measured_requests=1))
        report['request_checks'].append(dict(report['request_checks'][1], arm='control'))
        with self.assertRaises(ValueError):
            summarize(report)

    def test_failed_audit_rejected(self):
        report = self.fixture()
        report['request_checks'][0]['state_exact'] = False
        with self.assertRaises(ValueError):
            summarize(report)

    def test_unknown_timing_rejected(self):
        report = self.fixture()
        del report['request_checks'][1]['instrumented_timing']
        with self.assertRaises(ValueError):
            summarize(report)

    def test_nonfinite_measurement_rejected(self):
        report = self.fixture()
        report['request_checks'][1]['blocks'][0]['draft_ms'] = float('nan')
        with self.assertRaises(ValueError):
            summarize(report)


if __name__ == '__main__':
    unittest.main()
