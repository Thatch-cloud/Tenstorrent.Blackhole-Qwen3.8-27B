import unittest
from unittest.mock import patch

from dflash_native_comparison_report import summarize, qualify_report, latency_budget
import test_proposal_native_request as proposal_fixtures
import test_drafter_comparison_report as target_fixtures


class NativeComparisonTests(unittest.TestCase):
    def test_budget_uses_committed_tokens_not_draft_rows(self):
        entries = [dict(committed_decode_tokens=11, decode_ms=110.,
            blocks=[dict(verify_readback_ms=62.5)])] * 2
        result = latency_budget(entries)
        self.assertEqual(result['allowed_mean_cycle_ms'], 55.)
        self.assertEqual(result['verification_only_upper_bound_tg'], 176.)
        self.assertEqual(result['required_decode_reduction_fraction'], .5)
        self.assertTrue(result['verifier_alone_exceeds_budget'])

    def test_independent_validator_rejects_incomplete_or_rewritten_report(self):
        report = dict(passed=True, closed_cleanly=True, drafter_comparison_sources={'source': 'hash'},
            drafter_comparison_sources_after={'source': 'hash'}, request_checks=[], dflash_native_comparison={'checked': True})
        with patch('dflash_native_comparison_report.summarize', return_value={'checked': True}):
            self.assertEqual(qualify_report(report), {'checked': True})
            for name, value in (('passed', False), ('closed_cleanly', False), ('error', 'failed'),
                    ('drafter_comparison_sources_after', {}), ('dflash_native_comparison', {})):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    qualify_report({**report, name: value})

    def records(self):
        records = proposal_fixtures.NativeProposalRequestTests().requests()
        target = target_fixtures.ComparisonReportTests().records()[0]
        for entry in records:
            entry.update(prompt_tokens=[10] * 4096, max_new_tokens=256)
            entry['dflash']['block_rows'] = 16
            for block in entry['blocks']:
                block['position'] += 4094
            for name in ('fused_t16_mlp', 'gdn_shared_qk', 'norm_reader', 'gdn_direct_window', 'mlp_down_grid'):
                entry[name] = target[name]
        return records

    def check(self, records):
        def measured(group):
            return dict(committed_tokens_per_second=100, committed_tokens=4, target_reached=False)

        with patch('full_dflash_request.summarize_dflash_requests', side_effect=measured) as validate, \
                patch('cumulative_fusion_validation.validate_fusion_policy'):
            result = summarize(records)
            self.assertEqual(validate.call_count, 2)
            self.assertEqual(validate.call_args_list[0].args[0], [records[index] for index in (0, 2, 5)])
            self.assertEqual(validate.call_args_list[1].args[0], [records[index] for index in (1, 3, 4)])
            return result

    def test_different_proposals_share_exact_committed_output(self):
        result = self.check(self.records())
        self.assertTrue(result['proposal_trajectories_may_differ'])
        self.assertEqual(result['control']['acceptance']['acceptance_fraction'], 1)
        self.assertEqual(result['candidate']['acceptance']['acceptance_fraction'], 0)
        self.assertEqual(result['candidate']['mean_block_costs_ms']['draft_ms'], 1)

    def test_context_target_sources_and_accounting_cannot_drift(self):
        for name, value in (('prompt_tokens', [1] * 2048), ('max_new_tokens', 513),
                ('norm_reader', {}), ('gdn_direct_window', {}), ('mlp_down_grid', {}),
                ('sources', {}), ('emitted', [12, 90, 14]), ('accepted', 500)):
            records = self.records()
            records[3][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.check(records)

    def test_t8_or_incomplete_schedule_cannot_be_relabelled(self):
        records = self.records()
        for entry in records:
            entry['dflash']['block_rows'] = 8
        with self.assertRaises(ValueError):
            self.check(records)
        with self.assertRaises(ValueError):
            self.check(self.records()[:-1])

    def test_each_policy_must_repeat_its_own_audit(self):
        records = self.records()
        records[3]['blocks'][0]['input_tokens'][1] = 98
        with self.assertRaisesRegex(ValueError, 'trajectory'):
            self.check(records)


if __name__ == '__main__':
    unittest.main()
