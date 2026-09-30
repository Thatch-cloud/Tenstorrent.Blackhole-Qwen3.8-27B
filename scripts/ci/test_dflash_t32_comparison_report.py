import unittest
from unittest.mock import patch

from dflash_native_comparison_report import acceptance
from dflash_t32_comparison_report import summarize, qualify_report, WIDTHS
from test_dflash_native_comparison import NativeComparisonTests


class T32ComparisonTests(unittest.TestCase):
    def records(self):
        records = NativeComparisonTests().records()
        for entry, rows in zip(records, WIDTHS):
            entry['native_proposal_attention'] = True
            entry['dflash'].update(block_rows=rows, native_proposal_attention=True)
            for block in entry['blocks']:
                block['input_tokens'] += [99] * (rows - block['rows'])
                block['rows'] = rows
            entry['proposed'] = sum(block['rows'] - 1 for block in entry['blocks'])
        return records

    def check(self, records):
        with patch('dflash_t32_comparison_report.validate_target_components') as control, \
                patch('dflash_t32_comparison_report.validate_t32') as candidate, \
                patch('full_dflash_request.summarize_dflash_requests', side_effect=lambda group: dict(
                    committed_tokens_per_second=100., committed_tokens=4, target_reached=False)) as measured:
            result = summarize(records)
            self.assertEqual(control.call_count, 3)
            self.assertEqual(candidate.call_count, 3)
            self.assertEqual(measured.call_count, 2)
            return result

    def test_widths_have_independent_acceptance_and_same_committed_output(self):
        result = self.check(self.records())
        self.assertTrue(result['proposal_trajectories_may_differ'])
        self.assertFalse(result['performance_promoted'])
        self.assertEqual(result['candidate']['acceptance']['accepted'], 0)
        self.assertEqual(result['control']['acceptance']['accepted'], 2)
        self.assertEqual(result['candidate']['latency_budget']['allowed_mean_cycle_ms'], 5.)

    def test_default_acceptance_still_rejects_t32(self):
        entry = self.records()[1]
        with self.assertRaises(ValueError):
            acceptance(entry)
        self.assertEqual(acceptance(entry, max_rows=32)['proposed'], 62)

    def test_mismatched_target_or_timing_order_rejected(self):
        for key, value in (('emitted', [12, 98, 14]), ('sources', {}),
                ('instrumented_timing', True), ('max_new_tokens', 513), ('accepted', 999)):
            records = self.records()
            records[3][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(records)

    def test_within_width_trajectory_must_repeat(self):
        records = self.records()
        records[3]['blocks'][0]['input_tokens'][1] = 98
        with self.assertRaisesRegex(ValueError, 'reproduce'):
            self.check(records)

    def test_report_recomputed_not_trusted(self):
        with self.assertRaises(ValueError):
            qualify_report(dict(passed=True, closed_cleanly=False))

    def test_report_requires_shared_pool_cleanup_and_recomputed_summary(self):
        report = dict(passed=True, closed_cleanly=True, streams=1, ctx_tokens=4096, sampler_links=4,
            drafter_comparison_sources={'source': 'hash'}, drafter_comparison_sources_after={'source': 'hash'},
            block_stream_pool=dict(released=True, native_bindings_unchanged=True, allocated_layers=64,
                serving_defaults_changed=False, admission=[{}, {}], setup_ms=1.),
            request_checks=[], dflash_t32_comparison={'checked': True})
        with patch('dflash_t32_comparison_report.summarize', return_value={'checked': True}):
            self.assertEqual(qualify_report(report), {'checked': True})
            for key, value in (('sampler_links', 1), ('dflash_t32_comparison', {}), ('block_stream_pool', {})):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    qualify_report({**report, key: value})


if __name__ == '__main__':
    unittest.main()
