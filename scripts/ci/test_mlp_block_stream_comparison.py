import copy
import unittest
from unittest.mock import patch

from dflash_native_comparison_report import summarize
from test_dflash_native_comparison import NativeComparisonTests


class TransportComparisonTests(unittest.TestCase):
    def records(self):
        native = NativeComparisonTests().records()[1]
        records = [copy.deepcopy(native) for index in range(6)]
        for index, entry in enumerate(records):
            entry['instrumented_timing'] = index < 2
            if index in (1, 3, 4):
                entry['block_stream'] = dict(enabled=True)
        return records

    def test_only_transport_changes_and_complete_groups_are_measured(self):
        records = self.records()
        with patch('dflash_native_comparison_report.validate_target_components') as target, \
                patch('full_dflash_request.summarize_dflash_requests', return_value=dict(
                    committed_tokens_per_second=100, committed_tokens=4, target_reached=False)) as measured:
            result = summarize(records, weight_transport=True)
            self.assertFalse(result['proposal_trajectories_may_differ'])
            self.assertEqual(measured.call_count, 2)
            self.assertEqual([call.kwargs['block_stream'] for call in target.call_args_list],
                [False, True, False, True, True, False])
            records[3]['blocks'][0]['input_tokens'][-1] = 98
            with self.assertRaisesRegex(ValueError, 'proposals or acceptance'):
                summarize(records, weight_transport=True)

    def test_no_composed_draft_or_missing_transport_arm(self):
        for mutate in (lambda records: records[0].update(native_proposal_attention=False),
                lambda records: records[3].pop('block_stream')):
            records = self.records()
            mutate(records)
            with self.assertRaises(ValueError):
                summarize(records, weight_transport=True)


if __name__ == '__main__':
    unittest.main()
