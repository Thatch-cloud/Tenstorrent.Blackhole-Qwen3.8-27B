import copy
import unittest

from dspark_native_request_variants import SCHEDULE, summarize_variants
from test_dspark_request_variants import DSparkVariantTests


class NativeRequestVariantTests(unittest.TestCase):
    def requests(self):
        existing = DSparkVariantTests().requests()
        records = []
        for arm, audit in SCHEDULE:
            value = copy.deepcopy(next(value for value in existing
                if value['arm'] == 'trace_commit' and value['instrumented_timing'] is audit))
            value['arm'] = arm
            value['dspark']['native_attention'] = arm == 'native'
            records.append(value)
        return records

    def test_complete_matched_matrix_qualifies(self):
        result = summarize_variants(self.requests())
        self.assertEqual(set(result['arms']), {'composed', 'native'})
        self.assertEqual(result['committed_tg_change_percent'], 0)
        self.assertFalse(result['serving_qualified'])

    def test_corruption_or_wrong_backend_never_qualifies(self):
        for field in ('exact', 'state_exact', 'inactive_exact', 'commit_only_gdn'):
            records = self.requests()
            records[3][field] = False
            with self.assertRaises(ValueError):
                summarize_variants(records)
        records = self.requests()
        records[3]['dspark']['native_attention'] = False
        with self.assertRaises(ValueError):
            summarize_variants(records)

    def test_different_native_proposals_allowed_but_each_arm_must_repeat(self):
        records = self.requests()
        for value in records:
            if value['arm'] == 'native':
                value['blocks'][0]['input_tokens'][-1] = 99
        summarize_variants(records)
        records[3]['blocks'][0]['input_tokens'][-1] = 98
        with self.assertRaises(ValueError):
            summarize_variants(records)
