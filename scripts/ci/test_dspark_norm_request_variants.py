import copy
import unittest

from dspark_norm_request_variants import SCHEDULE, summarize_variants
from test_dspark_request_variants import DSparkVariantTests


class NormRequestVariantTests(unittest.TestCase):
    def requests(self):
        existing = DSparkVariantTests().requests()
        records = []
        for arm, audit in SCHEDULE:
            value = copy.deepcopy(next(value for value in existing
                if value['arm'] == 'trace_commit' and value['instrumented_timing'] is audit))
            value['arm'] = arm
            value['dspark']['native_attention'] = True
            value['norm_scatter_kernel'] = dict(restored=True, loads=[{}]) if arm == 'scatter' else None
            records.append(value)
        return records

    def test_complete_matrix(self):
        result = summarize_variants(self.requests())
        self.assertEqual(set(result['arms']), {'control', 'scatter'})
        self.assertEqual(result['committed_tg_change_percent'], 0)
        self.assertFalse(result['serving_qualified'])

    def test_missing_override_or_restoration(self):
        for evidence in (None, {}, dict(restored=False, loads=[{}]), dict(restored=True, loads=[])):
            records = self.requests()
            records[1]['norm_scatter_kernel'] = evidence
            with self.assertRaises(ValueError):
                summarize_variants(records)

    def test_control_contamination(self):
        records = self.requests()
        records[0]['norm_scatter_kernel'] = dict(restored=True, loads=[{}])
        with self.assertRaises(ValueError):
            summarize_variants(records)

    def test_different_proposals_between_arms_rejected(self):
        records = self.requests()
        for value in records:
            if value['arm'] == 'scatter':
                value['blocks'][0]['input_tokens'][-1] = 99
        with self.assertRaises(ValueError):
            summarize_variants(records)

    def test_wrong_backend_or_state(self):
        for field in ('exact', 'state_exact', 'inactive_exact', 'commit_only_gdn'):
            records = self.requests()
            records[3][field] = False
            with self.assertRaises(ValueError):
                summarize_variants(records)
        records = self.requests()
        records[3]['dspark']['native_attention'] = False
        with self.assertRaises(ValueError):
            summarize_variants(records)


if __name__ == '__main__':
    unittest.main()
