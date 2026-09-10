import copy
import unittest

from dspark_target_attention_variants import SCHEDULE, summarize_variants
from test_dspark_request_variants import DSparkVariantTests


class TargetAttentionVariantTests(unittest.TestCase):
    def requests(self):
        existing = DSparkVariantTests().requests()
        result = []
        for arm, audit in SCHEDULE:
            value = copy.deepcopy(next(entry for entry in existing
                if entry['arm'] == 'trace_commit' and entry['instrumented_timing'] is audit))
            value['arm'] = arm
            value['dspark']['native_attention'] = True
            for field in ('target_attention_t16', 'attention_replay', 'family_routing'):
                value[field] = arm == 'parallel'
            result.append(value)
        return result

    def test_complete_matched_schedule(self):
        result = summarize_variants(self.requests())
        self.assertEqual(set(result['arms']), {'control', 'parallel'})
        self.assertEqual(result['committed_tg_change_percent'], 0)
        self.assertFalse(result['serving_qualified'])

    def test_wrong_route_or_failed_state(self):
        for field in ('target_attention_t16', 'attention_replay', 'family_routing', 'exact', 'state_exact', 'inactive_exact'):
            records = self.requests()
            records[1][field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize_variants(records)

    def test_proposals_must_match_across_arms(self):
        records = self.requests()
        for value in records:
            if value['arm'] == 'parallel':
                value['blocks'][0]['input_tokens'][-1] = 99
        with self.assertRaises(ValueError):
            summarize_variants(records)


if __name__ == '__main__':
    unittest.main()
