import unittest

from dspark_combined_variants import POLICIES, summarize_variants
from test_dspark_norm_request_variants import NormRequestVariantTests


class CombinedVariantTests(unittest.TestCase):
    def requests(self):
        records = NormRequestVariantTests().requests()
        for value in records:
            value.update(target_attention_t16=True, attention_replay=True, family_routing=True)
        return records

    def test_both_policies_keep_parallel_attention_enabled(self):
        self.assertEqual(set(POLICIES), {'control', 'scatter'})
        for policy in POLICIES.values():
            self.assertIs(policy['target_attention_t16'], True)
            self.assertIs(policy['native_attention'], True)
            self.assertIs(policy['commit_only_gdn'], True)
        result = summarize_variants(self.requests())
        self.assertEqual(result['committed_tg_change_percent'], 0)
        self.assertFalse(result['serving_qualified'])

    def test_missing_parallel_attention_rejected(self):
        for field in ('target_attention_t16', 'attention_replay', 'family_routing'):
            records = self.requests()
            records[0][field] = False
            with self.assertRaises(ValueError):
                summarize_variants(records)

    def test_missing_norm_engagement_rejected(self):
        records = self.requests()
        records[1]['norm_scatter_kernel'] = None
        with self.assertRaises(ValueError):
            summarize_variants(records)


if __name__ == '__main__':
    unittest.main()
