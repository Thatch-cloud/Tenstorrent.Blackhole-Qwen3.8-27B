import unittest

from dspark_mlp_down_variants import POLICIES, summarize_variants
from test_dspark_target_attention_variants import TargetAttentionVariantTests


class DownVariantsTests(unittest.TestCase):
    def requests(self):
        records = TargetAttentionVariantTests().requests()
        for value in records:
            if value['arm'] == 'parallel':
                value['arm'] = 'down'
                value['down_mlp'] = dict(restored=True, native_bindings_unchanged=True,
                    layers=64, rows=16, hits=[3] * 64, setup_ms=100)
            for name in ('target_attention_t16', 'attention_replay', 'family_routing'):
                value[name] = True
        return records

    def test_only_mlp_changes_with_complete_matched_schedule(self):
        self.assertEqual(POLICIES['control'], POLICIES['down'])
        self.assertTrue(POLICIES['control']['target_attention_t16'])
        result = summarize_variants(self.requests())
        self.assertEqual(set(result['arms']), {'control', 'down'})
        self.assertEqual(result['committed_tg_change_percent'], 0)
        self.assertFalse(result['serving_qualified'])

    def test_missing_layer_restore_or_setup_cost_rejected(self):
        for failure in ('hit', 'restore', 'binding', 'setup', 'control', 'attention', 'proposal'):
            records = self.requests()
            audit = records[1]['down_mlp']
            if failure == 'hit':
                audit['hits'][17] = 0
            elif failure == 'restore':
                audit['restored'] = False
            elif failure == 'binding':
                audit['native_bindings_unchanged'] = False
            elif failure == 'setup':
                audit['setup_ms'] = float('nan')
            elif failure == 'control':
                records[0]['down_mlp'] = audit
            elif failure == 'attention':
                records[0]['target_attention_t16'] = False
            else:
                records[1]['blocks'][0]['input_tokens'][-1] = 99
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                summarize_variants(records)


if __name__ == '__main__':
    unittest.main()
