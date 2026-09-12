import unittest

from dspark_mlp_down_variants import POLICIES, summarize_variants
from test_dspark_target_attention_variants import TargetAttentionVariantTests


class DownVariantsTests(unittest.TestCase):
    def footprint_requests(self):
        records = self.requests()
        for value in records:
            enabled = value['arm'] == 'down'
            value['down_mlp_equal_footprint'] = True
            value['down_mlp'] = dict(restored=True, native_bindings_unchanged=True, layers=64, rows=16,
                hits=[3 if enabled else 0] * 64, setup_ms=100, enabled=enabled,
                prepared_bindings_unchanged=True, prepared_bindings=[[index * 1024] * 2 for index in range(64)])
        return records

    def test_equal_footprint_is_labelled_and_addresses_reported(self):
        records = self.footprint_requests()
        result = summarize_variants(records)
        self.assertTrue(result['equal_resident_weight_footprint'])
        self.assertTrue(result['timed_weight_addresses_match'])
        self.assertFalse(result['serving_qualified'])
        records[3]['down_mlp']['prepared_bindings'][0][0] += 32
        self.assertFalse(summarize_variants(records)['timed_weight_addresses_match'])

    def test_footprint_control_cannot_execute_candidate_or_skip_preparation(self):
        for failure in ('hit', 'enabled', 'bindings', 'policy', 'stable'):
            records = self.footprint_requests()
            audit = records[0]['down_mlp']
            if failure == 'hit':
                audit['hits'][0] = 1
            elif failure == 'enabled':
                audit['enabled'] = True
            elif failure == 'bindings':
                audit['prepared_bindings'].pop()
            elif failure == 'policy':
                records[0]['down_mlp_equal_footprint'] = False
            else:
                audit['prepared_bindings_unchanged'] = False
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                summarize_variants(records)

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
