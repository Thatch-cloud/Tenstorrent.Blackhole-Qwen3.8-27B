import unittest

from dspark_center_fill_timed import SCREEN_RUN, summarize_timed
from test_target_t16_64k_timed import FoldedTimingTests


class CenterFillTimingTests(unittest.TestCase):
    def fixture(self):
        audited, values = FoldedTimingTests().fixture()
        for value in [audited, *values]:
            value['native_attention_kernel'] = {'original': {'header': 'native'},
                'patched': {'header': 'qualified'}, 'signature': {'0': 1}}
        return audited, values

    def test_full_decode_loop_and_corrected_audit(self):
        audited, values = self.fixture()
        result = summarize_timed(values, audited)
        self.assertEqual(result['audit_run'], SCREEN_RUN)
        self.assertTrue(result['direct_fp32_stage'])
        self.assertTrue(result['normalization_direct_stage'])
        self.assertTrue(result['center_tile_fill'])
        self.assertEqual(result['arms']['scatter']['committed_tg'], 19)
        self.assertFalse(result['performance_qualified'])

    def test_changed_kernel_or_failed_state_rejected(self):
        for name, setting in (('native_attention_kernel', {}), ('state_exact', False),
                ('max_new_tokens', 17), ('target_attention_t16', False)):
            audited, values = self.fixture()
            values[0][name] = setting
            with self.assertRaises(ValueError):
                summarize_timed(values, audited)

    def test_live_integer_signature_matches_serialized_audit(self):
        audited, values = self.fixture()
        values[0]['native_attention_kernel']['signature'] = {0: 1}
        self.assertTrue(summarize_timed(values, audited)['direct_fp32_stage'])
        values[0]['native_attention_kernel']['signature'] = {0: 2}
        with self.assertRaises(ValueError):
            summarize_timed(values, audited)


if __name__ == '__main__':
    unittest.main()
