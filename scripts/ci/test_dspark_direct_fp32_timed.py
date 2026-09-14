import unittest

from dspark_direct_fp32_timed import SCREEN_RUN, summarize_timed
from test_target_t16_64k_timed import FoldedTimingTests


class DirectTimingTests(unittest.TestCase):
    def fixture(self):
        audited, values = FoldedTimingTests().fixture()
        for value in [audited, *values]:
            value['native_attention_kernel'] = {'patched': {'header': 'qualified'}}
        return audited, values

    def test_full_decode_loop_and_corrected_audit(self):
        audited, values = self.fixture()
        result = summarize_timed(values, audited)
        self.assertEqual(result['audit_run'], SCREEN_RUN)
        self.assertTrue(result['direct_fp32_stage'])
        self.assertEqual(result['arms']['scatter']['committed_tg'], 19)
        self.assertFalse(result['performance_qualified'])

    def test_changed_kernel_or_failed_state_rejected(self):
        for name, setting in (('native_attention_kernel', {}), ('state_exact', False),
                ('max_new_tokens', 17), ('target_attention_t16', False)):
            audited, values = self.fixture()
            values[0][name] = setting
            with self.assertRaises(ValueError):
                summarize_timed(values, audited)


if __name__ == '__main__':
    unittest.main()
