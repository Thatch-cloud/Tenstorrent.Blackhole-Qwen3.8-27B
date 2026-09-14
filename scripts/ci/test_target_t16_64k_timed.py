import unittest

from test_dspark_sfpu_timed_requests import TimedRequestTests
from target_t16_64k_timed import SCREEN_RUN, summarize_timed


class FoldedTimingTests(unittest.TestCase):
    def fixture(self):
        audited, values = TimedRequestTests().fixture()
        for value in values:
            value.update(target_attention_t16=True, attention_replay=True, family_routing=True,
                capture_count=5, max_new_tokens=256)
        return audited, values

    def test_complete_loop_metrics_and_candidate_identity(self):
        audited, values = self.fixture()
        summary = summarize_timed(values, audited)
        self.assertEqual(summary['audit_run'], SCREEN_RUN)
        self.assertTrue(summary['target_attention_t16'])
        self.assertEqual(summary['arms']['scatter']['committed_tg'], 19)
        self.assertFalse(summary['performance_qualified'])

    def test_native_fallback_or_shortened_budget_is_rejected(self):
        for name, setting in (('target_attention_t16', False), ('attention_replay', False),
                ('family_routing', False), ('capture_count', 4), ('max_new_tokens', 17),
                ('state_exact', False)):
            audited, values = self.fixture()
            values[0][name] = setting
            with self.assertRaises(ValueError):
                summarize_timed(values, audited)


if __name__ == '__main__':
    unittest.main()
