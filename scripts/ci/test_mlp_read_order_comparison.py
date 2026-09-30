import copy
import unittest

from dspark_request_experiment import summarize as summarize_requests
from mlp_read_order_comparison import SCHEDULE, summarize


def fixture():
    return [dict(weight_read_order=dict(staggered=enabled), instrumented_timing=audit,
        arm='publication', length=4096, exact=True, state_exact=True, inactive_exact=True,
        prompt_tokens=[1] * 4096, emitted=list(range(13)), committed_decode_tokens=12,
        committed_tokens_per_second=None if audit else (120.0 if enabled else 100.0),
        decode_ms=100 if enabled else 120, prefill_ms=1200, feature_setup_ms=900,
        engine_setup_ms=2700, prefill_setup_decode_ms=5000, proposed=15, accepted=11,
        dspark=dict(proposal_trace=True), commit_only_gdn=True,
        blocks=[dict(position=4096, rows=16, input_tokens=list(range(16)), accepted=11, committed=12)])
        for enabled, audit in SCHEDULE]


class ReadOrderComparisonTests(unittest.TestCase):
    def test_complete_cycle_accounting_and_abba_pairs(self):
        audits, routes = [], []
        result = summarize(fixture(), summarize_requests, audits.append,
            lambda request, arm: routes.append((request, arm)))
        self.assertEqual(len(audits), 2)
        self.assertEqual(len(routes), 6)
        self.assertEqual(result['arms']['unchanged']['committed_tg'], 100)
        self.assertEqual(result['arms']['staggered']['committed_tg'], 120)
        self.assertTrue(result['improvement_screen_passed'])
        self.assertFalse(result['performance_promoted'])
        self.assertFalse(result['held_out_coding_quality'])

    def test_changed_acceptance_incomplete_schedule_and_bad_time_rejected(self):
        mutations = (
            lambda requests: requests.pop(),
            lambda requests: requests[1]['blocks'][0].update(accepted=10),
            lambda requests: requests[2].update(committed_tokens_per_second=float('nan')),
            lambda requests: requests[3].update(committed_tokens_per_second=200.0),
            lambda requests: requests[3].update(state_exact=False),
            lambda requests: requests[4].update(emitted=[42]),
        )
        for mutate in mutations:
            requests = copy.deepcopy(fixture())
            mutate(requests)
            with self.assertRaises(ValueError):
                summarize(requests, summarize_requests, lambda request: None, lambda request, arm: None)

    def test_one_regressing_pair_cannot_pass_improvement_screen(self):
        requests = fixture()
        requests[4].update(committed_tokens_per_second=90.0, decode_ms=1000 * 12 / 90)
        result = summarize(requests, summarize_requests, lambda request: None, lambda request, arm: None)
        self.assertFalse(result['improvement_screen_passed'])
