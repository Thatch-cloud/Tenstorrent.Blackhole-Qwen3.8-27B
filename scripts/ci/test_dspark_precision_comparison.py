import unittest

from dspark_precision_comparison import SCHEDULE, summarize
from dspark_request_experiment import summarize as summarize_requests


def fixture():
    requests = []
    for precision, audit in SCHEDULE:
        candidate = precision == 'hifi2'
        blocks = ([dict(position=4096, rows=16, input_tokens=list(range(16)), accepted=5, committed=6),
            dict(position=4102, rows=16, input_tokens=list(range(1, 17)), accepted=5, committed=6)]
            if candidate else [dict(position=4096, rows=16, input_tokens=list(range(16)), accepted=11, committed=12)])
        requests.append(dict(drafter_precision=precision, instrumented_timing=audit,
            arm='publication', length=4096, exact=True, state_exact=True, inactive_exact=True,
            prompt_tokens=[1] * 4096, emitted=list(range(13)), committed_decode_tokens=12,
            committed_tokens_per_second=None if audit else (120.0 if candidate else 100.0),
            decode_ms=100 if candidate else 120, prefill_ms=1200, feature_setup_ms=900,
            engine_setup_ms=2700, prefill_setup_decode_ms=5000, proposed=30 if candidate else 15,
            accepted=10 if candidate else 11, dspark=dict(proposal_trace=True),
            commit_only_gdn=True, blocks=blocks))
    return requests


class PrecisionComparisonTests(unittest.TestCase):
    def test_different_proposals_and_acceptance_allowed_between_arms(self):
        audits, routes = [], []
        result = summarize(fixture(), summarize_requests, audits.append,
            lambda request, arm: routes.append((request, arm)))
        self.assertEqual(len(audits), 2)
        self.assertEqual(len(routes), 6)
        self.assertEqual(result['arms']['hifi4']['committed_tg'], 100)
        self.assertEqual(result['arms']['hifi2']['committed_tg'], 120)
        self.assertNotEqual(result['arms']['hifi4']['acceptance'], result['arms']['hifi2']['acceptance'])
        self.assertTrue(result['improvement_screen_passed'])
        self.assertFalse(result['performance_promoted'])
        self.assertFalse(result['held_out_coding_quality'])

    def test_incomplete_audit_drift_target_mismatch_and_false_tg_rejected(self):
        for mutate in (lambda requests: requests.pop(),
                lambda requests: requests[3]['blocks'][0]['input_tokens'].append(99),
                lambda requests: requests[1].update(state_exact=False),
                lambda requests: requests[4].update(emitted=[42]),
                lambda requests: requests[4].update(committed_tokens_per_second=200),
                lambda requests: requests[3].update(accepted=11),
                lambda requests: requests[5].update(committed_tokens_per_second=float('nan'))):
            requests = fixture()
            mutate(requests)
            with self.assertRaises(ValueError):
                summarize(requests, summarize_requests, lambda request: None, lambda request, arm: None)

    def test_native_audit_failure_propagates(self):
        def reject(request):
            raise ValueError('Native state or replay evidence failed')
        with self.assertRaisesRegex(ValueError, 'Native state'):
            summarize(fixture(), summarize_requests, reject, lambda request, arm: None)

    def test_one_regressing_pair_rejects_improvement(self):
        requests = fixture()
        requests[4].update(committed_tokens_per_second=90, decode_ms=1000 * 12 / 90)
        result = summarize(requests, summarize_requests, lambda request: None, lambda request, arm: None)
        self.assertFalse(result['improvement_screen_passed'])
