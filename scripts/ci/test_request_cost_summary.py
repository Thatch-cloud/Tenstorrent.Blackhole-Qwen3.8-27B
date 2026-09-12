import unittest

from request_cost_summary import summarize


def fixture():
    return dict(length=200, replay_group_rows=8, exact=True, state_exact=True, inactive_exact=True,
        committed_decode_tokens=5, decode_ms=100, blocks=[
            dict(rows=1, committed=1, draft_ms=1, input_ms=2, verify_readback_ms=30, select_commit_ms=3),
            dict(rows=8, committed=4, draft_ms=1, input_ms=2, verify_readback_ms=40, select_commit_ms=3)])


class RequestCostSummaryTests(unittest.TestCase):
    def test_committed_not_proposed_throughput_and_nonoverlapping_costs(self):
        result = summarize(fixture())
        self.assertEqual(result['width_histogram'], {1: 1, 8: 1})
        self.assertEqual(result['committed_tokens_per_second'], 50)
        self.assertEqual(result['committed_per_verification'], 2.5)
        self.assertEqual(result['unassigned_decode_ms'], 18)
        self.assertEqual(result['width8_verify_readback_ms'], 40)
        self.assertEqual(result['target_200_decode_budget_ms'], 25)

    def test_reject_inconsistent_or_failed_request(self):
        for key, value in (('committed_decode_tokens', 9), ('exact', False),
                           ('state_exact', False), ('inactive_exact', False)):
            request = fixture()
            request[key] = value
            with self.assertRaises(ValueError):
                summarize(request)

    def test_seed_only_request_has_no_decode_rate(self):
        request = fixture()
        request.update(blocks=[], committed_decode_tokens=0, decode_ms=0)
        result = summarize(request)
        self.assertIsNone(result['committed_tokens_per_second'])
        self.assertIsNone(result['committed_per_verification'])
