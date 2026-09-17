import unittest
from unittest.mock import patch

import workers_combined_timed as candidate
from workers_combined_gate import BASE_QUALIFY, validate_calls


class WorkerTimingTests(unittest.TestCase):
    def test_fresh_identity_propagates_and_restores(self):
        before = (candidate.request_gate.qualify, candidate.timing.SCREEN_RUN)
        with patch.dict('os.environ', QWEN_MATCHED_COMBINED='1'):
            with candidate.identity_scope():
                self.assertEqual(candidate.timing.baseline.shared.SCREEN_RUN, 35041737747)
                self.assertEqual(candidate.timing.baseline.shared.SCREEN_SHA256, candidate.SCREEN_SHA256)
                self.assertIs(candidate.request_gate.qualify, candidate.qualify_combined)
                self.assertIsNot(BASE_QUALIFY, candidate.request_gate.qualify)
        self.assertEqual((candidate.request_gate.qualify, candidate.timing.SCREEN_RUN), before)

    def test_reject_missing_or_wrong_worker_execution(self):
        call = dict(key_chunk_size=256, requested_worker_limit=8, selected_worker_limit=16,
            stripe_keys=False, fp32_dest_acc=True)
        validate_calls([call] * 25, 25)
        for calls, count in (([], 0), ([call], 25), ([dict(call, selected_worker_limit=8)], 1)):
            with self.assertRaises(ValueError):
                validate_calls(calls, count)


if __name__ == '__main__':
    unittest.main()
