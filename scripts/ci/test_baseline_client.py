"""Host-only checks for token accounting in the full-model baseline."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("baseline", Path(__file__).with_name("baseline-client.py"))
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class BaselineTests(unittest.TestCase):
    def stream(self, usage=3, done=True):
        events = [dict(choices=[dict(index=0, text="ab", logprobs=dict(tokens=["a", "b"]))]),
                  dict(choices=[dict(index=0, text="c", logprobs=dict(tokens=["c"]), finish_reason="length")]),
                  dict(choices=[], usage=dict(completion_tokens=usage))]
        response = mock.MagicMock()
        response.__enter__.return_value = [f"data: {json.dumps(event)}\n".encode() for event in events]
        if done:
            response.__enter__.return_value.append(b"data: [DONE]\n")
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(baseline.urllib.request, "urlopen", return_value=response), \
                mock.patch.object(baseline.time, "perf_counter", side_effect=[0., 1., 2., 3.]), \
                mock.patch("builtins.print"):
            return baseline.request_stream([1, 2], 3, "test", Path(directory))

    def test_coalesced_events_use_token_counts(self):
        report = self.stream()
        self.assertEqual(report["client_decode_estimate_tok_s"], 1.)
        self.assertEqual(report["coalesced_events"], 1)
        self.assertIsNone(report["engine_committed_tok_s"])
        self.assertEqual(report["ttft_s"], 1.)

    def test_usage_disagreement_fails(self):
        with self.assertRaises(AssertionError):
            self.stream(usage=4)

    def test_unterminated_stream_fails(self):
        with self.assertRaises(AssertionError):
            self.stream(done=False)

    def test_quantiles_handle_empty_input(self):
        self.assertIsNone(baseline.quantile([], .99))
        self.assertEqual(baseline.quantile([3, 1, 2], .5), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
