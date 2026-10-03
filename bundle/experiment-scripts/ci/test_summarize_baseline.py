"""Cache fractions, missing series and timing windows remain distinct."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("summary", Path(__file__).with_name("summarize-baseline.py"))
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def summarize(self, metrics, rate=30.):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "baseline-summary.json").write_text('{"passed":true}')
            (root / "run-0-length-128-batch-1-user-0.json").write_text(json.dumps(dict(
                label="run-0-length-128-batch-1-user-0", passed=True, prompt_tokens=120,
                client_decode_estimate_tok_s=rate, token_strings_sha256="tokens", ttft_s=1., event_gap_p99_s=.05)))
            (root / "metrics.jsonl").write_text(json.dumps(dict(metrics=metrics)) + "\n")
            return summary.summarize(root)

    def test_fraction_is_not_already_percent(self):
        result = self.summarize('vllm:kv_cache_usage_perc{engine="0"} 0.008\nvllm:num_requests_running{engine="0"} 1\n')
        self.assertEqual(result["gauges"]["kv_cache_usage_perc"]["maximum"], .008)
        self.assertEqual(result["cache_classification"], "cache occupancy observed nonzero")

    def test_missing_series_is_not_zero(self):
        result = self.summarize('vllm:num_requests_running 1\n')
        self.assertEqual(result["busy_zero_cache_samples"], 0)
        self.assertIn("insufficient", result["cache_classification"])

    def test_active_zero_is_flagged_not_diagnosed(self):
        result = self.summarize('vllm:num_requests_running 1\nvllm:kv_cache_usage_perc 0\n')
        self.assertEqual(result["busy_zero_cache_samples"], 1)
        self.assertIn("investigate", result["cache_classification"])

    def test_empty_decode_window_remains_unavailable(self):
        result = self.summarize('', rate=None)
        self.assertIsNone(result["workloads"][0]["client_decode_median_tok_s"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
