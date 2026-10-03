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
    def stream(self, usage=3, done=True, ids=True, **kwargs):
        events = [dict(choices=[dict(index=0, text="ab", token_ids=[11, 12] if ids else None, logprobs=dict(tokens=["a", "b"]))]),
                  dict(choices=[dict(index=0, text="c", token_ids=[13] if ids else None, logprobs=dict(tokens=["c"]), finish_reason="length")]),
                  dict(choices=[], usage=dict(completion_tokens=usage, prompt_tokens=2))]
        response = mock.MagicMock()
        response.__enter__.return_value = [f"data: {json.dumps(event)}\n".encode() for event in events]
        if done:
            response.__enter__.return_value.append(b"data: [DONE]\n")
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(baseline.urllib.request, "urlopen", return_value=response) as open_request, \
                mock.patch.object(baseline.time, "perf_counter", side_effect=[0., 1., 2., 3.]), \
                mock.patch("builtins.print"):
            result = baseline.request_stream([1, 2], 3, "test", Path(directory), **kwargs)
            self.payload = json.loads(open_request.call_args.args[0].data)
            return result

    def test_sampling_probe_can_omit_seed_and_label_request(self):
        result = self.stream(seed=None, request_id="qwen-sampling-device-test")
        self.assertNotIn("seed", self.payload)
        self.assertNotIn("logprobs", self.payload)
        self.assertEqual(self.payload["request_id"], "qwen-sampling-device-test")
        self.assertIsNone(result["seed"])

    def test_coalesced_events_use_token_counts(self):
        report = self.stream()
        self.assertEqual(report["client_decode_estimate_tok_s"], 1.)
        self.assertEqual(report["coalesced_events"], 1)
        self.assertIsNone(report["engine_committed_tok_s"])
        self.assertEqual(report["ttft_s"], 1.)
        self.assertEqual(report["token_ids"], [11, 12, 13])
        self.assertFalse(report["logprobs_requested"])
        self.assertEqual(report["sampling_options"], {"temperature": 0})

    def test_fallback_options_reach_endpoint_and_report(self):
        options = dict(temperature=.7, top_k=10, top_p=.9, presence_penalty=1.2,
                       frequency_penalty=1.3, repetition_penalty=1.1)
        report = self.stream(seed=456, **options)
        self.assertEqual(report["sampling_options"], options)
        self.assertEqual(self.payload["seed"], 456)
        for key, value in options.items():
            self.assertEqual(self.payload[key], value)

    def test_missing_ids_cannot_fall_back_to_token_strings(self):
        with self.assertRaises(AssertionError):
            self.stream(ids=False)

    def test_usage_disagreement_fails(self):
        with self.assertRaises(AssertionError):
            self.stream(usage=4)

    def test_unterminated_stream_fails(self):
        with self.assertRaises(AssertionError):
            self.stream(done=False)

    def test_quantiles_handle_empty_input(self):
        self.assertIsNone(baseline.quantile([], .99))
        self.assertEqual(baseline.quantile([3, 1, 2], .5), 2)

    def test_template_mapping_uses_actual_token_length(self):
        tokenizer = mock.Mock()
        tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: {
            "input_ids": list(range(20 + messages[-1]["content"].count("def lookup") * 10))}
        prompt = baseline.make_prompt(tokenizer, 128, 0)
        self.assertIsInstance(prompt, list)
        self.assertEqual(len(prompt), 120)

    def test_template_flat_list_remains_supported(self):
        tokenizer = mock.Mock()
        tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: list(
            range(20 + messages[-1]["content"].count("def lookup") * 10))
        self.assertEqual(len(baseline.make_prompt(tokenizer, 128, 0)), 120)


if __name__ == "__main__":
    unittest.main(verbosity=2)
