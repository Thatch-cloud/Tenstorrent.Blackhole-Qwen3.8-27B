"""Exercise stream accounting without a model or accelerator."""

import io
import json
import unittest
from unittest.mock import patch

from interleave_bench import quantile, stream


def response(chunks, done=True):
    lines = []
    for tokens in chunks:
        payload = {"choices": [{"index": 0, "text": "".join(tokens), "token_ids": [ord(token) for token in tokens]}]}
        lines.append("data: " + json.dumps(payload))
    lines.append("data: " + json.dumps(dict(choices=[], usage=dict(prompt_tokens=1,
                       completion_tokens=sum(len(tokens) for tokens in chunks)))))
    if done:
        lines.append("data: [DONE]")
    return io.BytesIO(("\n\n".join(lines) + "\n").encode())


class StreamTests(unittest.TestCase):
    def test_coalesced_events_count_actual_tokens(self):
        with patch("urllib.request.urlopen", return_value=response([["a", "b"], ["c"]])), \
                patch("time.perf_counter", side_effect=[0.0, 1.0, 1.5]):
            result = stream("http://localhost:8000", "test", [1], 3)
        self.assertEqual(result["tokens"], [97, 98, 99])
        self.assertEqual(result["decode_tok_s"], 2.0)
        self.assertEqual(result["coalesced_events"], 1)
        self.assertEqual(result["ttft_s"], 1.0)

    def test_truncated_stream_fails(self):
        with patch("urllib.request.urlopen", return_value=response([["a"], ["b"]], done=False)):
            with self.assertRaises(RuntimeError):
                stream("http://localhost", "test", [1], 2)

    def test_missing_tokens_fail(self):
        with patch("urllib.request.urlopen", return_value=response([[], []])):
            with self.assertRaises(RuntimeError):
                stream("http://localhost", "test", [1], 2)

    def test_quantile(self):
        self.assertEqual(quantile([3, 1, 2], 0.99), 3)
        self.assertIsNone(quantile([], 0.99))


if __name__ == "__main__":
    unittest.main(verbosity=2)
