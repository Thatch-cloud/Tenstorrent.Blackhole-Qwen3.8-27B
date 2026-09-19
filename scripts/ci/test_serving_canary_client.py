import json
import unittest

from serving_canary_client import consume, endpoint


class CanaryClientTests(unittest.TestCase):
    def stream(self, tokens=(11, 12, 13), done=True):
        chunks = [dict(choices=[dict(token_ids=[tokens[0]], finish_reason=None)]),
            dict(choices=[dict(token_ids=list(tokens[1:]), finish_reason='stop')]),
            dict(choices=[], usage=dict(prompt_tokens=4096, completion_tokens=len(tokens)))]
        lines = [('data: ' + json.dumps(chunk) + '\n').encode() for chunk in chunks]
        return lines + ([b'data: [DONE]\n'] if done else [])

    def test_exact_stream_and_delivered_token_timing(self):
        timestamps = iter((12.0, 12.5, 12.6))
        report = consume(self.stream(), [11, 12, 13], 10.0, clock=lambda: next(timestamps))
        self.assertTrue(report['exact'])
        self.assertEqual(report['ttft_seconds'], 2.0)
        self.assertEqual(report['stream_delivery_tokens_per_second'], 4.0)
        self.assertIsNone(report['pp_tokens_per_second'])
        self.assertFalse(report['serving_qualified'])

    def test_wrong_tokens_or_incomplete_stream_are_rejected(self):
        for lines in (self.stream((11, 99, 13)), self.stream(done=False)):
            with self.assertRaises(ValueError):
                consume(lines, [11, 12, 13], 0.0, clock=lambda: 1.0)

    def test_no_remote_or_credentialed_endpoint(self):
        self.assertEqual(endpoint('http://127.0.0.1:8000'), 'http://127.0.0.1:8000/v1/completions')
        for base in ('https://example.com', 'http://user:password@localhost', 'http://localhost/path',
                'http://127.0.0.1:8000?token=secret'):
            with self.assertRaises(ValueError):
                endpoint(base)
