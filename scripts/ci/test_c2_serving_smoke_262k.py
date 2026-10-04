"""The 262k window's smoke arms (c2_serving_smoke.py): concurrent8_code_128k and stall8_cold262k, run for real against a fake OpenAI server that
also answers POST /tokenize with its own tokenizer (characters per token is not the estimate's 3.6, so the calibration has work to do).

concurrent8_code_128k is eight real-code prompts of about 120,000 tokens as the SERVER counts them (never above: a prompt past the profile's
limit is a 400, not a measurement), 800 out, the same arm on the 131k and the 262k eight-seat profiles. stall8_cold262k is seven decoding
users and one cold 253,920-token arrival, recording its time to first token and every seat's longest gap from the arrival on."""

import http.server
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_smoke_check as check  # noqa: E402

SMOKE = HERE / 'c2_serving_smoke.py'
MODEL = 'Qwen/Qwen3.8-27B'
CHARS_PER_TOKEN = 3.0               # the fake tokenizer's: the smoke's estimate is 3.6, so every first guess is 20% over


class TokenFake(object):
    """Records every chat request (seq, body, the chunks each earlier stream had sent when it arrived), streams `delay`-spaced
    deltas, answers /tokenize with ceil(characters / CHARS_PER_TOKEN) (or a 404 when `tokenize` is False)."""

    def __init__(self, tokenize=True, delay=0.0004, cap=50):
        self.tokenize, self.delay, self.cap = tokenize, delay, cap
        self.requests, self.sent, self.lock, self.tokenized = [], {}, threading.Lock(), []
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header('content-type', 'application/json')
                self.send_header('content-length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get('content-length') or 0)))
                if self.path == '/tokenize':
                    if not fake.tokenize:
                        self.respond(404, dict(error='no such endpoint'))
                        return
                    text = ''.join(message['content'] for message in body['messages'])
                    with fake.lock:
                        fake.tokenized.append(len(text))
                    self.respond(200, dict(count=int(math.ceil(len(text) / CHARS_PER_TOKEN)), max_model_len=262144))
                    return
                with fake.lock:
                    seq = len(fake.requests)
                    fake.requests.append(dict(seq=seq, body=body, at=time.time(), sent_at_arrival=dict(fake.sent)))
                    fake.sent[seq] = 0
                tokens = int(body.get('max_tokens') or 1)
                if not body.get('ignore_eos'):
                    tokens = min(tokens, fake.cap)
                if not body.get('stream'):
                    self.respond(200, dict(choices=[dict(message=dict(content='ok'), finish_reason='length')],
                                           usage=dict(completion_tokens=tokens, prompt_tokens=50)))
                    return
                self.send_response(200)
                self.send_header('content-type', 'text/event-stream')
                self.end_headers()
                cold = len(json.dumps(body)) > 600000
                if cold:
                    time.sleep(0.4)                          # a long prefill: no token for a while
                for number in range(1, tokens + 1):
                    self.wfile.write(('data: %s\n\n' % json.dumps(dict(choices=[dict(delta=dict(content='t%d ' % number))]))).encode())
                    self.wfile.flush()
                    with fake.lock:
                        fake.sent[seq] = number
                    time.sleep(fake.delay)
                final = dict(choices=[dict(delta={}, finish_reason='length')], usage=dict(completion_tokens=tokens, prompt_tokens=100))
                self.wfile.write(('data: %s\n\ndata: [DONE]\n\n' % json.dumps(final)).encode())
                self.wfile.flush()

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *failure):
        self.server.shutdown()
        self.server.server_close()
        return False

    @property
    def base(self):
        return 'http://127.0.0.1:%d' % self.server.server_address[1]


class SmokeRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name) / 'corpus' / 'pkg'
        root.mkdir(parents=True)
        # 3.4 MB of distinct python-looking text: eight disjoint windows of 120,000 tokens at 3 characters a token (360,000 each)
        lines = ['def function_%d(argument):\n    return argument + %d\n' % (index, index) for index in range(70000)]
        (root / 'module.py').write_text(''.join(lines), encoding='utf-8')
        cls.code_root = str(root)
        cls.cache = {}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_smoke(self, fake, tests, timeout=600):
        environment = dict(os.environ, SMOKE_CODE_ROOT=self.code_root)
        done = subprocess.run([sys.executable, '-B', str(SMOKE), fake.base, MODEL, ','.join(tests)], capture_output=True, text=True,
                              timeout=timeout, env=environment, cwd=str(HERE))
        self.assertEqual(done.returncode, 0, done.stdout[-1500:] + done.stderr[-1500:])
        line = [text for text in done.stdout.splitlines() if text.startswith('SMOKE_JSON ')][-1]
        return json.loads(line[len('SMOKE_JSON '):])

    def ran(self, name, **options):
        key = (name, tuple(sorted(options.items())))
        if key not in self.cache:
            with TokenFake(**options) as fake:
                self.cache[key] = (self.run_smoke(fake, (name,)), list(fake.requests), list(fake.tokenized))
        return self.cache[key]


class DeepArmTests(SmokeRuns):
    def test_eight_prompts_of_about_120000_server_counted_tokens_never_above(self):
        results, requests, tokenized = self.ran('concurrent8_code_128k')
        entry = results['concurrent8_code_128k']
        self.assertNotIn('error', entry, entry)
        fit = entry['fit']
        self.assertTrue(fit['calibrated'])
        self.assertEqual(fit['targets'], [120000] * 8)
        for count in fit['counts']:
            self.assertLessEqual(count, 120000)
            self.assertGreaterEqual(count, 120000 * 0.99)
        bodies = [request['body'] for request in requests]
        self.assertEqual(len(bodies), 8)
        self.assertEqual({body['max_tokens'] for body in bodies}, {800})
        prompts = [body['messages'][0]['content'] for body in bodies]
        self.assertEqual(len(set(prompts)), 8, 'eight different windows of the corpus')
        for prompt in prompts:
            self.assertTrue(prompt.startswith('<repository_context>'))
            self.assertLessEqual(math.ceil(len(prompt) / CHARS_PER_TOKEN), 120000)
        self.assertGreater(len(tokenized), 8, 'the estimate (3.6 characters a token) was over: it was refitted')

    def test_the_users_are_judged_as_eight_user_text_tests_and_pass(self):
        results, _requests, _tokenized = self.ran('concurrent8_code_128k')
        self.assertEqual(len(results['concurrent8_code_128k']['users']), 8)
        self.assertIn('concurrent8_code_128k', check.EIGHT_TESTS)
        self.assertEqual(check.smoke_problems(results), [])

    def test_without_a_tokenizer_the_estimate_stands_and_says_it_is_uncalibrated(self):
        results, requests, tokenized = self.ran('concurrent8_code_128k', tokenize=False)
        fit = results['concurrent8_code_128k']['fit']
        self.assertFalse(fit['calibrated'])
        self.assertEqual(fit['counts'], [None] * 8)
        self.assertEqual(tokenized, [])
        self.assertEqual(len(requests), 8)

    def test_the_arm_fits_the_131k_time_gate_profile_room(self):
        # the contract's room on c2-packed-tp4-8-time-gate: 131,328 - 256; 120,000 + 800 answer leaves no clamp
        import serving_c2_contract as contract

        profile = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']['c2-packed-tp4-8-time-gate']
        limits = contract.request_limits(profile)
        room = contract.prompt_room(profile['engine']['max-model-len'], limits['budget'], limits['max_prompt_tokens'],
                                    limits['min_answer_tokens'], limits.get('drafter_headroom_tokens', 0))
        self.assertGreaterEqual(room, 120000)
        self.assertGreaterEqual(profile['engine']['max-model-len'] - 120000, 800)
        import serving_kv_reservation as kv

        # on the 262k pool all eight are live at once: 8 x r blocks of 21,759
        self.assertLessEqual(8 * kv.request_blocks(120000, 800), 21760 - 1)
        self.assertEqual(8 * kv.request_blocks(120000, 800), 15112)


class SkewArmTests(SmokeRuns):
    def test_two_long_prompts_one_per_block_and_six_short_ones_as_the_server_counts_them(self):
        results, requests, tokenized = self.ran('concurrent8_skew')
        entry = results['concurrent8_skew']
        self.assertNotIn('error', entry, {key: value for key, value in entry.items() if key != 'users'})
        fit = entry['fit']
        self.assertEqual(fit['targets'], [253920, 4096, 4096, 4096, 253920, 4096, 4096, 4096])
        self.assertTrue(fit['calibrated'])
        for count, target in zip(fit['counts'], fit['targets']):
            self.assertLessEqual(count, target)
            self.assertGreaterEqual(count, target * 0.99)
        self.assertEqual(len(requests), 8)
        self.assertEqual({request['body']['max_tokens'] for request in requests}, {800})
        self.assertEqual(len({request['body']['messages'][0]['content'] for request in requests}), 8)
        self.assertIn('concurrent8_skew', check.EIGHT_TESTS)
        self.assertEqual(check.smoke_problems(results), [])


class StallArmTests(SmokeRuns):
    def test_seven_decoders_then_one_cold_arrival_is_recorded(self):
        results, requests, tokenized = self.ran('stall8_cold262k')
        entry = results['stall8_cold262k']
        self.assertNotIn('error', entry, {key: value for key, value in entry.items() if key != 'users'})
        self.assertEqual(len(requests), 8)
        short, cold = requests[:7], requests[7]
        for request in short:
            self.assertEqual((request['body']['max_tokens'], request['body']['ignore_eos']), (6000, True))
            self.assertLess(abs(len(request['body']['messages'][0]['content']) - 4096 * 3.6), 1500)
        self.assertEqual(cold['body']['max_tokens'], 200)
        self.assertLessEqual(math.ceil(len(cold['body']['messages'][0]['content']) / CHARS_PER_TOKEN), 253920)
        self.assertGreaterEqual(math.ceil(len(cold['body']['messages'][0]['content']) / CHARS_PER_TOKEN), 253920 * 0.99)
        for index in range(7):
            self.assertGreaterEqual(cold['sent_at_arrival'][index], 12, 'every seat was decoding when the cold prompt arrived')
        self.assertEqual(entry['fit']['targets'], [253920])
        self.assertTrue(entry['fit']['calibrated'])

    def test_the_arrivals_ttft_and_every_seats_longest_gap_are_recorded(self):
        results, _requests, _tokenized = self.ran('stall8_cold262k')
        entry = results['stall8_cold262k']
        self.assertIsInstance(entry['arrival_ttft_s'], float)
        self.assertGreaterEqual(entry['arrival_ttft_s'], 0.3, 'the fake\'s 0.4 s prefill delay')
        self.assertEqual(len(entry['seat_gaps']), 7)
        for gap in entry['seat_gaps']:
            self.assertIsNone(gap['error'])
            self.assertIsInstance(gap['longest_gap_s'], float)
            self.assertGreater(gap['tokens'], 0)
        self.assertEqual(entry['longest_gap_s'], max(gap['longest_gap_s'] for gap in entry['seat_gaps']))
        self.assertGreater(entry['arrival_started_at'], 0)

    def test_the_checker_reads_the_stall_record_and_refuses_a_missing_arrival_ttft(self):
        results, _requests, _tokenized = self.ran('stall8_cold262k')
        self.assertEqual(check.smoke_problems(results), [])
        broken = json.loads(json.dumps(results))
        broken['stall8_cold262k']['arrival_ttft_s'] = None
        problems = check.smoke_problems(broken)
        self.assertTrue(any('no time to first token' in problem for problem in problems), problems)
        errored = {'stall8_cold262k': dict(error='boom')}
        self.assertEqual(check.smoke_problems(errored), ['stall8_cold262k: boom'])


class PureTests(unittest.TestCase):
    def test_longest_gap_over_the_stamps_after_the_arrival(self):
        source = SMOKE.read_text(encoding='utf-8')
        namespace = {}
        start = source.index('def longest_gap')
        end = source.index('def stall8_cold262k')
        exec(compile(source[start:end], 'smoke', 'exec'), namespace)
        gap = namespace['longest_gap']
        self.assertEqual(gap([0.0, 0.1, 0.2, 5.2, 5.3]), (5.0, 0.2))
        self.assertEqual(gap([0.0, 0.1, 0.2, 5.2, 5.3], after=5.2), (0.1, 5.2))
        self.assertEqual(gap([0.0, 0.1, 0.2], after=1.0), (None, None))
        self.assertEqual(gap([1.0]), (None, None))
        self.assertEqual(gap([]), (None, None))


if __name__ == '__main__':
    unittest.main()
