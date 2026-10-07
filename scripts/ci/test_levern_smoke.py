"""Lever N at TP4: the smoke tests it adds to c2_serving_smoke.py, run for real against a fake OpenAI server.

levern_equal / levern_equal_long / levern_equal_busy send prompts of EXACTLY the asked token count (real code text tokenized by the server's /tokenize, cut at the
token) as token ids through /v1/completions; stall8_cold262k and stall8_cold128k record, beside the arrival's time to first token and every seat's longest gap, each
seat's progress INSIDE the arrival's prefill window; the hang shapes (levern_decoder_finishes, levern_all_decoders_finish, levern_cancel_mid_prefill,
levern_arrival_during_prefill, levern_seed_stops) drive seven or fewer decoding seats and the arrival shapes the design's G-N2 names, and levern_cancel_mid_prefill drops
the client's connection while its prefill runs. The fake answers /tokenize with its own tokenizer (three characters a token) and the chat and completions
endpoints with deterministic text."""

import ast
import hashlib
import http.server
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_smoke_check as check  # noqa: E402

SMOKE = HERE / 'c2_serving_smoke.py'
MODEL = 'Qwen/Qwen3.8-27B'
CHARS_PER_TOKEN = 3.0
COLD_CHARS = 150000          # a chat prompt longer than this is a long prefill to the fake: no byte for 0.4 s


class Fake(object):
    def __init__(self, tokenize=True, delay=0.0004, cap=60, completion_tokens=7):
        self.tokenize, self.delay, self.cap, self.completion_tokens = tokenize, delay, cap, completion_tokens
        self.chats, self.completions, self.tokenized, self.lock = [], [], [], threading.Lock()
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
                    if not fake.tokenize or 'prompt' not in body:
                        self.respond(404, dict(error='no such endpoint'))
                        return
                    text = body['prompt']
                    ids = [ord(char) % 1000 for char in text[::3]]
                    with fake.lock:
                        fake.tokenized.append(len(text))
                    self.respond(200, dict(count=len(ids), max_model_len=262144, tokens=ids))
                    return
                if self.path == '/v1/completions':
                    with fake.lock:
                        fake.completions.append(body)
                    ids = body['prompt']
                    if not isinstance(ids, list) or not all(isinstance(value, int) for value in ids):
                        self.respond(400, dict(error='token ids expected'))
                        return
                    digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:12]
                    self.respond(200, dict(choices=[dict(text='answer %s ' % digest * 8, finish_reason='length')],
                                           usage=dict(prompt_tokens=len(ids), completion_tokens=min(body['max_tokens'], fake.completion_tokens))))
                    return
                with fake.lock:
                    seq = len(fake.chats)
                    fake.chats.append(dict(seq=seq, body=body, at=time.time()))
                tokens = int(body.get('max_tokens') or 1)
                if not body.get('ignore_eos'):
                    tokens = min(tokens, fake.cap)
                self.send_response(200)
                self.send_header('content-type', 'text/event-stream')
                self.end_headers()
                try:
                    if len(json.dumps(body)) > COLD_CHARS:
                        time.sleep(0.4)
                    for number in range(1, tokens + 1):
                        self.wfile.write(('data: %s\n\n' % json.dumps(dict(choices=[dict(delta=dict(content='t%d ' % number))]))).encode())
                        self.wfile.flush()
                        time.sleep(fake.delay)
                    final = dict(choices=[dict(delta={}, finish_reason='length')], usage=dict(completion_tokens=tokens, prompt_tokens=100))
                    self.wfile.write(('data: %s\n\ndata: [DONE]\n\n' % json.dumps(final)).encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.server.handle_error = lambda *args: None
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
        # distinct python-looking text, 3 characters a token for the fake: far more than the longest boundary prompt (253,920 tokens = 762,000 characters)
        lines = ['def function_%d(argument):\n    return argument + %d\n' % (index, index) for index in range(110000)]
        (root / 'module.py').write_text(''.join(lines), encoding='utf-8')
        cls.code_root = str(root)
        cls.cache = {}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_smoke(self, fake, tests, extra_env=None, timeout=900):
        environment = dict(os.environ, SMOKE_CODE_ROOT=self.code_root, **(extra_env or {}))
        done = subprocess.run([sys.executable, '-B', str(SMOKE), fake.base, MODEL, ','.join(tests)], capture_output=True, text=True,
                              timeout=timeout, env=environment, cwd=str(HERE))
        self.assertEqual(done.returncode, 0, done.stdout[-1500:] + done.stderr[-1500:])
        line = [text for text in done.stdout.splitlines() if text.startswith('SMOKE_JSON ')][-1]
        return json.loads(line[len('SMOKE_JSON '):]), done.stdout

    def ran(self, name, extra_env=None, **options):
        key = (name, tuple(sorted(options.items())), tuple(sorted((extra_env or {}).items())))
        if key not in self.cache:
            with Fake(**options) as fake:
                results, stdout = self.run_smoke(fake, (name,), extra_env)
                self.cache[key] = (results, list(fake.chats), list(fake.completions), list(fake.tokenized), stdout)
        return self.cache[key]


class ExactLengthTests(SmokeRuns):
    def test_every_boundary_prompt_is_sent_at_exactly_its_token_count_as_ids(self):
        results, chats, completions, tokenized, _ = self.ran('levern_equal')
        entry = results['levern_equal']
        self.assertNotIn('error', entry, entry)
        lengths = [2047, 2048, 2049, 4095, 4096, 4097, 6143, 6145, 32785]
        self.assertEqual(entry['lengths'], lengths)
        self.assertEqual([len(body['prompt']) for body in completions], lengths)
        self.assertEqual({body['max_tokens'] for body in completions}, {256})
        self.assertEqual(chats, [], 'completions, not chat: the prompt is token ids')
        for length in lengths:
            row = entry['prompts'][str(length)]
            self.assertEqual((row['prompt_tokens'], row['prompt_tokens_sent']), (length, length))
            self.assertEqual(row['status'], 200)
            self.assertTrue(row['content_sha256'])
        self.assertEqual(len({tuple(body['prompt']) for body in completions}), len(lengths), 'different windows: different prompts')
        self.assertEqual(check.smoke_problems(results), [])

    def test_the_long_boundary_prompts(self):
        results, _chats, completions, _tokenized, _ = self.ran('levern_equal_long')
        entry = results['levern_equal_long']
        self.assertNotIn('error', entry, entry)
        self.assertEqual([len(body['prompt']) for body in completions], [131077, 253920])
        self.assertEqual(check.smoke_problems(results), [])

    def test_a_server_that_cannot_tokenize_is_an_error_not_an_estimate(self):
        results, _chats, completions, _tokenized, _ = self.ran('levern_equal', tokenize=False)
        entry = results['levern_equal']
        self.assertIn('cannot tokenize', entry['error'])
        self.assertEqual(completions, [])
        self.assertTrue(any('cannot tokenize' in problem for problem in check.smoke_problems(results)))

    def test_busy_runs_seven_decoders_while_the_boundary_prompts_arrive(self):
        results, chats, completions, _tokenized, _ = self.ran('levern_equal_busy')
        entry = results['levern_equal_busy']
        self.assertNotIn('error', entry, entry)
        self.assertEqual(len(entry['users']), 7)
        self.assertEqual({chat['body']['max_tokens'] for chat in chats}, {4000})
        self.assertTrue(all(chat['body']['ignore_eos'] for chat in chats))
        self.assertEqual([len(body['prompt']) for body in completions], [4097, 6145, 32785])
        self.assertEqual(sorted(entry['prompts']), ['32785', '4097', '6145'])
        self.assertEqual(check.smoke_problems(results), [])

    def test_a_row_whose_prompt_count_differs_is_a_problem(self):
        results, *_ = self.ran('levern_equal')
        broken = json.loads(json.dumps(results))
        broken['levern_equal']['prompts']['4097']['prompt_tokens'] = 4098
        self.assertTrue(any('the server counted 4098 tokens of the 4097' in problem for problem in check.smoke_problems(broken)))


def function_source(name):
    tree = ast.parse(SMOKE.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(SMOKE.read_text(encoding='utf-8'), node)
    raise AssertionError('no %s in the smoke' % name)


def helpers():
    namespace = {}
    for name in ('window_stats', 'window_aggregate', 'longest_gap'):
        exec(function_source(name), namespace)
    return namespace


class WindowTests(unittest.TestCase):
    def test_window_stats_counts_the_chunks_inside_the_window(self):
        window = helpers()['window_stats']([1.0, 2.0, 3.0, 11.0, 12.0, 13.0, 14.0, 50.0], 800, 80, 10.0, 20.0)
        self.assertEqual(window, dict(window_s=10.0, chunks=4, chunk_rate=0.4, est_tok_s=4.0))

    def test_a_frozen_seat_has_zero_progress_in_the_window(self):
        window = helpers()['window_stats']([1.0, 2.0, 3.0, 50.0, 51.0], 100, 10, 10.0, 40.0)
        self.assertEqual((window['chunks'], window['chunk_rate'], window['est_tok_s']), (0, 0.0, 0.0))

    def test_a_window_with_no_end_or_no_length_is_none(self):
        stats = helpers()['window_stats']
        for begin, end in ((10.0, None), (10.0, 10.0), (10.0, 5.0), (None, 20.0)):
            self.assertEqual(stats([11.0], 10, 10, begin, end), dict(window_s=None, chunks=None, chunk_rate=None, est_tok_s=None))

    def test_the_aggregate_counts_seats_that_progressed(self):
        namespace = helpers()
        seats = [namespace['window_stats']([11.0 + 0.5 * n for n in range(count)], 40, 10, 10.0, 20.0) for count in (0, 4, 8)]
        aggregate = namespace['window_aggregate'](seats)
        self.assertEqual((aggregate['seats'], aggregate['seats_progressing']), (3, 2))
        self.assertEqual(aggregate['min_chunk_rate'], 0.0)
        self.assertEqual(aggregate['total_est_tok_s'], 4.8)
        self.assertEqual(namespace['window_aggregate']([dict(chunks=None)]), dict(seats=0, seats_progressing=0, min_chunk_rate=None, total_est_tok_s=None))


class StallTests(SmokeRuns):
    def test_both_stall_tests_record_the_window_numbers_for_every_seat(self):
        for name, tokens in (('stall8_cold262k', 253920), ('stall8_cold128k', 120000)):
            with self.subTest(name=name):
                results, chats, _completions, _tokenized, _ = self.ran(name)
                entry = results[name]
                self.assertNotIn('error', entry, {key: value for key, value in entry.items() if key != 'users'})
                self.assertEqual(entry['fit']['targets'], [tokens])
                self.assertEqual(len(chats), 8)
                self.assertEqual(len(entry['seat_windows']), 7)
                for seat in entry['seat_windows']:
                    self.assertGreater(seat['window_s'], 0.3)
                    self.assertGreater(seat['chunks'], 0, 'the fake keeps every decoder streaming through the arrival window')
                self.assertEqual(entry['window']['seats'], 7)
                self.assertEqual(entry['window']['seats_progressing'], 7)
                self.assertIn('arrival_ttft_s', entry)
                self.assertEqual(check.smoke_problems(results), [])

    def test_the_existing_keys_are_all_still_there(self):
        results, *_ = self.ran('stall8_cold262k')
        for key in ('users', 'corpus', 'fit', 'arrival_started_at', 'arrival_ttft_s', 'arrival_prompt_tokens', 'seat_gaps', 'longest_gap_s'):
            self.assertIn(key, results['stall8_cold262k'])


class HangShapeTests(SmokeRuns):
    ENV = {'SMOKE_LEVERN_CANCEL_AFTER_S': '0.3', 'SMOKE_LEVERN_ARRIVAL_AFTER_S': '0.1'}

    def test_a_decoder_finishing_mid_prefill(self):
        results, chats, *_ = self.ran('levern_decoder_finishes')
        entry = results['levern_decoder_finishes']
        self.assertNotIn('error', entry, entry)
        self.assertEqual(entry['budgets'], [150] + [4000] * 6)
        self.assertEqual(len(entry['users']), 8)
        self.assertEqual(sorted(chat['body']['max_tokens'] for chat in chats), sorted([150] + [4000] * 6 + [300]))
        self.assertEqual(check.smoke_problems(results), [])

    def test_every_decoder_finishing_mid_prefill(self):
        results, chats, *_ = self.ran('levern_all_decoders_finish')
        entry = results['levern_all_decoders_finish']
        self.assertEqual(entry['budgets'], [120, 150, 180])
        self.assertEqual(len(entry['users']), 4)
        self.assertEqual(check.smoke_problems(results), [])

    def test_the_client_drops_the_cold_prefill_and_a_followup_completes(self):
        results, chats, *_ = self.ran('levern_cancel_mid_prefill', extra_env=self.ENV)
        entry = results['levern_cancel_mid_prefill']
        self.assertNotIn('error', entry, entry)
        self.assertEqual(entry['dropped']['outcome'], 'dropped')
        self.assertEqual(entry['dropped']['after_s'], 0.3)
        self.assertEqual(len(entry['users']), 4, 'three decoders and the follow-up')
        self.assertEqual(check.smoke_problems(results), [])

    def test_a_cancel_that_was_not_a_cancel_is_a_problem(self):
        results, *_ = self.ran('levern_cancel_mid_prefill', extra_env=self.ENV)
        broken = json.loads(json.dumps(results))
        broken['levern_cancel_mid_prefill']['dropped']['outcome'] = 'completed'
        self.assertTrue(any('was not dropped by the client' in problem for problem in check.smoke_problems(broken)))

    def test_an_arrival_during_the_prefill(self):
        results, chats, *_ = self.ran('levern_arrival_during_prefill', extra_env=self.ENV)
        entry = results['levern_arrival_during_prefill']
        self.assertNotIn('error', entry, entry)
        self.assertEqual(len(entry['users']), 5)
        self.assertEqual(entry['users'][3]['tokens'] and entry['users'][4]['tokens'] and True, True)
        self.assertEqual(check.smoke_problems(results), [])

    def test_a_one_token_chunked_prompt_ends_at_its_seed_and_the_next_request_completes(self):
        results, chats, *_ = self.ran('levern_seed_stops')
        entry = results['levern_seed_stops']
        self.assertNotIn('error', entry, entry)
        self.assertEqual(len(entry['users']), 4)
        one = [chat for chat in chats if chat['body']['max_tokens'] == 1]
        self.assertEqual(len(one), 1)
        self.assertEqual(entry['users'][2]['tokens'], 1)
        self.assertEqual(check.smoke_problems(results), [])

    def test_a_seed_stop_that_produced_more_than_one_token_is_a_problem(self):
        results, *_ = self.ran('levern_seed_stops')
        broken = json.loads(json.dumps(results))
        broken['levern_seed_stops']['users'][2]['tokens'] = 5
        self.assertTrue(any('the one-token request produced 5' in problem for problem in check.smoke_problems(broken)))


class ListedTests(unittest.TestCase):
    def test_every_levern_test_the_smoke_defines_is_one_the_check_judges(self):
        source = SMOKE.read_text(encoding='utf-8')
        names = [name for name in ('levern_equal', 'levern_equal_long', 'levern_equal_busy', 'levern_decoder_finishes', 'levern_all_decoders_finish',
                                   'levern_cancel_mid_prefill', 'levern_arrival_during_prefill', 'levern_seed_stops')]
        for name in names:
            self.assertIn('def %s(' % name, source)
            self.assertIn(name, check.LEVERN_ROW_TESTS + check.LEVERN_USER_TESTS)


if __name__ == '__main__':
    unittest.main()
