"""The agent_turns smoke tests (c2_serving_smoke: agent_turns, agent_turns_long), the module behind them (agent_turns.py) and their judge (c2_smoke_check.agent_turns_problems).

The smoke runs for real, as a subprocess, against a fake OpenAI server that behaves like a prefix-reuse one for the purposes of these tests: it counts a prompt's tokens
(three characters each), keeps the last prompt of every VERIFIED cache_salt and reports the page-rounded prefix a request shares with it as
usage.prompt_tokens_details.cached_tokens. test_levern_smoke's SmokeRuns drives it. The tests hold the SHAPE (append-only history, the answer carried verbatim, the salt per
session, the tags, the plan's determinism), the numbers' arithmetic (the live table, the log reader's window and shares) and the judge; the times are the card's.
"""

import ast
import hashlib
import hmac
import http.server
import json
import os
import re
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import agent_turns as at  # noqa: E402
import c2_smoke_check as check  # noqa: E402
import octo_markers  # noqa: E402
from test_levern_smoke import SmokeRuns  # noqa: E402
from test_w2ln_smoke_shapes import constants  # noqa: E402

SMOKE = HERE / 'c2_serving_smoke.py'
KEY = b'k' * 40
CHARS_PER_FAKE_TOKEN = 3
PAGE = 64


class AgentFake(object):
    """An OpenAI-shaped server with a salted prefix cache. fail_at=(session tag, status) answers that request with the status; reuse=False reports nothing cached;
    details=False leaves prompt_tokens_details out of the usage; key=None verifies no salt."""

    def __init__(self, key=KEY, reuse=True, details=True, fail=None, max_model_len=262144, completion=24):
        self.key, self.reuse, self.details, self.fail, self.max_model_len, self.completion = key, reuse, details, fail, max_model_len, completion
        self.chats, self.lock, self.last = [], threading.Lock(), {}
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

            def do_GET(self):
                if self.path == '/v1/models':
                    self.respond(200, dict(data=[dict(id='Qwen/Qwen3.8-27B', max_model_len=fake.max_model_len)]))
                else:
                    self.respond(404, dict(error='no'))

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get('content-length') or 0)))
                if self.path == '/tokenize':
                    self.respond(200, dict(count=fake.count(body.get('messages') or [])))
                    return
                tag = self.headers.get('X-Request-Id')
                if fake.fail and tag == fake.fail[0]:
                    self.respond(fake.fail[1], dict(error='injected'))
                    return
                prompt = fake.render(body['messages'])
                prompt_tokens = len(prompt) // CHARS_PER_FAKE_TOKEN + 1
                salt = body.get('cache_salt')
                verified = fake.verified(salt)
                cached = 0
                with fake.lock:
                    if verified and fake.reuse and salt in fake.last:
                        shared = len(os.path.commonprefix([fake.last[salt], prompt]))
                        cached = shared // CHARS_PER_FAKE_TOKEN // PAGE * PAGE
                    if verified:
                        fake.last[salt] = prompt
                digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
                tokens = min(int(body.get('max_tokens') or 1), fake.completion)
                thinking = ['think-%s-%d ' % (digest, index) for index in range(min(4, tokens))]
                answer = ['word-%s-%d ' % (digest, index) for index in range(tokens - len(thinking))]
                record = dict(seq=len(fake.chats), body=body, tag=tag, headers=dict(self.headers.items()), prompt_tokens=prompt_tokens, cached=cached,
                              verified=verified, content=''.join(answer), reasoning=''.join(thinking), tokens=tokens)
                with fake.lock:
                    fake.chats.append(record)
                self.send_response(200)
                self.send_header('content-type', 'text/event-stream')
                self.end_headers()
                try:
                    time.sleep(max(prompt_tokens - cached, 0) / 4.0e6)
                    for text in thinking:
                        self.wfile.write(('data: %s\n\n' % json.dumps(dict(choices=[dict(delta=dict(reasoning_content=text))]))).encode())
                        self.wfile.flush()
                        time.sleep(0.0004)
                    for text in answer:
                        self.wfile.write(('data: %s\n\n' % json.dumps(dict(choices=[dict(delta=dict(content=text))]))).encode())
                        self.wfile.flush()
                        time.sleep(0.0004)
                    usage = dict(prompt_tokens=prompt_tokens, completion_tokens=tokens)
                    if fake.details:
                        usage['prompt_tokens_details'] = dict(cached_tokens=cached)
                    final = dict(choices=[dict(delta={}, finish_reason='length')])
                    self.wfile.write(('data: %s\n\ndata: %s\n\ndata: [DONE]\n\n' % (json.dumps(final), json.dumps(dict(choices=[], usage=usage)))).encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.server.handle_error = lambda *args: None
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @staticmethod
    def render(messages):
        return ''.join('<%s>%s%s' % (message['role'], message.get('reasoning') or '', message.get('content') or '') for message in messages)

    def count(self, messages):
        return len(self.render(messages)) // CHARS_PER_FAKE_TOKEN + 1

    def verified(self, salt):
        if not salt or not self.key:
            return False
        parts = salt.split('.')
        return len(parts) == 3 and parts[0] == 'qps1' and hmac.compare_digest(at.mint_salt(self.key, parts[1]), salt)

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


def container_log(chats, nonce='abcd1234', noise=True, prefix='', round_live=4):
    """A container log of the fake's requests in the engine's formats: per request its route steps (a cold prompt is split in two), a rebind line for a returning
    turn, and decode rounds after it; with `noise`, rounds before the run and another request's prefill and rounds after it, which the window must leave out."""
    def line(text):
        return prefix + text

    out = []
    if noise:
        out += [line(octo_markers.ROUND_LINE.format(round=1, shape='octo', planned='octo', live=8, rows=8, eligible=1, counted=1, octo_rounds=1, m3_rounds=0,
                                                   committed=40, step_ms=100.0, gap_ms=10.0))]
    for chat in chats:
        req = 'chatcmpl-%s-1a2b3c4d' % chat['tag']
        prompt, cached = chat['prompt_tokens'], chat['cached']
        source = 'CHECKPOINT' if cached else 'COLD'
        if cached:
            out.append(line('[PINDIAG] sticky engine built req=%s ms=540.0 frontier=%d prompt=%d kind=rebind' % (req, cached, prompt)))
            out.append(line('[PINDIAG] lever N route req=%s start=%d end=%d prompt=%d final=1 wrote_slot=1 ms=%.1f programs=1->1 window=0 source=%s captured=-'
                            % (req, cached, prompt, prompt, 600.0 + (prompt - cached) / 4.0, source)))
        else:
            half = prompt // 2
            out.append(line('[PINDIAG] lever N route req=%s start=0 end=%d prompt=%d final=0 wrote_slot=0 ms=2000.0 programs=1->1 window=0 source=COLD captured=-'
                            % (req, half, prompt)))
            out.append(line('[PINDIAG] lever N route req=%s start=%d end=%d prompt=%d final=1 wrote_slot=1 ms=3000.0 programs=1->1 window=0 source=SCRATCH captured=-'
                            % (req, half, prompt, prompt)))
        for number in range(6):
            out.append(line(octo_markers.ROUND_LINE.format(round=2 + number, shape='octo', planned='octo', live=round_live, rows=8, eligible=1, counted=1,
                                                          octo_rounds=2 + number, m3_rounds=0, committed=chat['tokens'] * 4 // 6 + 4, step_ms=100.0, gap_ms=20.0)))
        out.append(line('[PREFIX] row=%d req=%s path=traced registry=present grant=committed Q=%d L=%d plan=[] restored_ms=- captured=[] ms=1.0 programs=1->1' % (
            chat['seq'], req, cached, prompt)))
    if noise:
        out += [line('[PINDIAG] lever N route req=chatcmpl-other-1a2b3c4d start=0 end=900 prompt=900 final=1 wrote_slot=1 ms=50.0 programs=1->1 window=0'),
                line(octo_markers.ROUND_LINE.format(round=99, shape='octo', planned='octo', live=1, rows=8, eligible=1, counted=1, octo_rounds=99, m3_rounds=0,
                                                   committed=500, step_ms=100.0, gap_ms=10.0))]
    return '\n'.join(out) + '\n'


class AgentRuns(SmokeRuns):
    ENV = {'SMOKE_AGENT_SESSIONS': '3', 'SMOKE_AGENT_TURNS': '3', 'SMOKE_AGENT_MIX': '1500,2500', 'SMOKE_AGENT_GAP_SCALE': '0.02',
           'SMOKE_AGENT_STAGGER_S': '0.05', 'SMOKE_AGENT_NONCE': 'abcd1234'}

    @classmethod
    def key_file(cls):
        if not hasattr(cls, '_key_file'):
            cls._key_file = Path(cls.tmp.name) / 'salt.key'
            cls._key_file.write_bytes(KEY + b'\n')
        return str(cls._key_file)

    def agent(self, test='agent_turns', env=None, salted=True, fake_options=None):
        key = (test, tuple(sorted((env or {}).items())), salted, tuple(sorted((fake_options or {}).items(), key=lambda item: item[0])))
        if key not in self.cache:
            environment = dict(self.ENV, **(env or {}))
            if salted:
                environment['SMOKE_SALT_KEY_FILE'] = self.key_file()
            with AgentFake(**(fake_options or {})) as fake:
                results, stdout = self.run_smoke(fake, (test,), environment)
                self.cache[key] = (results, list(fake.chats), stdout)
        return self.cache[key]


class RunShapeTests(AgentRuns):
    def test_sessions_append_one_message_after_the_previous_answer_verbatim(self):
        results, chats, _ = self.agent()
        entry = results['agent_turns']
        self.assertNotIn('error', entry, {key: value for key, value in entry.items() if key != 'sessions'})
        self.assertEqual((entry['sessions_planned'], entry['turns_planned']), (3, 3))
        self.assertEqual(len(chats), 3 * 4, 'the opening request and three returning turns a session')
        by_session = {}
        for chat in chats:
            match = re.match(r'agt-abcd1234-s(\d+)-t(\d+)$', chat['tag'])
            self.assertIsNotNone(match, chat['tag'])
            by_session.setdefault(int(match.group(1)), {})[int(match.group(2))] = chat
        self.assertEqual(sorted(by_session), [0, 1, 2])
        for session, turns in by_session.items():
            self.assertEqual(sorted(turns), [0, 1, 2, 3])
            self.assertEqual(len(turns[0]['body']['messages']), 1)
            self.assertIn('<repository_context>', turns[0]['body']['messages'][0]['content'])
            for turn in (1, 2, 3):
                previous, current = turns[turn - 1]['body']['messages'], turns[turn]['body']['messages']
                self.assertEqual(current[:len(previous)], previous, 'append-only: the earlier messages are byte for byte what was sent')
                self.assertEqual(len(current), len(previous) + 2, 'the assistant answer and one new user message')
                answer = current[len(previous)]
                self.assertEqual(answer['role'], 'assistant')
                self.assertEqual(answer['content'], turns[turn - 1]['content'], 'the previous answer, verbatim')
                self.assertEqual(answer['reasoning'], turns[turn - 1]['reasoning'], 'with the reasoning the server returned')
                self.assertEqual(current[-1]['role'], 'user')
                self.assertTrue(current[-1]['content'].startswith('[tool result]'))
                self.assertTrue(200 <= turns[turn]['body']['max_tokens'] <= 600)
            self.assertNotIn('ignore_eos', turns[1]['body'])

    def test_every_request_carries_its_tag_and_its_sessions_verified_salt(self):
        _results, chats, _ = self.agent()
        salts = {}
        for chat in chats:
            self.assertEqual(chat['headers'].get('X-Request-Id'), chat['tag'])
            self.assertTrue(chat['verified'], 'minted under the key file')
            session = int(re.search(r'-s(\d+)-t', chat['tag']).group(1))
            salts.setdefault(session, set()).add(chat['body']['cache_salt'])
        self.assertEqual([len(values) for values in salts.values()], [1, 1, 1], 'one salt a session')
        self.assertEqual(len({next(iter(values)) for values in salts.values()}), 3, 'a different salt for every session')

    def test_returning_turns_report_cached_tokens_and_the_row_fields(self):
        results, chats, _ = self.agent()
        entry = results['agent_turns']
        for session in entry['sessions']:
            rows = session['turns']
            self.assertEqual([row['turn'] for row in rows], [0, 1, 2, 3])
            self.assertEqual(rows[0]['cached_tokens'], 0)
            for row in rows[1:]:
                self.assertGreater(row['cached_tokens'], 0)
                self.assertEqual(row['reprefill_tokens'], row['prompt_tokens'] - row['cached_tokens'])
                self.assertGreater(row['appended_tokens'], 0)
            for row in rows:
                for field in ('prompt_tokens', 'tokens', 'finish', 'ttft', 'decode_tok_s', 'max_tokens', 'gap_after_s', 'content_sha256', 'started_at', 'first_at', 'ended_at'):
                    self.assertIn(field, row)
                self.assertNotIn('delta_stamps', row)
            self.assertEqual(session['turns_done'], 3)
            self.assertIsNone(session['stopped'])
        aggregate = entry['aggregate']
        self.assertEqual((aggregate['requests_done'], aggregate['errors']), (12, 0))
        self.assertEqual(aggregate['reuse']['returning_turns'], 9)
        self.assertEqual(aggregate['reuse']['cached_positive'], 9)
        self.assertGreater(aggregate['committed_tok_s'], 0)
        self.assertTrue(aggregate['live'], 'the client-side per-live table')
        self.assertEqual(entry['plan_digest'], at.plan_digest(at.plan(at.settings('agent_turns', dict(self.ENV)))))

    def test_agent_turns8_runs_and_is_judged_under_its_own_name(self):
        results, chats, stdout = self.agent('agent_turns8', env={'SMOKE_AGENT_SESSIONS': '8', 'SMOKE_AGENT_TURNS': '2'})
        entry = results['agent_turns8']
        self.assertNotIn('error', entry, {key: value for key, value in entry.items() if key != 'sessions'})
        self.assertEqual((entry['sessions_planned'], entry['turns_planned']), (8, 2))
        self.assertEqual(len(chats), 8 * 3)
        self.assertEqual(sorted({int(re.search(r'-s(\d+)-t', chat['tag']).group(1)) for chat in chats}), list(range(8)))
        problems, facts = check.check(stdout, container_log(chats, round_live=8), False, env={'QWEN_PREFIX_REUSE': '1'})
        self.assertEqual([item for item in problems if 'agent_turns' in item], [], problems)
        self.assertIn('agent_turns8', facts['agent_turns'])
        self.assertEqual((facts['agent_turns']['agent_turns8']['requests'], facts['agent_turns']['agent_turns8']['errors']), (24, 0))

    def test_the_requests_are_the_same_on_a_second_run(self):
        _results, chats, _ = self.agent()
        with AgentFake() as fake:
            environment = dict(self.ENV, SMOKE_SALT_KEY_FILE=self.key_file())
            self.run_smoke(fake, ('agent_turns',), environment)
            again = list(fake.chats)
        first = {chat['tag']: chat['body']['messages'] for chat in chats}
        second = {chat['tag']: chat['body']['messages'] for chat in again}
        self.assertEqual(first, second, 'same seed, same fake: the same messages on every request')

    def test_the_judge_passes_the_run_and_reads_the_engine_log(self):
        results, chats, stdout = self.agent()
        text = container_log(chats)
        problems, facts = check.check(stdout, text, False, env={'QWEN_PREFIX_REUSE': '1'})
        self.assertEqual([item for item in problems if 'agent_turns' in item], [], problems)
        found = facts['agent_turns']['agent_turns']
        json.dumps(facts)       # the facts line c2_smoke_check prints
        self.assertEqual((found['requests'], found['errors']), (12, 0))
        self.assertTrue(found['salted'])
        self.assertEqual(found['reuse']['returning_turns'], 9)
        self.assertEqual(found['reuse']['reused_zero'], 0)
        self.assertEqual(found['reuse']['source'], 'log+usage')
        self.assertIsNotNone(found['log']['prefill_share'])
        self.assertIsNotNone(found['log']['decode_share'])
        self.assertEqual(found['log']['rebind_p50_ms'], 540.0)
        self.assertEqual([row['live'] for row in found['log']['live']], [4])

    def test_a_server_that_reports_no_cached_tokens_is_read_from_its_log(self):
        results, chats, stdout = self.agent(fake_options=dict(details=False))
        entry = results['agent_turns']
        for session in entry['sessions']:
            for row in session['turns']:
                self.assertNotIn('cached_tokens', row)
        problems, facts = check.check(stdout, container_log(chats), False, env={'QWEN_PREFIX_REUSE': '1'})
        self.assertEqual([item for item in problems if 'agent_turns' in item], [])
        self.assertEqual(facts['agent_turns']['agent_turns']['reuse']['source'], 'log')
        problems, facts = check.check(stdout, '', False, env={'QWEN_PREFIX_REUSE': '1'})
        self.assertEqual([item for item in problems if 'agent_turns' in item], [], 'no usage detail and no log: reuse is unread, not failed')
        self.assertTrue(facts['agent_turns']['agent_turns']['reuse']['unread'])

    def test_a_request_that_fails_ends_its_session_and_fails_the_judge(self):
        results, chats, stdout = self.agent(fake_options=dict(fail=('agt-abcd1234-s1-t02', 500)))
        entry = results['agent_turns']
        self.assertNotIn('error', entry)
        self.assertEqual(entry['sessions'][1]['stopped'], 'error')
        self.assertIn('error', entry['sessions'][1]['turns'][-1])
        self.assertEqual([len(session['turns']) for session in entry['sessions']], [4, 3, 4], 'the other sessions ran on')
        problems, _ = check.check(stdout, '', False, env=None)
        self.assertTrue(any('agent_turns session 1 turn 2' in item for item in problems), problems)

    def test_no_salt_key_means_an_unsalted_run_the_judge_refuses_on_a_reuse_profile(self):
        results, chats, stdout = self.agent(salted=False)
        entry = results['agent_turns']
        self.assertFalse(entry['salted'])
        self.assertTrue(all('cache_salt' not in chat['body'] for chat in chats))
        self.assertTrue(all(row['cached_tokens'] == 0 for session in entry['sessions'] for row in session['turns']))
        problems, _ = check.check(stdout, '', False, env={'QWEN_PREFIX_REUSE': '1'})
        self.assertTrue(any('sent no cache_salt on a prefix-reuse profile' in item for item in problems), problems)
        problems, _ = check.check(stdout, '', False, env={'QWEN_PREFIX_REUSE': '0'})
        self.assertEqual([item for item in problems if 'agent_turns' in item], [], 'a profile without reuse is judged on its streams alone')
        problems, _ = check.check(stdout, '', False, env=None)
        self.assertEqual([item for item in problems if 'agent_turns' in item], [])

    def test_asking_for_no_salt_is_the_control_arm_and_passes(self):
        results, chats, stdout = self.agent(env={'SMOKE_AGENT_SALT': 'none'})
        entry = results['agent_turns']
        self.assertEqual((entry['salted'], entry['salt_mode']), (False, 'none'))
        self.assertTrue(all('cache_salt' not in chat['body'] for chat in chats), 'even with the key file named')
        problems, _ = check.check(stdout, '', False, env={'QWEN_PREFIX_REUSE': '1'})
        self.assertEqual([item for item in problems if 'agent_turns' in item], [])

    def test_a_salted_run_whose_server_never_reuses_fails_on_a_reuse_profile(self):
        results, chats, stdout = self.agent(fake_options=dict(reuse=False))
        problems, _ = check.check(stdout, container_log([dict(chat, cached=0) for chat in chats]), False, env={'QWEN_PREFIX_REUSE': '1'})
        self.assertTrue(any('returning turns reused nothing' in item for item in problems), problems)

    def test_the_context_is_clamped_to_what_the_server_says_it_serves(self):
        results, chats, _ = self.agent(env={'SMOKE_AGENT_MIX': '4k,200k'}, fake_options=dict(max_model_len=16384))
        entry = results['agent_turns']
        self.assertNotIn('error', entry)
        room = at.context_room(at.settings('agent_turns', dict(self.ENV)), 16384)
        self.assertGreaterEqual(entry['clamped'], 1)
        self.assertLessEqual(max(entry['targets']), room)
        self.assertEqual(entry['max_model_len'], 16384)

    def test_the_budget_stops_a_session_between_turns_without_an_error(self):
        results, chats, stdout = self.agent(env={'SMOKE_AGENT_BUDGET_S': '60', 'SMOKE_AGENT_GAP_SCALE': '0', 'SMOKE_AGENT_TURNS': '2'})
        self.assertNotIn('error', results['agent_turns'])
        self.assertEqual([session['turns_done'] for session in results['agent_turns']['sessions']], [2, 2, 2])


class ReportTests(AgentRuns):
    def test_the_report_prints_every_table_from_the_smoke_log_and_the_container_log(self):
        import contextlib
        import io
        _results, chats, stdout = self.agent()
        with tempfile.TemporaryDirectory() as directory:
            smoke_log, container, out_json = Path(directory) / 'smoke.log', Path(directory) / 'container.log', Path(directory) / 'out.json'
            smoke_log.write_text(stdout, encoding='utf-8')
            container.write_text(container_log(chats), encoding='utf-8')
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                code = at.main(['report', '--smoke-log', str(smoke_log), '--container-log', str(container), '--json', str(out_json)])
            self.assertEqual(code, 0)
            text = captured.getvalue()
            for heading in ('PER SESSION', 'PER TURN', 'AGGREGATE', 'CLIENT LIVE TABLE', 'CONTAINER LOG', 'SERVER LIVE TABLE', 'steady window', 'rebind'):
                self.assertIn(heading, text)
            saved = json.loads(out_json.read_text(encoding='utf-8'))
            self.assertEqual(saved['entry']['test'], 'agent_turns')
            self.assertTrue(saved['log']['readable'])
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                self.assertEqual(at.main(['report', '--smoke-log', str(smoke_log), '--test', 'agent_turns_long']), 1)
            self.assertIn('no usable agent_turns_long entry', captured.getvalue())

    def test_the_plan_command_needs_no_server(self):
        import contextlib
        import io
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            self.assertEqual(at.main(['plan', '--test', 'agent_turns_long', '--max-model-len', '131072']), 0)
        shown = json.loads(captured.getvalue())
        self.assertEqual((shown['sessions'], shown['turns']), (6, 30))
        self.assertGreater(shown['clamped'], 0)
        self.assertIn('minutes', shown['estimate'])


class PlanTests(unittest.TestCase):
    def cfg(self, name='agent_turns', **env):
        return at.settings(name, dict({'SMOKE_AGENT_NONCE': 'abcd1234'}, **env))

    def test_defaults_are_six_sessions_of_twelve_turns_and_the_long_one_is_thirty(self):
        self.assertEqual((self.cfg()['sessions'], self.cfg()['turns']), (6, 12))
        self.assertEqual((self.cfg('agent_turns_long')['sessions'], self.cfg('agent_turns_long')['turns']), (6, 30))
        self.assertEqual(self.cfg()['mix'], (64000, 128000, 200000))
        self.assertEqual(self.cfg()['seed'], at.SEED)

    def test_the_existing_tests_plans_and_settings_are_frozen(self):
        # agent_turns and agent_turns_long are the baseline of past results (the 2026-10-19 window's T1A/T1B): adding agent_turns8 must not move what they measure
        for name, digest in (('agent_turns', '9fb35dd44ea15464'), ('agent_turns_long', 'e0a2af94ab80930f')):
            cfg = self.cfg(name)
            self.assertEqual(at.plan_digest(at.plan(cfg, 262144)), digest, name)
            self.assertEqual((cfg['sessions'], cfg['budget_s'], cfg['mix'], cfg['seed'], cfg['answer_range'], cfg['message_range'], cfg['gap_range']),
                             (6, 2700.0, (64000, 128000, 200000), 20261012, (200, 600), (150, 400), (1.0, 8.0)), name)
        self.assertEqual((self.cfg()['turns'], self.cfg('agent_turns_long')['turns']), (12, 30))

    def test_agent_turns8_is_the_same_plan_with_eight_sessions_and_a_budget_that_fits_a_half_hour_box(self):
        self.assertEqual(at.EIGHT_NAME, 'agent_turns8')
        self.assertEqual(at.TESTS, ('agent_turns', 'agent_turns_long', 'agent_turns8'))
        cfg, base = self.cfg('agent_turns8'), self.cfg()
        self.assertEqual((cfg['sessions'], cfg['turns'], cfg['budget_s']), (8, 12, 1500.0))
        for key in ('mix', 'seed', 'answer_range', 'message_range', 'gap_range', 'gap_scale', 'stagger_s', 'timeout_s', 'salt_mode', 'ignore_eos'):
            self.assertEqual(cfg[key], base[key], key)
        eight, six = at.plan(cfg, 262144), at.plan(base, 262144)
        self.assertEqual(eight['targets'][:6], six['targets'], 'sessions 0 to 5 are agent_turns\' six contexts')
        self.assertEqual(eight['sessions'][:6], six['sessions'], 'and their turns, gaps and answer budgets')
        self.assertEqual(len(eight['targets']), 8)
        self.assertEqual(eight['clamped'], 0, 'the 262k profile holds all eight with 12 turns of growth')
        self.assertEqual(len(set(eight['targets'])), 8)
        # an environment parameter still wins over the variant's default, for either
        self.assertEqual(self.cfg('agent_turns8', SMOKE_AGENT_BUDGET_S='900')['budget_s'], 900.0)
        self.assertEqual(self.cfg('agent_turns8', SMOKE_AGENT_SESSIONS='7')['sessions'], 7)

    def test_agent_turns8_estimate_fits_a_thirty_minute_box(self):
        cfg = self.cfg('agent_turns8')
        found = at.estimate(cfg, at.session_targets(cfg, 262144)[0])
        self.assertTrue(12 <= found['minutes'] <= 18, found)
        self.assertTrue(found['within_budget'])
        self.assertLessEqual(cfg['budget_s'] / 60.0, 25.0, 'no new turn starts after 25 minutes: the run ends inside 30 with its last turn (a turn is under 3 minutes at 8 live)')
        self.assertGreater(found['minutes'], at.estimate(self.cfg())['minutes'], 'two more contexts to build')

    def test_the_plan_is_deterministic_in_the_seed_and_changes_with_it(self):
        one, two = at.plan(self.cfg()), at.plan(self.cfg())
        self.assertEqual(one, two)
        self.assertEqual(at.plan_digest(one), at.plan_digest(two))
        other = at.plan(self.cfg(SMOKE_AGENT_SEED='7'))
        self.assertNotEqual(at.plan_digest(one), at.plan_digest(other))
        self.assertEqual(one['targets'], at.plan(self.cfg(SMOKE_AGENT_NONCE='feedbeef'))['targets'], 'the nonce names requests, it does not move the plan')

    def test_the_plan_stays_in_the_ranges_it_promises(self):
        the_plan = at.plan(self.cfg())
        self.assertEqual(len(the_plan['sessions']), 6)
        for target, turns in zip(the_plan['targets'], the_plan['sessions']):
            self.assertEqual(target % 64, 0)
            self.assertTrue(any(abs(target - base) <= 0.045 * base for base in (64000, 128000, 200000)), target)
            self.assertEqual(len(turns), 13)
            self.assertEqual(turns[0]['message_tokens'], 0)
            for draws in turns:
                self.assertTrue(1.0 <= draws['gap_s'] <= 8.0, draws)
                self.assertTrue(200 <= draws['max_tokens'] <= 600, draws)
                self.assertIn(draws['kind'], ('read_file', 'grep'))
                if draws['turn']:
                    self.assertTrue(150 <= draws['message_tokens'] <= 400, draws)
        self.assertEqual(sorted(the_plan['targets']), sorted(the_plan['targets']))
        self.assertEqual(len({value for value in the_plan['targets']}), 6, 'six different contexts')
        scaled = at.plan(self.cfg(SMOKE_AGENT_GAP_SCALE='0.5'))
        for turns, original in zip(scaled['sessions'], the_plan['sessions']):
            for draws, first in zip(turns, original):
                self.assertAlmostEqual(draws['gap_s'], first['gap_s'] * 0.5, delta=0.002)
                self.assertEqual(draws['max_tokens'], first['max_tokens'])

    def test_the_contexts_are_clamped_under_a_smaller_profile(self):
        cfg = self.cfg()
        targets, clamped = at.session_targets(cfg, 131072)
        self.assertGreater(clamped, 0)
        self.assertLessEqual(max(targets), at.context_room(cfg, 131072))
        self.assertEqual(at.session_targets(cfg, 262144)[1], 0, 'the 262k profile holds the 200k mix with 12 turns of growth')
        self.assertEqual(at.session_targets(self.cfg('agent_turns_long'), 262144)[1], 0, 'and 30 turns')

    def test_the_run_time_estimate_lands_where_the_docstring_says(self):
        short, long = at.estimate(self.cfg()), at.estimate(self.cfg('agent_turns_long'))
        self.assertTrue(8 <= short['minutes'] <= 18, short)
        self.assertTrue(20 <= long['minutes'] <= 30, long)
        self.assertTrue(short['within_budget'] and long['within_budget'])
        self.assertGreater(at.estimate(self.cfg(SMOKE_AGENT_GAP_SCALE='4'))['minutes'], short['minutes'])

    def test_settings_refuse_what_they_cannot_run(self):
        for key, value in (('SESSIONS', '0'), ('SESSIONS', '40'), ('TURNS', 'x'), ('GAP_SCALE', '-1'), ('MIX', '10'), ('SALT', 'maybe'), ('NONCE', 'NOT OK'),
                           ('BUDGET_S', '5')):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    self.cfg(**{'SMOKE_AGENT_' + key: value})
        with self.assertRaises(ValueError):
            at.settings('agent_turns_huge', {})
        self.assertEqual(self.cfg(SMOKE_AGENT_MIX='32k, 64000,1.5k')['mix'], (32000, 64000, 1500))

    def test_the_tool_message_is_real_code_of_the_planned_size_and_deterministic(self):
        corpus = ''.join('File: pkg/module_%d.py\n%s\n\n' % (index, '\n'.join('def function_%d_%d(argument):\n    return argument + <|im_start|> %d' % (index, line, line)
                                                                              for line in range(120))) for index in range(40))
        cfg = self.cfg()
        the_plan = at.plan(cfg)
        sizes = []
        for session in range(2):
            for turn in range(1, 13):
                draws = the_plan['sessions'][session][turn]
                message = at.tool_message(corpus, cfg, session, turn, draws)
                self.assertEqual(message, at.tool_message(corpus, cfg, session, turn, draws))
                self.assertTrue(message.startswith('[tool result] '), message[:60])
                self.assertIn('[end of tool result]', message)
                self.assertTrue(message.rstrip().endswith(at.FOLLOWUPS[draws['instruction']]))
                self.assertNotIn('<|', message, 'chat-template markers in real code are neutralised')
                self.assertIn('pkg/module_', message)
                sizes.append(len(message) / at.CHARS_PER_TOKEN)
        self.assertTrue(80 <= min(sizes) and max(sizes) <= 420, (min(sizes), max(sizes)))
        self.assertNotEqual(at.tool_message(corpus, cfg, 0, 1, the_plan['sessions'][0][1]), at.tool_message(corpus, cfg, 1, 1, the_plan['sessions'][1][1]))

    def test_the_line_numbers_of_an_excerpt_are_the_files(self):
        corpus = 'File: a/b.py\n' + ''.join('line %d\n' % number for number in range(1, 200)) + '\nFile: c/d.py\nother\n'
        position = corpus.index('line 50\n')
        self.assertEqual(at.file_at(corpus, position), ('a/b.py', 50))
        self.assertEqual(at.file_at(corpus, corpus.index('other')), ('c/d.py', 1))
        self.assertEqual(at.file_at('no header here\nat all', 5), ('src/module.py', 1))


class SaltTests(unittest.TestCase):
    def test_the_salt_is_the_contracts(self):
        import serving_c2_contract
        for tag in ('agentturns-abcd1234-s0', 'agentturns-abcd1234-s5'):
            self.assertEqual(at.mint_salt(KEY, tag), serving_c2_contract.mint_salt(KEY, tag))
            self.assertEqual(serving_c2_contract.salt_verdict(at.mint_salt(KEY, tag), KEY), 'verified')
        self.assertEqual(serving_c2_contract.salt_verdict(at.mint_salt(KEY, 'agentturns-abcd1234-s0'), b'x' * 40), 'dropped-unverified')

    def test_a_short_key_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'short.key'
            path.write_bytes(b'tiny\n')
            with self.assertRaises(ValueError):
                at.read_salt_key(str(path))
            path.write_bytes(b'0123456789abcdef0123456789abcdef\n')
            self.assertEqual(at.read_salt_key(str(path)), b'0123456789abcdef0123456789abcdef')

    def test_the_tag_is_what_the_engine_log_carries(self):
        tag = at.request_tag('abcd1234', 3, 7)
        self.assertEqual(tag, 'agt-abcd1234-s3-t07')
        line = '(EngineCore pid=67) [PINDIAG] lever N route req=chatcmpl-%s-1a2b3c4d start=0 end=10 prompt=10 final=1' % tag
        self.assertEqual(at.tag_of(line, 'abcd1234'), (3, 7))
        self.assertIsNone(at.tag_of(line, 'feedbeef'), 'another run\'s tag is not this run\'s')
        self.assertIsNone(at.tag_of('[PINDIAG] lever N route req=chatcmpl-other-1a2b3c4d start=0', 'abcd1234'))


class NumbersTests(unittest.TestCase):
    def test_percentiles(self):
        self.assertIsNone(at.percentile([], 50))
        self.assertIsNone(at.percentile([None], 50))
        self.assertEqual(at.percentile([4.0], 90), 4.0)
        self.assertEqual(at.percentile([1.0, 2.0, 3.0, 4.0, None], 50), 2.5)
        self.assertAlmostEqual(at.percentile(list(range(1, 11)), 90), 9.1)

    def test_the_live_table_counts_users_streaming_and_their_tokens(self):
        # A streams 0-10 s with a stamp every 0.5 s (20 stamps, 40 tokens: 2 each); B streams 5-10 s with a stamp every 0.5 s (10 stamps, 10 tokens: 1 each)
        stamps_a = [0.5 * number for number in range(1, 21)]
        stamps_b = [5.0 + 0.5 * number for number in range(1, 11)]
        table = at.live_table([(0.0, 10.0, stamps_a, 40), (5.0, 10.0, stamps_b, 10)])
        self.assertEqual([row['live'] for row in table], [1, 2])
        one, two = table
        self.assertEqual((one['seconds'], one['tokens']), (5.0, 20))
        self.assertEqual(one['user_tok_s'], 4.0)
        self.assertEqual((two['seconds'], two['tokens']), (5.0, 30))
        self.assertEqual(two['aggregate_tok_s'], 6.0)
        self.assertEqual(two['user_tok_s'], 3.0)
        windowed = at.live_table([(0.0, 10.0, stamps_a, 40), (5.0, 10.0, stamps_b, 10)], 6.0, 9.0)
        self.assertEqual([row['live'] for row in windowed], [2])
        self.assertEqual(windowed[0]['seconds'], 3.0)
        self.assertEqual(at.live_table([(None, None, [], 0)]), [])
        self.assertAlmostEqual(at.window_tokens([(0.0, 10.0, stamps_a, 40)], 0.0, 5.0), 20.0)

    def test_the_server_live_table_is_the_round_arithmetic(self):
        rounds = [dict(live=4, step_ms=100.0, gap_ms=20.0, committed=40), dict(live=4, step_ms=100.0, gap_ms=20.0, committed=40),
                  dict(live=2, step_ms=80.0, gap_ms=0.0, committed=20), dict(live=4, step_ms=100.0, gap_ms=90000.0, committed=40)]
        table, idle = at.live_rows(rounds)
        self.assertEqual(idle, 1, 'a round that followed 90 s of idle is left out')
        self.assertEqual([row['live'] for row in table], [2, 4])
        two, four = table
        self.assertEqual((four['rounds'], four['step_ms'], four['gap_ms'], four['committed_per_live']), (2, 100.0, 20.0, 10.0))
        self.assertAlmostEqual(four['user_tok_s'], 10.0 / 0.12, delta=0.1)
        self.assertAlmostEqual(four['aggregate_tok_s'], 40.0 / 0.12, delta=0.1)
        self.assertAlmostEqual(two['user_tok_s'], 10.0 / 0.08, delta=0.1)


class LogTests(unittest.TestCase):
    def entry(self, turns=2, sessions=2, wall_s=100.0):
        rows = []
        for session in range(sessions):
            rows.append(dict(session=session, turns=[dict(session=session, turn=turn, tokens=25, prompt_tokens=1000 + 100 * turn, cached_tokens=0 if turn == 0 else 900) for turn in range(turns + 1)]))
        return dict(test='agent_turns', nonce='abcd1234', sessions=rows, aggregate=dict(wall_s=wall_s))

    def chats(self, entry):
        out = []
        for session in entry['sessions']:
            for row in session['turns']:
                out.append(dict(seq=len(out), tag=at.request_tag('abcd1234', row['session'], row['turn']), prompt_tokens=row['prompt_tokens'], cached=row['cached_tokens'],
                                tokens=row['tokens']))
        return out

    def test_the_window_is_the_run_and_the_tail_of_its_rounds_not_the_noise(self):
        entry = self.entry()
        text = container_log(self.chats(entry))
        facts = at.container_facts(entry, text)
        self.assertTrue(facts['readable'])
        # 6 requests x 6 rounds of 100 ms; the round before the run and the round after the foreign prefill are not counted
        self.assertEqual(facts['decode']['rounds'], 36)
        self.assertEqual(facts['decode']['step_ms'], 3600.0)
        self.assertAlmostEqual(facts['decode']['share'], 3600.0 / 100000.0, places=3)
        # cold builds: 2 sessions x (2000 + 3000) ms; resumed turns: 4 x (600 + 100/4 * ... ) from the lines
        self.assertEqual(facts['prefill']['cold_ms'], 10000.0)
        self.assertEqual(facts['prefill']['route_steps'], 2 * 2 + 4)
        self.assertGreater(facts['prefill']['resumed_ms'], 0)
        self.assertAlmostEqual(facts['prefill']['share'], (10000.0 + facts['prefill']['resumed_ms']) / 100000.0, places=2)
        self.assertAlmostEqual(facts['other_share'], 1.0 - facts['prefill']['share'] - facts['decode']['share'], places=2)
        self.assertEqual(facts['turns'][(0, 0)]['reused'], 0)
        self.assertEqual(facts['turns'][(1, 2)]['reused'], 900)
        self.assertEqual(facts['rebind']['count'], 4)

    def test_a_run_ends_where_its_tokens_are_all_committed(self):
        entry = self.entry(turns=0, sessions=1)
        text = container_log(self.chats(entry), noise=False) + ''.join(octo_markers.ROUND_LINE.format(
            round=50 + number, shape='octo', planned='octo', live=1, rows=8, eligible=1, counted=1, octo_rounds=50 + number, m3_rounds=0, committed=40, step_ms=100.0,
            gap_ms=10.0) + '\n' for number in range(10))
        # one request of 25 tokens: its own six rounds commit 6 * (25 * 4 // 6 + 4) = 114 >= 24: nothing past them belongs to it
        self.assertEqual(at.container_facts(entry, text)['decode']['rounds'], 6)

    def test_a_loguru_or_docker_prefix_does_not_hide_the_lines(self):
        entry = self.entry()
        prefix = '2026-10-10T20:45:03.000000Z (EngineCore pid=67) 2026-10-10 20:45:03.000 | INFO | x:y:1 - '
        plain = at.container_facts(entry, container_log(self.chats(entry)))
        prefixed = at.container_facts(entry, container_log(self.chats(entry), prefix=prefix))
        self.assertEqual((plain['decode'], plain['prefill']), (prefixed['decode'], prefixed['prefill']))
        self.assertEqual(plain['live'], prefixed['live'])

    def test_a_log_that_never_names_the_run_is_unread_not_empty(self):
        entry = self.entry()
        facts = at.container_facts(entry, container_log(self.chats(entry), nonce='feedbeef').replace('agt-abcd1234', 'agt-feedbeef'))
        self.assertFalse(facts['readable'])
        self.assertIn('X-Request-Id', facts['reason'])
        problems, found = at.judge(entry, {'QWEN_PREFIX_REUSE': '1'}, 'nothing useful\n')
        self.assertTrue(str(found['log']).startswith('unread'))

    def test_the_prefix_row_is_the_reuse_where_no_route_line_exists(self):
        entry = self.entry(turns=1, sessions=1)
        text = ''.join('[PREFIX] row=%d req=chatcmpl-%s-1a2b3c4d path=traced registry=present grant=committed Q=%d L=%d plan=[] ms=1.0\n' % (
            number, at.request_tag('abcd1234', 0, number), 0 if number == 0 else 900, 1000 + 100 * number) for number in range(2))
        facts = at.container_facts(entry, text)
        self.assertEqual((facts['turns'][(0, 0)]['reused'], facts['turns'][(0, 1)]['reused']), (0, 900))
        self.assertEqual(facts['turns'][(0, 1)]['source'], 'PREFIX')
        self.assertIsNone(facts['prefill']['share'], 'no route line, no prefill time')


class JudgeTests(unittest.TestCase):
    def entry(self, cached=900, salted=True, salt_mode='auto', returning=10):
        rows = [dict(session=0, turn=0, tokens=25, finish='length', text='fine words here ' * 4, prompt_tokens=1000, cached_tokens=0)]
        rows += [dict(session=0, turn=turn, tokens=25, finish='length', text='fine words here ' * 4, prompt_tokens=1100 + turn, cached_tokens=cached if callable(cached) is False else cached(turn))
                 for turn in range(1, returning + 1)]
        return dict(test='agent_turns', nonce='abcd1234', salted=salted, salt_mode=salt_mode, sessions=[dict(session=0, turns=rows, stopped=None)],
                    aggregate=dict(requests_done=len(rows), errors=0, sessions=1, wall_s=10.0, committed_tok_s=25.0, returning=dict(turns=returning)))

    def results(self, **options):
        return {'agent_turns': self.entry(**options)}

    def test_a_good_salted_run_on_a_reuse_profile_passes(self):
        problems, facts = check.agent_turns_problems(self.results(), {'QWEN_PREFIX_REUSE': '1'})
        self.assertEqual(problems, [])
        self.assertEqual(facts['agent_turns']['reuse'], dict(source='usage', returning_turns=10, read=10, reused_zero=0))

    def test_one_missed_turn_in_ten_is_tolerated_two_are_not(self):
        one = self.results(cached=lambda turn: 0 if turn == 3 else 900)
        self.assertEqual(check.agent_turns_problems(one, {'QWEN_PREFIX_REUSE': '1'})[0], [])
        two = self.results(cached=lambda turn: 0 if turn in (3, 4) else 900)
        problems, _ = check.agent_turns_problems(two, {'QWEN_PREFIX_REUSE': '1'})
        self.assertEqual(len(problems), 1)
        self.assertIn('2 of 10 returning turns reused nothing', problems[0])

    def test_reuse_is_not_judged_without_the_profile_saying_so(self):
        broken = self.results(cached=0)
        self.assertEqual(check.agent_turns_problems(broken, None)[0], [])
        self.assertEqual(check.agent_turns_problems(broken, {})[0], [])
        self.assertEqual(check.agent_turns_problems(broken, {'QWEN_PREFIX_REUSE': '0'})[0], [])
        self.assertEqual(len(check.agent_turns_problems(broken, {'QWEN_PREFIX_REUSE': '1'})[0]), 1)

    def test_unsalted_on_a_reuse_profile_fails_unless_it_asked_for_nothing(self):
        problems, _ = check.agent_turns_problems(self.results(salted=False, cached=0), {'QWEN_PREFIX_REUSE': '1'})
        self.assertEqual(len(problems), 1)
        self.assertIn('sent no cache_salt', problems[0])
        self.assertEqual(check.agent_turns_problems(self.results(salted=False, salt_mode='none', cached=0), {'QWEN_PREFIX_REUSE': '1'})[0], [])

    def test_errors_timeouts_and_empty_answers_fail(self):
        entry = self.entry()
        entry['sessions'][0]['turns'][4] = dict(session=0, turn=4, tag='t', error="TimeoutError('timed out')")
        entry['aggregate']['errors'] = 1
        problems, _ = check.agent_turns_problems({'agent_turns': entry}, None)
        self.assertEqual(problems, ["agent_turns session 0 turn 4: TimeoutError('timed out')"])
        empty = self.entry()
        empty['sessions'][0]['turns'][2].update(tokens=0, finish=None)
        problems, _ = check.agent_turns_problems({'agent_turns': empty}, None)
        self.assertTrue(any('no tokens' in item for item in problems), problems)
        self.assertTrue(any('no finish reason' in item for item in problems), problems)
        self.assertEqual(check.agent_turns_problems({'agent_turns': dict(error='boom')}, None)[0], ['agent_turns: boom'])
        self.assertEqual(check.agent_turns_problems({'agent_turns': 'oops'}, None)[0], ['agent_turns: no result'])
        self.assertEqual(check.agent_turns_problems({}, None), ([], {}))

    def test_only_opening_requests_is_a_failure(self):
        entry = self.entry(returning=0)
        entry['aggregate']['returning'] = dict(turns=0)
        problems, _ = check.agent_turns_problems({'agent_turns': entry}, None)
        self.assertEqual(problems, ['agent_turns: no returning turn completed (the opening requests only)'])

    def test_a_budget_stop_is_a_fact_not_a_failure(self):
        entry = self.entry()
        entry['sessions'][0]['stopped'] = 'budget'
        problems, facts = check.agent_turns_problems({'agent_turns': entry}, None)
        self.assertEqual(problems, [])
        self.assertEqual(facts['agent_turns']['stopped'], ['budget'])


class RegistrationTests(unittest.TestCase):
    def test_the_smoke_dispatches_both_tests_opt_in_and_declares_their_seats(self):
        source = SMOKE.read_text(encoding='utf-8')
        self.assertIn("for _agent_name in ('agent_turns', 'agent_turns_long', 'agent_turns8'):", source)
        self.assertIn('if ONLY and _agent_name in ONLY:', source)
        self.assertEqual(at.TESTS, check.AGENT_TURNS_TESTS)
        seats = constants()['SHAPE_SEATS']
        for name in at.TESTS:
            self.assertEqual(seats[name], at.VARIANTS[name][0], name)
            self.assertLessEqual(seats[name], 8)

    def test_agent_turns8_is_dispatched_judged_and_given_a_salt_key_by_the_smoke_step(self):
        self.assertIn('agent_turns8', check.AGENT_TURNS_TESTS)
        self.assertIn('agent_turns8', check.LEVERN_SCHED_RETURNING_TESTS)
        self.assertEqual(constants()['SHAPE_SEATS']['agent_turns8'], 8, 'every seat of the profile')
        text = (HERE.parent.parent / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
        step = text[text.index('- name: Smoke on the four-card set'):text.index('- name: Replay the node agent')]
        self.assertIn('*,agent_turns8,*', step, 'a job that names only agent_turns8 still gets the throwaway salt key (without it the run is unsalted and measures a server with reuse off)')

    def test_the_default_test_list_does_not_include_them(self):
        # opt-in: with no test list the smoke records only warmup and its default set (nothing here names the agent tests outside the dispatch)
        source = SMOKE.read_text(encoding='utf-8')
        self.assertEqual(source.count("'agent_turns'"), 2, 'the dispatch tuple and SHAPE_SEATS')

    def test_the_modules_stay_python_37_syntax_for_the_rig_host(self):
        for name in ('agent_turns.py', 'c2_serving_smoke.py', 'c2_smoke_check.py'):
            with self.subTest(name=name):
                ast.parse((HERE / name).read_text(encoding='utf-8'), feature_version=(3, 7))

    def test_the_smoke_step_hands_the_container_a_salt_key_for_the_agent_tests_only(self):
        text = (HERE.parent.parent / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
        step = text[text.index('- name: Smoke on the four-card set'):text.index('- name: Replay the node agent')]
        self.assertIn('agent_turns', step)
        self.assertIn('SMOKE_SALT_KEY_FILE', step)
        self.assertIn('QWEN_PREFIX_SALT_KEY_FILE', step)
        self.assertIn('rm -f "$salt_key"', step, 'the throwaway key is removed with the container')
        self.assertNotIn('c2-smoke-salt', text[:text.index('- name: Smoke on the four-card set')], 'no other step names the smoke key')


if __name__ == '__main__':
    unittest.main()
