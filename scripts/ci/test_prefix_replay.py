"""prefix_replay: the prefix gates' driver, on CPU.

The streaming client is held against a real local HTTP server (server-sent events, slow first
tokens, closed sockets); the scenarios against FakeEngine, a served general-prefix engine in
miniature: a deterministic word 'tokenizer' whose prompts extend turn by turn the way the chat
template's do, greedy answers that depend only on the prompt, grants from the design's rules (the
judge's own oracle - these tests hold the plumbing, test_prefix_oracle_check holds the rules to the
real graft), and the marker lines the engine prints (prefix_markers' contract). Faults are switches
on it. test_c2_prefix_gate reuses it."""

import http.server
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import time
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prefix_agent_corpus as pc  # noqa: E402
import prefix_judge as judge  # noqa: E402
import prefix_markers as pm  # noqa: E402
import prefix_replay as replay  # noqa: E402
import prefix_scheduler_graft as graft  # noqa: E402
import test_prefix_agent_corpus as corpus_fixture  # noqa: E402
import test_prefix_markers as marker_fixture  # noqa: E402

GEN = [9001, 9002]
ROLE = dict(system=9101, user=9102, assistant=9103, tool=9104)
VOCAB = ['alpha', 'beta', 'gamma', 'delta', 'return', 'value', 'list', 'None', 'file', 'test', 'call', 'fix']


def word(text):
    return zlib.crc32(text.encode('utf-8')) % 150000 + 1000


def words(text):
    """About one token per 3.2 characters, as Qwen3.8's tokenizer on this text: each whitespace-
    separated word in 3-character pieces."""
    return [word(w[i:i + 3]) for w in (text or '').split() for i in range(0, len(w), 3)]


class FakeEngine(object):
    """A general-prefix (or general) engine in miniature; see the module docstring."""

    def __init__(self, prefix=True, profile='general-prefix', path='traced', diverge_hits=False, grow_programs=False,
                 drop_rows=False, audit=False, store=None, answer_len=24, dev_mode=True, preempt_on_ignore_eos=0,
                 unsalted_differs=False, kill_line=True, store_gib=None, kv_tokens=262144):
        self.prefix, self.profile, self.path = prefix, profile, path
        self.diverge_hits, self.grow_programs, self.drop_rows = diverge_hits, grow_programs, drop_rows
        self.audit, self.store, self.answer_len, self.dev_mode = audit, store, answer_len, dev_mode
        self.preempt_on_ignore_eos, self.unsalted_differs, self.kill_line = preempt_on_ignore_eos, unsalted_differs, kill_line
        self.store_gib, self.kv_tokens = store_gib, kv_tokens
        self.lock = threading.Lock()
        self.lines = []
        self.count = 0
        self.programs = 500
        self.preemptions = 0
        self.flag = False
        self.stats = dict(dropped_attempts=2, evicted_lru=3, pins=0, commit_mismatch=0)
        self.boot()

    # -- the log ---------------------------------------------------------------------------------
    def say(self, line):
        self.lines.append('2026-09-26T10:%02d:%02d.000000000Z %s' % (len(self.lines) // 60 % 60, len(self.lines) % 60, line))

    def boot(self):
        self.oracle = judge.Oracle(self.store)
        self.killed = False
        profile = marker_fixture.prefix_profile() if self.prefix else marker_fixture.PROFILES['profiles']['general']
        self.say('[QWEN-C2] profile %s: vLLM argv %s' % (self.profile, json.dumps(marker_fixture.launched_argv(profile))))
        if self.prefix:
            self.say('INFO platform.py:83] Chunked prefill is not supported for `model_type=qwen3_5`; disabling it.')
        self.say('INFO platform.py:1153] Automatic prefix caching is %s' % ('enabled' if self.prefix else 'disabled'))
        self.say('INFO kv_cache_utils.py:2146] GPU KV cache size: {:,} tokens'.format(self.kv_tokens))
        if self.prefix:
            line = marker_fixture.install_line()
            if self.store_gib is not None:
                line = line.replace('store_gib=8.0', 'store_gib=%.1f' % self.store_gib)
            self.say('(EngineCore pid=9) ' + line)
            self.say('[PINDIAG] dram after kv: chip0 allocated=20.0GB free=8.0GB largest_free=900MB of 32GB')

    # -- tokens ----------------------------------------------------------------------------------
    def render_message(self, message):
        if message['role'] == 'assistant':
            calls = ' '.join('%s %s' % (c['function']['name'], c['function']['arguments'])
                             for c in message.get('tool_calls') or ())
            return (GEN + words(message.get('reasoning')) + [9005] + words(message.get('content')) + words(calls)
                    + [9003])
        return [9100, ROLE[message['role']]] + words(message.get('content')) + [9003]

    def render(self, body):
        ids = words(json.dumps(body.get('tools') or []))
        for message in body['messages']:
            ids += self.render_message(message)
        return ids + GEN

    def tokenize(self, body):
        return len(self.render(body))

    # -- the API ---------------------------------------------------------------------------------
    def chat(self, body, tag, salt=None, max_tokens=64, timeout=None, abort_after_s=None, abort_after_tokens=None,
             extra=None, on_first=None, abort_signal=None):
        if abort_signal is not None:
            abort_signal.wait(10)
            return dict(content='', reasoning=None, tool_calls=[], token_ids=None, prompt_ids=None, prompt_sha=None,
                        prompt_tokens=None, completion_tokens=0, finish=None, error=None, ttft_s=None, wall_s=0.5,
                        status=200, aborted='closed on signal', ok=False)
        ids = self.render(body)
        with self.lock:
            if self.flag and not self.killed:
                self.killed = True
                self.oracle.kill()
                if self.kill_line:
                    self.say(graft.KILL_SWITCH_PATH and '[PINDIAG] prefix: kill switch %s present: no grants' % graft.KILL_SWITCH_PATH)
            want = self.oracle.admit(salt if self.prefix else None, ids)
            self.count += 1
            req = 'chatcmpl-%s-%08x' % (tag, self.count)
            if self.prefix:
                if salt and (want['q'] or want['plan']):
                    self.say(marker_fixture.grant_line(req, want['h'], want['q'], want['plan']))
                if self.grow_programs and want['q']:
                    self.programs += 1
                if not self.drop_rows:
                    self.say('[PREFIX] req=%s Q=%d L=%d path=%s restored_ms=%.1f captured=[%s] capture_ms=%.1f programs=%d'
                             % (req, want['q'], len(ids), self.path, 120.0 if want['q'] else 0.0,
                                ','.join(str(p) for p in want['plan']), 300.0 if want['plan'] else 0.0, self.programs))
                if self.audit:
                    digest = judge.token_sha(ids)[:16]
                    self.say('[PREFIX-AUDIT] req=%s Q=%d L=%d kv_range=0:%d kv_sha=%s slot_sha=%s' % (
                        req, want['q'], len(ids), len(ids), digest, digest))
            if len(ids) >= judge.CHUNK and self.path == 'traced':
                self.say('INFO [TP chunk-replay] 1/1 chunks')
            ignore_eos = bool((extra or {}).get('ignore_eos'))
            if ignore_eos and self.preempt_on_ignore_eos:
                self.preemptions += self.preempt_on_ignore_eos
        rng = random.Random(judge.token_sha(ids))
        length = int(max_tokens) if ignore_eos else min(int(max_tokens), self.answer_len)
        answer = [VOCAB[rng.randrange(len(VOCAB))] for _ in range(length)]
        calls = []
        if rng.random() < 0.4:
            calls = [dict(id='chatcmpl-tool-%d' % self.count, type='function', function=dict(
                name='read', arguments=json.dumps({'file_path': '/w/scripts/ci/mod_1.py'})))]
        token_ids = [word(w) for w in answer]
        if (self.diverge_hits and want['q']) or (self.unsalted_differs and not salt):
            token_ids[-1] += 1
            answer[-1] += 'x'
        if abort_after_s is not None:
            return dict(content='', reasoning=None, tool_calls=[], token_ids=None, prompt_ids=ids,
                        prompt_sha=judge.token_sha(ids), prompt_tokens=len(ids), completion_tokens=0, finish=None,
                        error=None, ttft_s=None, wall_s=abort_after_s, status=200,
                        aborted='closed after %.1f s' % abort_after_s, ok=False)
        if on_first is not None:
            on_first()
        half = len(answer) // 2
        return dict(content=' '.join(answer[half:]), reasoning=' '.join(answer[:half]) or None, tool_calls=calls,
                    token_ids=token_ids, prompt_ids=ids, prompt_sha=judge.token_sha(ids), prompt_tokens=len(ids),
                    completion_tokens=len(token_ids), finish='length' if length == int(max_tokens) else 'stop',
                    error=None, ttft_s=0.05 if not want['q'] else 0.01, wall_s=0.5, status=200, aborted=None, ok=True)

    def metrics(self):
        return {'vllm:num_requests_waiting': 1.0, 'vllm:num_preemptions': float(self.preemptions)}

    def healthy(self):
        return True

    def ready(self):
        return True

    def reset_prefix_cache(self):
        if not self.dev_mode:
            return 404
        with self.lock:
            self.oracle.reset_prefix_cache()
        return 200

    def restart(self):
        with self.lock:
            self.boot()


class FakeLog(object):
    def __init__(self, engine):
        self.engine = engine
        self.started = 0

    def start(self, since=None):
        self.started += 1

    def stop(self, wait=0):
        pass

    def mark(self):
        return len(self.engine.lines)

    def lines(self):
        return list(self.engine.lines)

    def last_time(self):
        return pm.split_timestamp(self.engine.lines[-1])[0] if self.engine.lines else None


class FakeContainer(object):
    def __init__(self, engine):
        self.engine = engine
        self.scripts = []
        self.stops = 0

    def running(self):
        return True

    def exec_shell(self, script, timeout=60):
        self.scripts.append(script)
        if ('echo %s >' % replay.KILL_SWITCH_OWNER) in script:
            self.engine.flag = True
        elif 'rm -f' in script:
            self.engine.flag = False
        return 0, ''

    def read_file(self, path):
        return json.dumps(self.engine.stats) if path == pm.STATS_FILE else None

    def stop(self, grace=60):
        self.stops += 1
        return 0, ''

    def start(self):
        self.engine.restart()
        return 0, ''

    def rss_gb(self):
        return 14.5


def corpus():
    root = tempfile.mkdtemp()
    corpus_fixture.make_tree(root, copies=12)
    try:
        return pc.Corpus(pc.load_sources(root))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def driver_for(engine, arm='arm', strict=True):
    return replay.Driver(engine, arm, CORPUS, log=FakeLog(engine), container=FakeContainer(engine), seed=1,
                         sleep=lambda seconds: None, say=lambda text: None, pods=lambda: 3, strict=strict)


CORPUS = corpus()


# -- the streaming client against a real local server ------------------------------------------

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    seen = []
    first_delay = 0.0
    chunk_delay = 0.0
    chunks = 3

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == '/metrics':
            body = b'vllm:num_requests_waiting{engine="0"} 2.0\n'
        elif self.path in ('/health', '/v1/models'):
            body = b'{}'
        else:
            self.send_response(404)
            self.send_header('content-length', '0')
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('content-length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get('content-length') or 0)
        body = json.loads(self.rfile.read(length) or b'null')
        Handler.seen.append((self.path, dict(self.headers), body))
        if self.path == '/tokenize':
            payload = json.dumps(dict(count=len(body['messages']) * 10, tokens=[])).encode()
            self.send_response(200)
            self.send_header('content-length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == '/reset_prefix_cache':
            self.send_response(200)
            self.send_header('content-length', '0')
            self.end_headers()
            return
        if body.get('max_tokens') == 999:
            payload = b'{"error": "bad"}'
            self.send_response(400)
            self.send_header('content-length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        self.end_headers()
        try:
            time.sleep(Handler.first_delay)
            self.event(dict(prompt_token_ids=[1, 2, 3], choices=[dict(index=0, delta=dict(role='assistant'))]))
            self.event(dict(choices=[dict(index=0, delta=dict(reasoning='think '), token_ids=[10])]))
            for index in range(Handler.chunks):
                time.sleep(Handler.chunk_delay)
                self.event(dict(choices=[dict(index=0, delta=dict(content='w%d ' % index), token_ids=[20 + index])]))
            self.event(dict(choices=[dict(index=0, delta=dict(tool_calls=[dict(
                index=0, id='call-1', function=dict(name='read', arguments='{"file_'))]), token_ids=[30])]))
            self.event(dict(choices=[dict(index=0, delta=dict(tool_calls=[dict(
                index=0, function=dict(arguments='path": "/a"}'))]), token_ids=[31], finish_reason='tool_calls')]))
            self.event(dict(choices=[], usage=dict(prompt_tokens=3, completion_tokens=Handler.chunks + 3)))
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
        except OSError:
            pass
        self.close_connection = True

    def event(self, chunk):
        self.wfile.write(('data: %s\n\n' % json.dumps(chunk)).encode())
        self.wfile.flush()


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.daemon = True
        cls.thread.start()
        cls.client = replay.Client(cls.server.server_address[1])

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        Handler.seen[:] = []
        Handler.first_delay = Handler.chunk_delay = 0.0
        Handler.chunks = 3

    def body(self):
        return dict(messages=[dict(role='user', content='hi')], tools=[dict(type='function', function=dict(name='read'))],
                    tool_choice='auto')

    def test_a_streamed_greedy_request_and_everything_it_returns(self):
        result = self.client.chat(self.body(), 'pfx-a-0001-hit', salt='tenant', max_tokens=64)
        path, headers, sent = Handler.seen[-1]
        self.assertEqual(path, '/v1/chat/completions')
        self.assertEqual(headers.get('X-Request-Id'), 'pfx-a-0001-hit')
        self.assertEqual((sent['temperature'], sent['top_p'], sent['stream'], sent['return_token_ids']), (0.0, 1.0, True, True))
        self.assertEqual((sent['cache_salt'], sent['max_tokens'], sent['tool_choice']), ('tenant', 64, 'auto'))
        self.assertEqual(sent['stream_options'], dict(include_usage=True))
        self.assertTrue(result['ok'])
        self.assertEqual((result['prompt_tokens'], result['token_ids'], result['finish']), (3, [10, 20, 21, 22, 30, 31], 'tool_calls'))
        self.assertEqual((result['content'], result['reasoning']), ('w0 w1 w2 ', 'think '))
        self.assertEqual(result['tool_calls'], [dict(id='call-1', type='function', function=dict(
            name='read', arguments='{"file_path": "/a"}'))])
        self.assertEqual(result['prompt_sha'], judge.token_sha([1, 2, 3]))
        self.assertIsNotNone(result['ttft_s'])
        self.client.chat(self.body(), 'pfx-a-0002-hit', max_tokens=8, extra=dict(ignore_eos=True))
        self.assertNotIn('cache_salt', Handler.seen[-1][2], 'no salt: fail-closed tenancy')
        self.assertTrue(Handler.seen[-1][2]['ignore_eos'])

    def test_an_abort_before_the_first_token_closes_the_socket(self):
        """On Linux (the rig host, CI) shutdown() wakes the reader blocked in getresponse(), so the
        server sees the disconnect at once. Windows wakes it only when the server answers, so there
        only the verdict is held, not the timing."""
        Handler.first_delay = 1.5
        started = time.time()
        result = self.client.chat(self.body(), 'pfx-a-0003-hit', abort_after_s=0.3)
        if sys.platform.startswith('linux'):
            self.assertLess(time.time() - started, 1.4)
        self.assertEqual((result['ok'], result['error'], result['ttft_s']), (False, None, None))
        self.assertIn('closed after 0.3 s', result['aborted'])

    def test_an_abort_on_signal_and_after_tokens(self):
        Handler.first_delay = 1.5
        signal = threading.Event()
        threading.Timer(0.2, signal.set).start()
        result = self.client.chat(self.body(), 'pfx-a-0004-hit', abort_signal=signal)
        self.assertEqual(result['aborted'], 'closed on signal')
        Handler.first_delay, Handler.chunk_delay, Handler.chunks = 0.0, 0.05, 10
        result = self.client.chat(self.body(), 'pfx-a-0005-hit', abort_after_tokens=3)
        self.assertIn('after 3 tokens', result['aborted'])
        self.assertFalse(result['ok'])

    def test_an_http_error_is_an_error_not_an_abort(self):
        result = self.client.chat(self.body(), 'pfx-a-0006-hit', max_tokens=999)
        self.assertIn('HTTP 400', result['error'])
        self.assertFalse(result['ok'])

    def test_tokenize_metrics_health_and_reset(self):
        self.assertEqual(self.client.tokenize(self.body()), 10)
        path, _, sent = Handler.seen[-1]
        self.assertEqual((path, sent['add_generation_prompt'], len(sent['tools'])), ('/tokenize', True, 1))
        self.assertEqual(self.client.metrics(), {'vllm:num_requests_waiting': 2.0})
        self.assertTrue(self.client.healthy() and self.client.ready())
        self.assertEqual(self.client.reset_prefix_cache(), 200)

    def test_a_dead_server_is_not_ready(self):
        dead = replay.Client(1)
        self.assertFalse(dead.ready())
        self.assertEqual(dead.metrics(), {})
        result = dead.chat(self.body(), 'x')
        self.assertFalse(result['ok'])
        self.assertIsNotNone(result['error'])


class EngineHandler(http.server.BaseHTTPRequestHandler):
    """vLLM's OpenAI routes the driver uses, over a FakeEngine: the request id is the X-Request-Id
    header, the stream carries prompt_token_ids first and token_ids per chunk (return_token_ids)."""
    protocol_version = 'HTTP/1.1'
    engine = None

    def log_message(self, *args):
        pass

    def reply(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('content-length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply({})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('content-length') or 0)) or b'null')
        if self.path == '/tokenize':
            self.reply(dict(count=self.engine.tokenize(body)))
            return
        assert body['return_token_ids'] and body['temperature'] == 0.0 and body['stream']
        result = self.engine.chat(dict(messages=body['messages'], tools=body.get('tools')), self.headers['X-Request-Id'],
                                  salt=body.get('cache_salt'), max_tokens=body['max_tokens'],
                                  extra=dict(ignore_eos=body.get('ignore_eos')))
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        self.end_headers()
        ids = result['token_ids']
        half = len((result['reasoning'] or '').split())
        chunks = [dict(prompt_token_ids=result['prompt_ids'], choices=[dict(index=0, delta=dict(role='assistant'))]),
                  dict(choices=[dict(index=0, delta=dict(reasoning=result['reasoning']), token_ids=ids[:half])]),
                  dict(choices=[dict(index=0, delta=dict(content=result['content']), token_ids=ids[half:],
                                     finish_reason=None if result['tool_calls'] else result['finish'])])]
        if result['tool_calls']:
            call = result['tool_calls'][0]
            chunks.append(dict(choices=[dict(index=0, delta=dict(tool_calls=[dict(index=0, id=call['id'], function=dict(
                name=call['function']['name'], arguments=call['function']['arguments']))]), finish_reason='tool_calls')]))
        chunks.append(dict(choices=[], usage=dict(prompt_tokens=len(result['prompt_ids']), completion_tokens=len(ids))))
        for chunk in chunks:
            self.wfile.write(('data: %s\n\n' % json.dumps(chunk)).encode())
        self.wfile.write(b'data: [DONE]\n\n')
        self.wfile.flush()
        self.close_connection = True


class EndToEndTests(unittest.TestCase):
    def test_a_scenario_over_http_matches_every_marker_by_its_request_id(self):
        engine = FakeEngine()
        EngineHandler.engine = engine
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), EngineHandler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        try:
            client = replay.Client(server.server_address[1])
            driver = replay.Driver(client, 'e2e', CORPUS, log=FakeLog(engine), container=FakeContainer(engine), seed=2,
                                   sleep=lambda seconds: None, say=lambda text: None, pods=lambda: None)
            replay.scenario_bringup_prefix(driver)
            replay.boundary_cases(driver, prompts=(2048,))
        finally:
            server.shutdown()
            server.server_close()
        records = resolved(driver, engine)
        self.assertTrue(all(r['ok'] for r in records))
        self.assertTrue(all(r['markers']['matched'] == 'tag' and r['markers']['rows'] for r in records))
        self.assertEqual([text for r in records for _, text in judge.reuse_problems(r)], [])
        self.assertTrue(all(p['verdict'] == 'IDENTICAL' for p in driver.pairs), driver.pairs)
        self.assertTrue(any(r['markers']['q'] for r in records if r['role'] == 'hit'))
        self.assertTrue(any(r['tool_calls'] for r in records), 'a tool call came back and was sent back')
        self.assertEqual(driver.events['boundary-2048']['tokens'], 2048)


class StreamStateTests(unittest.TestCase):
    def test_an_error_event_ends_the_stream(self):
        state = replay.StreamState()
        self.assertFalse(state.feed(b'data: {"choices": []}'))
        self.assertTrue(state.feed('data: {"error": {"message": "boom"}}'))
        self.assertIn('boom', state.result()['error'])

    def test_comments_and_bad_json_are_skipped(self):
        state = replay.StreamState()
        self.assertFalse(state.feed(': keep-alive'))
        self.assertFalse(state.feed('data: {not json'))
        self.assertTrue(state.feed('data: [DONE]'))
        self.assertIsNone(state.result()['token_ids'])


class PlumbingTests(unittest.TestCase):
    def test_the_log_follower_reads_docker_logs(self):
        class Process(object):
            stdout = [b'2026-09-26T10:00:00.1Z first\n', b'2026-09-26T10:00:01.2Z second\n']

            def terminate(self):
                pass

        calls = []

        def popen(arguments, **kwargs):
            calls.append(arguments)
            return Process()

        directory = tempfile.mkdtemp()
        try:
            follower = replay.LogFollower('c', os.path.join(directory, 'server.log'), popen=popen)
            follower.start()
            follower.thread.join(5)
            self.assertEqual(follower.lines(), ['2026-09-26T10:00:00.1Z first', '2026-09-26T10:00:01.2Z second'])
            self.assertEqual((follower.mark(), follower.last_time()), (2, '2026-09-26T10:00:01.2Z'))
            follower.stop()
            follower.start(since=follower.last_time())
            follower.thread.join(5)
            self.assertEqual(calls[1], ['docker', 'logs', '-f', '--timestamps', '--since', '2026-09-26T10:00:01.2Z', 'c'])
            with open(os.path.join(directory, 'server.log'), encoding='utf-8') as handle:
                self.assertEqual(len(handle.read().splitlines()), 4)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_container_operations(self):
        class Result(object):
            def __init__(self, out, code=0):
                self.stdout, self.returncode = out, code

        outputs = {'stats': Result(b'17.5GiB / 80GiB\n'), 'inspect': Result(b'true\n')}
        seen = []

        def run(arguments, **kwargs):
            seen.append(arguments)
            return outputs['stats' if 'stats' in arguments else 'inspect']

        container = replay.Container('c', run=run)
        self.assertAlmostEqual(container.rss_gb(), 17.5 * 1024 ** 3 / 1e9)
        self.assertTrue(container.running())
        outputs['stats'] = Result(b'512MiB / 80GiB')
        self.assertAlmostEqual(container.rss_gb(), 512 * 1024 ** 2 / 1e9)
        outputs['stats'] = Result(b'', 1)
        self.assertIsNone(container.rss_gb())

    def test_the_kill_switch_file_is_removed_only_when_the_gate_wrote_it(self):
        container = FakeContainer(FakeEngine())
        replay.kill_switch_on(container)
        replay.kill_switch_off(container)
        on, off = container.scripts
        self.assertIn('echo %s > %s' % (replay.KILL_SWITCH_OWNER, replay.KILL_SWITCH_PATH), on)
        self.assertIn('if [ "$(cat %s 2>/dev/null)" = %s ]; then rm -f %s; fi' % (
            replay.KILL_SWITCH_PATH, replay.KILL_SWITCH_OWNER, replay.KILL_SWITCH_PATH), off)
        self.assertEqual(replay.KILL_SWITCH_PATH, graft.KILL_SWITCH_PATH, 'the path the scheduler graft polls')

    def test_ci_pods_counts_running_pods_or_says_nothing(self):
        class Result(object):
            returncode, stdout = 0, b'a 1/1 Running 0 1m\nb 0/1 Pending 0 1m\nc 1/1 Running 0 2m\n'

        self.assertEqual(replay.ci_pods(run=lambda *a, **k: Result()), 2)

        def boom(*a, **k):
            raise OSError('no sudo')

        self.assertIsNone(replay.ci_pods(run=boom))

    def test_records_leave_out_the_prompt_ids(self):
        text = replay.records_jsonl([dict(tag='a', prompt_ids=[1, 2], token_ids=[3])])
        self.assertEqual(json.loads(text), dict(tag='a', token_ids=[3]))
        self.assertEqual(replay.records_jsonl([]), '')


class DriverTests(unittest.TestCase):
    def test_a_pair_is_cold_then_hit_on_the_same_messages(self):
        engine = FakeEngine()
        driver = driver_for(engine)
        conv = driver.conversation('c', first_tokens=900)
        hit = driver.pair(conv, 'case')
        cold = driver.records[0]
        self.assertEqual((cold['role'], hit['role']), ('cold', 'hit'))
        self.assertNotEqual(cold['salt'], hit['salt'])
        self.assertTrue(cold['salt'].startswith('pfx-cold-arm-1-'))
        self.assertEqual(hit['salt'], 'pfx-salt-arm-1-c')
        self.assertEqual(cold['prompt_sha'], hit['prompt_sha'])
        self.assertEqual(driver.pairs[0]['verdict'], 'IDENTICAL')
        self.assertTrue(hit['tag'].startswith('pfx-arm-') and hit['tag'].endswith('-hit'))
        self.assertEqual(hit['expected']['q'], 0)
        self.assertNotIn('prompt_ids', hit)

    def test_a_divergent_hit_is_rerun_and_fails_when_it_reproduces(self):
        engine = FakeEngine(diverge_hits=True)
        driver = driver_for(engine)
        conv = driver.conversation('c', first_tokens=8000)
        hit = driver.pair(conv, 'case')
        driver.answer(conv, hit)
        conv.extend(600)
        driver.pair(conv, 'case')
        second = driver.pairs[-1]
        self.assertEqual(second['verdict'], 'DIVERGED')
        self.assertIsNotNone(second['rerun'])
        self.assertEqual([r['case'] for r in driver.records[-2:]], ['case:rerun', 'case:rerun'])

    def test_a_dead_engine_stops_the_arm(self):
        engine = FakeEngine()
        driver = driver_for(engine)
        engine.healthy = lambda: False
        original = engine.chat
        engine.chat = lambda *a, **k: dict(original(*a, **k), ok=False, error='connection reset')
        with self.assertRaises(replay.EngineDead):
            driver.send(dict(messages=[dict(role='user', content='x')]), 'hit', 's', 'c')

    def test_the_arms_time_runs_out(self):
        engine = FakeEngine()
        driver = driver_for(engine)
        driver.deadline = driver.clock() - 1
        with self.assertRaises(replay.OutOfTime):
            driver.send(dict(messages=[dict(role='user', content='x')]), 'hit', 's', 'c')


def resolved(driver, engine):
    judge.resolve(driver.records, pm.scan(engine.lines))
    return driver.records


class ScenarioTests(unittest.TestCase):
    def test_exactness_traced_runs_every_case_and_every_pair_matches(self):
        engine = FakeEngine()
        driver = driver_for(engine, 'exactness-traced')
        replay.scenario_exactness(driver, 'traced')
        cases = set(pair['case'] for pair in driver.pairs)
        self.assertEqual(cases, {'chain', 'changed-suffix', 'early-divergence', 'boundary-2047', 'boundary-2048',
                                 'boundary-2049', 'shared-system'})
        self.assertTrue(all(pair['verdict'] == 'IDENTICAL' for pair in driver.pairs))
        self.assertEqual(len([p for p in driver.pairs if p['case'] == 'chain']), 1 + len(replay.CHAIN_HITS))
        for name in ('boundary-2047', 'boundary-2048', 'boundary-2049'):
            self.assertEqual(driver.events[name]['tokens'], int(name[-4:]))
        records = resolved(driver, engine)
        problems = [text for r in records for _, text in judge.reuse_problems(r)]
        self.assertEqual(problems, [])
        hit = dict((p['hit'], p) for p in driver.pairs)
        by_tag = dict((r['tag'], r) for r in records)
        second = [by_tag[p['hit']] for p in driver.pairs if p['case'] == 'boundary-2048'][1]
        self.assertEqual(second['markers']['q'], 2048)
        self.assertTrue(hit)

    def test_eager_and_audit_run_the_short_chain(self):
        for variant in ('eager', 'audit'):
            engine = FakeEngine(path='eager' if variant == 'eager' else 'traced', audit=variant == 'audit')
            driver = driver_for(engine, 'exactness-' + variant)
            replay.scenario_exactness(driver, variant)
            chain = [p for p in driver.pairs if p['case'] == 'chain']
            self.assertEqual(len(chain), 1 + len(replay.CHAIN_HITS_SHORT))
            self.assertNotIn('shared-system', set(p['case'] for p in driver.pairs))

    def test_bringup_runs_unsalted_then_salted(self):
        engine = FakeEngine()
        driver = driver_for(engine, 'bringup-prefix')
        replay.scenario_bringup_prefix(driver)
        roles = [r['role'] for r in driver.records]
        self.assertEqual(roles[:3], ['unsalted'] * 3)
        self.assertEqual(roles[3:], ['cold', 'hit'] * 3)
        reference = driver_for(FakeEngine(prefix=False, profile='general'), 'bringup-reference')
        replay.scenario_bringup_reference(reference)
        self.assertEqual([r['prompt_sha'] for r in reference.records], [r['prompt_sha'] for r in driver.records[:3]],
                         'the grants-disabled run and the reference send the same prompts')

    def test_lifecycle_evict_drives_every_event(self):
        engine = FakeEngine()
        driver = driver_for(engine, 'lifecycle-evict', strict=False)
        restarts = []

        def restart():
            driver.container.stop()
            driver.container.start()
            restarts.append(1)
            return 1.0

        replay.scenario_lifecycle_evict(driver, pool_tokens=262144, restart=restart)
        events = driver.events
        self.assertTrue(events['arrivals']['ok'])
        self.assertEqual(len(events['arrivals']['tags']), 4)
        self.assertEqual((events['abort-waiting']['phase'], events['abort-waiting']['aborted']), ('waiting', 'closed on signal'))
        self.assertFalse(events['abort-prefill']['first_token_before_abort'])
        self.assertEqual(events['flood']['floods'], 3)
        self.assertEqual(events['reset-prefix-cache']['status'], 200)
        self.assertEqual((events['kill-switch']['written'], events['kill-switch']['removed']), (True, True))
        self.assertEqual(restarts, [1])
        records = resolved(driver, engine)
        by_case = dict()
        for record in records:
            if record['role'] == 'hit':
                by_case.setdefault(record['case'], []).append(record['markers']['q'])
        self.assertEqual(by_case['kill-on'], [0])
        self.assertEqual(by_case['kill-latched'], [0])
        self.assertEqual(by_case['after-reset'], [0])
        self.assertEqual(by_case['after-reload'][0], 0)
        self.assertGreater(by_case['after-reload-2'][0], 0)
        self.assertEqual(sum(1 for p in driver.pairs if p['verdict'] != 'IDENTICAL'), 0)

    def test_store_and_tiny(self):
        engine = FakeEngine(store=3)
        driver = driver_for(engine, 'lifecycle-store', strict=False)
        replay.scenario_lifecycle_store(driver)
        self.assertEqual(driver.pairs[-1]['case'], 'store-after')
        engine = FakeEngine(preempt_on_ignore_eos=1)
        driver = driver_for(engine, 'lifecycle-tiny', strict=False)
        replay.scenario_lifecycle_tiny(driver, pool_tokens=81920)
        self.assertEqual(driver.events['tiny']['preemptions'], 4.0)
        self.assertEqual(len([p for p in driver.pairs if p['case'] == 'tiny']), 4)
        tiny = [r for r in driver.records if r['case'] == 'tiny']
        self.assertTrue(all(r['completion_tokens'] == replay.TINY_MAX_TOKENS for r in tiny), 'ignore_eos answers')

    def test_timing_phases(self):
        engine = FakeEngine()
        driver = driver_for(engine, 'timing-prefix', strict=False)
        replay.scenario_timing(driver, agents=(1, 2), turns=3, max_tokens=16, gap_mean_s=0.0)
        self.assertEqual(sorted(driver.phases), ['agents-1', 'agents-2'])
        self.assertEqual(len([r for r in driver.records if r['case'] == 'agents-2']), 6)
        self.assertEqual(driver.phases['agents-1']['ci_pods'], [3, 3])
        salts = set(r['salt'] for r in driver.records if r['case'] == 'agents-2')
        self.assertEqual(len(salts), 2, 'one tenant salt per agent')
        continued = [r for r in driver.records if r['continuation']]
        self.assertTrue(continued)


if __name__ == '__main__':
    unittest.main()
