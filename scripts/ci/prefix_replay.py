"""The prefix-reuse hardware harness's driver: chained multi-turn replay against a served OpenAI API.

It runs on the rig host against one serving container (c2_prefix_gate.py starts it in the node
agent's shape) and does what the design's G1 gates ask (2.2, gate table rows Bring-up, Exactness
traced/eager, Lifecycle, Timing):

  - every conversation is real text (prefix_agent_corpus) and CHAINED: turn N+1 is turn N's
    messages, the model's own answer to it (content, reasoning, tool calls) and the next input;
  - every turn is sent twice: COLD, with a fresh cache_salt (a miss, other physical pages), then
    HIT, with the conversation's salt; the two are compared in full (prefix_judge.pair_verdict:
    every output token id). A first divergence sends a second cold run and a second hit at the SAME
    Q: every request the conversation's salt served before the hit is replayed, max_tokens=1, under
    a fresh salt (the same publishes and captures, so the same grants), then the hit's messages
    under it. The chain continues from the first hit;
  - hits that run concurrently (the arrivals, the tiny pool) are compared with cold twins run alone;
    a divergence there runs a second solo cold run and a batch control (the same messages at once,
    fresh salts), so a batch-dependent engine reads NOT_COMPARABLE, not DIVERGED;
  - every request carries X-Request-Id = its tag, so the engine's markers name it (prefix_markers);
    return_token_ids gives the prompt and output token ids, from which prefix_judge.Oracle derives
    the Q each request should get (a request aborted before its first output returns none: its
    prompt ids come from /tokenize);
  - greedy decoding is asked for explicitly (temperature 0): the general profiles have no request
    contract to coerce it;
  - vllm:prefix_cache_hits and _queries are read around the requests whose publishing is the claim
    (the unsalted bring-up turns, the turn after the kill switch latched): vLLM's raw hit there must
    not exceed what the oracle says was published (record['expected_raw_h']);
  - lifecycle drivers: aborts (a socket closed while a request waits for a seat, or during a hit's
    prefill), a KV flood that evicts cached conversations, a small checkpoint store, a tiny pool
    (allocation failure after a grant, preemption), reset_prefix_cache (vLLM dev mode), the runtime
    kill switch file, an in-place engine restart; the timing driver runs busy agents in the metering
    shape and records TTFT and turn time per turn;
  - no request waits past the arm's deadline: each one's socket timeout is the time left.

Scenarios are functions of a Driver; everything the network, docker and the clock do goes through
objects the CPU tests replace (test_prefix_replay).

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""

import hashlib
import http.client
import json
import os
import random
import re
import socket
import subprocess
import threading
import time
from collections import Counter

import prefix_agent_corpus as corpus_module
import prefix_judge as judge
import prefix_markers as markers

SERVED_NAME = 'Qwen/Qwen3.8-27B'
STREAM_TIMEOUT_S = 1800
MIN_REQUEST_TIMEOUT_S = 10
DEFAULT_MAX_TOKENS = 1024
PRIME_MAX_TOKENS = 1
KILL_SWITCH_PATH = '/models/.qwen-c2/prefix-reuse.off'
KILL_SWITCH_OWNER = 'qwen-c2-prefix-gate'
KILL_SWITCH_POLL_S = 1.0
KILL_ON_INPUT_TOKENS = 2500      # past one chunk: what the kill-on turn would publish is observable
COUNTERS = ('vllm:prefix_cache_hits', 'vllm:prefix_cache_queries', 'vllm:num_preemptions')
COUNTER_SETTLE_S = 1.0
# Exactness: the chain's first turn, then one hit near each length the gate names (4k, 16k, 42k,
# 60k) and the steps between them; the eager and audit arms run the short end.
CHAIN_FIRST = 2600
CHAIN_HITS = (4200, 9000, 16500, 24500, 33000, 42500, 51000, 60000)
CHAIN_HITS_SHORT = (4200, 9000, 16500)
VARIANT_AFTER_TURN = 3           # the changed-suffix and early-divergence variants fork after this turn
BOUNDARY_PROMPTS = (2047, 2048, 2049)
BOUNDARY_FIRST_MAX_TOKENS = 256  # so turn 2 stays inside the next chunk: a tail-only hit
BOUNDARY_FOLLOWUP_TOKENS = 120
SHARED_CONVERSATIONS = 3
# Each shared-system conversation's own task text: past one chunk, so the first conversation's own
# capture lands beyond the ~4.2k-token shared block and the second has to capture the gap.
SHARED_ATTACHMENT_TOKENS = 2600
# Lifecycle. Conversations start past two chunks, so every continuation can hit.
LIFE_FIRST_TOKENS = 4000
EVICT_CONVERSATIONS = 4
EVICT_LENGTHS = (28000, 56000)   # a 56k conversation plus its next turn and answer stay under 65,536
SEAT_HOLD_MAX_TOKENS = 1500
ABORT_PREFILL_AFTER_S = 2.0
ABORT_PREFILL_INPUT_TOKENS = 20000
WAITING_POLL_S = 0.5
WAITING_TIMEOUT_S = 120
STORE_CONVERSATIONS = 4
# The tiny pool: 1280 blocks of 64. The TT worker overwrites num_gpu_blocks_override with its own
# count (plugin worker.py:388-390), so the arm sets the image model's QWEN36_MAX_TOKENS_ALL_USERS
# (qwen36_vllm.py:96-98) instead, and runs only when vLLM logs exactly this pool.
TINY_POOL_TOKENS = 81920
TINY_SHARE = 0.30                # each preemption conversation's first turn, as a share of the pool
TINY_PREEMPT_CONVERSATIONS = 3   # 3 x ~30%: all cached together, all admitted together
TINY_PREEMPT_MARGIN_BLOCKS = 40  # each answer's decode outgrows the pool left by this many blocks
CONTEXT_TOKENS = 65536           # the general profiles' max-model-len
# The dropped grant (F2), by block arithmetic on the 1280-block pool (tiny_dropped_grant): X's first
# turn (~20%) is cached; the filler takes every block but X's, TINY_GRANT_MARGIN_BLOCKS and its
# answer's; X's next turn needs TINY_GRANT_EXCESS_BLOCKS more than the filler can ever leave free.
TINY_GRANT_FIRST_SHARE = 0.20
TINY_GRANT_MARGIN_BLOCKS = 48
TINY_GRANT_EXCESS_BLOCKS = 32
TINY_FILLER_MAX_TOKENS = 1024
TINY_GRANT_MAX_TOKENS = 256
# Client-side clocks: X's first token may be stamped a little before the filler's stream end; a turn
# admitted beside the filler instead would get its first token a whole filler answer earlier.
TINY_WAIT_SLACK_S = 0.25
FLOOD_MAX_TOKENS = 32
# Timing (metering shape).
TIMING_AGENTS = (1, 4, 5, 6)
TIMING_TURNS = 8
TIMING_MAX_TOKENS = 1024
TIMING_FIRST_RANGE = (8000, 40000)
STARTUP_STAGGER_S = 30.0
SAMPLE_EVERY_S = 15.0


class EngineDead(RuntimeError):
    """The server stopped answering: the arm cannot continue."""


class OutOfTime(RuntimeError):
    """The arm's time ran out before its scenario finished."""


# -- the streamed chat completion ----------------------------------------------------------------

class StreamState(object):
    """A streamed chat completion, fed line by line (server-sent events): text, reasoning, tool
    calls, output and prompt token ids, usage, finish reason, and when the first token came."""

    def __init__(self, clock=time.time):
        self.clock = clock
        self.started = clock()
        self.first_token_at = None
        self.content, self.reasoning, self.token_ids = [], [], []
        self.prompt_token_ids = None
        self.calls = {}
        self.usage = None
        self.finish = None
        self.error = None
        self.done = False

    def feed(self, raw):
        line = raw.decode('utf-8', 'replace') if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith('data:'):
            return self.done
        body = line[5:].strip()
        if body == '[DONE]':
            self.done = True
            return True
        try:
            chunk = json.loads(body)
        except ValueError:
            return False
        if chunk.get('error') is not None:
            self.error = str(chunk['error'])[:400]
            self.done = True
            return True
        if chunk.get('prompt_token_ids') is not None and self.prompt_token_ids is None:
            self.prompt_token_ids = list(chunk['prompt_token_ids'])
        if chunk.get('usage'):
            self.usage = chunk['usage']
        for choice in chunk.get('choices') or ():
            delta = choice.get('delta') or {}
            ids = choice.get('token_ids') or []
            text = delta.get('content') or ''
            thought = delta.get('reasoning') or delta.get('reasoning_content') or ''
            if (ids or text or thought or delta.get('tool_calls')) and self.first_token_at is None:
                self.first_token_at = self.clock()
            self.token_ids.extend(ids)
            if text:
                self.content.append(text)
            if thought:
                self.reasoning.append(thought)
            for call in delta.get('tool_calls') or ():
                entry = self.calls.setdefault(call.get('index', len(self.calls)), dict(id=None, name='', arguments=[]))
                entry['id'] = call.get('id') or entry['id']
                function = call.get('function') or {}
                if function.get('name'):
                    entry['name'] += function['name']
                if function.get('arguments'):
                    entry['arguments'].append(function['arguments'])
            if choice.get('finish_reason'):
                self.finish = choice['finish_reason']
        return False

    def generated(self):
        return len(self.token_ids)

    def result(self):
        ended = self.clock()
        calls = [dict(id=entry['id'], type='function', function=dict(name=entry['name'],
                                                                      arguments=''.join(entry['arguments'])))
                 for _, entry in sorted(self.calls.items())]
        prompt_ids = self.prompt_token_ids
        usage = self.usage or {}
        return dict(content=''.join(self.content), reasoning=''.join(self.reasoning) or None, tool_calls=calls,
                    token_ids=list(self.token_ids) if self.token_ids or self.finish else None,
                    prompt_ids=prompt_ids, prompt_sha=judge.token_sha(prompt_ids) if prompt_ids is not None else None,
                    prompt_tokens=len(prompt_ids) if prompt_ids is not None else usage.get('prompt_tokens'),
                    completion_tokens=usage.get('completion_tokens', len(self.token_ids)), finish=self.finish,
                    error=self.error,
                    ttft_s=round(self.first_token_at - self.started, 3) if self.first_token_at else None,
                    wall_s=round(ended - self.started, 3))


class Client(object):
    """The served OpenAI API on 127.0.0.1:<port>, over http.client so a request's socket can be
    closed from another thread at any moment (an abort while it waits or prefills)."""

    def __init__(self, port, host='127.0.0.1', model=SERVED_NAME, clock=time.time):
        self.host, self.port, self.model, self.clock = host, int(port), model, clock

    def request(self, method, path, body=None, timeout=60):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            payload = json.dumps(body).encode('utf-8') if body is not None else None
            connection.request(method, path, body=payload, headers={'content-type': 'application/json'})
            response = connection.getresponse()
            return response.status, response.read().decode('utf-8', 'replace')
        except (OSError, http.client.HTTPException) as error:
            return None, repr(error)[:300]
        finally:
            connection.close()

    def json_request(self, method, path, body=None, timeout=60):
        status, text = self.request(method, path, body, timeout)
        try:
            return status, json.loads(text)
        except (TypeError, ValueError):
            return status, text

    def ready(self):
        status, _ = self.request('GET', '/v1/models', timeout=10)
        return status == 200

    def healthy(self):
        status, _ = self.request('GET', '/health', timeout=30)
        return status == 200

    def metrics(self):
        status, text = self.request('GET', '/metrics', timeout=30)
        return markers.parse_prometheus(text) if status == 200 else {}

    def reset_prefix_cache(self):
        status, _ = self.request('POST', '/reset_prefix_cache', timeout=120)
        return status

    def _tokenize(self, body):
        request = dict(model=self.model, messages=body['messages'], add_generation_prompt=True)
        if body.get('tools'):
            request['tools'] = body['tools']
        status, answer = self.json_request('POST', '/tokenize', request, timeout=120)
        if status != 200 or not isinstance(answer, dict):
            raise RuntimeError('/tokenize answered %s: %s' % (status, str(answer)[:300]))
        return answer

    def tokenize(self, body):
        """The prompt token count the chat endpoint will render for `body` (messages, tools)."""
        answer = self._tokenize(body)
        return int(answer.get('count', len(answer.get('tokens') or ())))

    def tokenize_ids(self, body):
        """The prompt token ids the chat endpoint will render for `body`."""
        tokens = self._tokenize(body).get('tokens')
        if not isinstance(tokens, list):
            raise RuntimeError('/tokenize returned no token list')
        return [int(token) for token in tokens]

    def chat(self, body, tag, salt=None, max_tokens=DEFAULT_MAX_TOKENS, timeout=STREAM_TIMEOUT_S,
             abort_after_s=None, abort_after_tokens=None, extra=None, on_first=None, abort_signal=None):
        """One streamed, greedy chat completion. -> the StreamState result plus status, ok, aborted.
        The socket is closed (the server sees a disconnect and aborts the request) after
        abort_after_s seconds, after abort_after_tokens output tokens, or when abort_signal (a
        threading.Event) is set - whichever comes first."""
        request = dict(model=self.model, messages=body['messages'], max_tokens=int(max_tokens), temperature=0.0,
                       top_p=1.0, stream=True, stream_options=dict(include_usage=True), return_token_ids=True)
        if body.get('tools'):
            request['tools'] = body['tools']
            request['tool_choice'] = body.get('tool_choice', 'auto')
        if salt:
            request['cache_salt'] = salt
        request.update(extra or {})
        state = StreamState(self.clock)
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        aborted = dict(reason=None, sock=None)
        timer = None

        def close(reason):
            if aborted['reason'] is None:
                aborted['reason'] = reason
            # The socket as it was when the request went out: getresponse() hands it to a response
            # that will close the connection and sets connection.sock to None.
            sock = aborted['sock'] or connection.sock
            if sock is None:
                return
            # shutdown wakes a recv blocked in another thread on Linux; close does it on Windows.
            for action in (lambda: sock.shutdown(socket.SHUT_RDWR), sock.close):
                try:
                    action()
                except OSError:
                    pass

        status = None
        finished = threading.Event()

        def watch():
            while not finished.is_set():
                if abort_signal.wait(0.2):
                    close('closed on signal')
                    return

        try:
            connection.request('POST', '/v1/chat/completions', body=json.dumps(request).encode('utf-8'),
                               headers={'content-type': 'application/json', 'X-Request-Id': tag})
            aborted['sock'] = connection.sock
            if abort_after_s is not None:
                timer = threading.Timer(abort_after_s, close, args=('closed after %.1f s' % abort_after_s,))
                timer.daemon = True
                timer.start()
            if abort_signal is not None:
                watcher = threading.Thread(target=watch)
                watcher.daemon = True
                watcher.start()
            response = connection.getresponse()
            status = response.status
            if status != 200:
                state.error = 'HTTP %s: %s' % (status, response.read().decode('utf-8', 'replace')[:400])
            else:
                while True:
                    raw = response.readline()
                    if not raw or aborted['reason'] is not None:
                        break
                    had_first = state.first_token_at is not None
                    if state.feed(raw):
                        break
                    if on_first is not None and not had_first and state.first_token_at is not None:
                        on_first()
                    if abort_after_tokens is not None and state.generated() >= abort_after_tokens:
                        close('closed after %d tokens' % state.generated())
                        break
        except (OSError, http.client.HTTPException, ValueError) as error:
            if aborted['reason'] is None:
                state.error = repr(error)[:300]
        finally:
            finished.set()
            if timer is not None:
                timer.cancel()
            connection.close()
        result = state.result()
        result.update(status=status, aborted=aborted['reason'])
        result['ok'] = (result['error'] is None and aborted['reason'] is None
                        and result['finish'] in ('stop', 'length', 'tool_calls'))
        return result


# -- the server log and the container ------------------------------------------------------------

def stamp_key(stamp):
    """A docker RFC3339Nano timestamp as a sortable key (docker trims the fraction's trailing zeros,
    so the text alone does not sort)."""
    match = re.match(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?Z$', stamp or '')
    if not match:
        return None
    return match.group(1), (match.group(2) or '').ljust(9, '0')[:9]


class LogFollower(object):
    """`docker logs -f --timestamps` of the serving container, into a list and a file. A restart
    with `since` re-reads the log from that time; lines the list already holds are not added twice
    (a duplicated [PREFIX] row would read as a second admission)."""

    def __init__(self, container, path, popen=subprocess.Popen):
        self.container, self.path, self.popen = container, path, popen
        self.lock = threading.Lock()
        self.buffer = []
        self.replayed = Counter()
        self.process = None
        self.thread = None

    def start(self, since=None):
        arguments = ['docker', 'logs', '-f', '--timestamps']
        if since:
            arguments += ['--since', since]
            floor = stamp_key(since)
            with self.lock:
                self.replayed = Counter(line for line in self.buffer
                                        if floor is not None and (stamp_key(markers.split_timestamp(line)[0]) or ('',))
                                        >= floor)
        self.process = self.popen(arguments + [self.container], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.thread = threading.Thread(target=self._read, args=(self.process,))
        self.thread.daemon = True
        self.thread.start()

    def _read(self, process):
        with open(self.path, 'a', encoding='utf-8') as handle:
            for raw in process.stdout:
                line = raw.decode('utf-8', 'replace') if isinstance(raw, bytes) else raw
                text = line.rstrip('\n')
                with self.lock:
                    if self.replayed.get(text):
                        self.replayed[text] -= 1
                        continue
                    self.buffer.append(text)
                handle.write(line if line.endswith('\n') else line + '\n')
                handle.flush()

    def mark(self):
        with self.lock:
            return len(self.buffer)

    def lines(self):
        with self.lock:
            return list(self.buffer)

    def last_time(self):
        with self.lock:
            for line in reversed(self.buffer):
                stamp, _ = markers.split_timestamp(line)
                if stamp:
                    return stamp
        return None

    def stop(self, wait=5.0):
        if self.process is not None:
            try:
                self.process.terminate()
            except OSError:
                pass
        if self.thread is not None:
            self.thread.join(wait)


class Container(object):
    """The docker operations a scenario needs on its serving container."""

    def __init__(self, name, run=subprocess.run):
        self.name, self.run = name, run

    def _run(self, arguments, timeout=120):
        try:
            result = self.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
            return result.returncode, (result.stdout or b'').decode('utf-8', 'replace')
        except (OSError, subprocess.SubprocessError) as error:
            return None, repr(error)[:300]

    def running(self):
        code, out = self._run(['docker', 'inspect', '-f', '{{.State.Running}}', self.name], timeout=60)
        return code == 0 and out.strip() == 'true'

    def exec_shell(self, script, timeout=60):
        return self._run(['docker', 'exec', self.name, 'sh', '-c', script], timeout=timeout)

    def read_file(self, path):
        code, out = self._run(['docker', 'exec', self.name, 'cat', path], timeout=60)
        return out if code == 0 else None

    def stop(self, grace=60):
        return self._run(['docker', 'stop', '-t', str(grace), self.name], timeout=grace + 120)

    def start(self):
        return self._run(['docker', 'start', self.name], timeout=300)

    def rss_gb(self):
        code, out = self._run(['docker', 'stats', '--no-stream', '--format', '{{.MemUsage}}', self.name], timeout=60)
        if code != 0 or not out.strip():
            return None
        used = out.strip().split('/')[0].strip()
        units = dict(B=1e-9, KiB=1024 / 1e9, MiB=1024 ** 2 / 1e9, GiB=1024 ** 3 / 1e9, kB=1e-6, MB=1e-3, GB=1.0)
        for unit in sorted(units, key=len, reverse=True):
            if used.endswith(unit):
                try:
                    return float(used[:-len(unit)]) * units[unit]
                except ValueError:
                    return None
        return None


def ci_pods(run=subprocess.run):
    """Busy ARC runner pods on the host (the priority step's count), or None."""
    try:
        result = run(['sudo', '-n', 'k3s', 'kubectl', '-n', 'arc-runners', 'get', 'pods', '--no-headers'],
                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return sum(1 for line in (result.stdout or b'').decode('utf-8', 'replace').splitlines() if 'Running' in line)


# -- the driver ----------------------------------------------------------------------------------

class Driver(object):
    """Sends tagged requests, keeps every record, feeds the oracle, compares pairs."""

    def __init__(self, client, arm, corpus, log=None, container=None, seed=0, max_tokens=DEFAULT_MAX_TOKENS,
                 deadline=None, clock=time.time, sleep=time.sleep, say=print, pods=ci_pods, strict=True,
                 store_capacity=None):
        self.client, self.arm, self.corpus, self.log, self.container = client, arm, corpus, log, container
        self.seed, self.max_tokens, self.deadline = seed, max_tokens, deadline
        self.clock, self.sleep, self.say, self.pods = clock, sleep, say, pods
        self.strict = strict
        self.oracle = judge.Oracle(store_capacity)
        self.records, self.pairs, self.events, self.phases = [], [], {}, {}
        self.history = {}
        self.lock = threading.Lock()
        self.counter = 0
        self.started = clock()

    # -- plumbing -------------------------------------------------------------------------------
    def tag(self, label):
        with self.lock:
            self.counter += 1
            return 'pfx-%s-%04d-%s' % (self.arm, self.counter, label)

    def salt(self, name):
        return 'pfx-salt-%s-%s-%s' % (self.arm, self.seed, name)

    def fresh_salt(self, kind='cold'):
        with self.lock:
            self.counter += 1
            return 'pfx-%s-%s-%s-%04d' % (kind, self.arm, self.seed, self.counter)

    def remaining(self):
        """Seconds before the arm's deadline, or None without one."""
        return None if self.deadline is None else self.deadline - self.clock()

    def check_time(self):
        left = self.remaining()
        if left is not None and left <= 0:
            raise OutOfTime('the arm\'s time ran out')

    def request_timeout(self):
        left = self.remaining()
        if left is None:
            return STREAM_TIMEOUT_S
        return max(MIN_REQUEST_TIMEOUT_S, min(STREAM_TIMEOUT_S, int(left)))

    def conversation(self, name, system='compact', first_tokens=0):
        conv = corpus_module.Conversation(self.corpus, name, self.seed, system=system, first_tokens=first_tokens)
        conv.salt = self.salt(name)
        return conv

    def counters(self):
        values = self.client.metrics() or {}
        return dict((name, values.get(name)) for name in COUNTERS)

    def send(self, body, role, salt, case, conv=None, max_tokens=None, admit=True, continuation=False,
             abort_after_s=None, abort_after_tokens=None, extra=None, on_first=None, abort_signal=None,
             counters=False):
        """One request. `admit` feeds the oracle (False for a request that never reached a seat).
        `counters` reads vLLM's prefix-cache counters around it (sequential requests only) and
        records the raw hit the oracle expects (what the salt has published, capped at L - 1)."""
        self.check_time()
        tag = self.tag(role)
        start = self.log.mark() if self.log else None
        before = self.counters() if counters else None
        sent = self.clock() - self.started
        result = self.client.chat(body, tag, salt=salt, max_tokens=max_tokens or self.max_tokens,
                                  timeout=self.request_timeout(), abort_after_s=abort_after_s,
                                  abort_after_tokens=abort_after_tokens, extra=extra, on_first=on_first,
                                  abort_signal=abort_signal)
        prompt_ids = result.pop('prompt_ids', None)
        record = dict(result, tag=tag, arm=self.arm, role=role, salt=salt, case=case, sent_s=round(sent, 3),
                      conv=getattr(conv, 'name', None), turn=getattr(conv, 'turn', None), continuation=continuation,
                      log_window=None)
        if counters:
            self.sleep(COUNTER_SETTLE_S)
            record['counters'] = dict(before=before, after=self.counters())
        if self.log:
            record['log_window'] = [start, self.log.mark() + 64]
        if admit and prompt_ids is None and record.get('aborted'):
            # The server sends prompt_token_ids with the first output only; a request aborted before
            # it was still admitted (published, maybe captured) when the abort landed in its prefill.
            try:
                prompt_ids = self.client.tokenize_ids(body)
                record['prompt_ids_from'] = 'tokenize'
            except Exception as error:     # noqa: BLE001 - recorded; the oracle then misses it
                record['prompt_ids_from'] = 'none (%s)' % str(error)[:120]
        if admit and prompt_ids is not None:
            with self.lock:
                owner = salt if role != 'unsalted' else None
                if counters:
                    capped = (max(0, len(prompt_ids) - 1) // judge.BLOCK) * judge.BLOCK
                    record['expected_raw_h'] = min(self.oracle.published_tokens(owner, prompt_ids), capped)
                record['expected'] = self.oracle.admit(owner, prompt_ids)
        if admit and salt and role not in judge.FRESH_ROLES:
            with self.lock:
                self.history.setdefault(salt, []).append((tag, body))
        with self.lock:
            self.records.append(record)
        status = 'ok' if record['ok'] else 'FAILED %s' % (record.get('error') or record.get('aborted'))
        self.say('[PREFIX-GATE] %s %s %s conv=%s turn=%s L=%s out=%s ttft=%s wall=%s %s' % (
            self.arm, tag, case, record['conv'], record['turn'], record.get('prompt_tokens'),
            record.get('completion_tokens'), record.get('ttft_s'), record.get('wall_s'), status))
        if not record['ok'] and not record.get('aborted'):
            self.check_alive()
        return record

    def check_alive(self):
        if self.container is not None and not self.container.running():
            raise EngineDead('the serving container is not running')
        if not self.client.healthy():
            raise EngineDead('the server\'s /health does not answer 200')

    def history_before(self, salt, tag):
        """The bodies this salt served before the request `tag`, in order."""
        with self.lock:
            entries = list(self.history.get(salt) or ())
        out = []
        for entry_tag, body in entries:
            if entry_tag == tag:
                break
            out.append(body)
        return out

    def pair(self, conv, case, max_tokens=None, rerun=True, extra=None, continuation=None, counters=False):
        """Cold (a fresh salt) then hit (the conversation's salt) on the same messages, compared in
        full. A divergence sends a second cold run and a second hit at the same Q: the salt's earlier
        requests replayed (max_tokens=1) under a fresh salt, then these messages. Returns the first
        hit record (the chain continues from it)."""
        body = conv.body()
        continuation = conv.turn > 0 if continuation is None else continuation
        cold = self.send(body, 'cold', self.fresh_salt(), case, conv, max_tokens, continuation=continuation,
                         extra=extra)
        hit = self.send(body, 'hit', conv.salt, case, conv, max_tokens, continuation=continuation, extra=extra,
                        counters=counters)
        result = judge.pair_verdict(cold, hit)
        entry = dict(case=case, conv=conv.name, turn=conv.turn, cold=cold['tag'], hit=hit['tag'], kind='sequential',
                     rerun=None, cold2=None, primes=[])
        if result['verdict'] == 'RERUN' and rerun:
            self.say('[PREFIX-GATE] %s: %s diverged from %s (%s) - a second cold run, then the same-Q re-run' % (
                self.arm, hit['tag'], cold['tag'], result['first'].get('detail')))
            cold2 = self.send(body, 'cold', self.fresh_salt(), case + ':rerun', conv, max_tokens,
                              continuation=continuation, extra=extra)
            replay_salt = self.fresh_salt('rerun')
            primes = [self.send(earlier, 'prime', replay_salt, case + ':prime', conv, PRIME_MAX_TOKENS,
                                continuation=True, extra=extra)
                      for earlier in self.history_before(conv.salt, hit['tag'])]
            hit2 = self.send(body, 'hit', replay_salt, case + ':rerun', conv, max_tokens, continuation=continuation,
                             extra=extra)
            result = judge.pair_verdict(cold, hit, cold2, hit2)
            entry.update(rerun=(cold2['tag'], hit2['tag']), cold2=cold2['tag'], primes=[p['tag'] for p in primes])
        entry.update(verdict=result['verdict'], detail=result.get('reason') or result['first'].get('detail'),
                     prompt_tokens=hit.get('prompt_tokens'))
        with self.lock:
            self.pairs.append(entry)
        self.say('[PREFIX-GATE] %s pair %s turn %s L=%s: %s%s' % (self.arm, case, conv.turn, hit.get('prompt_tokens'),
                                                                   entry['verdict'], ' (%s)' % entry['detail']
                                                                   if entry['detail'] else ''))
        return hit

    def answer(self, conv, record):
        conv.add_answer(record)

    def event(self, name, **values):
        with self.lock:
            self.events[name] = values
        self.say('[PREFIX-GATE] %s event %s %s' % (self.arm, name, json.dumps(values, sort_keys=True)[:600]))


# -- scenarios -----------------------------------------------------------------------------------

def run_chain(driver, conv, targets, case, max_tokens=None, after_turn=None):
    """A chained conversation: the first turn, then one turn per remaining target, every turn a
    cold/hit pair; the next input is sized from the served counts to reach the next target.
    after_turn(conv, turn) runs after each answer is appended (the variants fork there)."""
    hit = driver.pair(conv, case, max_tokens)
    driver.answer(conv, hit)
    if after_turn:
        after_turn(conv, 1)
    for index, target in enumerate(targets[1:]):
        if not hit.get('ok'):
            break
        conv.extend(corpus_module.growth_input(target, hit['prompt_tokens'], hit.get('completion_tokens') or 0))
        hit = driver.pair(conv, case, max_tokens)
        driver.answer(conv, hit)
        if after_turn:
            after_turn(conv, index + 2)
    return hit


def fitted_conversation(driver, name, target, system='compact'):
    """A conversation whose first prompt is exactly `target` tokens (the server's /tokenize)."""
    conv = driver.conversation(name, system=system)
    rng = random.Random('%s|%s|fit' % (driver.seed, name))
    task = driver.corpus.task(rng, 0)
    # Three times the estimate: the search only needs the pad to be long enough to overshoot.
    pad = driver.corpus.excerpt(rng, 3 * target * corpus_module.CHARS_PER_TOKEN)

    def build(pad_chars, filler):
        conv.messages[1] = {'role': 'user', 'content': task + '\n\n' + pad[:pad_chars].rstrip() + filler}
        return conv.body()

    body, tokens = corpus_module.fit_tokens(build, driver.client.tokenize, target, len(pad))
    conv.messages = body['messages']
    return conv, tokens


def boundary_cases(driver, prompts=BOUNDARY_PROMPTS):
    """Previous-prompt lengths 2047, 2048 and 2049, and the tail-only hit their second turns are.
    The event records what /tokenize fitted and what the chat endpoint then served: the gate
    judges only a case served at its target length."""
    for target in prompts:
        name = 'boundary-%d' % target
        try:
            conv, tokens = fitted_conversation(driver, name, target)
        except (corpus_module.FitError, RuntimeError) as error:
            driver.event(name, fitted=False, error=str(error)[:300])
            continue
        hit = driver.pair(conv, name, BOUNDARY_FIRST_MAX_TOKENS, continuation=False)
        driver.event(name, fitted=True, tokens=tokens, served=hit.get('prompt_tokens'))
        driver.answer(conv, hit)
        conv.extend(BOUNDARY_FOLLOWUP_TOKENS)
        driver.pair(conv, name, BOUNDARY_FIRST_MAX_TOKENS)


def shared_system(driver, count=SHARED_CONVERSATIONS):
    """Conversations of one tenant (one salt) sharing the full system block and tools, each with its
    own task of ~2.6k tokens: the first publishes and captures past the shared block, the second
    misses and captures the gap boundary floor2048(h) inside it, the third hits exactly that
    (design 2.0.1 item 2a.5)."""
    salt = driver.salt('shared')
    for index in range(count):
        conv = driver.conversation('shared-%d' % index, system='full', first_tokens=SHARED_ATTACHMENT_TOKENS)
        conv.salt = salt
        driver.pair(conv, 'shared-system', continuation=False)


def bringup_turns(driver, capture, turns=3):
    """The bring-up conversation's three unsalted turns, each followed (capture=True) by the same
    messages under a fresh salt: a first request that captures at floor2048(L) and publishes, which
    must equal the baseline too. The unsalted turns read vLLM's prefix-cache counters around them."""
    conv = driver.conversation('bringup', first_tokens=corpus_module.first_attachment('compact', CHAIN_FIRST))
    targets = corpus_module.chain_targets(CHAIN_FIRST, CHAIN_HITS_SHORT[:turns - 1])
    role = 'unsalted' if capture else 'reference'
    record = None
    for index, target in enumerate(targets):
        if index:
            if not record.get('ok'):
                break
            conv.extend(corpus_module.growth_input(target, record['prompt_tokens'], record.get('completion_tokens') or 0))
        body = conv.body()
        record = driver.send(body, role, None, 'bringup', conv, continuation=index > 0, counters=capture)
        if capture:
            driver.send(body, 'capture', driver.fresh_salt(), 'bringup-capture', conv, continuation=index > 0)
        driver.answer(conv, record)
    return targets


def scenario_bringup_reference(driver, turns=3):
    """On the baseline profile (general): the bring-up conversation, unsalted, as the reference the
    prefix profile's unsalted and capturing turns must equal byte for byte."""
    bringup_turns(driver, capture=False, turns=turns)


def scenario_bringup_prefix(driver, turns=3):
    """On the prefix profile: the bring-up turns unsalted (fail-closed tenancy: no grant, no publish -
    reuse off inside a reuse engine) and under fresh salts (the capture path), then a salted chain as
    cold/hit pairs: the first hit, whose program cache must not grow (F3)."""
    targets = bringup_turns(driver, capture=True, turns=turns)
    salted = driver.conversation('bringup-salted', first_tokens=corpus_module.first_attachment('compact', CHAIN_FIRST))
    run_chain(driver, salted, targets, 'bringup-salted')


def scenario_exactness(driver, variant='traced'):
    """Traced: the long chain (4k..60k) with its changed-suffix and early-divergence variants, the
    boundary cases and the shared system block. Eager and audit: the short chain and the boundaries."""
    full = variant == 'traced'
    targets = corpus_module.chain_targets(CHAIN_FIRST, CHAIN_HITS if full else CHAIN_HITS_SHORT)
    conv = driver.conversation('chain', first_tokens=corpus_module.first_attachment('compact', targets[0]))

    def variants(state, turn):
        if not full or turn != VARIANT_AFTER_TURN:
            return
        target = targets[turn] if turn < len(targets) else targets[-1]
        last = driver.records[-1]
        budget = corpus_module.growth_input(target, last.get('prompt_tokens') or 0, last.get('completion_tokens') or 0)
        suffix = state.fork('suffix')
        suffix.extend(budget)
        driver.pair(suffix, 'changed-suffix')
        # The second input after the first answer: the prompt then shares turn 2 and falls back to its
        # checkpoint (or turn 1's), below the changed suffix's turn-3 boundary.
        inputs = [i for i, m in enumerate(state.messages) if i > 1 and m['role'] in ('tool', 'user')
                  and state.messages[i - 1]['role'] == 'assistant']
        early_index = inputs[1] if len(inputs) > 1 else None
        if early_index is not None:
            early = state.diverge_at(early_index)
            early.extend(budget)
            driver.pair(early, 'early-divergence')
        else:
            driver.event('early-divergence', exercised=False, reason='no tool or user message after the first turn')

    run_chain(driver, conv, targets, 'chain', after_turn=variants)
    boundary_cases(driver)
    if full:
        shared_system(driver)


def _burst(driver, jobs):
    """Run callables at once (threads); the first exception is re-raised after all have ended."""
    return _join(_spawn(jobs))


def concurrent_pairs(driver, pending, case):
    """Hits that ran concurrently, each against a cold twin (a fresh salt) run alone. When any
    diverged: a second solo cold run of each divergent one, and a batch control - every message set
    again at once under fresh salts, the burst's shape - so concurrent_verdict can tell a hit that
    differs from a batch that does. pending: [(hit record, body, conv, max_tokens, extra)]."""
    colds = []
    for record, body, conv, max_tokens, extra in pending:
        colds.append(driver.send(body, 'cold', driver.fresh_salt(), case, conv, max_tokens, extra=extra,
                                 continuation=record.get('continuation')))
    first = [judge.compare(cold, record) for cold, (record, _, _, _, _) in zip(colds, pending)]
    diverged = [index for index, result in enumerate(first) if result['verdict'] == 'DIVERGED']
    seconds, batch = {}, [None] * len(pending)
    if diverged:
        driver.say('[PREFIX-GATE] %s: %d concurrent hits diverged from their solo cold runs - a second cold run '
                   'and a batch control' % (driver.arm, len(diverged)))
        for index in diverged:
            record, body, conv, max_tokens, extra = pending[index]
            seconds[index] = driver.send(body, 'cold', driver.fresh_salt(), case + ':rerun', conv, max_tokens,
                                         extra=extra, continuation=record.get('continuation'))
        batch = _burst(driver, [lambda body=body, conv=conv, max_tokens=max_tokens, extra=extra, record=record:
                                driver.send(body, 'cold-batch', driver.fresh_salt(), case + ':batch', conv, max_tokens,
                                            extra=extra, continuation=record.get('continuation'))
                                for record, body, conv, max_tokens, extra in pending])
    for index, ((record, body, conv, max_tokens, extra), cold) in enumerate(zip(pending, colds)):
        second = seconds.get(index)
        result = judge.concurrent_verdict(cold, record, second, batch[index] if second is not None else None)
        entry = dict(case=case, conv=getattr(conv, 'name', None), turn=getattr(conv, 'turn', None), cold=cold['tag'],
                     hit=record['tag'], kind='concurrent', rerun=None, primes=[],
                     cold2=second['tag'] if second is not None else None,
                     batch=batch[index]['tag'] if second is not None and batch[index] is not None else None,
                     verdict=result['verdict'], detail=result.get('reason') or result['first'].get('detail'),
                     prompt_tokens=record.get('prompt_tokens'))
        with driver.lock:
            driver.pairs.append(entry)


def wait_waiting(driver, minimum=1, timeout=WAITING_TIMEOUT_S, until=None):
    """vLLM's waiting gauge once it reaches `minimum`, or None after `timeout` seconds or as soon as
    until() is true (the request it waits for already ended)."""
    deadline = driver.clock() + timeout
    while driver.clock() < deadline:
        waiting = driver.client.metrics().get('vllm:num_requests_waiting')
        if waiting is not None and waiting >= minimum:
            return waiting
        if until is not None and until():
            return None
        driver.sleep(WAITING_POLL_S)
    return None


def sized_message(driver, conv, index, target):
    """Give message `index` of `conv` a real excerpt so that the prompt is about `target` tokens,
    measured with the server's /tokenize (an estimate, then one correction). -> the prompt's tokens."""
    base = conv.messages[index]['content']
    empty = driver.client.tokenize(conv.body())
    chars = token_chars(max(1, target - empty))
    got = empty
    for attempt in range(2):
        conv.messages[index]['content'] = base + '\n\n' + driver.corpus.excerpt(conv.rng('size'), chars)
        got = driver.client.tokenize(conv.body())
        if got <= empty:
            break
        chars = max(corpus_module.MIN_INPUT_CHARS, int(chars * float(target - empty) / (got - empty)))
    return got


def read_stats(driver):
    """The registry's stats export now: the container's STATS_FILE, else the log's last stats line."""
    text = driver.container.read_file(markers.STATS_FILE) if driver.container is not None else None
    if text:
        try:
            return json.loads(text)
        except ValueError:
            pass
    return markers.scan(driver.log.lines()).get('stats') if driver.log is not None else None


def lifecycle_arrivals(driver, convs):
    """Four arrivals in one scheduler step: two hit continuations and two new conversations of one
    tenant sharing a fresh system block (the same-step rule), then their cold twins."""
    first = []
    for conv in convs[:2]:
        hit = driver.send(conv.body(), 'hit', conv.salt, 'arrivals-setup', conv)
        driver.answer(conv, hit)
        conv.extend(600)
        first.append(conv)
    tenant = driver.salt('same-step')
    fresh = []
    for index in range(2):
        conv = driver.conversation('same-step-%d' % index, system='full')
        conv.salt = tenant
        fresh.append(conv)
    everyone = first + fresh
    bodies = [conv.body() for conv in everyone]
    records = _burst(driver, [lambda conv=conv, body=body: driver.send(
        body, 'hit', conv.salt, 'arrivals', conv, continuation=conv.turn > 0) for conv, body in zip(everyone, bodies)])
    driver.event('arrivals', sent=len(records), tags=[record['tag'] for record in records],
                 ok=all(record.get('ok') for record in records))
    concurrent_pairs(driver, [(record, body, conv, None, None) for record, body, conv in zip(records, bodies, everyone)],
                     'arrivals')
    for conv, record in zip(first, records[:2]):
        driver.answer(conv, record)


def lifecycle_abort_waiting(driver, conv, seats=4):
    """`seats` long requests (ignore_eos) hold every seat; a hit turn of `conv` then queues, and is
    aborted once vLLM's waiting gauge shows it waiting; after the seats drain the same turn is sent
    again (a fresh grant) with its cold twin. The aborted request never reached a seat, so it
    must leave no grant and no [PREFIX] row (the judge checks its tag)."""
    holders = [driver.conversation('seat-%d' % index) for index in range(seats)]
    busy = threading.Event()
    firsts = []
    lock = threading.Lock()

    def first_token():
        with lock:
            firsts.append(1)
            if len(firsts) >= seats:
                busy.set()

    held = _spawn([lambda holder=holder: driver.send(
        holder.body(), 'seat', driver.fresh_salt(), 'seat', holder, SEAT_HOLD_MAX_TOKENS,
        extra=dict(ignore_eos=True), on_first=first_token) for holder in holders])
    left = driver.remaining()
    busy.wait(900 if left is None else max(1, min(900, left)))
    body = conv.body()
    signal = threading.Event()
    waiter = _spawn([lambda: driver.send(body, 'hit', conv.salt, 'abort-waiting', conv, admit=False,
                                         continuation=True, abort_signal=signal)])
    waiting = wait_waiting(driver, until=lambda: not waiter['threads'][0].is_alive())
    signal.set()
    record = _join(waiter)[0]
    driver.event('abort-waiting', seats_busy=busy.is_set(), waiting_gauge=waiting,
                 phase='waiting' if (busy.is_set() and waiting) else 'unknown', tag=record['tag'] if record else None,
                 aborted=(record or {}).get('aborted'), first_token=(record or {}).get('ttft_s') is not None)
    _join(held)
    hit = driver.pair(conv, 'abort-waiting-retry')
    driver.answer(conv, hit)
    return record


def _spawn(jobs):
    """Start callables on threads; _join collects their results (re-raising the first error)."""
    box = dict(results=[None] * len(jobs), errors=[], threads=[])

    def one(index, job):
        try:
            box['results'][index] = job()
        except Exception as error:     # noqa: BLE001 - re-raised by _join
            box['errors'].append(error)

    for index, job in enumerate(jobs):
        thread = threading.Thread(target=one, args=(index, job))
        thread.daemon = True
        thread.start()
        box['threads'].append(thread)
    return box


def _join(box, timeout=None):
    for thread in box['threads']:
        thread.join(timeout)
    if box['errors']:
        raise box['errors'][0]
    return box['results']


def lifecycle_abort_prefill(driver, conv):
    """A hit whose resumed prefill is long (~20k new tokens) is aborted ~2 s after it is sent, before
    its first token; the same turn is then sent again with its cold twin."""
    conv.extend(ABORT_PREFILL_INPUT_TOKENS)
    body = conv.body()
    record = driver.send(body, 'hit', conv.salt, 'abort-prefill', conv, abort_after_s=ABORT_PREFILL_AFTER_S,
                         continuation=True)
    driver.event('abort-prefill', tag=record['tag'], first_token_before_abort=record.get('ttft_s') is not None,
                 aborted=record.get('aborted'))
    hit = driver.pair(conv, 'abort-prefill-retry')
    driver.answer(conv, hit)


def lifecycle_flood(driver, pool_tokens):
    """Forced KV eviction under 4 x 60k: four conversations built to ~60k, then fresh-salt floods of
    ~60k each sized to push about one and a half of them out of the pool, then every conversation's
    next turn (some must miss) against its cold twin."""
    convs = [driver.conversation('evict-%d' % index) for index in range(EVICT_CONVERSATIONS)]
    for conv in convs:
        record = driver.send(conv.body(), 'hit', conv.salt, 'evict-build', conv)
        driver.answer(conv, record)
        for target in EVICT_LENGTHS:
            conv.extend(corpus_module.growth_input(target, record['prompt_tokens'], record.get('completion_tokens') or 0))
            record = driver.send(conv.body(), 'hit', conv.salt, 'evict-build', conv, continuation=True)
            driver.answer(conv, record)
    length = EVICT_LENGTHS[-1]
    floods = max(1, -(-(int(pool_tokens or 0) - int(2.5 * length)) // length)) if pool_tokens else 2
    for index in range(floods):
        flood = driver.conversation('flood-%d' % index)
        flood.extend(length)
        driver.send(flood.body(), 'flood', driver.fresh_salt(), 'flood', flood, FLOOD_MAX_TOKENS)
    driver.event('flood', pool_tokens=pool_tokens, floods=floods, flood_tokens=floods * length)
    for conv in convs:
        conv.extend(1500)
        hit = driver.pair(conv, 'after-flood')
        driver.answer(conv, hit)
    return convs


def lifecycle_reset(driver, conv):
    status = driver.client.reset_prefix_cache()
    driver.event('reset-prefix-cache', status=status)
    if status == 200:
        with driver.lock:
            driver.oracle.reset_prefix_cache()
    conv.extend(800)
    hit = driver.pair(conv, 'after-reset')
    driver.answer(conv, hit)


def kill_switch_on(container, path=KILL_SWITCH_PATH, owner=KILL_SWITCH_OWNER):
    return container.exec_shell('mkdir -p "$(dirname %s)" && echo %s > %s' % (path, owner, path))


def kill_switch_off(container, path=KILL_SWITCH_PATH, owner=KILL_SWITCH_OWNER):
    """Remove the flag only if this harness wrote it (its content is the owner string)."""
    return container.exec_shell('if [ "$(cat %s 2>/dev/null)" = %s ]; then rm -f %s; fi; test ! -e %s' % (
        path, owner, path, path))


def lifecycle_kill_switch(driver):
    """The runtime kill switch: with the flag present a continuation gets no grant and publishes
    nothing; after the flag is removed reuse stays off until the engine restarts (it latches). The
    kill-on turn crosses a chunk boundary, so the kill-latched turn's raw vLLM hit (the counters
    around it) shows whether it published: it must not exceed what was published before the kill."""
    conv = driver.conversation('kill-switch', first_tokens=LIFE_FIRST_TOKENS)
    before = driver.pair(conv, 'kill-before')
    driver.answer(conv, before)
    code, out = kill_switch_on(driver.container)
    driver.sleep(KILL_SWITCH_POLL_S * 2 + 0.5)
    with driver.lock:
        driver.oracle.kill()
    try:
        conv.extend(KILL_ON_INPUT_TOKENS)
        on = driver.pair(conv, 'kill-on')
        driver.answer(conv, on)
    finally:
        removed = kill_switch_off(driver.container)
    driver.sleep(KILL_SWITCH_POLL_S * 2 + 0.5)
    conv.extend(800)
    latched = driver.pair(conv, 'kill-latched', counters=True)
    driver.answer(conv, latched)
    driver.event('kill-switch', written=code == 0, write_output=(out or '')[:200], removed=removed[0] == 0,
                 before_prompt=before.get('prompt_tokens'), on_prompt=on.get('prompt_tokens'),
                 latched_tag=latched.get('tag'))


def lifecycle_reload(driver, restart):
    """An in-place engine restart with a warm kernel cache, after the kill switch latched reuse off:
    a conversation's turn before it, then the first turn after it misses (the pool and the registry
    died with the engine; the latch too), and the next one hits again; all against cold twins.
    restart() returns the seconds to ready, or {seconds, log_window}: the log lines of the stop and
    the new engine's boot, whose failure lines the gate records instead of failing on."""
    conv = driver.conversation('reload', first_tokens=LIFE_FIRST_TOKENS)
    hit = driver.pair(conv, 'before-reload')
    driver.answer(conv, hit)
    info = restart()
    seconds, window = (info.get('seconds'), info.get('log_window')) if isinstance(info, dict) else (info, None)
    driver.oracle = judge.Oracle(driver.oracle.capacity)
    driver.event('reload', seconds=seconds, log_window=window)
    conv.extend(800)
    hit = driver.pair(conv, 'after-reload')
    driver.answer(conv, hit)
    conv.extend(800)
    hit = driver.pair(conv, 'after-reload-2')
    driver.answer(conv, hit)


def scenario_lifecycle_evict(driver, pool_tokens=None, restart=None):
    """Mixed arrivals, aborts, the KV flood, reset_prefix_cache, the kill switch and the reload."""
    convs = [driver.conversation('life-%d' % index, first_tokens=LIFE_FIRST_TOKENS) for index in range(2)]
    for conv in convs:
        hit = driver.pair(conv, 'life-first')
        driver.answer(conv, hit)
        conv.extend(1200)
    lifecycle_arrivals(driver, convs)
    convs[0].extend(1200)
    lifecycle_abort_waiting(driver, convs[0])
    lifecycle_abort_prefill(driver, convs[1])
    evicted = lifecycle_flood(driver, pool_tokens)
    lifecycle_reset(driver, evicted[-1])
    lifecycle_kill_switch(driver)
    # The restart starts a new registry: what the counters saw so far is read now.
    driver.event('stats-before-restart', stats=read_stats(driver))
    if restart is not None:
        lifecycle_reload(driver, restart)


def scenario_lifecycle_store(driver):
    """A checkpoint store of a few entries: a conversation's checkpoints are pushed out of the LRU
    by others', and its next turn falls back (to an older checkpoint, or a miss) - exact either way."""
    main = driver.conversation('store-main')
    hit = driver.pair(main, 'store')
    driver.answer(main, hit)
    for target in (6000, 9000):
        main.extend(corpus_module.growth_input(target, hit['prompt_tokens'], hit.get('completion_tokens') or 0))
        hit = driver.pair(main, 'store')
        driver.answer(main, hit)
    for index in range(STORE_CONVERSATIONS):
        other = driver.conversation('store-other-%d' % index)
        other.extend(2500)
        driver.pair(other, 'store-other')
    main.extend(1000)
    driver.pair(main, 'store-after')


def tiny_dropped_grant(driver, pool):
    """An allocation failure after a grant (F2), built rather than hoped for, in blocks of the pool
    (every size measured with /tokenize): conversation X's first turn (~20%) is cached and captured,
    its blocks at the free queue's recent end; a fresh-salt filler then takes all but X's blocks, a
    margin and its answer's, and decodes (ignore_eos); X's next turn needs more blocks than the pool
    has left beside it, and fits once the filler ends. TTScheduler prefers the prefill: the graft
    stages a grant with Q > 0 on X's intact cached prefix, the allocation fails and the step falls
    back to decode (plugin scheduler.py, default mode) - so every step while the filler runs drops
    that grant uncommitted. X's first token comes after the filler's last; then it restores Q > 0 on
    a fresh grant and is compared with a cold twin."""
    blocks = pool // judge.BLOCK
    conv = driver.conversation('tiny-grant')
    sized_message(driver, conv, 1, int(pool * TINY_GRANT_FIRST_SHARE))
    first = driver.send(conv.body(), 'hit', conv.salt, 'tiny-grant-build', conv, 64)
    driver.answer(conv, first)
    if not first.get('ok'):
        driver.event('tiny-grant', ok=False, reason='the first turn failed: %s' % first.get('error'))
        return
    held_by_x = -(-(first['prompt_tokens'] + (first.get('completion_tokens') or 0)) // judge.BLOCK)
    answer_blocks = -(-TINY_FILLER_MAX_TOKENS // judge.BLOCK)
    filler_blocks = blocks - held_by_x - TINY_GRANT_MARGIN_BLOCKS - answer_blocks
    filler_target = min(filler_blocks * judge.BLOCK - judge.BLOCK, CONTEXT_TOKENS - TINY_FILLER_MAX_TOKENS - judge.BLOCK)
    x2_target = (held_by_x + TINY_GRANT_MARGIN_BLOCKS + answer_blocks + TINY_GRANT_EXCESS_BLOCKS) * judge.BLOCK
    filler = driver.conversation('tiny-filler')
    filler_tokens = sized_message(driver, filler, 1, filler_target)
    conv.extend(corpus_module.MIN_INPUT_CHARS // 3)
    x2_tokens = sized_message(driver, conv, len(conv.messages) - 1, x2_target)
    running = threading.Event()
    held = _spawn([lambda: driver.send(filler.body(), 'seat', driver.fresh_salt(), 'tiny-filler', filler,
                                       TINY_FILLER_MAX_TOKENS, extra=dict(ignore_eos=True), on_first=running.set)])
    left = driver.remaining()
    running.wait(900 if left is None else max(1, min(900, left)))
    body = conv.body()
    waiter = _spawn([lambda: driver.send(body, 'hit', conv.salt, 'tiny-grant', conv, TINY_GRANT_MAX_TOKENS,
                                         continuation=True)])
    waiting = wait_waiting(driver, until=lambda: not waiter['threads'][0].is_alive()) if running.is_set() else None
    record = _join(waiter)[0]
    filler_record = _join(held)[0]
    first_token = (record['sent_s'] + record['ttft_s']) if record.get('ttft_s') is not None else None
    filler_end = filler_record['sent_s'] + (filler_record.get('wall_s') or 0)
    driver.event('tiny-grant', filler_running=running.is_set(), waiting_gauge=waiting, tag=record['tag'],
                 filler=filler_record['tag'], blocks=blocks, held_by_x=held_by_x, filler_tokens=filler_tokens,
                 x2_tokens=x2_tokens, first_token_s=first_token, filler_end_s=round(filler_end, 3),
                 waited_for_filler=bool(first_token is not None and first_token >= filler_end - TINY_WAIT_SLACK_S),
                 ok=bool(record.get('ok') and filler_record.get('ok')))
    concurrent_pairs(driver, [(record, body, conv, TINY_GRANT_MAX_TOKENS, None)], 'tiny-grant')


def scenario_lifecycle_tiny(driver, pool_tokens=None, expected_pool=TINY_POOL_TOKENS):
    """A tiny pool: the dropped grant (tiny_dropped_grant), then preemption (tiny_preemption). Nothing
    is sent unless vLLM logged exactly the pool the arm asked for: on any other pool the shares mean
    nothing (a pool the TT worker sized from max_model_len x max_num_seqs makes every first turn
    longer than the context)."""
    if pool_tokens != expected_pool:
        driver.event('tiny', skipped=True, pool_tokens=pool_tokens, expected_pool=expected_pool,
                     reason='vLLM logged a %s-token KV pool, not the %d the arm set: the derived profile did not size '
                            'the pool' % (pool_tokens, expected_pool))
        return
    pool = int(expected_pool)
    tiny_dropped_grant(driver, pool)
    tiny_preemption(driver, pool)


def tiny_preemption(driver, pool):
    """Preemption, by block arithmetic too: TINY_PREEMPT_CONVERSATIONS conversations (~30% of the
    pool each, all cached at once) send their next turns together; each prompt's exact size comes
    from /tokenize, and the ignore_eos answer budget is sized so that every one is admitted and their
    decode then needs TINY_PREEMPT_MARGIN_BLOCKS more blocks each than the pool has left - vLLM must
    preempt a running request, which resumes later (a second [PREFIX] row)."""
    convs = [driver.conversation('tiny-%d' % index) for index in range(TINY_PREEMPT_CONVERSATIONS)]
    first_target = int(pool * TINY_SHARE)
    for conv in convs:
        conv.messages[1]['content'] += '\n\n' + driver.corpus.excerpt(conv.rng('tiny'), token_chars(first_target))
        record = driver.send(conv.body(), 'hit', conv.salt, 'tiny-build', conv, 64)
        driver.answer(conv, record)
        conv.extend(1000)
    bodies = [conv.body() for conv in convs]
    lengths = [driver.client.tokenize(body) for body in bodies]
    free = pool // judge.BLOCK - sum(-(-length // judge.BLOCK) for length in lengths)
    room = CONTEXT_TOKENS - max(lengths) - judge.BLOCK
    max_tokens = min(room, (max(0, free) // len(bodies) + TINY_PREEMPT_MARGIN_BLOCKS) * judge.BLOCK)
    if free <= 0 or max_tokens * len(bodies) <= free * judge.BLOCK:
        driver.event('tiny', skipped=True, lengths=lengths, free_blocks=free, max_tokens=max_tokens,
                     reason='the next turns (%s tokens) leave %d free blocks of %d: no answer budget both admits them '
                            'all and runs the pool out' % (lengths, free, pool // judge.BLOCK))
        return
    before = driver.client.metrics().get('vllm:num_preemptions', 0.0)
    extra = dict(ignore_eos=True)
    records = _burst(driver, [lambda conv=conv, body=body: driver.send(
        body, 'hit', conv.salt, 'tiny', conv, max_tokens, continuation=True, extra=extra)
        for conv, body in zip(convs, bodies)])
    after = driver.client.metrics().get('vllm:num_preemptions', 0.0)
    driver.event('tiny', pool_tokens=pool, lengths=lengths, free_blocks=free, max_tokens=max_tokens,
                 preemptions=after - before, ok=all(r.get('ok') for r in records), tags=[r['tag'] for r in records])
    concurrent_pairs(driver, [(record, body, conv, max_tokens, extra)
                              for record, body, conv in zip(records, bodies, convs)], 'tiny')


def token_chars(tokens):
    return corpus_module.token_chars(tokens)


# -- timing --------------------------------------------------------------------------------------

def agent_loop(driver, phase, index, turns, max_tokens, gap_mean_s, stop):
    """One busy agent: conversations in the metering shape, compacted past the prompt limit."""
    rng = random.Random('%s|%s|agent|%d' % (driver.seed, phase, index))
    salt = driver.salt('%s-agent-%d' % (phase, index))
    shape = corpus_module.METERING
    driver.sleep(rng.uniform(0, STARTUP_STAGGER_S))
    conversation_index, done = 0, 0
    conv = None
    while done < turns and not stop.is_set():
        if conv is None:
            conv = driver.conversation('%s-a%d-c%d' % (phase, index, conversation_index), system='full')
            conv.salt = salt
            first = rng.randint(*TIMING_FIRST_RANGE) if conversation_index == 0 else shape['compaction_tokens']
            conv.messages[1]['content'] += '\n\n' + driver.corpus.excerpt(conv.rng('first'), token_chars(first))
            conversation_index += 1
        record = driver.send(conv.body(), 'agent', salt, phase, conv, max_tokens, continuation=conv.turn > 0)
        done += 1
        if not record.get('ok'):
            conv = None
            continue
        driver.answer(conv, record)
        size = corpus_module.metering_input(rng, shape)
        if record['prompt_tokens'] + (record.get('completion_tokens') or 0) + size + max_tokens > shape['max_prompt_tokens']:
            conv = None
        else:
            conv.extend(size)
        driver.sleep(corpus_module.think_gap(rng, gap_mean_s))


def scenario_timing(driver, agents=TIMING_AGENTS, turns=TIMING_TURNS, max_tokens=TIMING_MAX_TOKENS,
                    gap_mean_s=None):
    """Phases of 1, 4, 5 and 6 busy agents on one engine (new salts per phase), each agent running
    `turns` turns; per phase the wall time, host RSS samples and CI pod counts are recorded."""
    gap = corpus_module.METERING['gap_mean_s'] if gap_mean_s is None else gap_mean_s
    for count in agents:
        phase = 'agents-%d' % count
        stop = threading.Event()
        samples = []

        def sample():
            while not stop.wait(SAMPLE_EVERY_S):
                if driver.container is not None:
                    samples.append(driver.container.rss_gb())

        sampler = threading.Thread(target=sample)
        sampler.daemon = True
        pods_before = driver.pods()
        started = driver.clock()
        sampler.start()
        try:
            _burst(driver, [lambda index=index: agent_loop(driver, phase, index, turns, max_tokens, gap, stop)
                            for index in range(count)])
        finally:
            stop.set()
            sampler.join(SAMPLE_EVERY_S + 5)
        seconds = driver.clock() - started
        rss = [value for value in samples if value is not None]
        driver.phases[phase] = dict(agents=count, seconds=seconds, rss_gb_max=max(rss) if rss else None,
                                    ci_pods=[pods_before, driver.pods()])
        driver.event(phase, **driver.phases[phase])


SCENARIOS = dict(bringup_reference=scenario_bringup_reference, bringup_prefix=scenario_bringup_prefix,
                 exactness_traced=lambda driver, **_: scenario_exactness(driver, 'traced'),
                 exactness_audit=lambda driver, **_: scenario_exactness(driver, 'audit'),
                 exactness_eager=lambda driver, **_: scenario_exactness(driver, 'eager'),
                 lifecycle_evict=scenario_lifecycle_evict, lifecycle_store=scenario_lifecycle_store,
                 lifecycle_tiny=scenario_lifecycle_tiny, timing=scenario_timing)


def records_jsonl(records):
    """The records as JSON lines, without the bulky fields (output token ids kept: they are the
    comparison's evidence)."""
    out = []
    for record in records:
        slim = dict((key, value) for key, value in record.items() if key not in ('prompt_ids',))
        out.append(json.dumps(slim, sort_keys=True))
    return '\n'.join(out) + ('\n' if out else '')


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def results_path(results, name):
    os.makedirs(results, exist_ok=True)
    return os.path.join(results, name)
