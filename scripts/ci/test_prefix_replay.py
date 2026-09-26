"""prefix_replay: the prefix gates' driver, on CPU.

The streaming client is held against a real local HTTP server (server-sent events, slow first
tokens, closed sockets); the scenarios against FakeEngine, a served general-prefix engine in
miniature whose GRANTS COME FROM THE REAL SCHEDULER GRAFT, not from the harness's oracle:

  - qwen_prefix_scheduler_patch.SchedulerGraft (the trim, the cap, the per-step commit, eviction
    coupling, the kill switch) and qwen_prefix_registry.PrefixRegistry (the checkpoint LRU, pins,
    grants; mid-loop captures declared, as the served model graft does at warmup) are driven over
    a model of vLLM's KV side - FakePool (a free queue in LRU order, a hash map where the first block
    cached for a hash wins, blocks freed tail first, a cached block evicted when reallocated),
    FakeManager.get_computed_blocks (the longest cached run of the request's block hashes, salt in
    the first, capped at num_tokens - 1) and FakeCoordinator.cache_blocks (publish at allocation and
    after every output) - so the harness's oracle is checked against the graft, never against itself;
  - a scheduler loop: steps admit waiting requests FCFS while seats (max-num-seqs) and blocks last
    (an allocation failure leaves the staged grant uncommitted), print a [PREFIX] row per admission,
    and decode one token per running request, preempting the last one when a block runs out (it
    resumes later: a second row whose L is prompt + output so far); concurrent arrivals share a step
    (burst_aware tells the engine a burst is coming, as vLLM would see it on hardware);
  - the KV pool is sized as the TT worker sizes it (pool_blocks: QWEN36_MAX_TOKENS_ALL_USERS or
    max_model_len x max_num_seqs, plus a block per sequence) - num-gpu-blocks-override is ignored,
    as the worker overwrites it;
  - a request aborted before its first output gets no prompt ids back, as vLLM streams them with the
    first output; the vLLM counters (prefix_cache_hits/queries before the trim, preemptions, the
    waiting gauge) are served on metrics();
  - greedy answers depend only on the prompt; faults are switches (see FakeEngine.FAULTS).

test_c2_prefix_gate reuses it."""

import copy
import hashlib
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
from array import array
from collections import OrderedDict, deque
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prefix_agent_corpus as pc  # noqa: E402
import prefix_judge as judge  # noqa: E402
import prefix_markers as pm  # noqa: E402
import prefix_replay as replay  # noqa: E402
import qwen_prefix_registry as prefix_registry  # noqa: E402
import qwen_prefix_scheduler_patch as graft  # noqa: E402
import test_prefix_agent_corpus as corpus_fixture  # noqa: E402
import test_prefix_markers as marker_fixture  # noqa: E402

GEN = [9001, 9002]
ROLE = dict(system=9101, user=9102, assistant=9103, tool=9104)
VOCAB = ['alpha', 'beta', 'gamma', 'delta', 'return', 'value', 'list', 'None', 'file', 'test', 'call', 'fix']
BLOCK = judge.BLOCK
PACE_S = 0.0005        # a seat's decode step, real time: long enough to see a request wait behind it


def word(text):
    return zlib.crc32(text.encode('utf-8')) % 150000 + 1000


def words(text):
    """About one token per 3.2 characters, as Qwen3.8's tokenizer on this text: each whitespace-
    separated word in 3-character pieces."""
    return [word(w[i:i + 3]) for w in (text or '').split() for i in range(0, len(w), 3)]


def served_profile(name='general-prefix'):
    if name == 'general-prefix':
        return marker_fixture.prefix_profile()
    return copy.deepcopy(marker_fixture.PROFILES['profiles'][name])


def pool_blocks(profile):
    """The TT worker's KV block count (plugin worker.py:479-575): the model's all-user tokens -
    QWEN36_MAX_TOKENS_ALL_USERS, else max_model_len x max_num_seqs (qwen36_vllm.py:96-100) - plus a
    block per sequence, in blocks. num-gpu-blocks-override is overwritten by it (worker.py:388-390)."""
    engine, env = profile.get('engine') or {}, profile.get('env') or {}
    seqs = int(engine.get('max-num-seqs', 1))
    size = int(engine.get('block-size', BLOCK))
    tokens = int(env['QWEN36_MAX_TOKENS_ALL_USERS']) if env.get('QWEN36_MAX_TOKENS_ALL_USERS') else (
        int(engine.get('max-model-len', 65536)) * seqs)
    return -(-(tokens + size * seqs) // size)


# -- vLLM's KV side, in miniature ----------------------------------------------------------------

class SaltMintTests(unittest.TestCase):
    """prefix_replay.mint_salt is serving_c2_contract.mint_salt (the rig host has no contract import)."""

    def test_the_minted_salt_is_the_contract_s(self):
        import serving_c2_contract as contract

        key = b'0123456789abcdef' * 4
        for name in ('pfx-salt-exactness-traced-0-chain', 'pfx-cold-bringup-prefix-0-0001', 'a.b c', 'x' * 300):
            with self.subTest(name=name):
                minted = replay.mint_salt(key, name)
                tag = minted.split('.')[1]
                self.assertEqual(minted, contract.mint_salt(key, tag))
                self.assertEqual(contract.salt_verdict(minted, key), 'verified')
                self.assertRegex(tag, '^[A-Za-z0-9_-]{8,128}$')

    def test_without_a_key_the_driver_sends_the_raw_names(self):
        driver = replay.Driver(None, 'arm', None)
        self.assertEqual(driver.salt('s'), 'pfx-salt-arm-0-s')
        self.assertTrue(driver.fresh_salt().startswith('pfx-cold-arm-0-'))


class FakeRequest(object):
    """The Request fields the scheduler graft reads, and the fake scheduler's own bookkeeping."""

    def __init__(self, request_id, tag, prompt_ids, salt, answer, calls, max_tokens):
        self.request_id, self.tag = request_id, tag
        self.prompt_token_ids = list(prompt_ids)
        self.tokens = list(prompt_ids)
        self.output_ids, self.emitted = [], []
        self.cache_salt = salt
        self.num_prompt_tokens = len(prompt_ids)
        self.num_preemptions = 0
        self.hashes = []
        self.answer, self.calls, self.max_tokens = answer, calls, max_tokens
        self.admissions = 0
        self.first_q = None
        self.captured = []
        self.resumed_at = None
        self.perturb = False
        self.on_first = None
        self.abort_before_first = None
        self.pace = False
        self.started = time.time()
        self.first_at = None
        self.done = threading.Event()
        self.result = None

    @property
    def all_token_ids(self):
        return self.tokens

    @property
    def num_tokens(self):
        return len(self.tokens)

    def append(self, token):
        self.output_ids.append(token)
        self.tokens.append(token)

    @property
    def block_hashes(self):
        """vLLM's chain of full-block hashes, the salt in the first (kv_cache_utils.py:560-568)."""
        ids = self.tokens
        while len(self.hashes) < len(ids) // BLOCK:
            index = len(self.hashes)
            digest = hashlib.sha256(self.hashes[-1] if self.hashes else ('salt:%s' % (self.cache_salt or '')).encode())
            digest.update(array('q', ids[index * BLOCK:(index + 1) * BLOCK]).tobytes())
            self.hashes.append(digest.digest())
        return self.hashes


class FakeBlock(object):
    __slots__ = ('block_id', 'block_hash', 'ref')

    def __init__(self, block_id):
        self.block_id, self.block_hash, self.ref = block_id, None, 0


class FakePool(object):
    """vLLM's block pool: a free queue in LRU order (its head is reallocated, so evicted, first), a
    hash -> block map where the first block cached for a hash wins (block_pool.py:47-72), blocks
    freed tail first (so a request's last blocks leave the cache before its first)."""

    def __init__(self, count):
        self.blocks = [FakeBlock(index) for index in range(count)]
        self.free = OrderedDict((block.block_id, block) for block in self.blocks)
        self.cached = {}
        self.cached_block_hashes_by_block = {}
        self.cached_block_hash_to_block = self
        self.evict = self._maybe_evict_cached_block

    def get_one_block(self, key):
        return self.cached.get(key)

    def _maybe_evict_cached_block(self, block):
        if block.block_hash is None:
            return False
        if self.cached.get(block.block_hash) is block:
            del self.cached[block.block_hash]
        block.block_hash = None
        return True

    def evictable(self, blocks):
        return sum(1 for block in blocks if block.ref == 0)

    def touch(self, blocks):
        for block in blocks:
            if block.ref == 0:
                self.free.pop(block.block_id, None)
            block.ref += 1

    def allocate(self, count):
        if count > len(self.free):
            return None
        out = []
        for _ in range(count):
            _, block = self.free.popitem(last=False)
            self.evict(block)
            block.ref = 1
            out.append(block)
        return out

    def release(self, blocks):
        for block in reversed(blocks):
            block.ref -= 1
            if block.ref == 0:
                self.free[block.block_id] = block

    def reset(self):
        for block in self.blocks:
            block.block_hash = None
        self.cached.clear()


class FakeBlocks(object):
    def __init__(self, groups):
        self.blocks = groups


class FakeCoordinator(object):
    """cache_blocks: publish a request's full blocks up to num_tokens (the graft caps what it asks)."""

    def __init__(self, pool, single):
        self.pool, self.single = pool, single
        self.single_type_managers = [single]

    def cache_blocks(self, request, num_tokens):
        owned = self.single.req_to_blocks.get(request.request_id) or []
        start = self.single.num_cached_block.get(request.request_id, 0)
        hashes = request.block_hashes
        end = min(num_tokens // BLOCK, len(owned), len(hashes))
        for index in range(start, end):
            block = owned[index]
            if block.block_hash is None and hashes[index] not in self.pool.cached:
                block.block_hash = hashes[index]
                self.pool.cached[hashes[index]] = block
        if end > start:
            self.single.num_cached_block[request.request_id] = end


class FakeManager(object):
    """get_computed_blocks: the longest run of the request's block hashes the pool has cached,
    capped at num_tokens - 1, counted into the prefix-cache stats before any trim
    (kv_cache_manager.py:206-246; a preempted request's attempts are counted apart)."""

    def __init__(self, pool, coordinator):
        self.block_pool, self.coordinator = pool, coordinator
        self.empty_kv_cache_blocks = FakeBlocks(([],))
        self.hits = self.queries = 0

    def create_kv_cache_blocks(self, groups):
        return FakeBlocks(groups)

    def get_computed_blocks(self, request):
        limit = (request.num_tokens - 1) // BLOCK
        found = []
        for key in request.block_hashes[:limit]:
            block = self.block_pool.cached.get(key)
            if block is None:
                break
            found.append(block)
        if not request.num_preemptions:
            self.queries += request.num_tokens
            self.hits += len(found) * BLOCK
        return FakeBlocks((found,)), len(found) * BLOCK


# -- the engine ----------------------------------------------------------------------------------

class FakeEngine(object):
    """A served engine in miniature (see the module docstring). `profile` is what the contract would
    serve: prefix reuse, the loop path (trace_mode), audit, dev mode, the store and the pool are read
    from it. FAULTS switch on one defect each."""

    FAULTS = ('diverge_hits', 'diverge_once', 'diverge_after_resume', 'unsalted_differs', 'capture_differs',
              'grow_programs', 'grow_on_capture', 'drop_rows', 'drop_resumed_rows', 'no_digests', 'bad_slot_on_hit',
              'publish_unsalted', 'publish_when_killed', 'no_stats', 'no_dropped_hits', 'no_counters', 'kill_line_off',
              'no_dram', 'no_capture_dram', 'dram_unavailable', 'no_eager_warm')
    # The model graft's G2 reading (qwen_prefix_model_patch._qwen_prefix_dram), in serving_buffer_pool's text.
    DRAM_TEXT = ('chip0 allocated=25.90GB free=8.01GB largest_free=7877.5MB of 33.91GB; '
                 'chip1 allocated=25.90GB free=8.01GB largest_free=7877.5MB of 33.91GB')

    def __init__(self, profile=None, name='general-prefix', answer_len=24, path=None, **faults):
        unknown = set(faults) - set(self.FAULTS)
        if unknown:
            raise TypeError('unknown faults %s' % sorted(unknown))
        self.faults = set(key for key, value in faults.items() if value)
        self.name = name
        self.profile = copy.deepcopy(profile) if profile is not None else served_profile(name)
        env, engine = self.profile.get('env') or {}, self.profile.get('engine') or {}
        tt = (engine.get('additional-config') or {}).get('tt') or {}
        self.prefix = str(env.get('QWEN_PREFIX_REUSE')) == '1' and engine.get('enable-prefix-caching') is True
        self.path = path or ('eager' if tt.get('trace_mode') == 'decode_only' else 'traced')
        self.audit = str(env.get('QWEN_PREFIX_AUDIT')) == '1'
        self.dev_mode = str(env.get('VLLM_SERVER_DEV_MODE')) == '1'
        self.store_gib = float(env.get('QWEN_PREFIX_STORE_GIB', prefix_registry.DEFAULT_STORE_GIB))
        self.max_num_seqs = int(engine.get('max-num-seqs', 4))
        self.max_model_len = int(engine.get('max-model-len', 65536))
        self.num_blocks = pool_blocks(self.profile)
        self.answer_len = answer_len
        self.kill_path = os.path.join(tempfile.gettempdir(), 'pfx-kill-%d-%d' % (os.getpid(), id(self)))
        self.cv = threading.Condition()
        self.thread = None
        self.lines = []
        self.count = 0
        self.programs = 500
        self.grew_capture = False
        self.diverged_once = False
        self.boot()

    # -- the log ---------------------------------------------------------------------------------
    def say(self, line):
        self.lines.append('2026-09-26T%02d:%02d:%02d.000000001Z %s' % (
            len(self.lines) // 3600 % 24, len(self.lines) // 60 % 60, len(self.lines) % 60, line))

    def dram(self, point):
        """G2's reading at `point`, once per engine process, as the model graft logs it: the registry
        point on the first prefix-route prefill, the capture point after the first stored checkpoint."""
        if point in self.dram_logged or 'no_dram' in self.faults or (
                point == pm.DRAM_FIRST_CAPTURE and 'no_capture_dram' in self.faults):
            return
        self.dram_logged.add(point)
        text = ('unavailable (RuntimeError: no allocator on this device)' if 'dram_unavailable' in self.faults
                else self.DRAM_TEXT)
        self.say('(EngineCore pid=9) INFO | models.demos.blackhole.qwen36.tt.model:_qwen_prefix_dram:180 - '
                 '[PINDIAG] dram after %s: %s' % (point, text))

    def graft_say(self, message, *values):
        if 'kill switch' in message and 'kill_line_off' in self.faults:
            return
        self.say('[PINDIAG] prefix: ' + (message % values if values else message))

    def boot(self):
        self.pool = FakePool(self.num_blocks)
        self.single = SimpleNamespace(num_cached_block={}, req_to_blocks={})
        self.coordinator = FakeCoordinator(self.pool, self.single)
        self.manager = FakeManager(self.pool, self.coordinator)
        self.registry = graft.PrefixRegistry(environ=dict(QWEN_PREFIX_STORE_GIB=str(self.store_gib)))
        # The served model graft declares its mid-loop captures at warmup (qwen_prefix_model_patch).
        self.registry.enable_mid_loop_capture()
        self.graft = graft.SchedulerGraft(SimpleNamespace(kv_cache_manager=self.manager), self.registry,
                                          graft.KillSwitch(self.kill_path, 0.0, time.monotonic), self.graft_say)
        self.graft.original.update(get_computed_blocks=self.manager.get_computed_blocks,
                                   cache_blocks=self.coordinator.cache_blocks,
                                   _maybe_evict_cached_block=self.pool._maybe_evict_cached_block)
        self.graft.get_block_hash = lambda key: key
        self.pool.evict = self.graft.evict
        self.waiting, self.running = deque(), []
        self.preemptions = 0
        self.dropped_hits = 0
        self.expected, self.expect_since = 0, None
        self.dead = None
        self.dram_logged = set()
        profile = self.profile
        self.say('(APIServer pid=1) [QWEN-C2] profile %s: vLLM argv %s' % (
            self.name, json.dumps(marker_fixture.launched_argv(profile))))
        if self.prefix:
            self.say('INFO platform.py:83] Chunked prefill is not supported for `model_type=qwen3_5`; disabling it.')
        self.say('INFO platform.py:1153] Automatic prefix caching is %s' % ('enabled' if self.prefix else 'disabled'))
        self.say('INFO kv_cache_utils.py:2146] GPU KV cache size: {:,} tokens'.format(self.num_blocks * BLOCK))
        if self.prefix and self.path == 'eager' and 'no_eager_warm' not in self.faults:
            # The model graft's eager warm (qwen_prefix_model_patch._qwen_prefix_warm_eager), before the decode trace.
            self.say('(EngineCore pid=9) INFO | models.demos.blackhole.qwen36.tt.qwen36_vllm:_qwen_prefix_warm_eager:505 - '
                     '[PINDIAG] prefix: eager prefill warmed before the decode trace: page_table_blocks=4128 '
                     'programs=115->554')
        if self.prefix:
            line = marker_fixture.install_line().replace('store_gib=8.0', 'store_gib=%.1f' % self.store_gib)
            self.say('(EngineCore pid=9) ' + line)

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

    def tokenize_ids(self, body):
        return self.render(body)

    # -- the API ---------------------------------------------------------------------------------
    def chat(self, body, tag, salt=None, max_tokens=64, timeout=None, abort_after_s=None, abort_after_tokens=None,
             extra=None, on_first=None, abort_signal=None):
        ids = self.render(body)
        if len(ids) + int(max_tokens) > self.max_model_len:
            return dict(content='', reasoning=None, tool_calls=[], token_ids=None, prompt_ids=None, prompt_sha=None,
                        prompt_tokens=None, completion_tokens=0, finish=None, ttft_s=None, wall_s=0.1, status=400,
                        aborted=None, ok=False, error='HTTP 400: This model\'s maximum context length is %d tokens '
                        '(%d in the messages, %d in the completion)' % (self.max_model_len, len(ids), int(max_tokens)))
        ignore_eos = bool((extra or {}).get('ignore_eos'))
        rng = random.Random(judge.token_sha(ids))
        length = int(max_tokens) if ignore_eos else min(int(max_tokens), self.answer_len)
        answer = [VOCAB[rng.randrange(len(VOCAB))] for _ in range(length)]
        calls = []
        if rng.random() < 0.4:
            calls = [dict(id='chatcmpl-tool-%d' % len(ids), type='function', function=dict(
                name='read', arguments=json.dumps({'file_path': '/w/scripts/ci/mod_1.py'})))]
        with self.cv:
            self.count += 1
            request = FakeRequest('chatcmpl-%s-%08x' % (tag, self.count), tag, ids, salt if self.prefix else None,
                                  answer, calls, int(max_tokens))
            request.on_first = on_first
            request.abort_before_first = abort_after_s
            request.pace = tag.endswith('-seat')
            if self.dead:
                request.result = self.failed(request, self.dead)
                return request.result
            self.waiting.append(request)
            self.cv.notify_all()
            self.ensure_thread()
        while not request.done.wait(0.01):
            if abort_signal is not None and abort_signal.is_set():
                with self.cv:
                    if not request.done.is_set():
                        self.abort(request, 'closed on signal')
        return request.result

    def metrics(self):
        with self.cv:
            values = {'vllm:num_requests_waiting': float(len(self.waiting)),
                      'vllm:num_preemptions': float(self.preemptions)}
            if 'no_counters' not in self.faults:
                values.update({'vllm:prefix_cache_hits': float(self.manager.hits),
                               'vllm:prefix_cache_queries': float(self.manager.queries)})
            return values

    def healthy(self):
        return not self.dead

    def ready(self):
        return True

    def reset_prefix_cache(self):
        if not self.dev_mode:
            return 404
        with self.cv:
            self.pool.reset()
            self.registry.clear()
        return 200

    def restart(self):
        with self.cv:
            for request in list(self.waiting) + list(self.running):
                self.abort(request, 'engine stopped')
            self.say('INFO launcher.py] Shutting down the engine (docker stop)')
            self.boot()

    def expect(self, count):
        """A burst of `count` requests is coming: admit them in one step (vLLM sees simultaneous
        arrivals in one schedule() call)."""
        with self.cv:
            self.expected, self.expect_since = count, time.monotonic()

    def stats(self):
        if 'no_stats' in self.faults:
            return None
        with self.cv:
            values = self.registry.snapshot()
            if 'no_dropped_hits' in self.faults:
                values.pop('dropped_hits', None)     # a registry without the counter (the P0a prototype)
            return values

    def kill_switch(self, on):
        if on:
            with open(self.kill_path, 'w') as handle:
                handle.write(replay.KILL_SWITCH_OWNER + '\n')
        elif os.path.exists(self.kill_path):
            os.remove(self.kill_path)

    # -- the scheduler loop ----------------------------------------------------------------------
    def ensure_thread(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.loop)
            self.thread.daemon = True
            self.thread.start()

    def loop(self):
        idle = 0
        while True:
            with self.cv:
                if not self.waiting and not self.running:
                    self.cv.wait(0.05)
                    if not self.waiting and not self.running:
                        idle += 1
                        if idle > 20:
                            self.thread = None
                            return
                    continue
                idle = 0
                progressed = self.step()
                if not progressed or (self.running and all(request.pace for request in self.running)):
                    # Held for a burst, or only paced seats running: let the clock (and the callers) move.
                    self.cv.wait(0.002)

    def step(self):
        """One schedule() and its model step. -> whether anything was admitted or decoded."""
        holding = bool(self.expected and len(self.waiting) < self.expected
                       and time.monotonic() - (self.expect_since or 0) < 2.0)
        admitted = []
        if self.waiting and not holding:
            # The registry logs through the module's log (commit refused, capture skipped).
            with mock.patch.object(prefix_registry, 'log', self.graft_say):
                self.registry.begin_step()
                admitted = self.admit()
                for request, q in admitted:
                    self.prefill(request, q)
        else:
            self.registry.begin_step()
        return self.decode() or bool(admitted)

    def admit(self):
        new, seats = [], self.max_num_seqs - len(self.running)
        for request in list(self.waiting):
            if seats <= 0:
                break
            if self.prefix:
                blocks, h = self.manager.get_computed_blocks(request)
                trimmed, q = self.graft.trim(request, blocks, h)
                hit = list(trimmed.blocks[0])
            else:
                hit, q = [], 0
            need = -(-request.num_tokens // BLOCK) - len(hit)
            if need + self.pool.evictable(hit) > len(self.pool.free):
                break      # vLLM stops admitting at an allocation failure; a staged grant is dropped at commit
            self.pool.touch(hit)
            self.single.req_to_blocks[request.request_id] = hit + self.pool.allocate(need)
            self.single.num_cached_block[request.request_id] = len(hit)
            self.publish(request, request.num_tokens)
            self.waiting.remove(request)
            self.running.append(request)
            seats -= 1
            new.append((request, q))
        if new:
            self.expected = 0
        if self.prefix:
            ids = set(request.request_id for request, _ in new)
            self.dropped_hits += sum(1 for rid, grant in self.registry.staged.items() if rid not in ids and grant.q)
            fresh = [(r, q) for r, q in new if not r.admissions]
            resumed = [(r, q) for r, q in new if r.admissions]
            self.graft.commit(SimpleNamespace(
                scheduled_new_reqs=[SimpleNamespace(req_id=r.request_id, num_computed_tokens=q) for r, q in fresh],
                scheduled_cached_reqs=SimpleNamespace(req_ids=[r.request_id for r, _ in resumed],
                                                      resumed_req_ids=set(r.request_id for r, _ in resumed),
                                                      num_computed_tokens=[q for _, q in resumed])))
        return new

    def publish(self, request, tokens):
        if not self.prefix:
            return
        bypass = (('publish_unsalted' in self.faults and not request.cache_salt)
                  or ('publish_when_killed' in self.faults and self.graft.killed))
        if bypass:
            self.coordinator.cache_blocks(request, min(tokens, judge.floor_chunk(request.num_prompt_tokens)))
        else:
            self.graft.cap(request, tokens)

    def prefill(self, request, q):
        """The model's prefill row: the committed grant must be the row's start (the model graft's
        assertion), the planned boundaries are captured, and the row is printed."""
        request.admissions += 1
        first = request.admissions == 1
        if not first:
            request.resumed_at = len(request.output_ids)
        captured = []
        if self.prefix:
            self.dram(pm.DRAM_REGISTRY)
            grant = self.registry.grant_for(request.request_id)
            granted = grant.q if grant is not None else 0
            if granted != q:
                self.say('ERROR AssertionError: the committed grant Q=%s is not the row start %s (%s)' % (
                    granted, q, request.request_id))
            for position, _ in (grant.plan if grant is not None else ()):
                # The model graft captures inside its chunk loop, after exactly `position` tokens.
                if self.registry.capture(request.request_id, position, loop_pos=position) is not None:
                    captured.append(position)
        if first:
            request.first_q, request.captured = q, captured
            request.perturb = bool(
                ('diverge_hits' in self.faults and q) or ('unsalted_differs' in self.faults and not request.cache_salt)
                or ('capture_differs' in self.faults and captured)
                or ('diverge_once' in self.faults and q and not self.diverged_once))
            if 'diverge_once' in self.faults and q:
                self.diverged_once = True
        if not self.prefix:
            return
        before = self.programs
        if 'grow_programs' in self.faults and q:
            self.programs += 1
        if 'grow_on_capture' in self.faults and captured and not self.grew_capture:
            self.programs += 1
            self.grew_capture = True
        ids = list(request.all_token_ids)
        slot = judge.token_sha(ids)[:16]
        if 'bad_slot_on_hit' in self.faults and q:
            slot = 'ff' + slot[2:]
        digests = '' if 'no_digests' in self.faults else ' slot_sha=%s logits_sha=%s' % (
            slot, hashlib.sha256(('logits:%s' % judge.token_sha(ids)).encode()).hexdigest()[:16])
        if not ('drop_rows' in self.faults or ('drop_resumed_rows' in self.faults and not first)):
            self.say('[PREFIX] req=%s Q=%d L=%d path=%s restored_ms=%.1f captured=[%s] capture_ms=%.1f '
                     'programs_before=%d programs=%d%s' % (
                         request.request_id, q, len(ids), self.path, 120.0 if q else 0.0,
                         ','.join(str(p) for p in captured), 300.0 if captured else 0.0, before, self.programs,
                         digests))
        if captured:
            self.dram(pm.DRAM_FIRST_CAPTURE)
        if self.audit:
            digest = judge.token_sha(ids)[:16]
            self.say('[PREFIX-AUDIT] req=%s Q=%d L=%d kv_range=0:%d kv_sha=%s slot_sha=%s' % (
                request.request_id, q, len(ids), len(ids), digest, digest))
        if len(ids) >= judge.CHUNK and self.path == 'traced':
            self.say('INFO [TP chunk-replay] %d/%d chunks' % (len(ids) // judge.CHUNK, len(ids) // judge.CHUNK))

    def decode(self):
        """A token for each running request; a seat (a request whose tag ends -seat) decodes at one
        token per PACE_S of real time instead, so a waiting request can be seen and aborted behind it
        as on hardware. -> whether any token was emitted."""
        now, emitted = time.time(), False
        for request in list(self.running):
            if request not in self.running:
                continue
            if request.abort_before_first is not None and not request.output_ids:
                self.finish(request, 'closed after %.1f s' % request.abort_before_first)
                continue
            count = 1
            if request.pace and request.first_at is not None:
                allowed = int((now - request.first_at) / PACE_S) + 1
                count = min(allowed, len(request.answer)) - len(request.output_ids)
            for _ in range(max(0, count)):
                emitted = True
                if not self.emit(request):
                    break
        return emitted

    def emit(self, request):
        """One output token for a running request (a block for it first, preempting the last running
        request when none is free). -> whether the request is still running."""
        owned = self.single.req_to_blocks[request.request_id]
        while len(owned) * BLOCK < request.num_tokens + 1:
            fresh = self.pool.allocate(1)
            if fresh:
                owned.extend(fresh)
                break
            victim = self.running[-1]
            self.preempt(victim)
            if victim is request:
                return False
        index = len(request.output_ids)
        text = request.answer[index]
        token = word(text)
        last = index == len(request.answer) - 1
        if (request.perturb and last) or ('diverge_after_resume' in self.faults and request.resumed_at is not None
                                          and index >= request.resumed_at):
            token, text = token + 1, text + 'x'
        request.append(token)
        request.emitted.append(text)
        if index == 0:
            request.first_at = time.time()
            if request.on_first is not None:
                request.on_first()
        self.publish(request, request.num_tokens)
        if len(request.output_ids) >= len(request.answer):
            self.finish(request)
            return False
        return True

    def preempt(self, victim):
        self.release(victim)
        self.running.remove(victim)
        victim.num_preemptions += 1
        self.waiting.appendleft(victim)
        self.preemptions += 1

    def release(self, request):
        self.pool.release(self.single.req_to_blocks.pop(request.request_id, []))
        self.single.num_cached_block.pop(request.request_id, None)

    def finish(self, request, aborted=None):
        self.release(request)
        if self.prefix:
            self.registry.forget_request(request.request_id)
        if request in self.running:
            self.running.remove(request)
        request.result = self.result(request, aborted)
        request.done.set()

    def abort(self, request, reason):
        if request in self.waiting:
            self.waiting.remove(request)
        self.finish(request, reason)

    def failed(self, request, error):
        request.done.set()
        return dict(content='', reasoning=None, tool_calls=[], token_ids=None, prompt_ids=None, prompt_sha=None,
                    prompt_tokens=None, completion_tokens=0, finish=None, error=error, ttft_s=None, wall_s=0.1,
                    status=None, aborted=None, ok=False)

    def result(self, request, aborted):
        emitted = len(request.output_ids)
        texts = request.emitted
        half = len(texts) // 2
        finish, calls = None, []
        if aborted is None:
            if emitted >= request.max_tokens:
                finish = 'length'
                # A call cut off by max_tokens streams half its arguments.
                calls = [dict(call, function=dict(call['function'], arguments=call['function']['arguments'][:-6]))
                         for call in request.calls]
            elif request.calls:
                finish, calls = 'tool_calls', copy.deepcopy(request.calls)
            else:
                finish = 'stop'
        prompt = list(request.prompt_token_ids) if emitted else None
        return dict(content=' '.join(texts[half:]), reasoning=' '.join(texts[:half]) or None, tool_calls=calls,
                    token_ids=list(request.output_ids) if emitted or finish else None, prompt_ids=prompt,
                    prompt_sha=judge.token_sha(prompt) if prompt is not None else None,
                    prompt_tokens=len(prompt) if prompt is not None else None, completion_tokens=emitted,
                    finish=finish, error=None,
                    ttft_s=round(request.first_at - request.started, 4) if request.first_at is not None else None,
                    wall_s=round(time.time() - request.started, 4), status=200, aborted=aborted, ok=aborted is None)


def burst_aware(get_engine):
    """replay._burst, telling the engine how many requests are about to arrive at once."""
    original = replay._burst

    def burst(driver, jobs):
        engine = get_engine()
        if engine is not None:
            engine.expect(len(jobs))
        return original(driver, jobs)

    return mock.patch.object(replay, '_burst', burst)


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
            self.engine.kill_switch(True)
        elif 'rm -f' in script:
            self.engine.kill_switch(False)
        return 0, ''

    def read_file(self, path):
        if path != pm.STATS_FILE:
            return None
        stats = self.engine.stats()
        return json.dumps(stats) if stats is not None else None

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


def resolved(driver, engine):
    judge.resolve(driver.records, pm.scan(engine.lines))
    return driver.records


# -- the fake itself: what the gates' fixes rest on ----------------------------------------------

class FakeEngineTests(unittest.TestCase):
    def body(self, text, salt=None):
        return dict(messages=[dict(role='system', content='s'), dict(role='user', content=text)], tools=[])

    def test_the_pool_is_the_workers_not_the_override(self):
        """The TT worker overwrites num_gpu_blocks_override (plugin worker.py:388-390): the pool comes
        from the model's all-user tokens, which QWEN36_MAX_TOKENS_ALL_USERS sets."""
        profile = served_profile('general-prefix')
        self.assertEqual(pool_blocks(profile), 4100, '65536 x 4 + 4 blocks of padding')
        profile['engine']['num-gpu-blocks-override'] = 1280
        self.assertEqual(pool_blocks(profile), 4100)
        profile['env']['QWEN36_MAX_TOKENS_ALL_USERS'] = '81664'
        self.assertEqual(pool_blocks(profile), 1280)

    def test_grants_come_from_the_real_graft(self):
        engine = FakeEngine()
        first = engine.chat(self.body('alpha ' * 3000), 't1-hit', salt='s', max_tokens=8)
        second_body = self.body('alpha ' * 3000)
        second_body['messages'].append(dict(role='assistant', content=first['content'], reasoning=first['reasoning']))
        second_body['messages'].append(dict(role='user', content='beta ' * 500))
        engine.chat(second_body, 't2-hit', salt='s', max_tokens=8)
        rows = pm.scan(engine.lines)['rows']
        self.assertEqual([row['q'] for row in rows], [0, judge.floor_chunk(first['prompt_tokens'])])
        self.assertEqual(engine.registry.stats['grants'], 1, 'the real registry committed the grant')
        self.assertTrue(pm.scan(engine.lines)['grants'])

    def test_an_abort_before_the_first_output_returns_no_prompt_ids(self):
        engine = FakeEngine()
        result = engine.chat(self.body('gamma ' * 100), 'a-hit', salt='s', abort_after_s=2.0)
        self.assertEqual((result['aborted'], result['prompt_ids'], result['prompt_tokens']), ('closed after 2.0 s', None, None))
        self.assertEqual(len(pm.scan(engine.lines)['rows']), 1, 'it was admitted and prefilled')

    def test_a_preempted_request_is_readmitted_with_prompt_plus_output(self):
        profile = served_profile('general-prefix')
        profile['env']['QWEN36_MAX_TOKENS_ALL_USERS'] = str(12 * BLOCK - 4 * BLOCK)   # a 12-block pool
        engine = FakeEngine(profile=profile)
        bodies = [self.body('%s ' % name * 120) for name in ('alpha', 'beta')]
        engine.expect(2)
        results = replay._join(replay._spawn([lambda body=body, index=index: engine.chat(
            body, 'p%d-hit' % index, salt='s%d' % index, max_tokens=300, extra=dict(ignore_eos=True))
            for index, body in enumerate(bodies)]))
        self.assertTrue(all(r['ok'] for r in results))
        self.assertGreater(engine.preemptions, 0)
        rows = pm.scan(engine.lines)['rows']
        again = [row for row in rows if row['l'] > min(r['prompt_tokens'] for r in results)]
        self.assertTrue(again, 'a second row at prompt + output')


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
            payload = json.dumps(dict(count=len(body['messages']) * 10, tokens=list(range(len(body['messages']) * 10)))).encode()
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
        self.assertEqual(self.client.tokenize_ids(self.body()), list(range(10)))
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
            ids = self.engine.tokenize_ids(body)
            self.reply(dict(count=len(ids), tokens=ids))
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
                name=call['function']['name'], arguments=call['function']['arguments']))]),
                finish_reason=result['finish'])]))
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
        self.assertEqual([text for r in records for severity, text in judge.reuse_problems(r) if severity != 'NOTE'], [])
        self.assertTrue(all(p['verdict'] == 'IDENTICAL' for p in driver.pairs), driver.pairs)
        self.assertTrue(any(r['markers']['q'] for r in records if r['role'] == 'hit'))
        self.assertTrue(any(r['tool_calls'] for r in records), 'a tool call came back and was sent back')
        self.assertEqual(driver.events['boundary-2048']['served'], 2048)


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
    def test_the_log_follower_reads_docker_logs_and_a_restart_adds_no_line_twice(self):
        outputs = [[b'2026-09-26T10:00:00.1Z first\n', b'2026-09-26T10:00:01.2Z second\n'],
                   [b'2026-09-26T10:00:01.2Z second\n', b'2026-09-26T10:00:05Z after the restart\n']]

        class Process(object):
            def __init__(self, lines):
                self.stdout = lines

            def terminate(self):
                pass

        calls = []

        def popen(arguments, **kwargs):
            calls.append(arguments)
            return Process(outputs[len(calls) - 1])

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
            self.assertEqual(follower.lines()[2:], ['2026-09-26T10:00:05Z after the restart'],
                             'the line --since re-read is not added twice')
            with open(os.path.join(directory, 'server.log'), encoding='utf-8') as handle:
                self.assertEqual(len(handle.read().splitlines()), 3)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_docker_timestamps_sort_as_times(self):
        self.assertLess(replay.stamp_key('2026-09-26T10:00:00.1234Z'), replay.stamp_key('2026-09-26T10:00:00.12345Z'))
        self.assertLess(replay.stamp_key('2026-09-26T10:00:00Z'), replay.stamp_key('2026-09-26T10:00:00.5Z'))
        self.assertIsNone(replay.stamp_key('nope'))

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
        self.assertTrue(os.path.exists(container.engine.kill_path))
        replay.kill_switch_off(container)
        self.assertFalse(os.path.exists(container.engine.kill_path))
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

    def chain_of_two(self, engine):
        driver = driver_for(engine)
        conv = driver.conversation('c', first_tokens=8000)
        hit = driver.pair(conv, 'case')
        driver.answer(conv, hit)
        conv.extend(600)
        driver.pair(conv, 'case')
        return driver

    def test_a_divergent_hit_reruns_at_the_same_q_and_fails(self):
        engine = FakeEngine(diverge_hits=True)
        driver = self.chain_of_two(engine)
        second = driver.pairs[-1]
        self.assertEqual(second['verdict'], 'DIVERGED')
        self.assertIsNotNone(second['rerun'])
        records = resolved(driver, engine)
        by_tag = dict((r['tag'], r) for r in records)
        primes = [by_tag[tag] for tag in second['primes']]
        self.assertEqual([p['role'] for p in primes], ['prime'])
        self.assertEqual(primes[0]['completion_tokens'], replay.PRIME_MAX_TOKENS)
        first_hit, rerun_hit = by_tag[second['hit']], by_tag[second['rerun'][1]]
        self.assertEqual(rerun_hit['case'], 'case:rerun')
        self.assertNotEqual(rerun_hit['salt'], first_hit['salt'], 'the re-run replays under a fresh salt')
        self.assertEqual(rerun_hit['markers']['q'], first_hit['markers']['q'], 'the same Q, not a tail-only hit')
        self.assertGreater(first_hit['markers']['q'], 0)
        self.assertIn('diverged again', second['detail'])

    def test_a_hit_that_diverges_once_from_agreeing_colds_still_fails(self):
        driver = self.chain_of_two(FakeEngine(diverge_once=True))
        second = driver.pairs[-1]
        self.assertEqual(second['verdict'], 'DIVERGED')
        self.assertIn('did not reproduce', second['detail'])

    def test_a_dead_engine_stops_the_arm(self):
        engine = FakeEngine()
        driver = driver_for(engine)
        engine.healthy = lambda: False
        original = engine.chat
        engine.chat = lambda *a, **k: dict(original(*a, **k), ok=False, error='connection reset')
        with self.assertRaises(replay.EngineDead):
            driver.send(dict(messages=[dict(role='user', content='x')]), 'hit', 's', 'c')

    def test_the_arms_time_runs_out_and_bounds_every_request(self):
        engine = FakeEngine()
        clock = [1000.0]
        driver = replay.Driver(engine, 'arm', CORPUS, clock=lambda: clock[0], sleep=lambda s: None, say=lambda t: None,
                               pods=lambda: None, deadline=1000.0 + 300)
        self.assertEqual(driver.request_timeout(), 300)
        clock[0] += 295
        self.assertEqual(driver.request_timeout(), replay.MIN_REQUEST_TIMEOUT_S)
        seen = []
        engine.chat = lambda *a, **k: seen.append(k.get('timeout')) or dict(
            content='', reasoning=None, tool_calls=[], token_ids=[1], prompt_ids=[1], prompt_sha='x', prompt_tokens=1,
            completion_tokens=1, finish='stop', error=None, ttft_s=0.1, wall_s=0.1, status=200, aborted=None, ok=True)
        driver.send(dict(messages=[dict(role='user', content='x')]), 'hit', 's', 'c')
        self.assertEqual(seen, [replay.MIN_REQUEST_TIMEOUT_S])
        clock[0] += 10
        with self.assertRaises(replay.OutOfTime):
            driver.send(dict(messages=[dict(role='user', content='x')]), 'hit', 's', 'c')

    def test_an_aborted_request_is_admitted_from_its_tokenized_prompt(self):
        engine = FakeEngine()
        driver = driver_for(engine)
        conv = driver.conversation('c', first_tokens=3000)
        record = driver.send(conv.body(), 'hit', conv.salt, 'abort', conv, abort_after_s=2.0)
        self.assertEqual((record['aborted'], record['prompt_ids_from']), ('closed after 2.0 s', 'tokenize'))
        self.assertEqual(record['expected']['q'], 0)
        self.assertEqual(driver.oracle.published_tokens(conv.salt, engine.render(conv.body())),
                         judge.floor_chunk(engine.tokenize(conv.body())), 'the oracle saw what it published')


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.engine = None
        patcher = burst_aware(lambda: self.engine)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_exactness_traced_runs_every_case_and_every_pair_matches(self):
        self.engine = engine = FakeEngine()
        driver = driver_for(engine, 'exactness-traced')
        replay.scenario_exactness(driver, 'traced')
        cases = set(pair['case'] for pair in driver.pairs)
        self.assertEqual(cases, {'chain', 'changed-suffix', 'early-divergence', 'boundary-2047', 'boundary-2048',
                                 'boundary-2049', 'shared-system'})
        self.assertTrue(all(pair['verdict'] == 'IDENTICAL' for pair in driver.pairs))
        self.assertEqual(len([p for p in driver.pairs if p['case'] == 'chain']), 1 + len(replay.CHAIN_HITS))
        for name in ('boundary-2047', 'boundary-2048', 'boundary-2049'):
            self.assertEqual((driver.events[name]['tokens'], driver.events[name]['served']), (int(name[-4:]),) * 2)
        records = resolved(driver, engine)
        problems = [text for r in records for severity, text in judge.reuse_problems(r) if severity != 'NOTE']
        self.assertEqual(problems, [], 'the oracle agrees with the real graft on every request')
        by_tag = dict((r['tag'], r) for r in records)
        second = [by_tag[p['hit']] for p in driver.pairs if p['case'] == 'boundary-2048'][1]
        self.assertEqual(second['markers']['q'], 2048)
        shared = [by_tag[p['hit']] for p in driver.pairs if p['case'] == 'shared-system']
        own = judge.floor_chunk(shared[1]['prompt_tokens'])
        gap = [pos for pos in shared[1]['markers']['row']['captured'] if pos < own]
        self.assertTrue(gap, 'the second shared conversation captured a gap boundary')
        self.assertEqual(shared[2]['markers']['q'], gap[0], 'the third restored exactly it')
        self.assertEqual(shared[1]['markers']['q'], 0)

    def test_eager_and_audit_run_the_short_chain(self):
        for kind in ('eager', 'audit'):
            profile = served_profile()
            if kind == 'eager':
                profile['engine']['additional-config'].setdefault('tt', {})['trace_mode'] = 'decode_only'
            else:
                profile['env']['QWEN_PREFIX_AUDIT'] = '1'
            self.engine = engine = FakeEngine(profile=profile)
            driver = driver_for(engine, 'exactness-' + kind)
            replay.scenario_exactness(driver, kind)
            chain = [p for p in driver.pairs if p['case'] == 'chain']
            self.assertEqual(len(chain), 1 + len(replay.CHAIN_HITS_SHORT))
            self.assertNotIn('shared-system', set(p['case'] for p in driver.pairs))
            self.assertEqual(engine.path, 'eager' if kind == 'eager' else 'traced')

    def test_bringup_runs_unsalted_and_capturing_turns_then_salted_pairs(self):
        self.engine = engine = FakeEngine()
        driver = driver_for(engine, 'bringup-prefix')
        replay.scenario_bringup_prefix(driver)
        roles = [r['role'] for r in driver.records]
        self.assertEqual(roles[:6], ['unsalted', 'capture'] * 3)
        self.assertEqual(roles[6:], ['cold', 'hit'] * 3)
        records = resolved(driver, engine)
        for record in records[:6]:
            if record['role'] == 'capture':
                self.assertEqual(record['markers']['row']['captured'], [judge.floor_chunk(record['prompt_tokens'])])
            else:
                self.assertEqual((record['markers']['row']['captured'], record['expected_raw_h']), ([], 0))
                self.assertEqual(judge.raw_hit_per_attempt(record), (0.0, 1))
        self.engine = reference_engine = FakeEngine(name='general')
        reference = driver_for(reference_engine, 'bringup-reference')
        replay.scenario_bringup_reference(reference)
        self.assertEqual([r['prompt_sha'] for r in reference.records], [r['prompt_sha'] for r in driver.records[:6:2]],
                         'the grants-disabled run and the reference send the same prompts')

    def test_an_unsalted_request_that_publishes_is_seen_in_the_counters(self):
        self.engine = engine = FakeEngine(publish_unsalted=True)
        driver = driver_for(engine, 'bringup-prefix')
        replay.scenario_bringup_prefix(driver)
        unsalted = [r for r in driver.records if r['role'] == 'unsalted']
        raw = [judge.raw_hit_per_attempt(r)[0] for r in unsalted]
        self.assertEqual(raw[0], 0)
        self.assertGreater(raw[1], 0, 'turn 2 found what turn 1 published')
        self.assertEqual([r['expected_raw_h'] for r in unsalted], [0, 0, 0])

    def test_lifecycle_evict_drives_every_event(self):
        self.engine = engine = FakeEngine(profile=dict(served_profile(), env=dict(
            served_profile()['env'], VLLM_SERVER_DEV_MODE='1')))
        driver = driver_for(engine, 'lifecycle-evict', strict=False)
        restarts = []

        def restart():
            driver.container.stop()
            driver.container.start()
            restarts.append(1)
            return dict(seconds=1.0, log_window=[len(engine.lines) - 3, len(engine.lines)])

        with mock.patch.object(replay, 'EVICT_LENGTHS', (12000, 24000)):
            replay.scenario_lifecycle_evict(driver, pool_tokens=engine.num_blocks * BLOCK, restart=restart)
        events = driver.events
        self.assertTrue(events['arrivals']['ok'])
        self.assertEqual(len(events['arrivals']['tags']), 4)
        self.assertEqual((events['abort-waiting']['phase'], events['abort-waiting']['aborted']), ('waiting', 'closed on signal'))
        self.assertFalse(events['abort-prefill']['first_token_before_abort'])
        self.assertEqual(events['reset-prefix-cache']['status'], 200)
        self.assertEqual((events['kill-switch']['written'], events['kill-switch']['removed']), (True, True))
        self.assertEqual(len(events['reload']['log_window']), 2, 'the restart window reaches the gate')
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
        before = events['stats-before-restart']['stats']
        self.assertGreater(before['same_step_rejects'], 0, 'the burst shared a step')
        self.assertGreater(before['evicted_coupled'], 0, 'the flood took checkpoints with their blocks')
        self.assertEqual(engine.registry.stats['same_step_rejects'], 0, 'the restart began a new registry')
        latched = dict((r['tag'], r) for r in records)[events['kill-switch']['latched_tag']]
        raw, _ = judge.raw_hit_per_attempt(latched)
        self.assertLessEqual(raw, latched['expected_raw_h'])
        self.assertGreater(judge.floor_chunk(events['kill-switch']['on_prompt']), latched['expected_raw_h'])

    def test_store_evicts_by_lru(self):
        profile = served_profile()
        profile['env']['QWEN_PREFIX_STORE_GIB'] = '0.5'
        self.engine = engine = FakeEngine(profile=profile)
        driver = driver_for(engine, 'lifecycle-store', strict=False)
        replay.scenario_lifecycle_store(driver)
        self.assertEqual(driver.pairs[-1]['case'], 'store-after')
        self.assertGreater(engine.registry.stats['evicted_lru'], 0)

    def tiny_engine(self, **faults):
        profile = served_profile()
        profile['env']['QWEN36_MAX_TOKENS_ALL_USERS'] = str(replay.TINY_POOL_TOKENS - 4 * BLOCK)
        return FakeEngine(profile=profile, **faults)

    def test_tiny_builds_a_dropped_grant_and_a_preemption(self):
        self.engine = engine = self.tiny_engine()
        self.assertEqual(engine.num_blocks * BLOCK, replay.TINY_POOL_TOKENS)
        driver = driver_for(engine, 'lifecycle-tiny', strict=False)
        replay.scenario_lifecycle_tiny(driver, pool_tokens=replay.TINY_POOL_TOKENS)
        grant, tiny = driver.events['tiny-grant'], driver.events['tiny']
        self.assertTrue(grant['ok'] and grant['filler_running'] and grant['waiting_gauge'])
        self.assertGreater(engine.dropped_hits, 0, 'a staged grant with Q > 0 was dropped')
        self.assertEqual(engine.registry.stats['dropped_hits'], engine.dropped_hits,
                         'the registry counts the dropped hits the fake saw')
        self.assertGreater(tiny['preemptions'], 0)
        records = resolved(driver, engine)
        waited = dict((r['tag'], r) for r in records)[grant['tag']]
        self.assertGreater(waited['markers']['q'], 0)
        self.assertTrue(any(judge.admissions(r) > 1 for r in records), 'a preempted request printed a second row')
        self.assertEqual(len([p for p in driver.pairs if p['case'] == 'tiny']), replay.TINY_PREEMPT_CONVERSATIONS)
        self.assertTrue(all(r['completion_tokens'] == tiny['max_tokens'] for r in records if r['case'] == 'tiny'))

    def test_tiny_on_the_wrong_pool_sends_nothing(self):
        self.engine = engine = FakeEngine()
        driver = driver_for(engine, 'lifecycle-tiny', strict=False)
        replay.scenario_lifecycle_tiny(driver, pool_tokens=engine.num_blocks * BLOCK)
        self.assertTrue(driver.events['tiny']['skipped'])
        self.assertEqual(driver.records, [])

    def test_timing_phases(self):
        self.engine = engine = FakeEngine()
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
