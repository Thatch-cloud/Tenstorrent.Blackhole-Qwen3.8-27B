"""Conversation prefix reuse on the TT general path: the checkpoint registry and scheduler graft.

This is the shared mechanism of the TT prefix-reuse design (revision 2, 2026-09-26; its section
2.0.1, items 1 and 2), cut so that the P0a probe (prefix_p0a_probe.py) drives it against the real
vLLM 0.25.1 scheduler in the serving image before any model code exists. The probe's checks 5-17 are its unit tests. G1
delivers it by an AST stage of the plugin's scheduler.py that calls maybe_install(self) at the end
of TTScheduler.__init__ (not a scheduler_cls subclass: the platform assigns scheduler_cls
unconditionally, vllm_tt_plugin/platform.py:1078-1081).

vLLM keeps owning the attention KV pages: it hashes, ref-counts and LRU-evicts them. What it cannot
know is that a TT prefill is exact only from a 2048-token chunk boundary with the GatedDeltaNet
(GDN) state the model saved there. Five wrappers on the live scheduler instance make every hit
exact and every grant current:

a. kv_cache_manager.get_computed_blocks (the trim). vLLM calls it on every attempt to admit a
   waiting request (v1/core/sched/scheduler.py:684-729). The wrapper cuts vLLM's 64-aligned hit h
   to Q, the largest 2048-multiple <= h such that the registry holds a checkpoint keyed by the
   block hash at Q, no block below Q was cached in this scheduler step (the TT analogue of vLLM's
   Mamba rule, single_type_kv_cache_manager.py:1196-1203: rows run in sequence, so a same-step
   block may not be written yet), and the checkpoint's stored token ids equal the request's
   (a mismatch lowers Q here instead of killing the engine in the model). It stages the grant
   and the captures the model should take; staging is idempotent and takes no pin. A request
   without cache_salt gets no hit (fail closed, S4).
b. kv_cache_manager.coordinator.cache_blocks (the cap). Both publish paths reach it: allocation
   (kv_cache_manager.py:452-462) and every output (async_scheduler.py:67-73 through
   kv_cache_manager.py:620-629). Only prompt blocks below floor2048(prompt) are hashed into the
   cache, i.e. blocks a full prefill chunk wrote; decode and prompt-tail blocks never are, because
   vLLM returns the first block cached for a hash (block_pool.py:47-72). An unsalted request
   publishes nothing. The block ids it newly caches feed the same-step rule.
c. schedule (the per-step commit, F2/S6). vLLM may discard an admission attempt after the hit was
   taken - a budget break, an allocation failure, TTScheduler's decode fallback. So grants staged
   during one schedule() call are committed only for requests the returned SchedulerOutput
   actually admits (new or resumed) at start_pos == Q, and pinned for that step; the rest drop.
d. kv_cache_manager.block_pool._maybe_evict_cached_block (eviction coupling, F8/S9). When vLLM
   evicts a block and its hash no longer maps to any cached block, the checkpoint keyed by that
   hash goes too, so the registry stays a subset of the cached KV.
e. reset_prefix_cache clears the registry; _free_request drops the request's grants and pins
   (this covers a waiting request that is aborted).

A runtime kill switch - the file KILL_SWITCH_PATH on the persistent /models mount, polled at most
once per second - turns grants and publishing off and clears the registry. It latches: turning
reuse back on needs an engine restart, so nothing published under a suspect build is served.

install() refuses to start unless async scheduling is off (blocks hashed at allocation in step t
are unwritten when step t+1 reads them), chunked prefill is off with a whole-prompt token budget
(Lever N's chunking gives start_pos > 0 a second meaning), and the coordinator is the unitary one
(HybridKVCacheCoordinator calls manager.cache_blocks directly and would bypass the cap,
kv_cache_coordinator.py:602-628). It also refuses prefix caching off, a KV spec block size other
than 64, a KV connector and speculative lookahead.

The model side (G1's model graft) reads grant_for(request_id) per prefill row, asserts
grant.q == start_pos, restores the checkpoint, runs the chunk loop from Q/2048 and calls capture()
for each planned boundary before the tail. A row with start_pos > 0 and no committed grant is an
assertion there, never a silent rewrite of shared blocks.
"""

import os
import sys
import time
from array import array
from collections import OrderedDict

CHUNK = 2048
BLOCK = 64
REGISTRY_KEY = '_qwen_prefix_registry'
KILL_SWITCH_PATH = '/models/.qwen-c2/prefix-reuse.off'
KILL_SWITCH_POLL_S = 1.0
DEFAULT_STORE_GIB = 8.0
# Host checkpoint, both chips: 48 layers x (fp32 rec_state [1,24,128,128] + bf16 conv_carry
# [1,3,5120]) = 73.4 MiB per chip (design section 2.0.3). The model passes the real size.
CHECKPOINT_NBYTES = 2 * 48 * (24 * 128 * 128 * 4 + 3 * 5120 * 2)

STAT_NAMES = (
    'attempts', 'staged', 'dropped_attempts', 'commit_mismatch', 'admissions', 'grants',
    'grant_tokens', 'trim_loss_tokens', 'kv_hit_without_checkpoint', 'orphans',
    'same_step_rejects', 'token_mismatches', 'unsalted_denied', 'killed_denied', 'publish_capped',
    'captures', 'capture_replaced', 'capture_kept_pinned', 'capture_failures',
    'capture_skipped_budget', 'evicted_lru', 'evicted_coupled', 'dropped', 'clears',
    'freed_requests', 'restore_ms', 'capture_ms',
)


class PrefixInstallError(RuntimeError):
    """The scheduler is not one prefix reuse is exact under; the engine must not start."""


def log(message, *values):
    try:
        sys.stderr.write('[PINDIAG] prefix: ' + (message % values if values else message) + '\n')
        sys.stderr.flush()
    except Exception:
        pass


def floor_chunk(tokens):
    return (int(tokens) // CHUNK) * CHUNK


def token_array(tokens):
    return array('q', tokens)


class Checkpoint(object):
    """GDN state at a 2048-token boundary, keyed by vLLM's block hash at that boundary."""

    __slots__ = ('key', 'pos', 'token_ids', 'rec', 'carry', 'nbytes', 'pins')

    def __init__(self, key, pos, token_ids, rec=None, carry=None, nbytes=0):
        self.key = key
        self.pos = pos
        self.token_ids = token_ids
        self.rec = rec
        self.carry = carry
        self.nbytes = int(nbytes)
        self.pins = 0

    def matches(self, tokens):
        return len(tokens) == self.pos and self.token_ids == token_array(tokens)


class Grant(object):
    """One admission's reuse record: the trimmed hit Q (0 on a miss) and the planned captures.

    Staged on every admission attempt; committed only when the scheduler output admits the
    request at start_pos == Q. The model reads the committed grant for its row."""

    __slots__ = ('req_id', 'q', 'h', 'key', 'checkpoint', 'plan', 'request', 'tokens', 'orphans')

    def __init__(self, req_id, q, h, key, checkpoint, plan, request, orphans=0):
        self.req_id = req_id
        self.q = q
        self.h = h
        self.key = key
        self.checkpoint = checkpoint
        self.plan = plan
        self.request = request
        self.tokens = None
        self.orphans = orphans

    def describe(self):
        return 'req=%s h=%d Q=%d plan=[%s]' % (self.req_id, self.h, self.q,
                                                ','.join(str(pos) for pos, _ in self.plan))


class PrefixRegistry(object):
    """The checkpoint LRU (bounded by bytes, pinned entries never evicted) and the grant ledger.

    One per engine process, shared by the scheduler graft and the model graft through
    shared_registry(); tests make their own."""

    def __init__(self, budget_bytes=None, environ=None):
        environ = os.environ if environ is None else environ
        if budget_bytes is None:
            budget_bytes = int(float(environ.get('QWEN_PREFIX_STORE_GIB', DEFAULT_STORE_GIB)) * (1 << 30))
        self.budget_bytes = int(budget_bytes)
        self.entries = OrderedDict()
        self.bytes = 0
        self.staged = {}
        self.committed = {}
        self.same_step_blocks = set()
        self.stats = dict.fromkeys(STAT_NAMES, 0)

    # -- checkpoints -------------------------------------------------------------------------
    def get(self, key):
        return self.entries.get(key)

    def put(self, key, pos, token_ids, rec=None, carry=None, nbytes=0):
        if pos <= 0 or pos % CHUNK:
            raise ValueError('a checkpoint is only exact at a %d-token boundary, not %d' % (CHUNK, pos))
        if len(token_ids) != pos:
            raise ValueError('checkpoint at %d carries %d token ids' % (pos, len(token_ids)))
        if nbytes > self.budget_bytes:
            self.stats['capture_skipped_budget'] += 1
            return None
        old = self.entries.get(key)
        if old is not None:
            if old.pins:
                # In use by this step's restore. A capture of the same prefix is the same state
                # (design section 2.0.4, L3), so the pinned one stays.
                self.entries.move_to_end(key)
                self.stats['capture_kept_pinned'] += 1
                return old
            self._remove(key)
            self.stats['capture_replaced'] += 1
        ids = token_ids if isinstance(token_ids, array) else token_array(token_ids)
        checkpoint = Checkpoint(key, pos, ids, rec, carry, nbytes)
        self.entries[key] = checkpoint
        self.bytes += checkpoint.nbytes
        self.stats['captures'] += 1
        self._enforce_budget()
        return checkpoint

    def touch(self, key):
        if key in self.entries:
            self.entries.move_to_end(key)

    def _remove(self, key):
        checkpoint = self.entries.pop(key, None)
        if checkpoint is not None:
            self.bytes -= checkpoint.nbytes
        return checkpoint

    def drop(self, key, reason='dropped'):
        checkpoint = self._remove(key)
        if checkpoint is not None:
            name = 'evicted_' + reason if ('evicted_' + reason) in self.stats else 'dropped'
            self.stats[name] += 1
        return checkpoint

    def _enforce_budget(self):
        if self.bytes <= self.budget_bytes:
            return
        for key in list(self.entries):
            if self.bytes <= self.budget_bytes:
                break
            if self.entries[key].pins:
                continue
            self._remove(key)
            self.stats['evicted_lru'] += 1

    def clear(self):
        """Drop every checkpoint. Staged and committed grants hold their own checkpoint reference
        and die at the next begin_step, so a clear inside schedule() (the kill switch) or between
        steps (reset_prefix_cache) never breaks an admission vLLM has already been handed."""
        self.entries.clear()
        self.bytes = 0
        self.stats['clears'] += 1

    # -- grants ------------------------------------------------------------------------------
    def begin_step(self):
        for grant in self.committed.values():
            if grant.checkpoint is not None and grant.checkpoint.pins > 0:
                grant.checkpoint.pins -= 1
        self.committed.clear()
        self.staged.clear()
        self.same_step_blocks.clear()
        self._enforce_budget()

    def stage(self, grant):
        self.staged[grant.req_id] = grant
        self.stats['staged'] += 1

    def unstage(self, req_id):
        self.staged.pop(req_id, None)

    def commit(self, admitted):
        """admitted: request id -> start_pos for every request the step's output admits."""
        done = []
        for req_id, grant in self.staged.items():
            if req_id not in admitted:
                self.stats['dropped_attempts'] += 1
                continue
            if admitted[req_id] != grant.q:
                self.stats['commit_mismatch'] += 1
                log('commit refused req=%s start_pos=%d Q=%d', req_id, admitted[req_id], grant.q)
                continue
            if grant.plan:
                grant.tokens = token_array(grant.request.all_token_ids[0:max(pos for pos, _ in grant.plan)])
            if grant.checkpoint is not None:
                grant.checkpoint.pins += 1
                self.touch(grant.key)
                self.stats['grants'] += 1
                self.stats['grant_tokens'] += grant.q
            self.stats['admissions'] += 1
            self.stats['trim_loss_tokens'] += grant.h - grant.q
            if floor_chunk(grant.h) > grant.q:
                self.stats['kv_hit_without_checkpoint'] += 1
            self.stats['orphans'] += grant.orphans
            self.committed[req_id] = grant
            done.append(grant)
        self.staged.clear()
        return done

    def grant_for(self, req_id):
        return self.committed.get(req_id)

    def capture(self, req_id, pos, rec=None, carry=None, nbytes=None):
        """Model side: store the state captured at boundary pos of this step's prefill of req_id.
        A failure skips the checkpoint and is counted; it never fails the request (S7)."""
        try:
            grant = self.committed.get(req_id)
            if grant is None:
                raise KeyError('no committed grant for %s' % req_id)
            keys = dict(grant.plan)
            if pos not in keys:
                raise KeyError('boundary %d is not planned for %s' % (pos, req_id))
            return self.put(keys[pos], pos, grant.tokens[0:pos], rec, carry,
                            CHECKPOINT_NBYTES if nbytes is None else nbytes)
        except Exception as error:
            self.stats['capture_failures'] += 1
            log('capture skipped req=%s pos=%s: %s', req_id, pos, error)
            return None

    def forget_request(self, req_id):
        found = self.staged.pop(req_id, None) is not None
        grant = self.committed.pop(req_id, None)
        if grant is not None:
            found = True
            if grant.checkpoint is not None and grant.checkpoint.pins > 0:
                grant.checkpoint.pins -= 1
        if found:
            self.stats['freed_requests'] += 1
        return found

    def pins(self):
        return sum(checkpoint.pins for checkpoint in self.entries.values())

    def snapshot(self):
        values = dict(self.stats)
        values.update(entries=len(self.entries), bytes=self.bytes, pins=self.pins(),
                      staged_now=len(self.staged), committed_now=len(self.committed))
        return values


def shared_registry(environ=None):
    """The one registry of this process, parked under a fixed sys.modules key so the scheduler
    graft (the plugin tree) and the model graft (the model tree) reach it without importing each
    other by path (precedent: serving_lifecycle.PREFILL_GATE_KEY)."""
    import types
    holder = sys.modules.get(REGISTRY_KEY)
    if holder is None:
        holder = types.ModuleType(REGISTRY_KEY)
        holder.registry = PrefixRegistry(environ=environ)
        sys.modules[REGISTRY_KEY] = holder
    return holder.registry


def install_problems(scheduler):
    """Every reason prefix reuse would not be exact on this scheduler (F5, F6, F7)."""
    from vllm.v1.core.kv_cache_coordinator import UnitaryKVCacheCoordinator

    problems = []
    config = scheduler.scheduler_config
    max_model_len = scheduler.max_model_len
    if getattr(config, 'async_scheduling', False):
        problems.append('async scheduling is on: blocks hashed at allocation in step t are unwritten '
                        'when step t+1 reads them')
    if getattr(config, 'enable_chunked_prefill', False):
        problems.append('chunked prefill is on: start_pos > 0 would also mean "continue my own '
                        'suspended scratch" (Lever N)')
    if config.max_num_batched_tokens < max_model_len:
        problems.append('max_num_batched_tokens %d < max_model_len %d: a prefill could be split'
                        % (config.max_num_batched_tokens, max_model_len))
    if not getattr(scheduler.cache_config, 'enable_prefix_caching', False):
        problems.append('prefix caching is off')
    manager = scheduler.kv_cache_manager
    coordinator = manager.coordinator
    if not isinstance(coordinator, UnitaryKVCacheCoordinator):
        problems.append('the KV coordinator is %s, not UnitaryKVCacheCoordinator: the cap would '
                        'be bypassed' % type(coordinator).__name__)
    groups = scheduler.kv_cache_config.kv_cache_groups
    if len(groups) != 1:
        problems.append('%d KV cache groups, not one' % len(groups))
    else:
        block_size = groups[0].kv_cache_spec.block_size
        if block_size != BLOCK:
            problems.append('KV block size %d, not %d: the page tables and chunk arithmetic assume %d'
                            % (block_size, BLOCK, BLOCK))
    if getattr(scheduler, 'block_size', BLOCK) != BLOCK:
        problems.append('scheduler block size %s, not %d' % (scheduler.block_size, BLOCK))
    if getattr(scheduler, 'has_mamba_layers', False):
        problems.append('the KV config has Mamba layers: vLLM would run its own align-mode split')
    if getattr(scheduler, 'connector', None) is not None:
        problems.append('a KV connector is configured: external tokens would move start_pos past Q')
    if getattr(scheduler, 'num_lookahead_tokens', 0):
        problems.append('speculative lookahead is on')
    return problems


def maybe_install(scheduler, environ=None):
    """The G1 hook: TTScheduler.__init__ calls this last. Off unless QWEN_PREFIX_REUSE=1."""
    environ = os.environ if environ is None else environ
    if environ.get('QWEN_PREFIX_REUSE') != '1':
        return None
    return install(scheduler)


def install(scheduler, registry=None, kill_switch_path=KILL_SWITCH_PATH, poll_s=KILL_SWITCH_POLL_S,
            clock=time.monotonic, logger=log):
    existing = getattr(scheduler, '_qwen_prefix', None)
    if existing is not None:
        return existing
    problems = install_problems(scheduler)
    if problems:
        raise PrefixInstallError('prefix reuse refused: ' + '; '.join(problems))
    graft = SchedulerGraft(scheduler, registry if registry is not None else shared_registry(),
                           kill_switch_path, poll_s, clock, logger)
    graft.wrap()
    scheduler._qwen_prefix = graft
    return graft


class SchedulerGraft(object):
    def __init__(self, scheduler, registry, kill_switch_path, poll_s, clock, logger):
        self.scheduler = scheduler
        self.registry = registry
        self.manager = scheduler.kv_cache_manager
        self.coordinator = self.manager.coordinator
        self.block_pool = self.manager.block_pool
        self.single = self.coordinator.single_type_managers[0]
        self.kill_switch_path = kill_switch_path
        self.poll_s = poll_s
        self.clock = clock
        self.log = logger
        self.killed = False
        self._last_poll = None
        self.original = {}

    # -- kill switch -------------------------------------------------------------------------
    def kill_switch_engaged(self):
        if self.killed:
            return True
        now = self.clock()
        if self._last_poll is not None and now - self._last_poll < self.poll_s:
            return False
        self._last_poll = now
        if self.kill_switch_path and os.path.exists(self.kill_switch_path):
            self.killed = True
            # Checkpoints go. Grants already staged in this schedule() call keep their own
            # reference: vLLM holds their trimmed hits, so they must still commit and restore.
            self.registry.clear()
            self.log('kill switch %s present: no grants and no publishing until the engine restarts; '
                     'registry cleared', self.kill_switch_path)
        return self.killed

    # -- a. the trim -------------------------------------------------------------------------
    def plan(self, request, h, q):
        prompt_boundary = floor_chunk(request.num_prompt_tokens)
        gap_boundary = floor_chunk(h)
        positions = set()
        if prompt_boundary > q:
            positions.add(prompt_boundary)
        if gap_boundary - q >= CHUNK:
            positions.add(gap_boundary)
        hashes = request.block_hashes
        return [(pos, hashes[pos // BLOCK - 1]) for pos in sorted(positions) if pos // BLOCK <= len(hashes)]

    def trim(self, request, blocks, h):
        registry = self.registry
        stats = registry.stats
        stats['attempts'] += 1
        empty = self.manager.empty_kv_cache_blocks
        if not request.cache_salt:
            registry.unstage(request.request_id)
            if h:
                stats['unsalted_denied'] += 1
            return empty, 0
        if self.kill_switch_engaged():
            registry.unstage(request.request_id)
            if h:
                stats['killed_denied'] += 1
            return empty, 0
        group = blocks.blocks[0] if h else ()
        limit = h // BLOCK
        same_step = registry.same_step_blocks
        first_same = limit
        if same_step:
            for index in range(limit):
                if group[index].block_id in same_step:
                    first_same = index
                    break
        hashes = request.block_hashes
        q, key, checkpoint = 0, None, None
        rejected_same_step = False
        k = h // CHUNK
        while k > 0:
            candidate = k * CHUNK
            count = candidate // BLOCK
            k -= 1
            if count > first_same:
                rejected_same_step = True
                continue
            entry = registry.get(hashes[count - 1])
            if entry is None:
                continue
            if entry.pos != candidate or not entry.matches(request.all_token_ids[0:candidate]):
                stats['token_mismatches'] += 1
                continue
            q, key, checkpoint = candidate, hashes[count - 1], entry
            break
        if rejected_same_step:
            stats['same_step_rejects'] += 1
        orphans = 0
        for boundary in range(floor_chunk(h) + CHUNK, floor_chunk(request.num_tokens - 1) + 1, CHUNK):
            if boundary // BLOCK <= len(hashes) and registry.get(hashes[boundary // BLOCK - 1]) is not None:
                orphans += 1
        plan = self.plan(request, h, q)
        if q or plan:
            registry.stage(Grant(request.request_id, q, h, key, checkpoint, plan, request, orphans))
        else:
            registry.unstage(request.request_id)
        if q == 0:
            return empty, 0
        return self.manager.create_kv_cache_blocks((list(group[0:q // BLOCK]),)), q

    # -- b. the cap --------------------------------------------------------------------------
    def cap(self, request, num_computed_tokens):
        if not request.cache_salt or self.killed:
            limit = 0
        else:
            limit = floor_chunk(request.num_prompt_tokens)
        tokens = min(num_computed_tokens, limit)
        if tokens < num_computed_tokens:
            self.registry.stats['publish_capped'] += 1
        request_id = request.request_id
        before = self.single.num_cached_block.get(request_id, 0)
        self.original['cache_blocks'](request, tokens)
        after = self.single.num_cached_block.get(request_id, 0)
        if after > before:
            owned = self.single.req_to_blocks.get(request_id) or ()
            self.registry.same_step_blocks.update(block.block_id for block in owned[before:after])

    # -- c. the per-step commit --------------------------------------------------------------
    def commit(self, output):
        admitted = {}
        for data in output.scheduled_new_reqs:
            admitted[data.req_id] = data.num_computed_tokens
        cached = output.scheduled_cached_reqs
        resumed = cached.resumed_req_ids
        if resumed:
            for index, req_id in enumerate(cached.req_ids):
                if req_id in resumed:
                    admitted[req_id] = cached.num_computed_tokens[index]
        for grant in self.registry.commit(admitted):
            self.log('grant %s', grant.describe())

    # -- d. eviction coupling ----------------------------------------------------------------
    def evict(self, block):
        pool = self.block_pool
        keys = []
        if block.block_hash is not None:
            keys.append(block.block_hash)
        extra = pool.cached_block_hashes_by_block.get(block.block_id)
        if extra:
            keys.extend(extra)
        evicted = self.original['_maybe_evict_cached_block'](block)
        if evicted and keys and self.registry.entries:
            for key in keys:
                if pool.cached_block_hash_to_block.get_one_block(key) is None:
                    self.registry.drop(self.get_block_hash(key), 'coupled')
        return evicted

    # -- installation ------------------------------------------------------------------------
    def wrap(self):
        from vllm.v1.core.kv_cache_utils import get_block_hash

        self.get_block_hash = get_block_hash
        scheduler, manager, coordinator, pool = self.scheduler, self.manager, self.coordinator, self.block_pool
        registry = self.registry
        original = self.original
        original['get_computed_blocks'] = manager.get_computed_blocks
        original['cache_blocks'] = coordinator.cache_blocks
        original['schedule'] = scheduler.schedule
        original['_maybe_evict_cached_block'] = pool._maybe_evict_cached_block
        original['reset_prefix_cache'] = scheduler.reset_prefix_cache
        original['_free_request'] = scheduler._free_request

        def get_computed_blocks(request):
            blocks, h = original['get_computed_blocks'](request)
            return self.trim(request, blocks, h)

        def cache_blocks(request, num_computed_tokens):
            return self.cap(request, num_computed_tokens)

        def schedule(*args, **kwargs):
            registry.begin_step()
            output = original['schedule'](*args, **kwargs)
            self.commit(output)
            return output

        def _maybe_evict_cached_block(block):
            return self.evict(block)

        def reset_prefix_cache(*args, **kwargs):
            result = original['reset_prefix_cache'](*args, **kwargs)
            registry.clear()
            return result

        def _free_request(request, *args, **kwargs):
            registry.forget_request(request.request_id)
            return original['_free_request'](request, *args, **kwargs)

        manager.get_computed_blocks = get_computed_blocks
        coordinator.cache_blocks = cache_blocks
        scheduler.schedule = schedule
        pool._maybe_evict_cached_block = _maybe_evict_cached_block
        scheduler.reset_prefix_cache = reset_prefix_cache
        scheduler._free_request = _free_request

        spec = scheduler.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        module = sys.modules.get(type(scheduler).__module__)
        self.log('install scheduler=%s.%s plugin=%s coordinator=%s block_size=%d blocks=%d kv_spec_dtype=%s '
                 'QWEN_SDPA_BF8=%s store_gib=%.1f kill_switch=%s',
                 type(scheduler).__module__, type(scheduler).__name__, getattr(module, '__file__', '?'),
                 type(self.coordinator).__name__, spec.block_size, scheduler.kv_cache_config.num_blocks,
                 getattr(spec, 'dtype', '?'), os.environ.get('QWEN_SDPA_BF8', 'unset'),
                 registry.budget_bytes / float(1 << 30), self.kill_switch_path)

    def original_get_computed_blocks(self, request):
        """vLLM's own answer, untrimmed (for probes and audits)."""
        return self.original['get_computed_blocks'](request)
