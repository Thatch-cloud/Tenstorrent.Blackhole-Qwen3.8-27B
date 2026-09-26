"""Conversation prefix reuse on the TT general path (G1): the checkpoint registry.

The TT prefix-reuse design (revision 2, 2026-09-26), section 2.0.1 item 1. One registry per engine
process holds what vLLM cannot know about a hit: the GatedDeltaNet (GDN) state the model saved at a
2048-token prefill chunk boundary, and which admission may restore which checkpoint in this step.

- Checkpoints: an LRU keyed by vLLM's block hash at the boundary (the hash of block Q/64-1), each
  holding pos, the token ids [0, pos) it was captured from, the model's host tensors (rec_state and
  conv_carry for both chips) and their size. It is bounded by bytes (QWEN_PREFIX_STORE_GIB, default
  8 GiB, about 55 checkpoints of 154 MB); a pinned entry is never evicted.
- Grants: staged on every admission attempt inside one schedule() call, committed only for the
  requests the step's SchedulerOutput admits at start_pos == Q (F2, S6), pinned for that one step.
  The model reads grant_for(request_id) per prefill row.
- Kill switch: KillSwitch polls a flag file at most once a second; once engaged the registry is
  disabled for the life of the process (no grants, no captures, no publishing). Turning reuse back
  on needs an engine restart, so nothing published under a suspect build is ever served (S7).
- Counters (S5): grants, trim loss h-Q, KV hits without a checkpoint, distinct checkpoints found
  without their KV (orphans), token mismatches, capture failures and refusals, restore and capture
  ms, bytes, pins. StatsExport writes them from the scheduler process, rate-limited, as a
  "[PINDIAG] prefix: stats {...}" line and a JSON file on tmpfs (QWEN_PREFIX_STATS_PATH).

Shared through a fixed sys.modules key (REGISTRY_KEY; precedent serving_lifecycle.PREFILL_GATE_KEY)
so the scheduler graft (the plugin package's copy of this module) and the model graft (whichever
copy the model tree imports) reach the same object without importing each other by path. Copies of
this file may be loaded under two module names; nothing here depends on class identity.

The model side's contract (G1 model graft, qwen_prefix_model_patch). A prefill row covers tokens
[start_pos, L), L = the row's num_tokens; its chunk loop runs chunks start_pos/2048 .. L//2048 - 1
and DRAINS at floor2048(L), then the tail runs.
  registry = shared_registry()                  # or sys.modules[REGISTRY_KEY].registry
  grant = registry.grant_for(request_id)        # request ids arrive as prefill kwarg REQUEST_IDS_KWARG
  - start_pos > 0 and grant is None, or grant.q != start_pos: an assertion (never a silent rewrite).
  - grant.checkpoint.matches(row_tokens[0:grant.q]) as the last guard; restore grant.checkpoint.rec
    and .carry; run the chunk loop from grant.q // CHUNK; registry.note_restore(ms).
  - CAPTURES. A capture at pos is the GDN state after EXACTLY pos tokens: taken once chunk
    pos/2048 - 1 completes and before chunk pos/2048 starts - mid-loop whenever pos < floor2048(L)
    (grant.drain), and before the tail only when pos == grant.drain. For every pos in
    grant.capture_positions():
        registry.capture(request_id, pos, rec, carry, nbytes, ms, loop_pos=<tokens the loop has run>)
    and capture() refuses (counted as capture_wrong_position, never stored) unless loop_pos == pos,
    so a state taken at the wrong point can never be filed under another boundary's key (a later
    hit would restore it: silently inexact). A capture never raises (S7).
  - The scheduler plans only drain captures (pos == grant.drain) until the model declares it takes
    mid-loop captures: registry.enable_mid_loop_capture(), once, before serving. Only then does
    the plan carry the gap boundary floor2048(h) below the drain (the shared-prefix / sibling
    sub-agent case) and a resumed (preempted) request's prompt boundary below its drain. The model
    warms up before vLLM builds the scheduler (and so before any registry exists), so it may declare
    on the holder instead: sys.modules[REGISTRY_KEY].mid_loop_capture = True (creating the holder
    as a bare module when absent); shared_registry() honours that when it creates the registry.
  - The model also counts program_growth (a row whose program cache grew after warmup, F3) in
    registry.stats, and reports restore time through note_restore(ms).
Pure python; imports nothing outside the standard library.
"""

import gc
import itertools
import json
import os
import sys
import time
import types
import weakref
from array import array
from collections import OrderedDict

CHUNK = 2048
BLOCK = 64
REGISTRY_KEY = '_qwen_prefix_registry'
ENV_REUSE = 'QWEN_PREFIX_REUSE'
ENV_STORE_GIB = 'QWEN_PREFIX_STORE_GIB'
ENV_STATS_PATH = 'QWEN_PREFIX_STATS_PATH'
ENV_STATS_S = 'QWEN_PREFIX_STATS_S'
# The prefill kwarg the runner patch (qwen_prefix_runner_patch) adds: the request id of each row.
REQUEST_IDS_KWARG = 'request_ids'
KILL_SWITCH_PATH = '/models/.qwen-c2/prefix-reuse.off'
KILL_SWITCH_POLL_S = 1.0
DEFAULT_STORE_GIB = 8.0
# The stats export: /tmp is a container-private tmpfs in the node agent's container shape
# (--read-only --tmpfs /tmp, as the C2 smoke reproduces it) and must be writable there anyway
# (VLLM_CACHE_ROOT=/tmp/vllm-cache). /dev/shm is not used: under --ipc=host it is the host's, and
# two engines would share one file. Empty QWEN_PREFIX_STATS_PATH: the log line only.
DEFAULT_STATS_PATH = '/tmp/qwen-prefix-stats.json'
DEFAULT_STATS_S = 30.0
# Host checkpoint, both chips: 48 layers x (fp32 rec_state [1,24,128,128] + bf16 conv_carry
# [1,3,5120]) = 73.4 MiB per chip, about 154 MB (design section 2.0.3). The model passes the real size.
CHECKPOINT_NBYTES = 2 * 48 * (24 * 128 * 128 * 4 + 3 * 5120 * 2)

STAT_NAMES = (
    'attempts', 'staged', 'dropped_attempts', 'commit_mismatch', 'admissions', 'grants',
    'grant_tokens', 'trim_loss_tokens', 'kv_hit_without_checkpoint', 'orphans',
    'same_step_rejects', 'token_checks', 'token_mismatches', 'unsalted_denied', 'session_denied',
    'killed_denied', 'publish_capped', 'captures', 'capture_replaced', 'capture_kept_pinned',
    'capture_failures', 'capture_wrong_position', 'capture_skipped_budget', 'capture_disabled',
    'mid_loop_unplanned', 'evicted_lru', 'evicted_coupled', 'dropped', 'clears', 'reset_kept',
    'freed_requests', 'restores', 'restore_ms', 'capture_ms', 'dropped_hits', 'program_growth',
)


def log(message, *values):
    """A [PINDIAG] prefix marker on stderr (the engine's log). Never raises."""
    try:
        sys.stderr.write('[PINDIAG] prefix: ' + (message % values if values else message) + '\n')
        sys.stderr.flush()
    except Exception:
        pass


def reuse_enabled(environ=None):
    environ = os.environ if environ is None else environ
    return environ.get(ENV_REUSE) == '1'


def floor_chunk(tokens):
    return (int(tokens) // CHUNK) * CHUNK


def token_array(tokens):
    return array('q', tokens)


def store_budget_bytes(environ=None):
    """QWEN_PREFIX_STORE_GIB as bytes. A value that is not a non-negative number refuses to start."""
    environ = os.environ if environ is None else environ
    raw = environ.get(ENV_STORE_GIB)
    if raw is None or raw == '':
        return int(DEFAULT_STORE_GIB * (1 << 30))
    try:
        gib = float(raw)
    except ValueError:
        raise ValueError('%s=%r is not a number of GiB' % (ENV_STORE_GIB, raw))
    if not gib >= 0.0:
        raise ValueError('%s=%r must be >= 0' % (ENV_STORE_GIB, raw))
    return int(gib * (1 << 30))


class Checkpoint(object):
    """GDN state at a 2048-token boundary, keyed by vLLM's block hash at that boundary. serial is
    unique per process, so a remembered token check can never vouch for a later entry."""

    __slots__ = ('key', 'pos', 'token_ids', 'rec', 'carry', 'nbytes', 'pins', 'serial')

    def __init__(self, key, pos, token_ids, rec=None, carry=None, nbytes=0, serial=0):
        self.key = key
        self.pos = pos
        self.token_ids = token_ids
        self.rec = rec
        self.carry = carry
        self.nbytes = int(nbytes)
        self.pins = 0
        self.serial = serial

    def matches(self, tokens):
        """Whether tokens (a sequence of ints) are exactly the ids this checkpoint was captured from."""
        return len(tokens) == self.pos and self.token_ids == token_array(tokens)


class Grant(object):
    """One admission's reuse record: the trimmed hit Q (0 on a miss), vLLM's raw hit h, where the
    row's chunk loop drains (floor2048 of the tokens it prefills) and the boundaries the model
    should capture. Staged on every admission attempt; committed only when the step's output admits
    the request at start_pos == Q. The model reads the committed one."""

    __slots__ = ('req_id', 'q', 'h', 'key', 'checkpoint', 'plan', 'request', 'tokens', 'drain', 'unplanned')

    def __init__(self, req_id, q, h, key, checkpoint, plan, request, drain=None, unplanned=()):
        self.req_id = req_id
        self.q = q
        self.h = h
        self.key = key
        self.checkpoint = checkpoint
        self.plan = plan
        self.request = request
        self.tokens = None
        self.drain = max([q] + [pos for pos, _ in plan]) if drain is None else drain
        self.unplanned = tuple(unplanned)

    def capture_positions(self):
        return sorted(pos for pos, _ in self.plan)

    def mid_loop_positions(self):
        """Planned captures the chunk loop must take before it drains."""
        return [pos for pos in self.capture_positions() if pos < self.drain]

    def describe(self):
        return 'req=%s h=%d Q=%d drain=%d plan=[%s]' % (self.req_id, self.h, self.q, self.drain,
                                                        ','.join(str(pos) for pos in self.capture_positions()))


class KillSwitch(object):
    """The runtime kill switch (S7): a flag file, polled at most once per poll_s, that latches.

    The operator writes ~/hf-cache/hub/.qwen-c2/prefix-reuse.off on the host; the image mounts
    that directory at /models (design section 2.0.1 item 2, "runtime kill switch")."""

    def __init__(self, path=KILL_SWITCH_PATH, poll_s=KILL_SWITCH_POLL_S, clock=time.monotonic,
                 exists=os.path.exists):
        self.path = path
        self.poll_s = float(poll_s)
        self.clock = clock
        self.exists = exists
        self.engaged = False
        self._last_poll = None

    def poll(self):
        """True exactly once: on the poll that first sees the flag. Afterwards engaged stays True."""
        if self.engaged or not self.path:
            return False
        now = self.clock()
        if self._last_poll is not None and now - self._last_poll < self.poll_s:
            return False
        self._last_poll = now
        try:
            present = bool(self.exists(self.path))
        except Exception:
            present = False
        if present:
            self.engaged = True
        return present


class PrefixRegistry(object):
    """The checkpoint LRU (bounded by bytes, pinned entries never evicted) and the grant ledger."""

    def __init__(self, budget_bytes=None, environ=None):
        if budget_bytes is None:
            budget_bytes = store_budget_bytes(environ)
        self.budget_bytes = int(budget_bytes)
        self.entries = OrderedDict()
        self.bytes = 0
        self.staged = {}
        self.committed = {}
        self.same_step_blocks = set()
        self.disabled = None
        # Off until the model graft declares it takes captures inside its chunk loop.
        self.mid_loop_capture = False
        # Distinct checkpoint keys seen resident while their KV chain was broken below them.
        self.orphan_keys = set()
        # req_id -> {candidate Q: (checkpoint serial, tokens match)}: a waiting request is
        # re-attempted every step it stays blocked, and its token prefix never changes.
        self.token_checks = {}
        self.stats = dict.fromkeys(STAT_NAMES, 0)
        self._owner = None
        self._serials = itertools.count(1)

    # -- the scheduler that owns this registry --------------------------------------------------
    def _live_owner(self):
        return self._owner() if self._owner is not None else None

    def bind(self, owner):
        """Bind the one scheduler this registry serves. A second live scheduler (lane mode builds
        one TTScheduler per lane in one process) would share one grant ledger: refused. A dead
        owner (an engine rebuilt in the same process) is replaced, and the registry cleared."""
        current = self._live_owner()
        if current is owner:
            return
        if current is not None:
            # The graft and its scheduler reference each other; a dead engine is only a cycle.
            del current
            gc.collect()
            current = self._live_owner()
            if current is not None:
                raise RuntimeError('the prefix registry already serves another live scheduler (%s): '
                                   'one scheduler per process' % type(current).__name__)
        if self._owner is not None:
            self.clear()
            self.staged.clear()
            self.committed.clear()
            self.same_step_blocks.clear()
        self._owner = weakref.ref(owner)

    def enable_mid_loop_capture(self):
        """The model graft's declaration that it captures at pos after exactly pos tokens even when
        pos is below its loop's drain point. Until it is made, only drain captures are planned."""
        if not self.mid_loop_capture:
            self.mid_loop_capture = True
            log('model declares mid-loop captures: gap and resumed-prompt boundaries are planned')

    # -- checkpoints ---------------------------------------------------------------------------
    def get(self, key):
        return self.entries.get(key)

    def put(self, key, pos, token_ids, rec=None, carry=None, nbytes=0):
        if pos <= 0 or pos % CHUNK:
            raise ValueError('a checkpoint is only exact at a %d-token boundary, not %d' % (CHUNK, pos))
        if len(token_ids) != pos:
            raise ValueError('checkpoint at %d carries %d token ids' % (pos, len(token_ids)))
        if self.disabled is not None:
            self.stats['capture_disabled'] += 1
            return None
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
        checkpoint = Checkpoint(key, pos, ids, rec, carry, nbytes, next(self._serials))
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
            self.orphan_keys.discard(key)
        return checkpoint

    def drop(self, key, reason='dropped'):
        checkpoint = self._remove(key)
        if checkpoint is not None:
            name = 'evicted_' + reason
            self.stats[name if name in self.stats else 'dropped'] += 1
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
        self.orphan_keys.clear()
        self.token_checks.clear()
        self.stats['clears'] += 1

    def disable(self, reason):
        """Latch off for the life of the process: no new grants, captures or publishing."""
        if self.disabled is None:
            self.disabled = str(reason)
        self.clear()

    # -- the trim's helpers ---------------------------------------------------------------------
    def tokens_match(self, req_id, candidate, entry, tokens):
        """entry.matches(tokens()[0:candidate]), remembered per (request, candidate, checkpoint
        serial): the ~5 ms array build of a 62k-token check runs once per request and checkpoint,
        not on every step a blocked request is re-attempted. tokens is a callable returning the
        request's all_token_ids; a request's prefix never changes (streaming sessions, whose prompt
        is rewritten, never reach the trim)."""
        memo = self.token_checks.setdefault(req_id, {})
        seen = memo.get(candidate)
        if seen is not None and seen[0] == entry.serial:
            return seen[1]
        self.stats['token_checks'] += 1
        verdict = entry.pos == candidate and entry.matches(tokens()[0:candidate])
        if not verdict:
            self.stats['token_mismatches'] += 1
        memo[candidate] = (entry.serial, verdict)
        return verdict

    def note_orphan(self, key):
        """A resident checkpoint whose KV chain is broken below it (it cannot be granted until the
        missing block is re-cached; the LRU ages it out). Counted once per key."""
        if key in self.entries and key not in self.orphan_keys:
            self.orphan_keys.add(key)
            self.stats['orphans'] += 1

    # -- grants --------------------------------------------------------------------------------
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
        """admitted: request id -> start_pos for every request the step's output admits (new or
        resumed). Returns the grants committed; every other staged grant is dropped."""
        done = []
        for req_id, grant in self.staged.items():
            if req_id not in admitted:
                self.stats['dropped_attempts'] += 1
                if grant.q:
                    # A hit vLLM took and then discarded (an allocation failure after the grant, a
                    # budget break, the TT decode fallback; F2): the next attempt re-stages it.
                    self.stats['dropped_hits'] += 1
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
                self.orphan_keys.discard(grant.key)
                self.stats['grants'] += 1
                self.stats['grant_tokens'] += grant.q
            self.stats['admissions'] += 1
            self.stats['trim_loss_tokens'] += grant.h - grant.q
            self.stats['mid_loop_unplanned'] += len(grant.unplanned)
            if floor_chunk(grant.h) > grant.q:
                self.stats['kv_hit_without_checkpoint'] += 1
            grant.request = None
            self.committed[req_id] = grant
            done.append(grant)
        self.staged.clear()
        return done

    def grant_for(self, req_id):
        return self.committed.get(req_id)

    def capture(self, req_id, pos, rec=None, carry=None, nbytes=None, ms=None, loop_pos=None):
        """Model side: store the GDN state after exactly pos tokens of this step's prefill of req_id.
        loop_pos is how many tokens the model's chunk loop had run when it took rec and carry; the
        capture is refused unless it is pos (see the module's contract). A failure or refusal
        skips the checkpoint and is counted; it never fails the request (S7)."""
        try:
            grant = self.committed.get(req_id)
            if grant is None:
                raise KeyError('no committed grant for %s' % req_id)
            keys = dict(grant.plan)
            if pos not in keys:
                raise KeyError('boundary %d is not planned for %s' % (pos, req_id))
            if loop_pos != pos:
                self.stats['capture_wrong_position'] += 1
                raise ValueError('the state was taken after %r tokens, not %d: refused (it would be '
                                 'restored as the state at %d)' % (loop_pos, pos, pos))
            if ms is not None:
                self.stats['capture_ms'] += float(ms)
            return self.put(keys[pos], pos, grant.tokens[0:pos], rec, carry,
                            CHECKPOINT_NBYTES if nbytes is None else nbytes)
        except Exception as error:
            self.stats['capture_failures'] += 1
            log('capture skipped req=%s pos=%s: %s', req_id, pos, error)
            return None

    def note_restore(self, ms):
        self.stats['restores'] += 1
        self.stats['restore_ms'] += float(ms)

    def forget_request(self, req_id):
        self.token_checks.pop(req_id, None)
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
        values.update(entries=len(self.entries), bytes=self.bytes, budget_bytes=self.budget_bytes,
                      pins=self.pins(), staged_now=len(self.staged), committed_now=len(self.committed),
                      orphans_now=len(self.orphan_keys), mid_loop_capture=self.mid_loop_capture,
                      disabled=self.disabled)
        return values


class StatsExport(object):
    """The registry's counters out of the EngineCore process (S5): at most once per interval_s, a
    "[PINDIAG] prefix: stats {json}" line when they changed, and the same JSON written atomically
    to path (tmpfs) for whatever reads it next to the engine. Never raises; a write failure is
    logged once and the log line continues."""

    def __init__(self, path=None, interval_s=None, clock=time.monotonic, logger=log, environ=None):
        environ = os.environ if environ is None else environ
        self.path = environ.get(ENV_STATS_PATH, DEFAULT_STATS_PATH) if path is None else path
        if interval_s is None:
            try:
                interval_s = float(environ.get(ENV_STATS_S) or DEFAULT_STATS_S)
            except ValueError:
                interval_s = DEFAULT_STATS_S
        self.interval_s = max(0.0, float(interval_s))
        self.clock = clock
        self.log = logger
        self.last_time = None
        self.last_values = None
        self.write_failed = False
        self.exports = 0

    def maybe_export(self, registry, force=False):
        try:
            now = self.clock()
            if not force and self.last_time is not None and now - self.last_time < self.interval_s:
                return False
            self.last_time = now
            values = registry.snapshot()
            if values == self.last_values and not force:
                return False
            self.last_values = values
            text = json.dumps(values, sort_keys=True, separators=(',', ':'))
            self.log('stats %s', text)
            self.exports += 1
            if self.path:
                self._write(dict(values, pid=os.getpid(), time=time.time()))
            return True
        except Exception as error:
            if not self.write_failed:
                self.write_failed = True
                self.log('stats export failed: %s: %s', type(error).__name__, error)
            return False

    def _write(self, values):
        try:
            temporary = '%s.%d.tmp' % (self.path, os.getpid())
            with open(temporary, 'w', encoding='utf-8') as handle:
                json.dump(values, handle, sort_keys=True)
            os.replace(temporary, self.path)
        except Exception as error:
            if not self.write_failed:
                self.write_failed = True
                self.log('stats file %s not written (the log line continues): %s: %s',
                         self.path, type(error).__name__, error)


def shared_registry(environ=None):
    """The one registry of this process, parked under the fixed sys.modules key REGISTRY_KEY so the
    scheduler graft (the plugin tree) and the model graft (the model tree) reach it without
    importing each other by path (precedent: serving_lifecycle.PREFILL_GATE_KEY)."""
    holder = sys.modules.get(REGISTRY_KEY)
    if holder is None:
        holder = types.ModuleType(REGISTRY_KEY)
        sys.modules[REGISTRY_KEY] = holder
    registry = getattr(holder, 'registry', None)
    if registry is None:
        registry = PrefixRegistry(environ=environ)
        # The model graft warms up before vLLM builds the scheduler, so it declares mid-loop
        # captures on the holder (see the module's contract) before this registry exists.
        if getattr(holder, 'mid_loop_capture', False) is True:
            registry.enable_mid_loop_capture()
        holder.registry = registry
    return registry


def current_registry():
    """The shared registry if a scheduler has created one in this process, else None (warmup
    prefills run before the scheduler exists)."""
    holder = sys.modules.get(REGISTRY_KEY)
    return getattr(holder, 'registry', None) if holder is not None else None
