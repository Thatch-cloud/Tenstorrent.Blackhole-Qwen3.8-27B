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
- Counters for the metrics export (S5): grants, trim loss h-Q, KV hits without a checkpoint,
  checkpoints without KV (orphans), token mismatches, capture failures, restore and capture ms,
  bytes, pins.

Shared through a fixed sys.modules key (REGISTRY_KEY; precedent serving_lifecycle.PREFILL_GATE_KEY)
so the scheduler graft (the plugin package's copy of this module) and the model graft (whichever
copy the model tree imports) reach the same object without importing each other by path. Copies of
this file may be loaded under two module names; nothing here depends on class identity.

The model side's contract (G1 model graft, qwen_prefix_model_patch):
  registry = shared_registry()                  # or sys.modules[REGISTRY_KEY].registry
  grant = registry.grant_for(request_id)        # request ids arrive as prefill kwarg REQUEST_IDS_KWARG
  - start_pos > 0 and grant is None, or grant.q != start_pos: an assertion (never a silent rewrite).
  - grant.checkpoint.matches(row_tokens[0:grant.q]) as the last guard; restore grant.checkpoint.rec
    and .carry; run the chunk loop from grant.q // CHUNK; registry.note_restore(ms).
  - for pos in grant.capture_positions(): registry.capture(request_id, pos, rec, carry, nbytes, ms)
    right after the chunk loop drains and before the tail. A capture never raises (S7).
Pure python; imports nothing outside the standard library.
"""

import gc
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
# The prefill kwarg the runner patch (qwen_prefix_runner_patch) adds: the request id of each row.
REQUEST_IDS_KWARG = 'request_ids'
KILL_SWITCH_PATH = '/models/.qwen-c2/prefix-reuse.off'
KILL_SWITCH_POLL_S = 1.0
DEFAULT_STORE_GIB = 8.0
# Host checkpoint, both chips: 48 layers x (fp32 rec_state [1,24,128,128] + bf16 conv_carry
# [1,3,5120]) = 73.4 MiB per chip, about 154 MB (design section 2.0.3). The model passes the real size.
CHECKPOINT_NBYTES = 2 * 48 * (24 * 128 * 128 * 4 + 3 * 5120 * 2)

STAT_NAMES = (
    'attempts', 'staged', 'dropped_attempts', 'commit_mismatch', 'admissions', 'grants',
    'grant_tokens', 'trim_loss_tokens', 'kv_hit_without_checkpoint', 'orphans',
    'same_step_rejects', 'token_mismatches', 'unsalted_denied', 'killed_denied', 'publish_capped',
    'captures', 'capture_replaced', 'capture_kept_pinned', 'capture_failures',
    'capture_skipped_budget', 'capture_disabled', 'evicted_lru', 'evicted_coupled', 'dropped',
    'clears', 'freed_requests', 'restores', 'restore_ms', 'capture_ms',
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
        """Whether tokens (a sequence of ints) are exactly the ids this checkpoint was captured from."""
        return len(tokens) == self.pos and self.token_ids == token_array(tokens)


class Grant(object):
    """One admission's reuse record: the trimmed hit Q (0 on a miss), vLLM's raw hit h and the
    boundaries the model should capture. Staged on every admission attempt; committed only when
    the step's output admits the request at start_pos == Q. The model reads the committed one."""

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

    def capture_positions(self):
        return [pos for pos, _ in self.plan]

    def describe(self):
        return 'req=%s h=%d Q=%d plan=[%s]' % (self.req_id, self.h, self.q,
                                                ','.join(str(pos) for pos, _ in self.plan))


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
        self.stats = dict.fromkeys(STAT_NAMES, 0)
        self._owner = None

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
        self.stats['clears'] += 1

    def disable(self, reason):
        """Latch off for the life of the process: no new grants, captures or publishing."""
        if self.disabled is None:
            self.disabled = str(reason)
        self.clear()

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
            grant.request = None
            self.committed[req_id] = grant
            done.append(grant)
        self.staged.clear()
        return done

    def grant_for(self, req_id):
        return self.committed.get(req_id)

    def capture(self, req_id, pos, rec=None, carry=None, nbytes=None, ms=None):
        """Model side: store the state captured at boundary pos of this step's prefill of req_id.
        A failure skips the checkpoint and is counted; it never fails the request (S7)."""
        try:
            grant = self.committed.get(req_id)
            if grant is None:
                raise KeyError('no committed grant for %s' % req_id)
            keys = dict(grant.plan)
            if pos not in keys:
                raise KeyError('boundary %d is not planned for %s' % (pos, req_id))
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
                      disabled=self.disabled)
        return values


def shared_registry(environ=None):
    """The one registry of this process, parked under the fixed sys.modules key REGISTRY_KEY so the
    scheduler graft (the plugin tree) and the model graft (the model tree) reach it without
    importing each other by path (precedent: serving_lifecycle.PREFILL_GATE_KEY)."""
    holder = sys.modules.get(REGISTRY_KEY)
    if holder is None:
        holder = types.ModuleType(REGISTRY_KEY)
        holder.registry = PrefixRegistry(environ=environ)
        sys.modules[REGISTRY_KEY] = holder
    return holder.registry


def current_registry():
    """The shared registry if a scheduler has created one in this process, else None (warmup
    prefills run before the scheduler exists)."""
    holder = sys.modules.get(REGISTRY_KEY)
    return getattr(holder, 'registry', None) if holder is not None else None
