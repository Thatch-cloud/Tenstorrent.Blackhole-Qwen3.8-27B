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
  - A SPLIT prefill (Lever N, docs/lever-n-prefix-merged-route.md): a grant lives for one step, but a prompt the cap splits runs
    over several. commit() records such a request's capture plan as an IN-FLIGHT PLAN (inflight[req_id]) and capture() accepts a
    planned position from it in every later step, until the request's final step has run (note_scheduled marks it; begin_step drops
    it) or it is freed, killed or disabled. A grant that covers the rest of the prompt in one step never creates one, so a profile
    without Lever N behaves exactly as it did.
  - The model also counts program_growth (a row whose program cache grew after warmup, F3) in
    registry.stats, and reports restore time through note_restore(ms).
- Exact resume P0 (docs/tp4-exact-resume.md 5.2; QWEN_PREFIX_STICKY_UNDROP, QWEN_PREFIX_WINDOW_CARRY, default off): capture() remembers
  which checkpoints each request stored (captured_by), and the fast path, once the prefill has finished, files the drafter window rows
  below the prompt boundary on that checkpoint (attach_window: a Window, charged to the byte budget). A checkpoint carrying a Window that
  covers [P' - 2048, Q) may be granted above P' - 2048 (window_covers). With the switches off nothing here changes and the stats keys are
  the ones they always were.
- Turn telemetry (QWEN_PREFIX_TURN_TELEMETRY, on unless 0): TurnMemory counts how each continuation's prompt relates to the previous
  turn's prompt and answer (the turn_* counters; counts only, see the class).
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
    'inflight_started', 'inflight_captures', 'inflight_dropped',
)


# Exact resume P0 (docs/tp4-exact-resume.md 5.2; default off, gate-only profiles). QWEN_PREFIX_STICKY_UNDROP=1 (sticky sessions only):
# the scheduler graft asks vLLM for its hit WITHOUT the EAGLE last-block drop (DFlash on TT reads its window from the prefill capture,
# not from a recomputed last block) and plans the prompt boundary floor2048(P) as G1 always did. QWEN_PREFIX_WINDOW_CARRY=1 (needs the
# undrop): the checkpoint at floor2048(P) also carries the drafter window rows below it, so the next turn may resume above P' - 2048.
# QWEN_PREFIX_WINDOW_CARRY_AUDIT=1 (gate only, needs the carry) reads every carried window back and digests every sticky window.
ENV_UNDROP = 'QWEN_PREFIX_STICKY_UNDROP'
ENV_WINDOW_CARRY = 'QWEN_PREFIX_WINDOW_CARRY'
ENV_WINDOW_CARRY_AUDIT = 'QWEN_PREFIX_WINDOW_CARRY_AUDIT'
# Turn telemetry (docs/tp4-exact-resume.md 5.3; ON unless =0): how each continuation's prompt relates to the previous turn's prompt and
# answer, counted by class (no token, text, salt or tenant leaves this process); QWEN_PREFIX_TURN_MEMORY bounds the turns remembered.
ENV_TURN_TELEMETRY = 'QWEN_PREFIX_TURN_TELEMETRY'
ENV_TURN_MEMORY = 'QWEN_PREFIX_TURN_MEMORY'
DEFAULT_TURN_MEMORY = 2048
# The answer tokens a remembered turn keeps (from its anchor block on): a longer answer is compared this far only.
TURN_TAIL_TOKENS = 4096
WINDOW_STAT_NAMES = ('window_stored', 'window_missing', 'window_grants', 'window_grant_tokens')
TURN_STAT_NAMES = (
    'turn_records', 'turn_forgotten', 'turn_first', 'turn_earlier', 'turn_before_prompt_end', 'turn_at_p_minus_2',
    'turn_at_p_minus_1', 'turn_at_decode_start', 'turn_in_answer', 'turn_extends', 'turn_long_unresolved', 'turn_prev_length',
    'turn_in_answer_lt16', 'turn_in_answer_lt64', 'turn_in_answer_lt256', 'turn_in_answer_lt1024', 'turn_in_answer_ge1024',
    'turn_out_lt256', 'turn_out_lt1024', 'turn_out_lt4096', 'turn_out_ge4096',
    'turn_new_lt256', 'turn_new_lt1024', 'turn_new_lt4096', 'turn_new_ge4096',
)


def strict_switch(environ, name, default='0'):
    """A strict on/off switch: unset (or empty) is the default, '0' off, '1' on; anything else refuses to start (ValueError)."""
    environ = os.environ if environ is None else environ
    raw = environ.get(name)
    if raw is None or raw == '':
        raw = default
    if raw not in ('0', '1'):
        raise ValueError('%s=%r: only 0 or 1' % (name, raw))
    return raw == '1'


def undrop_enabled(environ=None):
    return strict_switch(environ, ENV_UNDROP)


def window_carry_enabled(environ=None):
    return strict_switch(environ, ENV_WINDOW_CARRY)


def window_carry_audit_enabled(environ=None):
    return strict_switch(environ, ENV_WINDOW_CARRY_AUDIT)


def turn_telemetry_enabled(environ=None):
    return strict_switch(environ, ENV_TURN_TELEMETRY, default='1')


def exact_resume_problems(environ=None):
    """Why the exact-resume switches cannot run as set ([] when they can): the carry needs the undrop (its Q is above the dropped hit)
    and the audit needs the carry."""
    environ = os.environ if environ is None else environ
    problems = []
    try:
        undrop, carry, audit = undrop_enabled(environ), window_carry_enabled(environ), window_carry_audit_enabled(environ)
    except ValueError as error:
        return [str(error)]
    if carry and not undrop:
        problems.append('%s=1 needs %s=1: a carried resume sits above the hit vLLM grants with the EAGLE drop' % (ENV_WINDOW_CARRY, ENV_UNDROP))
    if audit and not carry:
        problems.append('%s=1 without %s=1: there is no carried window to audit' % (ENV_WINDOW_CARRY_AUDIT, ENV_WINDOW_CARRY))
    return problems


def telemetry_problems(environ=None):
    """Why the turn telemetry's switches cannot be read ([] when they can): a malformed value refuses to start."""
    try:
        if turn_telemetry_enabled(environ):
            TurnMemory(environ)
    except ValueError as error:
        return [str(error)]
    return []


class Window(object):
    """The drafter window rows a checkpoint carries: the target taps of positions [start, end) (end = the checkpoint's pos) as the fast
    path's opaque host payload, its size and digest. A resume at end for a prompt of P tokens needs [max(0, P - 2048), end): covers()."""

    __slots__ = ('start', 'end', 'payload', 'nbytes', 'digest')

    def __init__(self, start, end, payload, nbytes, digest=None):
        self.start, self.end, self.payload, self.nbytes, self.digest = int(start), int(end), payload, int(nbytes), digest

    def covers(self, prompt, q):
        return q == self.end and self.start <= max(0, int(prompt) - CHUNK)


def window_covers(checkpoint, prompt, q):
    """Whether `checkpoint` carries the drafter window a resume at q of a prompt of `prompt` tokens needs below q."""
    window = getattr(checkpoint, 'window', None)
    return window is not None and window.covers(prompt, q)


def _bucket(value, bounds, names):
    for bound, name in zip(bounds, names):
        if value < bound:
            return name
    return names[-1]


class TurnMemory(object):
    """The previous turns of salted conversations, bounded, for the turn telemetry (docs/tp4-exact-resume.md 5.3). A finished request
    is remembered under the block hash of its ANCHOR, the last whole block at or below P - 2 (a prompt that diverges at P - 1 or P - 2 still
    contains it), with its prompt length P, its end E (prompt plus answer) and its tokens from the anchor on (at most TURN_TAIL_TOKENS); and
    under its first block's hash (the salt's chain head) for the 'earlier' class. A new salted request is classified ONCE, at its first
    admission attempt, by the longest common prefix L of its prompt with the newest remembered turn whose anchor its own chain contains:
      extends              L >= E: the prompt carries every token the previous answer generated (the client echoed the reasoning)
      in_answer            P < L < E: it diverged inside the answer (a trimmed or re-rendered tool call, a cut answer)
      at_decode_start      L == P
      at_p_minus_1         L == P - 1 (the reasoning was dropped: the template's '<think>' '\\n' became '<think>' '\\n\\n' '</think>')
      at_p_minus_2         L == P - 2 (the '<think>' itself went: preserve_thinking=false at a new user message)
      before_prompt_end    anchor <= L < P - 2
      earlier              no anchor in its chain, but its first block is a known turn's: it diverged further back (or a sibling)
      first                nothing known
    and the answer length E - P, the new content P' - E (extends) and the in-answer offset L - P are counted in buckets. Counts only."""

    def __init__(self, environ=None):
        environ = os.environ if environ is None else environ
        raw = environ.get(ENV_TURN_MEMORY)
        try:
            capacity = DEFAULT_TURN_MEMORY if raw in (None, '') else int(raw)
        except ValueError:
            raise ValueError('%s=%r is not a number of turns' % (ENV_TURN_MEMORY, raw))
        if capacity < 0:
            raise ValueError('%s=%r must be >= 0' % (ENV_TURN_MEMORY, raw))
        self.capacity = capacity
        self.anchors = OrderedDict()
        self.heads = {}
        self.seen = set()

    def record(self, request, stats):
        """Remember a finished request's turn (a normal finish only: stopped or length-capped). Never raises."""
        try:
            if not self.capacity or not getattr(request, 'cache_salt', None):
                return False
            status = getattr(getattr(request, 'status', None), 'name', str(getattr(request, 'status', '')))
            if 'STOPPED' not in status and 'LENGTH' not in status:
                return False
            prompt = int(request.num_prompt_tokens)
            tokens = request.all_token_ids
            end = len(tokens)
            anchor = ((prompt - 2) // BLOCK) * BLOCK
            hashes = request.block_hashes
            if anchor < BLOCK or end <= prompt or len(hashes) < anchor // BLOCK:
                return False
            key = hashes[anchor // BLOCK - 1]
            tail = array('i', tokens[anchor:min(end, anchor + TURN_TAIL_TOKENS)])
            self.anchors[key] = (prompt, end, anchor, tail, 'LENGTH' in status)
            self.anchors.move_to_end(key)
            self.heads[hashes[0]] = key
            stats['turn_records'] += 1
            while len(self.anchors) > self.capacity:
                old, _ = self.anchors.popitem(last=False)
                stats['turn_forgotten'] += 1
                for head in [head for head, value in self.heads.items() if value == old]:
                    del self.heads[head]
            return True
        except Exception:
            return False

    def classify(self, request, stats):
        """Count a new salted request's class once (its first admission attempt). Never raises."""
        try:
            request_id = request.request_id
            if request_id in self.seen:
                return None
            self.seen.add(request_id)
            hashes = request.block_hashes
            prompt_tokens = int(request.num_prompt_tokens)
            found = None
            for index in range(min(len(hashes), prompt_tokens // BLOCK) - 1, -1, -1):
                found = self.anchors.get(hashes[index])
                if found is not None:
                    break
            if found is None:
                name = 'earlier' if hashes and hashes[0] in self.heads else 'first'
                stats['turn_' + name] += 1
                return name
            prompt, end, anchor, tail, cut = found
            tokens = request.all_token_ids
            limit = min(len(tail), prompt_tokens - anchor)
            same = 0
            while same < limit and tokens[anchor + same] == tail[same]:
                same += 1
            common = anchor + same
            if cut:
                stats['turn_prev_length'] += 1
            stats['turn_' + _bucket(end - prompt, (256, 1024, 4096), ('out_lt256', 'out_lt1024', 'out_lt4096', 'out_ge4096'))] += 1
            if common >= end:
                name = 'extends'
                stats['turn_' + _bucket(prompt_tokens - end, (256, 1024, 4096),
                                        ('new_lt256', 'new_lt1024', 'new_lt4096', 'new_ge4096'))] += 1
            elif same == len(tail) and same < prompt_tokens - anchor:
                name = 'long_unresolved'
            elif common > prompt:
                name = 'in_answer'
                stats['turn_' + _bucket(common - prompt, (16, 64, 256, 1024),
                                        ('in_answer_lt16', 'in_answer_lt64', 'in_answer_lt256', 'in_answer_lt1024', 'in_answer_ge1024'))] += 1
            elif common == prompt:
                name = 'at_decode_start'
            elif common == prompt - 1:
                name = 'at_p_minus_1'
            elif common == prompt - 2:
                name = 'at_p_minus_2'
            else:
                name = 'before_prompt_end'
            stats['turn_' + name] += 1
            return name
        except Exception:
            return None

    def forget(self, request_id):
        self.seen.discard(request_id)


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

    __slots__ = ('key', 'pos', 'token_ids', 'rec', 'carry', 'nbytes', 'pins', 'serial', 'window')

    def __init__(self, key, pos, token_ids, rec=None, carry=None, nbytes=0, serial=0):
        self.key = key
        self.pos = pos
        self.token_ids = token_ids
        self.rec = rec
        self.carry = carry
        self.nbytes = int(nbytes)
        self.pins = 0
        self.serial = serial
        # Exact resume P0 (QWEN_PREFIX_WINDOW_CARRY): the drafter window rows below pos the capturing prefill computed (a Window),
        # attached after that prefill finished; None everywhere else (and always with the flag off).
        self.window = None

    def matches(self, tokens):
        """Whether tokens (a sequence of ints) are exactly the ids this checkpoint was captured from."""
        return len(tokens) == self.pos and self.token_ids == token_array(tokens)


class Grant(object):
    """One admission's reuse record: the trimmed hit Q (0 on a miss), vLLM's raw hit h, where the
    row's chunk loop drains (floor2048 of the tokens it prefills) and the boundaries the model
    should capture. Staged on every admission attempt; committed only when the step's output admits
    the request at start_pos == Q. The model reads the committed one."""

    __slots__ = ('req_id', 'q', 'h', 'key', 'checkpoint', 'plan', 'request', 'tokens', 'drain', 'unplanned', 'prompt')

    def __init__(self, req_id, q, h, key, checkpoint, plan, request, drain=None, unplanned=()):
        # The prompt length, kept past commit (which drops the request): the in-flight plan needs it.
        self.prompt = getattr(request, 'num_prompt_tokens', None)
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


class Inflight(object):
    """The capture plan of a request the cap splits across steps: {position: block hash key}, the token ids the captures are
    filed under (the grant's, taken when it committed), the prompt length and whether the request's final step has been
    scheduled (then begin_step drops it: that step's captures have run by the next schedule())."""

    __slots__ = ('plan', 'tokens', 'prompt', 'final')

    def __init__(self, plan, tokens, prompt):
        self.plan = dict(plan)
        self.tokens = tokens
        self.prompt = prompt
        self.final = False


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
        # req_id -> Inflight: the plans of the requests the cap is splitting (see the module docstring).
        self.inflight = {}
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
        # Exact resume P0 (docs/tp4-exact-resume.md 5.2) and the turn telemetry (5.3): their counters exist only while they are on,
        # so a registry with both off snapshots exactly the keys it always had.
        self.window_carry = window_carry_enabled(environ)
        if self.window_carry:
            self.stats.update(dict.fromkeys(WINDOW_STAT_NAMES, 0))
        # req_id -> {pos: block hash key} of the checkpoints this request's prefill stored (window carry only).
        self.captured_by = {}
        self.turns = TurnMemory(environ) if turn_telemetry_enabled(environ) else None
        if self.turns is not None:
            self.stats.update(dict.fromkeys(TURN_STAT_NAMES, 0))
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
            self.inflight.clear()
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
        self.inflight.clear()
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
        for req_id in [req_id for req_id, plan in self.inflight.items() if plan.final]:
            del self.inflight[req_id]
            self.stats['inflight_dropped'] += 1
        self._enforce_budget()

    def stage(self, grant):
        self.staged[grant.req_id] = grant
        self.stats['staged'] += 1

    def unstage(self, req_id):
        self.staged.pop(req_id, None)

    def commit(self, admitted, scheduled=None):
        """admitted: request id -> start_pos for every request the step's output admits (new or
        resumed). Returns the grants committed; every other staged grant is dropped.

        scheduled (optional): request id -> the tokens the step schedules for it. With it, an admitted request whose step stops short
        of its prompt (the Lever N cap split it) leaves its capture plan as an in-flight plan for the steps that follow; without it
        no plan outlives its step, exactly as before."""
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
                if self.window_carry and grant.prompt is not None and grant.q > max(0, grant.prompt - CHUNK):
                    # A carried resume: above the window ceiling floor2048(P - 2048) by these tokens.
                    self.stats['window_grants'] += 1
                    self.stats['window_grant_tokens'] += grant.q - floor_chunk(max(0, grant.prompt - CHUNK))
            self.stats['admissions'] += 1
            self.stats['trim_loss_tokens'] += grant.h - grant.q
            self.stats['mid_loop_unplanned'] += len(grant.unplanned)
            if floor_chunk(grant.h) > grant.q:
                self.stats['kv_hit_without_checkpoint'] += 1
            grant.request = None
            self.committed[req_id] = grant
            self._note_split(grant, admitted[req_id], scheduled)
            done.append(grant)
        self.staged.clear()
        return done

    def _note_split(self, grant, start, scheduled):
        """A grant admitted for fewer tokens than the rest of its prompt: keep its plan for the later steps."""
        if scheduled is None or not grant.plan or grant.prompt is None or grant.tokens is None:
            return
        tokens = scheduled.get(grant.req_id)
        if type(tokens) is not int or start + tokens >= grant.prompt:
            return
        self.inflight[grant.req_id] = Inflight(grant.plan, grant.tokens, grant.prompt)
        self.stats['inflight_started'] += 1

    def note_scheduled(self, req_id, start, tokens):
        """The scheduler's report that it scheduled `tokens` more tokens of a request already in flight from `start`. The step that
        reaches the prompt's end is the request's last: its plan is dropped at the next begin_step, after that step's captures ran."""
        plan = self.inflight.get(req_id)
        if plan is not None and start + tokens >= plan.prompt:
            plan.final = True
        return plan is not None

    def planned(self, req_id):
        """The positions this request may still capture, from its committed grant or its in-flight plan: sorted, [] when none."""
        grant = self.committed.get(req_id)
        if grant is not None:
            return grant.capture_positions()
        plan = self.inflight.get(req_id)
        return sorted(plan.plan) if plan is not None else []

    def grant_for(self, req_id):
        return self.committed.get(req_id)

    def capture(self, req_id, pos, rec=None, carry=None, nbytes=None, ms=None, loop_pos=None):
        """Model side: store the GDN state after exactly pos tokens of this step's prefill of req_id.
        loop_pos is how many tokens the model's chunk loop had run when it took rec and carry; the
        capture is refused unless it is pos (see the module's contract). A failure or refusal
        skips the checkpoint and is counted; it never fails the request (S7)."""
        try:
            grant = self.committed.get(req_id)
            carried = None
            if grant is not None:
                keys, tokens = dict(grant.plan), grant.tokens
            else:
                carried = self.inflight.get(req_id)
                if carried is None:
                    raise KeyError('no committed grant for %s' % req_id)
                keys, tokens = carried.plan, carried.tokens
            if pos not in keys:
                raise KeyError('boundary %d is not planned for %s' % (pos, req_id))
            if loop_pos != pos:
                self.stats['capture_wrong_position'] += 1
                raise ValueError('the state was taken after %r tokens, not %d: refused (it would be '
                                 'restored as the state at %d)' % (loop_pos, pos, pos))
            if ms is not None:
                self.stats['capture_ms'] += float(ms)
            stored = self.put(keys[pos], pos, tokens[0:pos], rec, carry,
                              CHECKPOINT_NBYTES if nbytes is None else nbytes)
            if carried is not None and stored is not None:
                self.stats['inflight_captures'] += 1
            if self.window_carry and stored is not None:
                self.captured_by.setdefault(req_id, {})[pos] = keys[pos]
            return stored
        except Exception as error:
            self.stats['capture_failures'] += 1
            log('capture skipped req=%s pos=%s: %s', req_id, pos, error)
            return None

    def note_restore(self, ms):
        self.stats['restores'] += 1
        self.stats['restore_ms'] += float(ms)

    # -- exact resume P0: the drafter window a checkpoint carries -------------------------------
    def stored_at(self, req_id, pos):
        """Whether req_id's prefill stored a checkpoint at pos that is still resident (window carry only)."""
        key = (self.captured_by.get(req_id) or {}).get(pos)
        entry = self.entries.get(key) if key is not None else None
        return entry is not None and entry.pos == pos

    def attach_window(self, req_id, pos, start, payload, nbytes, digest=None):
        """The fast path's report, once req_id's prefill has finished, of its drafter window rows [start, pos) (the target taps the
        capture recorded below the checkpoint this request stored at pos): filed on that checkpoint and charged to the byte budget.
        None (counted) when the request stored no checkpoint at pos or it is already gone. Never raises."""
        try:
            if not self.window_carry:
                return None
            key = (self.captured_by.get(req_id) or {}).get(pos)
            entry = self.entries.get(key) if key is not None else None
            if entry is None or entry.pos != pos or not 0 <= start < pos:
                self.stats['window_missing'] += 1
                return None
            if entry.window is not None:
                entry.nbytes -= entry.window.nbytes
                self.bytes -= entry.window.nbytes
            entry.window = Window(start, pos, payload, nbytes, digest)
            entry.nbytes += entry.window.nbytes
            self.bytes += entry.window.nbytes
            self.stats['window_stored'] += 1
            self._enforce_budget()
            return entry if self.entries.get(key) is entry else None
        except Exception as error:
            self.stats['window_missing'] += 1
            log('window not attached req=%s pos=%s: %s', req_id, pos, error)
            return None

    def forget_request(self, req_id):
        self.captured_by.pop(req_id, None)
        self.token_checks.pop(req_id, None)
        found = self.staged.pop(req_id, None) is not None
        if self.inflight.pop(req_id, None) is not None:
            found = True
            self.stats['inflight_dropped'] += 1
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
                      orphans_now=len(self.orphan_keys), inflight_now=len(self.inflight),
                      mid_loop_capture=self.mid_loop_capture,
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
