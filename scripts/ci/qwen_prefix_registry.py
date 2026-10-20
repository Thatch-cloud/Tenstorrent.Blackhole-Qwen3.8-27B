"""Conversation prefix reuse on the TT general path (G1): the checkpoint registry.

The TT prefix-reuse design (revision 2, 2026-09-26), section 2.0.1 item 1. One registry per engine
process holds what vLLM cannot know about a hit: the GatedDeltaNet (GDN) state the model saved at a
2048-token prefill chunk boundary, and which admission may restore which checkpoint in this step.

- Checkpoints: an LRU keyed by vLLM's block hash at the boundary (the hash of block Q/64-1), each
  holding pos, the token ids [0, pos) it was captured from, the model's host tensors (rec_state and
  conv_carry for the whole mesh) and their size. It is bounded by bytes (QWEN_PREFIX_STORE_GIB, default
  8 GiB). A checkpoint is as large as the GDN state dtype makes it (checkpoint_nbytes): 78446592 bytes
  with QWEN35_GDN_STATE_BF16=1 (about 109 in the default store), twice that with the fp32 state; a
  pinned entry is never evicted.
- Store hygiene (opt-in, each off unless its flag says so; off, the registry behaves exactly as before):
  QWEN_PREFIX_SUPERSEDE=1 retires a conversation's older checkpoints when a newer one of the same token
  path lands, keeping the newest older one (an edit or retry of the last turn still has it) and never a
  pinned, shared (gap-boundary) or branch-point (granted to two requests) one; QWEN_PREFIX_EVICT=fair
  evicts the oldest checkpoint of the tenant holding the most bytes (the tenant tag is the salt's)
  instead of the global oldest. Neither can serve anything the plain LRU could not: a retired or
  evicted checkpoint is simply a miss, and every served checkpoint is still token-verified.
- Telemetry (QWEN_PREFIX_TELEMETRY, default on; 0 turns every piece of it off): each admission is
  classified once, when the step admits it (class_* counters: returning_served, first_turn, rewritten,
  short, kv_evicted, ckpt_evicted, ckpt_missing, refused, unsalted, denied), a returning session's reuse
  distance (seconds and admitted prompt tokens since its previous turn) goes into two histograms, and
  host gauges (engine RSS, MemAvailable) ride with the stats. It reads the request's hashes and counts;
  it never changes a grant, a trim, a capture or an eviction (test_qwen_prefix_tiers).
- Host KV tier (QWEN_PREFIX_HOST_TIER_GIB, default 0 = off; the scheduler graft's TierHooks drive it, the model's
  tier IO moves the bytes): a cached attention-KV block that vLLM evicts from the device pool is first read, raw and
  packed, into host RAM under its block hash (TierStore), and the GDN checkpoint keyed at it is kept instead of dropped;
  a returning request whose device hit ends where the tier's consecutive blocks begin has them written back into free
  device blocks, hashed as cached-free blocks, before vLLM's own hit logic runs. The GiB is the whole prefix state's host
  budget: checkpoints (QWEN_PREFIX_STORE_GIB) plus KV pages take the rest. Blocks are keyed by vLLM's chained block hash,
  which carries the salt, so tenants never meet. A restored block is the evicted block's bytes (a digest checked on
  restore, a read-back compared under QWEN_PREFIX_HOST_TIER_AUDIT=1); any failure is a miss.
- Tenant quotas (QWEN_PREFIX_TENANT_QUOTA_*, default off, byte-identical off; docs/prefix-store-hygiene.md "Tenant quotas"): a per-tenant cap and floor on the cached prefix
  tokens in the device pool (the scheduler graft's DeviceQuota) and on the bytes in the host tier (TierStore), keyed by the salt's tenant tag. The cap makes a tenant that passes it
  lose its OWN oldest blocks (records) first; the floor spares a tenant's last tokens (bytes) from everyone else's allocations (spills). quota_config parses the four knobs strictly;
  the counters it adds (QUOTA_STAT_NAMES) are aggregates and exist only when a quota is on. No tenant, salt or digest is ever logged or exported.
- Grants: staged on every admission attempt inside one schedule() call, committed only for the
  requests the step's SchedulerOutput admits at start_pos == Q (F2, S6), pinned for that one step.
  The model reads grant_for(request_id) per prefill row.
- Kill switch: KillSwitch polls a flag file at most once a second; once engaged the registry is
  disabled for the life of the process (no grants, no captures, no publishing). Turning reuse back
  on needs an engine restart, so nothing published under a suspect build is ever served (S7).
- Counters (S5): grants, trim loss h-Q, KV hits without a checkpoint, distinct checkpoints found
  without their KV (orphans), token mismatches, capture failures and refusals, restore and capture
  ms, bytes, pins. StatsExport writes them from the scheduler process, rate-limited, as a
  "[PINDIAG] prefix: stats {...}" line and a JSON file on tmpfs (QWEN_PREFIX_STATS_PATH). Under the
  telemetry it also writes, at the same cadence and whether or not a counter moved, two compact
  key=value lines the host's log reader can parse: "[PINDIAG] prefix: tier rss=<bytes> avail=<bytes>
  reg_bytes=... reg_entries=... adm=... returning=... <class>=<n> ..." and "[PINDIAG] prefix: reuse
  bounds_s=... served_s=<counts>:<sum> missed_s=... bounds_tok=... served_tok=... missed_tok=...".

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

import bisect
import gc
import hashlib
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
# Store hygiene and telemetry (the module docstring). Each is strict: a value outside its set refuses to start.
ENV_EVICT = 'QWEN_PREFIX_EVICT'            # lru (default) | fair
ENV_SUPERSEDE = 'QWEN_PREFIX_SUPERSEDE'    # 0 (default) | 1
ENV_TELEMETRY = 'QWEN_PREFIX_TELEMETRY'    # 1 (default) | 0
ENV_GHOST = 'QWEN_PREFIX_GHOST_ENTRIES'    # the returning-session memory, entries (default 65536; 0 = none)
EVICT_POLICIES = ('lru', 'fair')
DEFAULT_GHOST_ENTRIES = 65536
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

# One host checkpoint is the GatedDeltaNet state of the whole mesh after a 2048-token boundary: per GDN layer the
# recurrent state [value heads, 128, 128] and the conv carry [3, channels]. The totals over the mesh do not depend on
# how many chips share them (two chips of 24 heads and 5120 channels, four of 12 and 2560). The recurrent state is
# fp32 unless QWEN35_GDN_STATE_BF16=1 (the production setting); the conv carry is bf16 either way. The model passes
# the real size of what it read (qwen_prefix_model_patch._qwen_prefix_read_scratch); this derivation is for the
# registry's default and for every capacity estimate made without a device (Lever N's host-memory guard, the gate).
ENV_GDN_STATE_BF16 = 'QWEN35_GDN_STATE_BF16'
GDN_LAYERS = 48
GDN_VALUE_HEADS = 48
GDN_HEAD_DIM = 128
GDN_CONV_TAPS = 3
GDN_CONV_CHANNELS = 10240
CARRY_ITEMSIZE = 2


def gdn_state_bf16(environ=None):
    """Whether the recurrent state is bf16 (QWEN35_GDN_STATE_BF16=1, exactly as the GDN layer reads it)."""
    environ = os.environ if environ is None else environ
    return environ.get(ENV_GDN_STATE_BF16) == '1'


def checkpoint_nbytes(environ=None, state_bf16=None):
    """Bytes of one checkpoint for the state dtype (state_bf16, else the environment's). 78446592 for bf16,
    153944064 for fp32."""
    if state_bf16 is None:
        state_bf16 = gdn_state_bf16(environ)
    itemsize = 2 if state_bf16 else 4
    return GDN_LAYERS * (GDN_VALUE_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM * itemsize
                         + GDN_CONV_TAPS * GDN_CONV_CHANNELS * CARRY_ITEMSIZE)


CHECKPOINT_NBYTES_FP32 = checkpoint_nbytes(state_bf16=False)
CHECKPOINT_NBYTES_BF16 = checkpoint_nbytes(state_bf16=True)
# What a checkpoint costs in this process (its environment's state dtype, read once at import).
CHECKPOINT_NBYTES = checkpoint_nbytes()

# A returning session's reuse distance, bucketed (upper bounds; one more bucket above the last): seconds since its
# previous turn, and prompt tokens admitted for OTHER requests since then (the quantity a cache of that many tokens
# would have had to hold across).
REUSE_SECONDS_BOUNDS = (10, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200, 14400, 28800, 86400)
REUSE_TOKENS_BOUNDS = (16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216)
REUSE_OUTCOMES = ('served', 'missed')
# Bounds on the telemetry's per-request memory (waiting requests with an attempt on file, requests already classified).
ATTEMPTS_LIMIT = 4096
CLASSIFIED_LIMIT = 8192

# How an admission is classified (the module docstring); every admission lands in exactly one. The classes that can
# still carry a partial hit (a prefix shared with another session, or an older boundary) also count class_<name>_hit
# when the trim granted q > 0.
CLASSES = ('returning_served', 'first_turn', 'rewritten', 'short', 'kv_evicted', 'ckpt_evicted', 'ckpt_missing',
           'refused', 'unsalted', 'denied')
CLASS_HIT = ('first_turn', 'rewritten', 'kv_evicted', 'ckpt_evicted', 'ckpt_missing', 'refused')
# Why a stored checkpoint left (Ghost.state): the budget (lru or fair), vLLM evicting its boundary block, supersession,
# a registry clear, or anything else.
GONE_REASONS = ('budget', 'coupled', 'superseded', 'cleared', 'other')

# The host KV tier (the module docstring). Each knob is strict.
ENV_TIER_GIB = 'QWEN_PREFIX_HOST_TIER_GIB'                  # 0 or unset: off
ENV_TIER_AUDIT = 'QWEN_PREFIX_HOST_TIER_AUDIT'              # gate instrument: read back every restore and compare
ENV_TIER_SPILL_MAX = 'QWEN_PREFIX_HOST_TIER_SPILL_MAX_BLOCKS'   # most blocks one allocation's evictions may spill
ENV_TIER_MIN_TOKENS = 'QWEN_PREFIX_HOST_TIER_MIN_TOKENS'    # sessions whose checkpoints stop below this are not worth a restore
ENV_TIER_MIN_AVAILABLE = 'QWEN_PREFIX_HOST_TIER_MIN_AVAILABLE_GIB'   # no spill while the host's MemAvailable is under this
ENV_TIER_VERIFY = 'QWEN_PREFIX_HOST_TIER_VERIFY'            # sample | all | off: digest checks at restore
ENV_TIER_OFF_PATH = 'QWEN_PREFIX_HOST_TIER_OFF_PATH'        # the tier's own kill-switch file (gate containers)
# The sliced restore (docs/prefix-store-hygiene.md "Sliced restore"): a returning session's blocks come back a bounded slice per engine step, the other seats decoding in between.
ENV_TIER_RESTORE_SLICE = 'QWEN_PREFIX_TIER_RESTORE_SLICE'       # most blocks one engine step restores (digits only; 0 or unset: the restore runs whole, in the trim)
ENV_TIER_RESTORE_SLICE_MS = 'QWEN_PREFIX_TIER_RESTORE_SLICE_MS'  # the milliseconds of restore work one engine step aims at (the slice adapts to the measured cost per block)
DEFAULT_TIER_SLICE_MS = 200.0
MAX_TIER_SLICE_BLOCKS = 65536
MAX_TIER_SLICE_MS = 60000.0
TIER_KILL_SWITCH_PATH = '/models/.qwen-c2/kv-tier.off'
DEFAULT_TIER_SPILL_MAX_BLOCKS = 512
# COMPLETE SPILLS (docs/tp4-sched-1.md section 3; default off): QWEN_PREFIX_HOST_TIER_SPILL_MODE=complete lifts the per-flush cap of ENV_TIER_SPILL_MAX blocks. The cap
# drops what it cuts for good: the blocks vLLM evicted are reused by the very allocation that evicted them, so their bytes cannot wait for a later step. The one flush
# that exceeds the cap is a restore (TierHooks.load allocates the whole session at once and evicts as many blocks as it restores: 3,393 of the 3,905 blocks a 254k
# session displaced in the TC1 card run were dropped). In complete mode the flush spills everything up to ENV_TIER_SPILL_HARD_MAX blocks (default: a 262,144-token session
# is 4,096 blocks), in batches of ENV_TIER_SPILL_BATCH blocks (each batch is one reservation and one device read, so a failed read loses its batch and not the flush), and
# logs a flush that held the engine longer than ENV_TIER_SPILL_WARN_MS. The cost of a block is measured, about 1.7 ms while the slab fills and 0.4 ms once it is resident.
ENV_TIER_SPILL_MODE = 'QWEN_PREFIX_HOST_TIER_SPILL_MODE'
ENV_TIER_SPILL_HARD_MAX = 'QWEN_PREFIX_HOST_TIER_SPILL_HARD_MAX_BLOCKS'
ENV_TIER_SPILL_BATCH = 'QWEN_PREFIX_HOST_TIER_SPILL_BATCH_BLOCKS'
ENV_TIER_SPILL_WARN_MS = 'QWEN_PREFIX_HOST_TIER_SPILL_WARN_MS'
TIER_SPILL_MODES = ('cap', 'complete')
DEFAULT_TIER_SPILL_HARD_MAX_BLOCKS = 4096
DEFAULT_TIER_SPILL_BATCH_BLOCKS = 512
DEFAULT_TIER_SPILL_WARN_MS = 1000
MAX_TIER_SPILL_BLOCKS = 65536
# The counters complete spills add to the registry's stats (only when the mode is on, so the stats of an engine without it are what they were): flushes in batches, blocks the
# cap would have dropped that were spilled instead, the largest flush (blocks, ms) and the flushes over the warning bound.
SPILL_STAT_NAMES = ('tier_spill_batches', 'tier_spill_cap_lifted', 'tier_spill_max_blocks', 'tier_spill_max_ms', 'tier_spill_slow')
DEFAULT_TIER_MIN_TOKENS = 8192
DEFAULT_TIER_MIN_AVAILABLE_GIB = 16.0
TIER_VERIFY_MODES = ('sample', 'all', 'off')
TIER_VERIFY_EVERY = 16
# A scan for dead records stops after this many of the oldest.
TIER_DEAD_SCAN = 256

# Per-tenant quotas (the module docstring, "Tenant quotas"): off unless one of these names a number. Every knob is strict: digits only, and a value the engine
# could not honour refuses to start. The tenant is the salt's (tenant_of_salt); the same numbers apply to every tenant.
ENV_QUOTA_DEVICE_CAP = 'QWEN_PREFIX_TENANT_QUOTA_DEVICE_CAP_TOKENS'       # most prefix tokens one tenant keeps cached in the device pool (0/unset: no cap)
ENV_QUOTA_DEVICE_FLOOR = 'QWEN_PREFIX_TENANT_QUOTA_DEVICE_FLOOR_TOKENS'   # prefix tokens of a tenant that other tenants' allocations may not evict (0/unset: none)
ENV_QUOTA_HOST_CAP = 'QWEN_PREFIX_TENANT_QUOTA_HOST_CAP_GIB'              # most host-tier GiB one tenant holds (0/unset: no cap)
ENV_QUOTA_HOST_FLOOR = 'QWEN_PREFIX_TENANT_QUOTA_HOST_FLOOR_GIB'          # host-tier GiB of a tenant that other tenants' spills may not evict (0/unset: none)
QUOTA_ENV_NAMES = (ENV_QUOTA_DEVICE_CAP, ENV_QUOTA_DEVICE_FLOOR, ENV_QUOTA_HOST_CAP, ENV_QUOTA_HOST_FLOOR)
# The counters the sliced restore adds to the registry's stats (only when QWEN_PREFIX_TIER_RESTORE_SLICE is on, so the stats of an engine without it are what they were).
SLICE_STAT_NAMES = ('tier_slice_jobs', 'tier_slice_steps', 'tier_slice_blocks', 'tier_slice_ms', 'tier_slice_max_ms', 'tier_slice_held', 'tier_slice_aborted',
                    'tier_slice_inline')
# The upload levers of the tier's host side (qwen_tier_upload; docs/tp4-upload1.md): threaded staging copies and parallel digest checks. The env names and the counters are
# pinned equal to the module's by test_qwen_tier_upload; the module is imported only when one of the names is set, so an engine without them runs none of its code.
ENV_UPLOAD_NAMES = ('QWEN_PREFIX_TIER_IO_THREADS', 'QWEN_PREFIX_TIER_IO_CHUNK_BLOCKS', 'QWEN_PREFIX_TIER_VERIFY_THREADS')
UPLOAD_STAT_NAMES = ('tier_io_write_calls', 'tier_io_write_blocks', 'tier_io_write_ms', 'tier_io_write_device_ms', 'tier_io_write_stage_ms',
                     'tier_io_read_calls', 'tier_io_read_blocks', 'tier_io_read_ms', 'tier_io_read_device_ms', 'tier_io_read_stage_ms',
                     'tier_io_latched', 'tier_verify_batches', 'tier_verify_blocks', 'tier_verify_ms')
# The counters a quota adds to the registry's stats (only when a quota is on, so the stats of an engine without one are what they were). Aggregates only: no per-tenant
# label, key or line ever carries a salt or a tenant tag.
QUOTA_STAT_NAMES = (
    'quota_dev_demoted_blocks', 'quota_dev_evictions_own', 'quota_dev_evictions_global', 'quota_dev_floor_protected', 'quota_dev_floor_overrides',
    'quota_host_evictions_own', 'quota_host_evictions_global', 'quota_host_floor_protected', 'quota_host_refused',
)
TIER_STAT_NAMES = (
    'tier_spill_flushes', 'tier_spill_blocks', 'tier_spill_bytes', 'tier_spill_ms', 'tier_spill_known',
    'tier_spill_dropped_useless', 'tier_spill_dropped_cap', 'tier_spill_dropped_governor', 'tier_spill_dropped_full',
    'tier_spill_failures', 'tier_restore_requests', 'tier_restore_blocks', 'tier_restore_bytes', 'tier_restore_ms',
    'tier_restore_refused_room', 'tier_restore_refused_digest', 'tier_restore_failures', 'tier_digest_checks',
    'tier_digest_failures', 'tier_evicted', 'tier_ckpt_kept', 'tier_ckpt_dropped', 'tier_audit_reads',
    'tier_audit_mismatches', 'tier_latched',
)

# The counters the registry kept before the store policies and the telemetry; a graft that does not feed those (an older one) leaves the
# rest at zero, so a parity check against it compares these.
LEGACY_STAT_NAMES = (
    'attempts', 'staged', 'dropped_attempts', 'commit_mismatch', 'admissions', 'grants',
    'grant_tokens', 'trim_loss_tokens', 'kv_hit_without_checkpoint', 'orphans',
    'same_step_rejects', 'token_checks', 'token_mismatches', 'unsalted_denied', 'session_denied',
    'killed_denied', 'publish_capped', 'captures', 'capture_replaced', 'capture_kept_pinned',
    'capture_failures', 'capture_wrong_position', 'capture_skipped_budget', 'capture_disabled',
    'mid_loop_unplanned', 'evicted_lru', 'evicted_coupled', 'dropped', 'clears', 'reset_kept',
    'freed_requests', 'restores', 'restore_ms', 'capture_ms', 'dropped_hits', 'program_growth',
    'inflight_started', 'inflight_captures', 'inflight_dropped',
)
STAT_NAMES = LEGACY_STAT_NAMES + (
    # store hygiene
    'evicted_fair', 'evicted_superseded', 'supersede_checks', 'supersede_kept_pinned', 'supersede_kept_shared',
    'supersede_kept_branch',
    # telemetry: admission classes, their losses, the reuse distance's population
    'admitted_prompt_tokens', 'returning_sessions', 'class_failures',
) + TIER_STAT_NAMES + tuple('class_' + name for name in CLASSES) + tuple('class_%s_hit' % name for name in CLASS_HIT) + (
    'class_refused_mismatch',
) + tuple('class_ckpt_evicted_' + reason for reason in GONE_REASONS) + (
    'lost_tokens_kv_evicted', 'lost_tokens_ckpt_evicted', 'lost_tokens_ckpt_missing', 'lost_tokens_refused',
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


def strict_flag(environ, name, default):
    """A 0/1 switch from the environment; unset or empty is the default, anything else but 0 and 1 refuses to start."""
    raw = environ.get(name)
    if raw is None or raw == '':
        return default
    if raw not in ('0', '1'):
        raise ValueError('%s=%r must be 0 or 1' % (name, raw))
    return raw == '1'


def evict_policy(environ=None):
    """QWEN_PREFIX_EVICT: 'lru' (default) or 'fair'. Anything else refuses to start."""
    environ = os.environ if environ is None else environ
    raw = environ.get(ENV_EVICT)
    if raw is None or raw == '':
        return 'lru'
    if raw not in EVICT_POLICIES:
        raise ValueError('%s=%r must be one of %s' % (ENV_EVICT, raw, ', '.join(EVICT_POLICIES)))
    return raw


def ghost_limit(environ=None):
    """QWEN_PREFIX_GHOST_ENTRIES: how many returning-session records the telemetry keeps (0: none)."""
    environ = os.environ if environ is None else environ
    raw = environ.get(ENV_GHOST)
    if raw is None or raw == '':
        return DEFAULT_GHOST_ENTRIES
    try:
        value = int(raw)
    except ValueError:
        raise ValueError('%s=%r is not a number of entries' % (ENV_GHOST, raw))
    if value < 0:
        raise ValueError('%s=%r must be >= 0' % (ENV_GHOST, raw))
    return value


def tenant_of_salt(salt):
    """The tenant a cache_salt partitions: the gateway's opaque tag of a platform salt 'qps1.<tag>.<mac>'
    (serving_c2_contract.mint_salt), else the whole salt (it is a partition key either way). Never logged and never
    a metric label: the fair policy reads it, and nothing else does."""
    if isinstance(salt, str):
        parts = salt.split('.')
        if len(parts) == 3 and parts[0] == 'qps1' and parts[1]:
            return parts[1]
    return salt or ''


def is_prefix(shorter, longer):
    """Whether the token array `shorter` is the start of `longer` (a byte comparison, no copy)."""
    count = len(shorter)
    if count > len(longer) or shorter.itemsize != longer.itemsize:
        return False
    width = count * shorter.itemsize
    return memoryview(shorter).cast('B') == memoryview(longer).cast('B')[0:width]


def host_gauges(proc='/proc'):
    """{'host_rss_bytes': this process's resident set, 'host_mem_available_bytes': the host's MemAvailable} from
    /proc, each only when readable. Never raises."""
    values = {}
    try:
        with open(os.path.join(proc, 'self', 'status'), encoding='ascii', errors='replace') as handle:
            for line in handle:
                if line.startswith('VmRSS:'):
                    values['host_rss_bytes'] = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open(os.path.join(proc, 'meminfo'), encoding='ascii', errors='replace') as handle:
            for line in handle:
                if line.startswith('MemAvailable:'):
                    values['host_mem_available_bytes'] = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError, IndexError):
        pass
    return values


class Checkpoint(object):
    """GDN state at a 2048-token boundary, keyed by vLLM's block hash at that boundary. serial is
    unique per process, so a remembered token check can never vouch for a later entry.

    chain (the hash of the prompt's first block, which carries the salt), tenant (the salt's tenant), shared (taken at
    a gap boundary: another session's prefix reaches it too) and grants (admissions this checkpoint served) feed the
    store hygiene only; no grant or restore reads them."""

    __slots__ = ('key', 'pos', 'token_ids', 'rec', 'carry', 'nbytes', 'pins', 'serial', 'chain', 'tenant', 'shared',
                 'grants', 'window')

    def __init__(self, key, pos, token_ids, rec=None, carry=None, nbytes=0, serial=0, chain=None, tenant=None,
                 shared=False):
        self.key = key
        self.pos = pos
        self.token_ids = token_ids
        self.rec = rec
        self.carry = carry
        self.nbytes = int(nbytes)
        self.pins = 0
        self.serial = serial
        self.chain = chain
        self.tenant = tenant
        self.shared = bool(shared)
        self.grants = 0
        # Exact resume P0 (QWEN_PREFIX_WINDOW_CARRY): the drafter window rows below pos the capturing prefill computed (a Window),
        # attached after that prefill finished; None everywhere else (and always with the flag off).
        self.window = None

    def matches(self, tokens):
        """Whether tokens (a sequence of ints) are exactly the ids this checkpoint was captured from."""
        return len(tokens) == self.pos and self.token_ids == token_array(tokens)


class Ghost(object):
    """What the telemetry remembers of a session's previous turn, under the block hash at that turn's last boundary
    (the key a checkpoint there would have): when it was admitted (seen), the boundary (pos), the prompt length, the
    engine's admitted-token counter after it (counter), and the checkpoint's fate (state): None while none was stored,
    'stored', or the reason it left (GONE_REASONS)."""

    __slots__ = ('seen', 'pos', 'prompt', 'counter', 'state')


class Attempt(object):
    """The telemetry's view of one waiting request's latest admission attempt: committed (or dropped with the request)
    when the step admits it."""

    __slots__ = ('kind', 'h', 'q', 'prompt', 'end', 'chain', 'end_key', 'ghost_key', 'target', 'target_key')


class Grant(object):
    """One admission's reuse record: the trimmed hit Q (0 on a miss), vLLM's raw hit h, where the
    row's chunk loop drains (floor2048 of the tokens it prefills) and the boundaries the model
    should capture. Staged on every admission attempt; committed only when the step's output admits
    the request at start_pos == Q. The model reads the committed one."""

    __slots__ = ('req_id', 'q', 'h', 'key', 'checkpoint', 'plan', 'request', 'tokens', 'drain', 'unplanned', 'prompt',
                 'chain', 'tenant', 'gap')

    def __init__(self, req_id, q, h, key, checkpoint, plan, request, drain=None, unplanned=(), chain=None, tenant=None,
                 gap=None):
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
        # Store hygiene only: the first block's hash, the salt's tenant, and the gap boundary (a prefix other sessions share).
        self.chain = chain
        self.tenant = tenant
        self.gap = gap

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

    __slots__ = ('plan', 'tokens', 'prompt', 'final', 'chain', 'tenant', 'gap')

    def __init__(self, plan, tokens, prompt, chain=None, tenant=None, gap=None):
        self.plan = dict(plan)
        self.tokens = tokens
        self.prompt = prompt
        self.final = False
        self.chain = chain
        self.tenant = tenant
        self.gap = gap


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


class TierConfig(object):
    """The host KV tier's settings (tier_config): total (the whole prefix state's host budget, bytes), kv_bytes (what the KV pages
    get: the total less the checkpoint store's budget), and the knobs."""

    __slots__ = ('total', 'kv_bytes', 'audit', 'spill_max', 'min_tokens', 'min_available', 'verify', 'off_path', 'restore_slice', 'restore_slice_ms')


def _tier_float(environ, name, default, minimum=0.0):
    raw = environ.get(name)
    if raw is None or raw == '':
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError('%s=%r is not a number' % (name, raw))
    if not value >= minimum:
        raise ValueError('%s=%r must be >= %s' % (name, raw, minimum))
    return value


def tier_config(environ, store_budget_bytes):
    """The host KV tier's configuration from the environment, None when QWEN_PREFIX_HOST_TIER_GIB is 0 or unset. The GiB is the whole prefix
    state's host budget (checkpoints plus KV pages), so it must exceed the checkpoint store's own budget; every knob is strict."""
    gib = _tier_float(environ, ENV_TIER_GIB, 0.0)
    if gib == 0.0:
        return None
    config = TierConfig()
    config.total = int(gib * (1 << 30))
    if config.total <= int(store_budget_bytes):
        raise ValueError('%s=%s leaves nothing for KV pages: it is the whole prefix state\'s host budget and must exceed the checkpoint '
                         'store\'s (%s)' % (ENV_TIER_GIB, environ.get(ENV_TIER_GIB), ENV_STORE_GIB))
    config.kv_bytes = config.total - int(store_budget_bytes)
    config.audit = strict_flag(environ, ENV_TIER_AUDIT, False)
    spill_max = _tier_float(environ, ENV_TIER_SPILL_MAX, float(DEFAULT_TIER_SPILL_MAX_BLOCKS))
    if spill_max != int(spill_max):
        raise ValueError('%s=%r is not a whole number of blocks' % (ENV_TIER_SPILL_MAX, environ.get(ENV_TIER_SPILL_MAX)))
    config.spill_max = int(spill_max)
    min_tokens = _tier_float(environ, ENV_TIER_MIN_TOKENS, float(DEFAULT_TIER_MIN_TOKENS))
    config.min_tokens = int(min_tokens)
    config.min_available = int(_tier_float(environ, ENV_TIER_MIN_AVAILABLE, DEFAULT_TIER_MIN_AVAILABLE_GIB) * (1 << 30))
    verify = environ.get(ENV_TIER_VERIFY) or 'sample'
    if verify not in TIER_VERIFY_MODES:
        raise ValueError('%s=%r must be one of %s' % (ENV_TIER_VERIFY, verify, ', '.join(TIER_VERIFY_MODES)))
    config.verify = 'all' if config.audit and verify != 'off' else verify
    config.off_path = environ.get(ENV_TIER_OFF_PATH) or TIER_KILL_SWITCH_PATH
    config.restore_slice, config.restore_slice_ms = restore_slice_config(environ)
    return config


def restore_slice_config(environ):
    """(blocks, ms) of the sliced restore from the environment: (0, DEFAULT_TIER_SLICE_MS) when unset. Strict: the blocks are plain digits (no sign, no exponent, no space) from 0 to
    MAX_TIER_SLICE_BLOCKS, the milliseconds a finite number above 0 and at most MAX_TIER_SLICE_MS, and the milliseconds without the blocks would be inert."""
    raw = environ.get(ENV_TIER_RESTORE_SLICE)
    blocks = 0
    if raw is not None and raw != '':
        if not (raw.isascii() and raw.isdigit()):
            raise ValueError('%s=%r is not a whole number of blocks (digits only)' % (ENV_TIER_RESTORE_SLICE, raw))
        blocks = int(raw)
        if blocks > MAX_TIER_SLICE_BLOCKS:
            raise ValueError('%s=%d is above %d blocks' % (ENV_TIER_RESTORE_SLICE, blocks, MAX_TIER_SLICE_BLOCKS))
    raw_ms = environ.get(ENV_TIER_RESTORE_SLICE_MS)
    millis = DEFAULT_TIER_SLICE_MS
    if raw_ms is not None and raw_ms != '':
        try:
            millis = float(raw_ms)
        except ValueError:
            raise ValueError('%s=%r is not a number of milliseconds' % (ENV_TIER_RESTORE_SLICE_MS, raw_ms))
        if not (millis > 0.0) or millis == float('inf') or millis > MAX_TIER_SLICE_MS:
            raise ValueError('%s=%r must be a finite number above 0 and at most %d' % (ENV_TIER_RESTORE_SLICE_MS, raw_ms, MAX_TIER_SLICE_MS))
        if not blocks:
            raise ValueError('%s is set and %s is not: the time budget would be inert' % (ENV_TIER_RESTORE_SLICE_MS, ENV_TIER_RESTORE_SLICE))
    return blocks, millis


class SpillPlan(object):
    """The complete-spill mode's numbers (spill_plan): hard_max (the most blocks one flush spills), batch (blocks per device read) and warn_ms."""

    __slots__ = ('hard_max', 'batch', 'warn_ms')


def _spill_whole(environ, name, default, low, high):
    raw = environ.get(name)
    if raw is None or raw == '':
        return default
    if not raw.isascii() or not raw.isdigit():
        raise ValueError('%s=%r must be a whole number in plain digits' % (name, raw))
    value = int(raw)
    if not low <= value <= high:
        raise ValueError('%s=%r must be from %d to %d' % (name, raw, low, high))
    return value


def spill_plan(environ, config):
    """The complete-spill mode (ENV_TIER_SPILL_MODE=complete) from the environment, None for the default cap mode. Strict: a mode that is neither, the mode without the host
    tier, a knob without the mode (an arm must not run the cap while believing otherwise), a hard maximum below the cap (the mode would spill LESS than the cap), a batch above the hard maximum."""
    mode = environ.get(ENV_TIER_SPILL_MODE)
    if mode is not None and mode != '' and mode not in TIER_SPILL_MODES:
        raise ValueError('%s=%r must be one of %s' % (ENV_TIER_SPILL_MODE, mode, ', '.join(TIER_SPILL_MODES)))
    knobs = [name for name in (ENV_TIER_SPILL_HARD_MAX, ENV_TIER_SPILL_BATCH, ENV_TIER_SPILL_WARN_MS) if environ.get(name) not in (None, '')]
    if mode != 'complete':
        if knobs:
            raise ValueError('%s set without %s=complete: the cap mode would ignore it' % (', '.join(knobs), ENV_TIER_SPILL_MODE))
        return None
    if config is None:
        raise ValueError('%s=complete needs the host KV tier (%s > 0): without it there is nothing to spill to' % (ENV_TIER_SPILL_MODE, ENV_TIER_GIB))
    plan = SpillPlan()
    plan.hard_max = _spill_whole(environ, ENV_TIER_SPILL_HARD_MAX, DEFAULT_TIER_SPILL_HARD_MAX_BLOCKS, 1, MAX_TIER_SPILL_BLOCKS)
    plan.batch = _spill_whole(environ, ENV_TIER_SPILL_BATCH, min(DEFAULT_TIER_SPILL_BATCH_BLOCKS, plan.hard_max), 1, MAX_TIER_SPILL_BLOCKS)   # (the default batch never exceeds a hard maximum below it)
    plan.warn_ms = _spill_whole(environ, ENV_TIER_SPILL_WARN_MS, DEFAULT_TIER_SPILL_WARN_MS, 1, 600000)
    if plan.hard_max < config.spill_max:
        raise ValueError('%s=%d is below %s=%d: the complete mode would spill less than the cap' % (ENV_TIER_SPILL_HARD_MAX, plan.hard_max, ENV_TIER_SPILL_MAX, config.spill_max))
    if plan.batch > plan.hard_max:
        raise ValueError('%s=%d is above %s=%d' % (ENV_TIER_SPILL_BATCH, plan.batch, ENV_TIER_SPILL_HARD_MAX, plan.hard_max))
    return plan


class QuotaConfig(object):
    """The per-tenant quotas (quota_config): device_cap_blocks and device_floor_blocks (0: none) in 64-token blocks, host_cap_bytes and host_floor_bytes (0: none)."""

    __slots__ = ('device_cap_blocks', 'device_floor_blocks', 'host_cap_bytes', 'host_floor_bytes')

    @property
    def device_on(self):
        return bool(self.device_cap_blocks or self.device_floor_blocks)

    @property
    def host_on(self):
        return bool(self.host_cap_bytes or self.host_floor_bytes)

    def describe(self):
        return dict(device_cap_tokens=self.device_cap_blocks * BLOCK, device_floor_tokens=self.device_floor_blocks * BLOCK,
                    host_cap_bytes=self.host_cap_bytes, host_floor_bytes=self.host_floor_bytes)


def _quota_tokens(environ, name):
    """A token count from the environment: unset or empty is 0 (off); anything else must be plain digits (no sign, no space, no exponent) and a whole number of blocks."""
    raw = environ.get(name)
    if raw is None or raw == '':
        return 0
    if not (raw.isascii() and raw.isdigit()):
        raise ValueError('%s=%r is not a whole number of tokens (digits only)' % (name, raw))
    value = int(raw)
    if value % BLOCK:
        raise ValueError('%s=%r is not a multiple of the %d-token block' % (name, raw, BLOCK))
    return value // BLOCK


def _quota_gib(environ, name):
    raw = environ.get(name)
    if raw is None or raw == '':
        return 0
    try:
        value = float(raw)
    except ValueError:
        raise ValueError('%s=%r is not a number of GiB' % (name, raw))
    if not (value >= 0.0) or value == float('inf'):
        raise ValueError('%s=%r must be a finite number >= 0' % (name, raw))
    return int(value * (1 << 30))


def quota_config(environ, tier_kv_bytes=None):
    """The per-tenant quotas from the environment, None when none is named (or all are 0). Strict: a cap below one 2048-token chunk (a tenant could never keep a
    checkpoint's worth of KV), a floor above its cap, a host quota without the host tier, and a host floor above half the tier's KV budget (two tenants could not both be
    guaranteed) refuse to start."""
    config = QuotaConfig()
    config.device_cap_blocks = _quota_tokens(environ, ENV_QUOTA_DEVICE_CAP)
    config.device_floor_blocks = _quota_tokens(environ, ENV_QUOTA_DEVICE_FLOOR)
    config.host_cap_bytes = _quota_gib(environ, ENV_QUOTA_HOST_CAP)
    config.host_floor_bytes = _quota_gib(environ, ENV_QUOTA_HOST_FLOOR)
    if not (config.device_on or config.host_on):
        return None
    if config.device_cap_blocks and config.device_cap_blocks * BLOCK < CHUNK:
        raise ValueError('%s=%d is below one %d-token chunk' % (ENV_QUOTA_DEVICE_CAP, config.device_cap_blocks * BLOCK, CHUNK))
    if config.device_cap_blocks and config.device_floor_blocks > config.device_cap_blocks:
        raise ValueError('%s=%d exceeds %s=%d' % (ENV_QUOTA_DEVICE_FLOOR, config.device_floor_blocks * BLOCK, ENV_QUOTA_DEVICE_CAP, config.device_cap_blocks * BLOCK))
    if config.host_on:
        if tier_kv_bytes is None:
            raise ValueError('%s / %s need the host KV tier (%s > 0): without it the quota would be inert' % (ENV_QUOTA_HOST_CAP, ENV_QUOTA_HOST_FLOOR, ENV_TIER_GIB))
        if config.host_cap_bytes and config.host_floor_bytes > config.host_cap_bytes:
            raise ValueError('%s exceeds %s' % (ENV_QUOTA_HOST_FLOOR, ENV_QUOTA_HOST_CAP))
        if config.host_floor_bytes * 2 > int(tier_kv_bytes):
            raise ValueError('%s=%s is more than half the tier\'s KV budget (%d bytes): two tenants could not both be guaranteed it' % (
                ENV_QUOTA_HOST_FLOOR, environ.get(ENV_QUOTA_HOST_FLOOR), int(tier_kv_bytes)))
    return config


class TierRecord(object):
    """One evicted attention-KV block held in host RAM: its raw packed bytes (payload), the chain it belongs to (the first block's hash),
    its position in it (index) and its tenant, and the sha256 of the bytes once the digest worker has computed it."""

    __slots__ = ('key', 'payload', 'nbytes', 'tenant', 'chain', 'index', 'digest', 'serial', 'slot')

    def __init__(self, key, payload, tenant, chain, index, serial, slot=None):
        self.key = key
        self.payload = payload
        self.slot = slot
        self.nbytes = len(payload)
        self.tenant = tenant
        self.chain = chain
        self.index = index
        self.digest = None
        self.serial = serial


class Slot(object):
    """One slot of the tier's slab, reserved for a block about to be read into it (view is writable); commit files it under a key, abort frees it."""

    __slots__ = ('index', 'view', 'tenant')

    def __init__(self, index, view, tenant=None):
        self.index = index
        self.view = view
        self.tenant = tenant


class TierStore(object):
    """The host KV tier's records: block hash -> TierRecord, bounded by bytes. The key is vLLM's chained block hash (it carries the salt,
    so a tenant cannot form another's key); a record is the bytes the device held for that block, read raw and packed. Eviction is the
    registry's policy: dead records first (those `dead` calls useless: no checkpoint of the chain reaches them), then the least recently
    used (lru) or the oldest record of the tenant holding the most bytes (fair), sparing the record just stored. Digests are computed
    off the scheduler thread (hashlib releases the GIL), or inline with digest_inline (tests, the audit).

    Two ways to hold the bytes. Heap mode (the default, what the unit tests use, and what a slab that cannot be mapped falls back to): a record keeps the payload object it was given. Slab mode
    (configure(slot_bytes), which attach_tier_io calls with the model's block size): one anonymous mapping, cap // slot_bytes slots, kept out of core
    dumps; a record is a slot, an eviction returns the slot, and the device read is made straight into a reserved slot - so the host does one copy
    of a block, into memory that is already resident after the first fill, instead of a fresh allocation and page faults per block."""

    def __init__(self, cap_bytes, policy='lru', fingerprint='', digest_inline=False, on_evict=None, dead=None, tenant_cap=0, tenant_floor=0, counters=None):
        self.cap = int(cap_bytes)
        # Per-tenant quotas in bytes (0: none): a tenant over tenant_cap loses its OWN oldest records first; a record whose removal would take its tenant below
        # tenant_floor is never another tenant's victim. counters is the dict the aggregate counters (QUOTA_STAT_NAMES) are kept in.
        self.tenant_cap = int(tenant_cap or 0)
        self.tenant_floor = int(tenant_floor or 0)
        self.counters = counters if counters is not None else {}
        self.reserved = {}
        self.policy = policy
        self.fingerprint = fingerprint
        self.records = OrderedDict()
        self.by_tenant = {}
        self.tenant_bytes = {}
        self.bytes = 0
        self.on_evict = on_evict
        self.dead = dead
        self.digest_inline = digest_inline
        self._executor = None
        self._serials = itertools.count(1)
        self.slot_bytes = 0
        self.slab_error = None
        self.pinned = set()
        self._map = None
        self._view = None
        self._free = []
        self.stats = {'puts': 0, 'duplicates': 0, 'refused_size': 0, 'refused_full': 0, 'evicted': 0, 'hits': 0, 'misses': 0,
                      'digested': 0}

    # -- the slab ----------------------------------------------------------------------------------
    def configure(self, slot_bytes):
        """Switch to slab mode for blocks of slot_bytes. The cap becomes a whole number of slots. Only on an empty store, once."""
        import mmap

        slot_bytes = int(slot_bytes)
        if self.slot_bytes == slot_bytes:
            return True
        if self.slot_bytes or self.records:
            raise ValueError('the tier store is already holding data in another layout')
        slots = self.cap // slot_bytes
        if slots < 1:
            raise ValueError('the tier cap (%d bytes) does not hold one block of %d bytes' % (self.cap, slot_bytes))
        try:
            mapping = mmap.mmap(-1, slots * slot_bytes)
        except (OSError, ValueError, OverflowError, MemoryError) as error:
            # No room for one mapping of the whole cap (a strict overcommit policy, a small host): the store stays in heap mode, one array per record,
            # and its cap still bounds the bytes. A tier that could not start would be worse than a slower one.
            self.slab_error = '%s: %s' % (type(error).__name__, error)
            return False
        self.slot_bytes = slot_bytes
        self.cap = slots * slot_bytes
        self._map = mapping
        try:
            self._map.madvise(mmap.MADV_DONTDUMP)
        except (AttributeError, OSError):
            pass
        self._view = memoryview(self._map)
        self._free = list(range(slots - 1, -1, -1))
        return True

    def _slot_view(self, index):
        return self._view[index * self.slot_bytes:(index + 1) * self.slot_bytes]

    def _take_slot(self, tenant=None):
        if self.tenant_cap and not self._enforce_cap(tenant, self.slot_bytes):
            self._count('quota_host_refused')
            return None
        if not self._free:
            victim = self._victim()
            if victim is None:
                return None
            self.remove(victim, evicted=True, reason='global')
        return Slot(self._free.pop(), None, tenant) if self._free else None

    def reserve(self, count, tenants=None):
        """Up to `count` slots to read blocks into, evicting by policy to make them (fewer when the store cannot: every slot is a
        reservation). `tenants` (one per slot) lets the per-tenant cap apply before the slot is taken. Each must be committed or aborted."""
        slots = []
        for position in range(count):
            tenant = tenants[position] if tenants else None
            slot = self._take_slot(tenant)
            if slot is None:
                break
            slot.view = self._slot_view(slot.index)
            if self.tenant_cap:
                self.reserved[tenant] = self.reserved.get(tenant, 0) + self.slot_bytes
            slots.append(slot)
        return slots

    def _release_reservation(self, slot):
        if self.tenant_cap:
            left = self.reserved.get(slot.tenant, 0) - self.slot_bytes
            if left > 0:
                self.reserved[slot.tenant] = left
            else:
                self.reserved.pop(slot.tenant, None)

    def abort(self, slots):
        for slot in slots:
            self._release_reservation(slot)
            self._free.append(slot.index)

    def commit(self, slot, key, tenant=None, chain=None, index=0):
        """File a reserved slot, now holding a block's bytes, under key. A key already held keeps its record and the slot goes back."""
        self._release_reservation(slot)
        existing = self.records.get(key)
        if existing is not None:
            self.records.move_to_end(key)
            self.by_tenant[existing.tenant].move_to_end(key)
            self.stats['duplicates'] += 1
            self._free.append(slot.index)
            return True
        return self._file(TierRecord(key, slot.view, tenant, chain, int(index), next(self._serials), slot.index))

    def _file(self, record):
        self.records[record.key] = record
        self.by_tenant.setdefault(record.tenant, OrderedDict())[record.key] = record
        self.tenant_bytes[record.tenant] = self.tenant_bytes.get(record.tenant, 0) + record.nbytes
        self.bytes += record.nbytes
        self.stats['puts'] += 1
        self._digest(record)
        return True

    # -- reads -------------------------------------------------------------------------------------
    def has(self, key):
        return key in self.records

    def get(self, key, touch=True):
        record = self.records.get(key)
        if record is None:
            self.stats['misses'] += 1
            return None
        self.stats['hits'] += 1
        if touch:
            self.records.move_to_end(key)
            self.by_tenant[record.tenant].move_to_end(key)
        return record

    # -- writes ------------------------------------------------------------------------------------
    def put(self, key, payload, tenant=None, chain=None, index=0):
        """Store payload under key. False (and nothing stored) when the payload alone exceeds the cap or no eviction can make room. A key
        already held keeps its record (the bytes of one hash are one block) and is touched."""
        size = len(payload)
        existing = self.records.get(key)
        if existing is not None:
            self.records.move_to_end(key)
            self.by_tenant[existing.tenant].move_to_end(key)
            self.stats['duplicates'] += 1
            return True
        if size > self.cap or (self.slot_bytes and size != self.slot_bytes):
            self.stats['refused_size'] += 1
            return False
        if self.slot_bytes:
            slot = self._take_slot(tenant)
            if slot is None:
                self.stats['refused_full'] += 1
                return False
            view = self._slot_view(slot.index)
            view[:] = memoryview(payload).cast('B')
            return self._file(TierRecord(key, view, tenant, chain, int(index), next(self._serials), slot.index))
        if self.tenant_cap and not self._enforce_cap(tenant, size):
            self._count('quota_host_refused')
            self.stats['refused_full'] += 1
            return False
        if not self._make_room(size):
            self.stats['refused_full'] += 1
            return False
        return self._file(TierRecord(key, payload, tenant, chain, int(index), next(self._serials)))

    def _make_room(self, need):
        while self.bytes + need > self.cap:
            victim = self._victim()
            if victim is None:
                return False
            self.remove(victim, evicted=True, reason='global')
        return True

    def _count(self, name, value=1):
        if name in self.counters:
            self.counters[name] += value

    def _enforce_cap(self, tenant, need):
        """The per-tenant cap: evict `tenant`'s own oldest unpinned records until its bytes (and the reservations it already holds) plus `need` fit under tenant_cap.
        True when they do; False when `need` alone exceeds the cap or what is left is pinned."""
        cap = self.tenant_cap
        if need > cap:
            return False
        group = self.by_tenant.get(tenant)
        while self.tenant_bytes.get(tenant, 0) + self.reserved.get(tenant, 0) + need > cap:
            victim = None
            for key in (group or ()):
                if key not in self.pinned:
                    victim = key
                    break
            if victim is None:
                return False
            self.remove(victim, evicted=True, reason='own')
            group = self.by_tenant.get(tenant)
        return True

    def _protected(self, record):
        """Whether removing `record` would take its tenant below the floor: it is then no other tenant's victim."""
        return bool(self.tenant_floor) and self.tenant_bytes.get(record.tenant, 0) - record.nbytes < self.tenant_floor

    def pin(self, keys):
        """Keep these records through any eviction until unpin (a restore is about to write their bytes to the device: in slab mode an eviction
        would let a spill reuse the very slot being read)."""
        self.pinned.update(keys)

    def unpin(self, keys):
        self.pinned.difference_update(keys)

    def _victim(self):
        pinned = self.pinned
        if self.dead is not None:
            for position, (key, record) in enumerate(self.records.items()):
                if position >= TIER_DEAD_SCAN:
                    break
                if key not in pinned and self.dead(record):
                    return key
        held = 0
        found = None
        if self.policy == 'fair' and self.tenant_bytes:
            # The tenants holding the most bytes first: the oldest unpinned record of the first that has one.
            for tenant in sorted(self.tenant_bytes, key=lambda name: -self.tenant_bytes[name]):
                for key in self.by_tenant[tenant]:
                    if key in pinned:
                        continue
                    if self.tenant_floor and self._protected(self.records[key]):
                        held += 1
                        continue
                    found = key
                    break
                if found is not None:
                    break
        else:
            for key, record in self.records.items():
                if key in pinned:
                    continue
                if self.tenant_floor and self._protected(record):
                    held += 1
                    continue
                found = key
                break
        if held:
            self._count('quota_host_floor_protected', held)
        return found

    def remove(self, key, evicted=False, reason=None):
        record = self.records.pop(key, None)
        if record is None:
            return None
        group = self.by_tenant.get(record.tenant)
        if group is not None:
            group.pop(key, None)
            if not group:
                del self.by_tenant[record.tenant]
        left = self.tenant_bytes.get(record.tenant, 0) - record.nbytes
        if left > 0:
            self.tenant_bytes[record.tenant] = left
        else:
            self.tenant_bytes.pop(record.tenant, None)
        self.bytes -= record.nbytes
        if record.slot is not None:
            self._free.append(record.slot)
        if evicted:
            self.stats['evicted'] += 1
            if reason is not None:
                self._count('quota_host_evictions_' + reason)
            if self.on_evict is not None:
                self.on_evict(record)
        return record

    def clear(self):
        for record in self.records.values():
            if record.slot is not None:
                self._free.append(record.slot)
        self.records.clear()
        self.by_tenant.clear()
        self.tenant_bytes.clear()
        self.bytes = 0

    # -- digests -----------------------------------------------------------------------------------
    def _digest(self, record):
        if self.digest_inline:
            record.digest = hashlib.sha256(record.payload).hexdigest()
            self.stats['digested'] += 1
            return
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor

            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='qwen-tier-digest')
        self._executor.submit(self._compute_digest, record)

    def _compute_digest(self, record):
        try:
            record.digest = hashlib.sha256(record.payload).hexdigest()
            self.stats['digested'] += 1
        except Exception:
            pass

    def verify(self, record):
        """True when the record's bytes still hash to the digest taken when it was stored, None when the digest is not ready yet (nothing
        is claimed), False on a difference."""
        digest = record.digest
        if digest is None:
            return None
        return hashlib.sha256(record.payload).hexdigest() == digest

    def close(self):
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True)

    def tenants_at_cap(self):
        """How many tenants hold their cap (an aggregate: the count, never who)."""
        cap = self.tenant_cap
        return sum(1 for held in self.tenant_bytes.values() if cap and held + (self.slot_bytes or 1) > cap)

    def snapshot(self):
        return dict(bytes=self.bytes, entries=len(self.records), cap=self.cap, tenants=len(self.tenant_bytes),
                    slots=(self.cap // self.slot_bytes) if self.slot_bytes else 0, free_slots=len(self._free))


class PrefixRegistry(object):
    """The checkpoint LRU (bounded by bytes, pinned entries never evicted) and the grant ledger."""

    def __init__(self, budget_bytes=None, environ=None, clock=time.monotonic):
        if budget_bytes is None:
            budget_bytes = store_budget_bytes(environ)
        flags = os.environ if environ is None else environ
        self.budget_bytes = int(budget_bytes)
        # Store hygiene and telemetry (the module docstring): strict, and off or default-equivalent when unset.
        self.evict_policy = evict_policy(flags)
        self.supersede = strict_flag(flags, ENV_SUPERSEDE, False)
        self.telemetry = strict_flag(flags, ENV_TELEMETRY, True)
        # What a capture is charged when the model does not say (the state dtype's checkpoint).
        self.checkpoint_nbytes = checkpoint_nbytes(flags)
        self.clock = clock
        # The host KV tier (the module docstring): off unless QWEN_PREFIX_HOST_TIER_GIB names a budget. tier_io is the model's side
        # (read_blocks / write_blocks), attached by the model graft; the scheduler graft's TierHooks refuse to install without it.
        self.tier_config = tier_config(flags, self.budget_bytes)
        # Complete spills (SpillPlan; None: the cap): the counters it adds exist only then.
        self.spill_plan = spill_plan(flags, self.tier_config)
        # Threaded staging and parallel digests (UploadPlan; None: the stock loops): the counters exist only then.
        self.upload_plan = None
        if any(flags.get(name) not in (None, '') for name in ENV_UPLOAD_NAMES):
            import qwen_tier_upload

            self.upload_plan = qwen_tier_upload.upload_plan(flags, self.tier_config)
        self.tier = None
        self.tier_io = None
        self.device_holds = None
        # Per-tenant quotas (QUOTA_ENV_NAMES): None unless named. The counters exist only then, so an engine without one keeps the stats it had.
        self.quota = quota_config(flags, self.tier_config.kv_bytes if self.tier_config is not None else None)
        # Blocks past a checkpoint's boundary a restore must still have: 1 when vLLM drops a hit's last block (sticky sessions), else 0.
        self.tier_slack = 0
        quota_counters = dict.fromkeys(QUOTA_STAT_NAMES, 0) if self.quota is not None else {}
        if self.tier_config is not None:
            self.tier = TierStore(self.tier_config.kv_bytes, self.evict_policy, digest_inline=self.tier_config.audit,
                                  on_evict=self._tier_evicted, dead=self._tier_dead,
                                  tenant_cap=self.quota.host_cap_bytes if self.quota is not None else 0,
                                  tenant_floor=self.quota.host_floor_bytes if self.quota is not None else 0, counters=quota_counters)
        self.entries = OrderedDict()
        self.bytes = 0
        # chain (first block hash) -> checkpoint keys, and tenant -> bytes held: the hygiene's indexes.
        self.by_chain = {}
        self.tenant_bytes = {}
        # Telemetry: the returning-session memory (block hash at a turn's last boundary -> Ghost), the first-block hashes
        # seen, each waiting request's latest attempt, the requests already classified (a preempted request is admitted
        # again), the engine's cumulative admitted prompt tokens and the two reuse-distance histograms.
        self.ghost = OrderedDict()
        self.ghost_limit = ghost_limit(flags) if self.telemetry else 0
        self.chains = OrderedDict()
        self.attempts = OrderedDict()
        self.classified = OrderedDict()
        self.admitted_tokens = 0
        self.reuse = {'seconds': {name: [[0] * (len(REUSE_SECONDS_BOUNDS) + 1), 0.0] for name in REUSE_OUTCOMES},
                      'tokens': {name: [[0] * (len(REUSE_TOKENS_BOUNDS) + 1), 0.0] for name in REUSE_OUTCOMES}}
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
        if self.tier_config is not None and self.tier_config.restore_slice:
            self.stats.update(dict.fromkeys(SLICE_STAT_NAMES, 0))
        elif flags.get(ENV_TIER_RESTORE_SLICE) not in (None, '', '0') or flags.get(ENV_TIER_RESTORE_SLICE_MS) not in (None, ''):
            raise ValueError('%s / %s need the host KV tier (%s > 0): without it the sliced restore would be inert' % (
                ENV_TIER_RESTORE_SLICE, ENV_TIER_RESTORE_SLICE_MS, ENV_TIER_GIB))
        if self.spill_plan is not None:
            self.stats.update(dict.fromkeys(SPILL_STAT_NAMES, 0))
        if self.upload_plan is not None:
            self.stats.update(dict.fromkeys(UPLOAD_STAT_NAMES, 0))
        if quota_counters:
            # One dict: the tier store counts into the registry's own stats.
            self.stats.update(quota_counters)
            if self.tier is not None:
                self.tier.counters = self.stats
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
            self._forget_sessions()
        self._owner = weakref.ref(owner)

    def enable_mid_loop_capture(self):
        """The model graft's declaration that it captures at pos after exactly pos tokens even when
        pos is below its loop's drain point. Until it is made, only drain captures are planned."""
        if not self.mid_loop_capture:
            self.mid_loop_capture = True
            log('model declares mid-loop captures: gap and resumed-prompt boundaries are planned')

    # -- the host KV tier ----------------------------------------------------------------------
    def attach_tier_io(self, io):
        """The model graft's side of the tier: io.read_blocks(block ids) -> one payload per block, io.write_blocks(block ids, payloads),
        io.block_bytes and io.fingerprint. Without it (or without QWEN_PREFIX_HOST_TIER_GIB) the tier does nothing."""
        plan = self.upload_plan
        if plan is not None and plan.io_threads:
            import qwen_tier_upload

            previous = self.tier_io
            if isinstance(previous, qwen_tier_upload.ThreadedTierIO):
                previous.close()
            if isinstance(io, qwen_tier_upload.ThreadedTierIO):
                io = io.io
            io = qwen_tier_upload.ThreadedTierIO(io, plan.io_threads, plan.io_chunk, self.stats, log=log)
        self.tier_io = io
        if self.tier is not None:
            self.tier.fingerprint = str(getattr(io, 'fingerprint', ''))
            block_bytes = int(getattr(io, 'block_bytes', 0) or 0)
            if block_bytes and not self.tier.configure(block_bytes):
                log('host tier slab not mapped (%s): the blocks are held as separate arrays, the cap still bounds the bytes', self.tier.slab_error)

    @property
    def tier_on(self):
        return self.tier is not None and self.tier_io is not None and not self.stats['tier_latched'] and self.disabled is None

    def tier_latch(self, reason):
        """Turn the tier off for the life of the process (a failing device read or write, its kill switch): the records go."""
        if not self.stats['tier_latched']:
            self.stats['tier_latched'] = 1
            log('host tier latched off: %s', reason)
        if self.tier is not None:
            self.tier.clear()

    def _tier_dead(self, record):
        """Whether no checkpoint reaches the block (the tier's eviction takes such records first, and the spill never stores them): a KV
        block is useful only with a checkpoint of its chain at or above it, and above the smallest session worth a restore."""
        keys = self.by_chain.get(record.chain)
        if not keys:
            return True
        need = max((record.index + 1 - self.tier_slack) * BLOCK, self.tier_config.min_tokens if self.tier_config is not None else 0)
        for key in keys:
            entry = self.entries.get(key)
            if entry is not None and entry.pos >= need:
                return False
        return True

    def _tier_evicted(self, record):
        """The tier gave a record up: the checkpoint at its key (the boundary block) has no KV left in the tier, so unless the device
        still holds the block it can never be granted - drop it."""
        holds = self.device_holds
        if self.entries.get(record.key) is not None and (holds is None or not holds(record.key)):
            self.drop(record.key, 'coupled')
            self.stats['tier_ckpt_dropped'] += 1
        self.stats['tier_evicted'] += 1

    # -- checkpoints ---------------------------------------------------------------------------
    def get(self, key):
        return self.entries.get(key)

    def put(self, key, pos, token_ids, rec=None, carry=None, nbytes=0, chain=None, tenant=None, shared=False):
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
        checkpoint = Checkpoint(key, pos, ids, rec, carry, nbytes, next(self._serials), chain, tenant, shared)
        self.entries[key] = checkpoint
        self.bytes += checkpoint.nbytes
        self._index(checkpoint)
        ghost = self.ghost.get(key)
        if ghost is not None:
            ghost.state = 'stored'
        self.stats['captures'] += 1
        self._enforce_budget(protect=key)
        return checkpoint

    def touch(self, key):
        if key in self.entries:
            self.entries.move_to_end(key)

    def _remove(self, key, reason=None):
        """Take a checkpoint out. reason (one of GONE_REASONS) is what the telemetry remembers of why it left; None for
        a replacement, which leaves a checkpoint of the same key in its place."""
        checkpoint = self.entries.pop(key, None)
        if checkpoint is not None:
            self.bytes -= checkpoint.nbytes
            self.orphan_keys.discard(key)
            self._unindex(checkpoint)
            if reason is not None:
                ghost = self.ghost.get(key)
                if ghost is not None:
                    ghost.state = reason
        return checkpoint

    def _index(self, checkpoint):
        if checkpoint.chain is not None:
            self.by_chain.setdefault(checkpoint.chain, set()).add(checkpoint.key)
        self.tenant_bytes[checkpoint.tenant] = self.tenant_bytes.get(checkpoint.tenant, 0) + checkpoint.nbytes

    def _unindex(self, checkpoint):
        if checkpoint.chain is not None:
            keys = self.by_chain.get(checkpoint.chain)
            if keys is not None:
                keys.discard(checkpoint.key)
                if not keys:
                    del self.by_chain[checkpoint.chain]
        left = self.tenant_bytes.get(checkpoint.tenant, 0) - checkpoint.nbytes
        if left > 0:
            self.tenant_bytes[checkpoint.tenant] = left
        else:
            self.tenant_bytes.pop(checkpoint.tenant, None)

    def drop(self, key, reason='dropped'):
        checkpoint = self._remove(key, reason if reason in GONE_REASONS else 'other')
        if checkpoint is not None:
            name = 'evicted_' + reason
            self.stats[name if name in self.stats else 'dropped'] += 1
        return checkpoint

    def _enforce_budget(self, protect=None):
        if self.bytes <= self.budget_bytes:
            return
        if self.evict_policy == 'fair':
            self._enforce_fair(protect)
            return
        for key in list(self.entries):
            if self.bytes <= self.budget_bytes:
                break
            if self.entries[key].pins:
                continue
            self._remove(key, 'budget')
            self.stats['evicted_lru'] += 1

    def _enforce_fair(self, protect=None):
        while self.bytes > self.budget_bytes:
            victim = self._fair_victim(protect)
            if victim is None:
                return
            self._remove(victim, 'budget')
            self.stats['evicted_fair'] += 1

    def _fair_victim(self, protect=None):
        """The key to evict under the fair policy: the least recently used unpinned checkpoint of the tenant holding the
        most bytes (a tie goes to the tenant whose candidate is older), so a tenant that fills the store pays for it
        before a tenant that holds a little does. The checkpoint just stored is spared unless nothing else can go;
        pinned ones never go."""
        oldest, age, spared = {}, {}, None
        for index, (key, entry) in enumerate(self.entries.items()):
            if entry.pins:
                continue
            if key == protect:
                spared = key
                continue
            if entry.tenant not in oldest:
                oldest[entry.tenant] = key
                age[entry.tenant] = index
        if not oldest:
            return spared
        tenant = max(oldest, key=lambda name: (self.tenant_bytes.get(name, 0), -age[name]))
        return oldest[tenant]

    def _supersede(self, newest):
        """Retire what a newer checkpoint of the same conversation makes dead weight (QWEN_PREFIX_SUPERSEDE=1).

        Candidates are the checkpoints of newest's chain that sit below it ON ITS OWN TOKEN PATH (their token ids are
        the start of newest's, compared byte for byte), so another conversation that merely shares the first block is
        never touched. The newest of them is the previous turn and stays: an edit or a retry of the last message
        resumes there. Of the older ones, a checkpoint stays when it is pinned (this step restores it), shared
        (taken at a gap boundary: other sessions' prefixes reach it) or a branch point (granted to two admissions:
        something else resumes from it). Dropping a checkpoint can only turn a later hit into a miss - what is served
        is still token-verified - and the telemetry names that miss (class_ckpt_evicted_superseded)."""
        chain = newest.chain
        if chain is None or self.entries.get(newest.key) is not newest:
            return
        stats = self.stats
        stats['supersede_checks'] += 1
        older = [entry for entry in (self.entries.get(key) for key in tuple(self.by_chain.get(chain, ())))
                 if entry is not None and entry.pos < newest.pos and is_prefix(entry.token_ids, newest.token_ids)]
        if len(older) < 2:
            return
        older.sort(key=lambda entry: entry.pos)
        for entry in older[:-1]:
            if entry.pins:
                stats['supersede_kept_pinned'] += 1
            elif entry.shared:
                stats['supersede_kept_shared'] += 1
            elif entry.grants > 1:
                stats['supersede_kept_branch'] += 1
            else:
                self._remove(entry.key, 'superseded')
                stats['evicted_superseded'] += 1

    def clear(self):
        """Drop every checkpoint. Staged and committed grants hold their own checkpoint reference
        and die at the next begin_step, so a clear inside schedule() (the kill switch) or between
        steps (reset_prefix_cache) never breaks an admission vLLM has already been handed."""
        for key in self.entries:
            ghost = self.ghost.get(key)
            if ghost is not None:
                ghost.state = 'cleared'
        self.entries.clear()
        self.bytes = 0
        self.by_chain.clear()
        self.tenant_bytes.clear()
        self.orphan_keys.clear()
        self.token_checks.clear()
        if self.tier is not None:
            self.tier.clear()
        self.stats['clears'] += 1

    def disable(self, reason):
        """Latch off for the life of the process: no new grants, captures or publishing."""
        if self.disabled is None:
            self.disabled = str(reason)
        self.inflight.clear()
        self.clear()
        self.attempts.clear()

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
                try:
                    grant.checkpoint.grants += 1
                except AttributeError:
                    pass
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
        self._classify_admitted(admitted)
        return done

    def _note_split(self, grant, start, scheduled):
        """A grant admitted for fewer tokens than the rest of its prompt: keep its plan for the later steps."""
        if scheduled is None or not grant.plan or grant.prompt is None or grant.tokens is None:
            return
        tokens = scheduled.get(grant.req_id)
        if type(tokens) is not int or start + tokens >= grant.prompt:
            return
        self.inflight[grant.req_id] = Inflight(grant.plan, grant.tokens, grant.prompt, grant.chain, grant.tenant, grant.gap)
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
                origin = grant
            else:
                carried = self.inflight.get(req_id)
                if carried is None:
                    raise KeyError('no committed grant for %s' % req_id)
                keys, tokens = carried.plan, carried.tokens
                origin = carried
            if pos not in keys:
                raise KeyError('boundary %d is not planned for %s' % (pos, req_id))
            if loop_pos != pos:
                self.stats['capture_wrong_position'] += 1
                raise ValueError('the state was taken after %r tokens, not %d: refused (it would be '
                                 'restored as the state at %d)' % (loop_pos, pos, pos))
            if ms is not None:
                self.stats['capture_ms'] += float(ms)
            kept = self.stats['capture_kept_pinned']
            stored = self.put(keys[pos], pos, tokens[0:pos], rec, carry,
                              self.checkpoint_nbytes if nbytes is None else nbytes,
                              chain=getattr(origin, 'chain', None), tenant=getattr(origin, 'tenant', None),
                              shared=getattr(origin, 'gap', None) is not None and pos == origin.gap)
            if carried is not None and stored is not None:
                self.stats['inflight_captures'] += 1
            if self.supersede and stored is not None and self.stats['capture_kept_pinned'] == kept:
                try:
                    self._supersede(stored)
                except Exception as error:
                    log('supersession skipped at %s: %s: %s', pos, type(error).__name__, error)
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
        self.attempts.pop(req_id, None)
        self.classified.pop(req_id, None)
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

    # -- telemetry: classify every admission, remember returning sessions ----------------------------------
    def note_attempt(self, req_id, hashes, prompt, end, h, q, kind=None):
        """The trim's latest answer for a waiting request: vLLM's raw hit h, the trimmed Q, and `kind` when the trim refused
        the request outright ('unsalted', or 'denied' for a streaming session or the kill switch). hashes are the request's
        block hashes, prompt its prompt length and end the boundary its own checkpoint would sit at (the prompt boundary,
        less one chunk under sticky sessions). A request is re-attempted every step it waits; only the latest answer is
        kept, and classify_admitted settles it when the step admits the request. Reads only; never raises (S7)."""
        if not self.telemetry or req_id in self.classified:
            return
        try:
            rec = self.attempts.get(req_id)
            if rec is None:
                rec = Attempt()
                rec.prompt = int(prompt or 0)
                rec.end = int(end)
                rec.chain = hashes[0] if len(hashes) else None
                rec.end_key = hashes[end // BLOCK - 1] if 0 < end and end // BLOCK <= len(hashes) else None
                rec.ghost_key = rec.target = rec.target_key = None
                if kind is None and self.ghost:
                    self._match_ghost(rec, hashes)
                self.attempts[req_id] = rec
                while len(self.attempts) > ATTEMPTS_LIMIT:
                    self.attempts.popitem(last=False)
            rec.kind, rec.h, rec.q = kind, int(h), int(q)
        except Exception:
            self.stats['class_failures'] += 1

    def _match_ghost(self, rec, hashes):
        """The highest boundary of this prompt that an earlier turn left as its last: the request returns to that session.
        target is the boundary a full hit could reach - less than the ghost's when the prompt is no longer than it (a resent
        prompt whose length is a whole chunk cannot hit its own last token)."""
        top = min(len(hashes) * BLOCK, rec.prompt) // CHUNK
        ghost = self.ghost
        for k in range(top, 0, -1):
            key = hashes[k * CHUNK // BLOCK - 1]
            if key in ghost:
                target = min(k * CHUNK, floor_chunk(((rec.prompt - 1) // BLOCK) * BLOCK))
                if target <= 0:
                    return
                rec.ghost_key, rec.target = key, target
                rec.target_key = key if target == k * CHUNK else hashes[target // BLOCK - 1]
                return

    def _classify_admitted(self, admitted):
        """commit()'s last step: settle the attempt of every request the step admits, once. Never raises."""
        if not self.telemetry or not self.attempts:
            return
        for req_id, start in admitted.items():
            rec = self.attempts.pop(req_id, None)
            if rec is None or req_id in self.classified:
                continue
            try:
                self._classify(req_id, rec, int(start))
            except Exception as error:
                self.stats['class_failures'] += 1
                log('classification skipped req=%s: %s: %s', req_id, type(error).__name__, error)
            self.classified[req_id] = None
            while len(self.classified) > CLASSIFIED_LIMIT:
                self.classified.popitem(last=False)

    def _classify(self, req_id, rec, q):
        stats = self.stats
        now = self.clock()
        ghost = self.ghost.get(rec.ghost_key) if rec.ghost_key is not None else None
        lost = 0
        if rec.kind is not None:
            name = rec.kind
        elif rec.end <= 0:
            name = 'short'
        elif ghost is None:
            name = 'rewritten' if rec.chain in self.chains else 'first_turn'
        else:
            target = rec.target
            if q >= target:
                name = 'returning_served'
            else:
                lost = target - q
                if rec.h < target:
                    name = 'kv_evicted'
                elif self.entries.get(rec.target_key) is not None:
                    name = 'refused'
                    verdict = self.token_checks.get(req_id, {}).get(target)
                    if verdict is not None and verdict[1] is False:
                        stats['class_refused_mismatch'] += 1
                elif ghost.state is None:
                    name = 'ckpt_missing'
                else:
                    name = 'ckpt_evicted'
                    stats['class_ckpt_evicted_' + (ghost.state if ghost.state in GONE_REASONS else 'other')] += 1
            stats['returning_sessions'] += 1
            outcome = 'served' if name == 'returning_served' else 'missed'
            self._observe('seconds', outcome, max(0.0, now - ghost.seen))
            self._observe('tokens', outcome, max(0, self.admitted_tokens - ghost.counter))
        stats['class_' + name] += 1
        if q > 0 and name in CLASS_HIT:
            stats['class_%s_hit' % name] += 1
        if lost:
            stats['lost_tokens_' + name] += lost
        # The session's new last boundary, and the engine's running count of prompt tokens admitted.
        self.admitted_tokens += rec.prompt
        stats['admitted_prompt_tokens'] += rec.prompt
        if rec.kind is None and rec.chain is not None:
            self.chains[rec.chain] = now
            self.chains.move_to_end(rec.chain)
            while len(self.chains) > max(self.ghost_limit, 1):
                self.chains.popitem(last=False)
            if rec.end_key is not None and self.ghost_limit:
                record = self.ghost.get(rec.end_key)
                if record is None:
                    record = self.ghost[rec.end_key] = Ghost()
                else:
                    self.ghost.move_to_end(rec.end_key)
                record.seen, record.pos, record.prompt, record.counter = now, rec.end, rec.prompt, self.admitted_tokens
                record.state = 'stored' if rec.end_key in self.entries else None
                while len(self.ghost) > self.ghost_limit:
                    self.ghost.popitem(last=False)

    def _observe(self, unit, outcome, value):
        bounds = REUSE_SECONDS_BOUNDS if unit == 'seconds' else REUSE_TOKENS_BOUNDS
        cell = self.reuse[unit][outcome]
        cell[0][bisect.bisect_left(bounds, value)] += 1
        cell[1] += value

    def _forget_sessions(self):
        """A rebuilt engine starts with no sessions: the telemetry's memory of the old one would misread them."""
        self.ghost.clear()
        self.chains.clear()
        self.attempts.clear()
        self.classified.clear()

    @staticmethod
    def host_gauges():
        """The host gauges of this process (module-level host_gauges), for the exporter, which reads a registry and nothing else."""
        return host_gauges()

    def histograms(self):
        """The reuse-distance histograms: {'reuse_seconds' | 'reuse_tokens': {'bounds': [...], 'served' | 'missed':
        {'counts': per-bucket counts (one more than bounds: the open top bucket), 'sum': ...}}}. Empty without telemetry."""
        if not self.telemetry:
            return {}
        out = {}
        for unit, bounds in (('seconds', REUSE_SECONDS_BOUNDS), ('tokens', REUSE_TOKENS_BOUNDS)):
            entry = {'bounds': list(bounds)}
            for outcome in REUSE_OUTCOMES:
                counts, total = self.reuse[unit][outcome]
                entry[outcome] = {'counts': list(counts), 'sum': total}
            out['reuse_' + unit] = entry
        return out

    def telemetry_lines(self, host=None):
        """The two compact key=value lines the periodic export logs (the module docstring), without the "[PINDIAG]
        prefix:" lead. host: the host gauges (default: read now)."""
        values = self.snapshot()
        host = host_gauges() if host is None else host
        head = ['tier']
        for key, name in (('host_rss_bytes', 'rss'), ('host_mem_available_bytes', 'avail')):
            if key in host:
                head.append('%s=%d' % (name, host[key]))
        head.extend('%s=%d' % (name, values[key]) for name, key in (
            ('reg_bytes', 'bytes'), ('reg_entries', 'entries'), ('reg_budget', 'budget_bytes'), ('ghost', 'ghost_now'),
            ('adm', 'admissions'), ('adm_tokens', 'admitted_prompt_tokens'), ('grants', 'grants'),
            ('returning', 'returning_sessions')))
        head.append('evict=%s' % self.evict_policy)
        head.append('supersede=%d' % int(self.supersede))
        if self.tier is not None:
            head.extend('%s=%d' % (name, values[key]) for name, key in (
                ('kv_bytes', 'tier_bytes'), ('kv_entries', 'tier_entries'), ('kv_spilled', 'tier_spill_blocks'),
                ('kv_restored', 'tier_restore_blocks'), ('kv_restore_reqs', 'tier_restore_requests'),
                ('kv_digest_fail', 'tier_digest_failures'), ('kv_on', 'tier_on')))
        head.extend('%s=%d' % (name, values['class_' + name]) for name in CLASSES)
        head.extend('%s=%d' % (name, values[key]) for name, key in (
            ('lost_kv', 'lost_tokens_kv_evicted'), ('lost_ckpt', 'lost_tokens_ckpt_evicted'),
            ('lost_missing', 'lost_tokens_ckpt_missing'), ('lost_refused', 'lost_tokens_refused'),
            ('superseded', 'evicted_superseded'), ('evicted_lru', 'evicted_lru'), ('evicted_fair', 'evicted_fair'),
            ('evicted_coupled', 'evicted_coupled')))
        lines = [' '.join(head)]
        reuse = ['reuse']
        histograms = self.histograms()
        for unit, short in (('reuse_seconds', 's'), ('reuse_tokens', 'tok')):
            entry = histograms.get(unit)
            if entry is None:
                continue
            reuse.append('bounds_%s=%s' % (short, ','.join(str(bound) for bound in entry['bounds'])))
            for outcome in REUSE_OUTCOMES:
                cell = entry[outcome]
                reuse.append('%s_%s=%s:%d' % (outcome, short, ','.join(str(count) for count in cell['counts']),
                                              round(cell['sum'])))
        if len(reuse) > 1:
            lines.append(' '.join(reuse))
        return lines

    def pins(self):
        return sum(checkpoint.pins for checkpoint in self.entries.values())

    def snapshot(self):
        values = dict(self.stats)
        values.update(entries=len(self.entries), bytes=self.bytes, budget_bytes=self.budget_bytes,
                      pins=self.pins(), staged_now=len(self.staged), committed_now=len(self.committed),
                      orphans_now=len(self.orphan_keys), inflight_now=len(self.inflight),
                      mid_loop_capture=self.mid_loop_capture,
                      disabled=self.disabled, ghost_now=len(self.ghost), evict_fair=int(self.evict_policy == 'fair'),
                      supersede=int(self.supersede), telemetry=int(self.telemetry))
        tier = self.tier.snapshot() if self.tier is not None else dict(bytes=0, entries=0, cap=0)
        values.update(tier_bytes=tier['bytes'], tier_entries=tier['entries'], tier_cap_bytes=tier['cap'], tier_on=int(self.tier_on))
        if self.quota is not None:
            # Aggregates only: how many tenants the quotas see and how many hold their cap, never which (no tenant tag, salt or digest is exported).
            values.update(quota_on=1, quota_dev_tenants=0, quota_dev_at_cap=0, quota_host_tenants=len(self.tier.tenant_bytes) if self.tier is not None else 0,
                          quota_host_at_cap=self.tier.tenants_at_cap() if self.tier is not None else 0)
            gauge = getattr(self, 'quota_gauge', None)
            if gauge is not None:
                values.update(gauge())
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
        self.tier_time = None
        self.tiers = 0

    def maybe_tier(self, registry, force=False):
        """The registry's telemetry lines (PrefixRegistry.telemetry_lines), at most once per interval_s, whether or not a
        counter moved (the host gauges drift while the engine decodes). Nothing without telemetry. Never raises."""
        try:
            if not getattr(registry, 'telemetry', False):
                return False
            now = self.clock()
            if not force and self.tier_time is not None and now - self.tier_time < self.interval_s:
                return False
            self.tier_time = now
            for line in registry.telemetry_lines():
                self.log('%s', line)
            self.tiers += 1
            return True
        except Exception as error:
            self.log('telemetry lines failed: %s: %s', type(error).__name__, error)
            return False

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
        # ... and its tier IO the same way (the model attaches it at warmup, before any registry exists).
        if getattr(holder, 'tier_io', None) is not None:
            registry.attach_tier_io(holder.tier_io)
        holder.registry = registry
    return registry


def current_registry():
    """The shared registry if a scheduler has created one in this process, else None (warmup
    prefills run before the scheduler exists)."""
    holder = sys.modules.get(REGISTRY_KEY)
    return getattr(holder, 'registry', None) if holder is not None else None
