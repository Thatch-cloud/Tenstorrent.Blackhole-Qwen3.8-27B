"""Conversation prefix reuse on the TT general path (G1): the scheduler graft and its AST stage.

The TT prefix-reuse design (revision 2, 2026-09-26), section 2.0.1 item 2, productionised from the
P0a prototype (prefix/p0a cbe730a7, scripts/ci/prefix_scheduler_graft.py, 18/18 on the P0a probe).

vLLM keeps owning the attention KV pages: it hashes, ref-counts and LRU-evicts them. What it cannot
know is that a TT prefill is exact only from a 2048-token chunk boundary with the GatedDeltaNet
(GDN) state the model saved there. Wrappers on the live TTScheduler instance make every hit exact
and every grant current:

a. kv_cache_manager.get_computed_blocks (the trim). vLLM calls it on every attempt to admit a
   waiting request (v1/core/sched/scheduler.py:684-729). The wrapper cuts vLLM's 64-aligned hit h
   to Q, the largest 2048-multiple <= h such that the registry holds a checkpoint keyed by the
   block hash at Q, no block below Q was cached in this scheduler step (the TT analogue of vLLM's
   Mamba rule, single_type_kv_cache_manager.py:1196-1203: a same-step block may not be written
   yet), and the checkpoint's stored token ids equal the request's (a mismatch lowers Q here
   instead of killing the engine in the model, S7; the verdict is remembered per request and
   checkpoint, so a request blocked at the queue head does not rebuild a 62k-token array every
   step). It stages the grant and the captures the model should take; staging is idempotent and
   takes no pin. A capture at pos is the state after exactly pos tokens (qwen_prefix_registry's
   model contract), so the plan is: the prompt boundary floor2048(P) above Q and the gap boundary
   floor2048(h) when it is a whole chunk above Q - each only if it is where the row's chunk loop
   drains (floor2048(num_tokens)), unless the model graft has declared mid-loop captures
   (registry.enable_mid_loop_capture); never anything above floor2048(P), which is all the cap
   publishes. A request without cache_salt gets no hit (fail closed, S4); so does a streaming-input
   session (request.resumable: vLLM folds its decode-written tokens into its prompt,
   scheduler.py:1213-1252) and every request once the kill switch is engaged.
b. kv_cache_manager.coordinator.cache_blocks (the cap). Both publish paths reach it: allocation
   (kv_cache_manager.py:452-462) and every output (async_scheduler.py:67-73 through
   kv_cache_manager.py:620-629). Only prompt blocks below floor2048(prompt) are hashed into the
   cache - blocks a full prefill chunk wrote; decode and prompt-tail blocks never are, because
   vLLM serves the first block cached for a hash (block_pool.py:47-72). An unsalted request or a
   streaming-input session publishes nothing. The block ids it newly caches feed the same-step
   rule.
c. schedule (the per-step commit, F2/S6). vLLM may discard an admission attempt after the hit was
   taken - a budget break (:833-840), an allocation failure (:917-924), TTScheduler's decode
   fallback (plugin scheduler.py:140). Grants staged during one schedule() call are committed only
   for requests the returned SchedulerOutput admits (new or resumed) at start_pos == Q, and pinned
   for that one step; the rest drop. The same wrapper exports the registry's counters
   (qwen_prefix_registry.StatsExport: a rate-limited "[PINDIAG] prefix: stats" line and a tmpfs
   JSON file).
d. kv_cache_manager.block_pool._maybe_evict_cached_block (eviction coupling, F8/S9). When vLLM
   evicts a block and its hash no longer maps to any cached block, the checkpoint keyed by that
   hash goes too, so the registry stays a subset of the cached KV.
e. reset_prefix_cache clears the registry when vLLM's reset succeeded (it returns False and keeps
   every cached block while running requests hold blocks, scheduler.py:2196-2240; the registry is
   then still a subset of the cached KV); _free_request drops the request's staged and committed
   grant, its pin and its remembered token checks (this covers a waiting request that is aborted).

f. The host KV tier (qwen_prefix_registry.TierStore, TierHooks below; QWEN_PREFIX_HOST_TIER_GIB > 0, off otherwise and then none of this runs).
   A block vLLM evicts from the device pool is queued (d) and, at the end of the schedule() call - before the model has written
   anything - read raw into host RAM under its block hash, unless nothing could use it; the checkpoint keyed at it stays. For a waiting
   request whose device hit stops where the tier's consecutive blocks begin, the trim (a) picks Q from the device hit plus those blocks;
   if Q is beyond the device hit it takes free blocks from the pool (touching the request's own hit first, so they cannot be
   the victims), writes the tier's bytes into them, registers their hashes and frees them as cached-free blocks - and then asks vLLM
   again, whose own hit logic, reservation rule and allocation do everything else. Nothing here uses an external-token path or a KV
   connector. A failure at any point is a miss.

h. The sliced restore (QWEN_PREFIX_TIER_RESTORE_SLICE blocks, QWEN_PREFIX_TIER_RESTORE_SLICE_MS; off unless named, and then the restore is item f's, whole, in the trim). A returning
   session's blocks come back a bounded slice per engine step instead of all in one: the one-fresh-prefill admission (serving_prefill_admission) asks SchedulerGraft.restore_hold
   for the head candidate before it admits it; the hold starts a RestoreJob (the tier's records pinned, the request's own device hit touched), and each call restores one slice -
   the digests of its records, an allocation of that many blocks (whose evictions are spilled at once, so a step's spill is bounded too), the raw device write, the read-back
   audit - and reports the request held, so the pass is a decode-only pass and the seats keep getting rounds. The slice is as many blocks as the time budget allows at the cost per
   block measured so far, at most the block cap. The request is admitted in the step its last slice lands; the blocks are hashed into the pool only then (all or nothing, so no other
   request ever sees a half restore) and freed as cached blocks, and vLLM's own hit logic takes them. A job ends with its request (the free hook), the kill switch, a latched tier or
   a failure (a miss: the request is admitted cold, and its tier hit is not retried); a trim that reaches a request whose job is unfinished finishes it whole. What it cannot slice
   is what a device-resident hit costs anyway: the tail prefill and the engine build.

i. The upload levers of the tier's host side (QWEN_PREFIX_TIER_VERIFY_THREADS here, QWEN_PREFIX_TIER_IO_THREADS and _IO_CHUNK_BLOCKS in the registry's attach_tier_io; qwen_tier_upload,
   docs/tp4-upload1.md; off unless named, and then none of it runs). The digest checks of a restore's records (item f and h) run on worker threads instead of one after another: the
   verdicts, the sampling, the counters and the record dropped on a failure are the sequential loop's (digest_failures); the staging copies of the raw device calls run on workers beside
   the device call (ThreadedTierIO). Neither changes a byte or a decision, only how long the same work takes.

j. The NVMe cold tier (QWEN_PREFIX_NVME*, qwen_kv_nvme, docs/tp4-kv-nvme.md; off unless named, and then none of it runs). Below the host tier: a record the tier evicts is queued to a
   raw NVMe namespace by the tier's spill sink, a checkpoint leaving the store likewise, both on a writer thread. The restore's reach grows with it: effective_hit counts the host
   tier's consecutive blocks AND the NVMe's, a checkpoint missing from the store is read back from the NVMe and filed as its own (adopt_checkpoint), and the NVMe-resident blocks of
   a restore are read into host-tier records first (promote, whole in the trim; fill, a slice at a time in the sliced restore) - after which the restore is item f's, unchanged,
   so every digest it checks it still checks. A direct restore backend (none is built) takes NVMe-resident blocks to the device without the host (load_direct). Any failure is a miss.

g. Tenant quotas (qwen_prefix_registry.QuotaConfig, DeviceQuota below; QWEN_PREFIX_TENANT_QUOTA_DEVICE_CAP_TOKENS / _FLOOR_TOKENS, off unless named and then none of
   this runs). The tenant is the salt's (tenant_of_salt); the same numbers apply to every tenant. A tenant's RESIDENT tokens are its cached prefix blocks in the
   device pool, running or idle. The CAP is soft by what is running (a running request's blocks cannot be evicted): when a tenant's resident blocks pass its cap,
   its OWN idle blocks, oldest first, are moved to the front of vLLM's free queue, so the next allocation of anyone evicts them before anything else - at publication
   and at every free. The FLOOR protects: an allocation (by another tenant) walks the front of the free queue and sets aside, at the tail, an idle block of a tenant
   whose resident blocks would fall below its floor by losing it. The eviction order is therefore: (1) a tenant's own idle blocks beyond its cap, oldest first; (2)
   vLLM's own LRU order over everything else - blocks never cached or no longer cached first - among tenants above their floor; (3) only when nothing else can
   satisfy the allocation, a protected block (counted as a floor override: the allocation is vLLM's and must not fail for a quota). Blocks of no tenant (never
   published by the cap) are in no quota. Counters are aggregates only (tenants, tenants at cap, evictions by reason own/global, blocks protected): nothing
   labelled by tenant is logged or exported. Off, none of it is installed and the pool, the stats and the log are what they were. The host tier has the same
   quotas in bytes (qwen_prefix_registry.TierStore).

The kill switch (qwen_prefix_registry.KillSwitch, the flag file /models/.qwen-c2/prefix-reuse.off,
polled at most once a second from the trim) disables the registry for the life of the process.

Sticky sessions (QWEN_FAST_STICKY_SESSIONS=1 beside QWEN_PREFIX_REUSE=1; the C2 fast path's phase 1,
serving_fast_policy.STICKY_SESSIONS_FLAG). The fast path decodes with DFlash (15 proposals), so vLLM
runs the scheduler with a lookahead of 16 and, because SpeculativeConfig.use_eagle() includes
dflash, its unitary coordinator drops the last matched block of every hit (use_eagle; vLLM 0.25.1
kv_cache_coordinator.py, single_type_kv_cache_manager.py). Under the flag, and only then:
  * install accepts that lookahead - exactly 16, for a dflash config with 15 proposals - and
    nothing else speculative;
  * the trim's ceiling is floor2048(P - 2048): the fast path's drafter window [P - 2048, P) must be
    this prefill's own chunks (serving_lifecycle refuses a resume above it);
  * the plan captures floor2048(P) - 2048 in place of the prompt boundary floor2048(P) when the
    coordinator drops the last block: the cap publishes KV up to floor2048(P), the next turn's hit
    is at most 64 tokens short of it, so its trim lands one chunk lower and a checkpoint at
    floor2048(P) could never be granted. The gap boundary is planned as before, and nothing
    above floor2048(P) ever is;
  * the cap is unchanged, and a second install line says "install sticky=1 lookahead=<n>
    drop_last=<bool>".
Unset, every path here is G1's own.

Exact resume P0 (docs/tp4-exact-resume.md 5.2; both default off, gate-only profiles, sticky sessions only):
  * QWEN_PREFIX_STICKY_UNDROP=1: the graft wraps the coordinator's find_longest_cache_hit so that vLLM's hit keeps its last matched
    block (the EAGLE drop exists for a drafter that recomputes the last block's hidden states; DFlash on TT reads its window from the
    prefill capture), and the plan captures the prompt boundary floor2048(P) as G1 does. Without the window carry the ceiling stays
    floor2048(P - 2048), so the plan also keeps C0 = floor2048(P) - 2048 (a mid-loop capture) for the turns that land under it;
  * QWEN_PREFIX_WINDOW_CARRY=1 (needs the undrop): the trim may land above P - 2048, up to floor2048(P - 1), on a checkpoint whose
    Window covers [P - 2048, Q) (qwen_prefix_registry.window_covers): the drafter window rows below Q come from the turn that captured
    it, the rows from Q on from this prefill; the plan captures only floor2048(P);
  * one line, "[PINDIAG] prefix: window carry engaged undrop=<0|1> carry=<0|1> audit=<0|1> ceiling=<...>", says which ran (the
    sticky install line above is unchanged: drop_last there is vLLM's coordinator's own setting).
Turn telemetry (QWEN_PREFIX_TURN_TELEMETRY, on unless 0): the trim classifies every salted request once (TurnMemory.classify) and
_free_request remembers every finished one (TurnMemory.record); counters only.

install() refuses to start - the engine dies in TTScheduler.__init__ - unless async scheduling is
off (blocks hashed at allocation in step t are unwritten when step t+1 reads them), chunked prefill
is off with a whole-prompt token budget and long_prefill_token_threshold 0 (vLLM splits a prefill
at that threshold before it looks at chunking, scheduler.py:828; Lever N's chunking gives
start_pos > 0 a second meaning, F5), the coordinator is the unitary one (HybridKVCacheCoordinator
calls manager.cache_blocks directly and would bypass the cap, kv_cache_coordinator.py:602-628),
prefix caching is on with a sha256 block hash (the exactness argument, design 2.0.4 L2, needs a
collision-resistant chain; vLLM itself warns about xxhash, config/cache.py:95-110), the executor is
the uniproc one (the registry is shared with the model through sys.modules, so the model must run
in this process), the KV spec, scheduler and hash block sizes are 64, there is no KV connector, no
speculative lookahead and no pipeline parallelism, the vLLM internals the wrappers bind exist, and
no other live scheduler in the process already owns the shared registry (lane mode builds one
TTScheduler per lane).

Delivery (the design's "an AST patch of the plugin's scheduler.py"; not a scheduler_cls subclass,
because platform.py:1078-1081 assigns scheduler_cls unconditionally): stage(<plugin package>)
refuses a scheduler.py whose sha256 is not bf77cd63's (the image's copy, dumped by CPU probe
35665853903, is that blob: a1bd6257...), appends a guarded hook to TTScheduler.__init__ and copies
this module and qwen_prefix_registry.py into the package beside it. A copy already there is
accepted only if it is this stage's own bytes: the C2 image lists both plugin copies as overlay
destinations (docker/qwen-c2-overlay.txt), so its provenance check hashes them against the
/experiment-scripts/ci copies this stage runs from (test_qwen_prefix_image_closure holds a profile
that turns QWEN_PREFIX_REUSE on to that plumbing). The hook imports nothing unless
QWEN_PREFIX_REUSE=1, so every other profile runs the plugin's own scheduler byte for byte.

    python3 -B qwen_prefix_scheduler_patch.py [--check] /opt/qwen-fast-plugin/src/vllm_tt_plugin

Markers: "[PINDIAG] prefix: install ..." when the wrappers go on (scheduler class, plugin path,
coordinator, block size, KV spec dtype, QWEN_SDPA_BF8, hash algorithm, executor); "[PINDIAG]
prefix: grant req=... h=... Q=... drain=... plan=[...]" from inside the per-step commit for every
admission that carries a grant; "[PINDIAG] prefix: stats {...}" at most every QWEN_PREFIX_STATS_S.
"""

import argparse
import ast
import hashlib
import os
import shutil
import sys
import time
import types
from collections import OrderedDict
from pathlib import Path

try:
    from . import qwen_prefix_registry as prefix_registry
except ImportError:
    import qwen_prefix_registry as prefix_registry

CHUNK = prefix_registry.CHUNK
BLOCK = prefix_registry.BLOCK
KILL_SWITCH_PATH = prefix_registry.KILL_SWITCH_PATH
KILL_SWITCH_POLL_S = prefix_registry.KILL_SWITCH_POLL_S
Checkpoint = prefix_registry.Checkpoint
Grant = prefix_registry.Grant
KillSwitch = prefix_registry.KillSwitch
PrefixRegistry = prefix_registry.PrefixRegistry
StatsExport = prefix_registry.StatsExport
floor_chunk = prefix_registry.floor_chunk
shared_registry = prefix_registry.shared_registry
tenant_of_salt = prefix_registry.tenant_of_salt
log = prefix_registry.log

PLUGIN_REVISION = 'bf77cd63756fc891b8fb7f7cb3f5c1420f0e044c'
# git show bf77cd63:src/vllm_tt_plugin/scheduler.py | sha256sum; the image's copy is this blob
# (scripts/ci/test_lever_n_scheduler_graft.py records sha256 a1bd6257d3a14c90 from probe 35665853903).
SCHEDULER_SHA256 = 'a1bd6257d3a14c904b41b4795b8e8b4b1b132c70fc3a4db9d4340a128a100de4'
RUNTIME_FILES = ('qwen_prefix_registry.py', 'qwen_prefix_scheduler_patch.py')
HOOK_TAG = 'qwen_prefix_scheduler_patch'
# Block-hash algorithms whose chain is collision resistant (vLLM config/cache.py:95-110).
HASH_ALGORITHMS = ('sha256', 'sha256_cbor')
# The registry reaches the model through sys.modules: the model must run in the scheduler's process.
EXECUTOR_BACKEND = 'uni'
# Sticky sessions (see the module docstring): the fast path's switch, and the one speculative
# shape it relaxes the lookahead refusal for - DFlash with 15 proposals, vLLM's lookahead 16
# (num_speculative_tokens + 1, v1/core/sched/scheduler.py).
STICKY_ENV = 'QWEN_FAST_STICKY_SESSIONS'
STICKY_METHOD = 'dflash'
STICKY_PROPOSALS = 15
STICKY_LOOKAHEAD = STICKY_PROPOSALS + 1
# Lever N's scheduler edits (lever_n_scheduler_patch, lever_n_model_patch.patch_scheduler) turn
# vLLM chunking on; prefix reuse is not exact beside them (design F5).
LEVER_N_TAGS = ('Lever N M2', '[PINDIAG] m2 one-in-flight:', '_qwen_saved_waiting')
# The merged route (Lever N with prefix reuse): the markers the runtime wraps carry, which this module reads as plain attribute names
# because it runs inside the plugin package, where the overlay modules are not importable (test_levern_prefix_graft pins each equal to
# levern_scheduler.WRAPPED, serving_prefill_admission.WRAPPED and levern_policy.FLAG).
LEVERN_ENV = 'QWEN_FAST_LEVER_N'
LEVERN_SCHEDULE_WRAPPED = '_qwen_levern_schedule'
ADMISSION_WRAPPED = '_qwen_one_fresh_prefill'

INIT_ANCHOR = (
    '    def __init__(self, *args, **kwargs):\n'
    '        super().__init__(*args, **kwargs)\n'
    '        self._forced_mode = TTSchedulingMode.DEFAULT\n'
)
INIT_HOOK = (
    '        # Qwen prefix reuse (G1): scripts/ci/qwen_prefix_scheduler_patch.py wraps this\n'
    '        # instance (trim, cap, per-step commit, eviction coupling, free and reset hooks) when\n'
    '        # QWEN_PREFIX_REUSE=1 and refuses to start where reuse would not be exact. Unset,\n'
    '        # nothing is imported and nothing changes.\n'
    '        import os as _qwen_prefix_os\n'
    '\n'
    '        if _qwen_prefix_os.environ.get("QWEN_PREFIX_REUSE") == "1":\n'
    '            from .qwen_prefix_scheduler_patch import maybe_install as _qwen_prefix_install\n'
    '\n'
    '            _qwen_prefix_install(self)\n'
)


class PrefixInstallError(RuntimeError):
    """The scheduler is not one prefix reuse is exact under; the engine must not start."""


# ------------------------------------------------------------------------------------------------
# The AST stage
# ------------------------------------------------------------------------------------------------
def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def method_span(source, class_name, method_name):
    """[start, end) line span of class_name.method_name, decorators included (Python >= 3.8)."""
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(classes) != 1:
        raise ValueError('expected one class %s, found %d' % (class_name, len(classes)))
    methods = [node for node in classes[0].body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name]
    if len(methods) != 1:
        raise ValueError('expected one %s.%s, found %d' % (class_name, method_name, len(methods)))
    node = methods[0]
    start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list]) - 1
    return start, node.end_lineno


def patch_scheduler(source):
    """Append the guarded install hook to TTScheduler.__init__, whose body must be exactly the
    pinned three lines. Refuses a tree that already carries the hook or Lever N's edits."""
    if HOOK_TAG in source:
        raise ValueError('scheduler.py already carries the prefix-reuse hook')
    for tag in LEVER_N_TAGS:
        if tag in source:
            raise ValueError('scheduler.py carries Lever N\'s scheduler edit (%r): prefix reuse is not '
                             'exact beside vLLM chunking (design F5)' % tag)
    start, end = method_span(source, 'TTScheduler', '__init__')
    lines = source.splitlines(keepends=True)
    region = ''.join(lines[start:end])
    if region != INIT_ANCHOR:
        raise ValueError('TTScheduler.__init__ is not the pinned body:\n%s' % region)
    lines[start:end] = [INIT_ANCHOR + INIT_HOOK]
    result = ''.join(lines)
    ast.parse(result)
    return result


def stage(package, source_dir=None, check_only=False):
    """Patch <package>/scheduler.py in place and copy the runtime modules beside it.

    Refuses, before writing anything: a scheduler.py that is not the pinned bf77cd63 blob, a package
    runtime module whose bytes are not the source's (a copy the image's overlay already installed
    from the same source is accepted and left alone), and a missing runtime source."""
    package = Path(package)
    source_dir = Path(source_dir) if source_dir else Path(__file__).resolve().parent
    target = package / 'scheduler.py'
    data = target.read_bytes()
    observed = sha256_hex(data)
    if observed != SCHEDULER_SHA256:
        raise ValueError('%s: sha256 %s is not the pinned plugin %s scheduler.py (%s)'
                         % (target, observed, PLUGIN_REVISION[:8], SCHEDULER_SHA256))
    sources = {}
    present = {}
    for name in RUNTIME_FILES:
        if not (source_dir / name).is_file():
            raise ValueError('runtime source %s is missing' % (source_dir / name))
        sources[name] = (source_dir / name).read_bytes()
        if (package / name).exists():
            present[name] = sha256_hex((package / name).read_bytes())
            if present[name] != sha256_hex(sources[name]):
                raise ValueError('refusing to overwrite %s (sha256 %s, not the source %s\'s %s)'
                                 % (package / name, present[name], source_dir / name, sha256_hex(sources[name])))
    patched = patch_scheduler(data.decode('utf-8')).encode('utf-8')
    report = {'scheduler.py': (observed, sha256_hex(patched))}
    for name in RUNTIME_FILES:
        report[name] = (present.get(name), sha256_hex(sources[name]))
    if check_only:
        return report
    for name in RUNTIME_FILES:
        if name not in present:
            (package / name).write_bytes(sources[name])
    target.write_bytes(patched)
    return report


# ------------------------------------------------------------------------------------------------
# The runtime graft
# ------------------------------------------------------------------------------------------------
REQUIRED_INTERNALS = (
    ('kv_cache_manager', ('get_computed_blocks', 'create_kv_cache_blocks', 'empty_kv_cache_blocks',
                          'coordinator', 'block_pool', 'enable_caching')),
    ('kv_cache_manager.coordinator', ('cache_blocks', 'single_type_managers')),
    ('kv_cache_manager.block_pool', ('_maybe_evict_cached_block', 'cached_block_hash_to_block',
                                     'cached_block_hashes_by_block', 'hash_block_size')),
    ('', ('schedule', 'reset_prefix_cache', '_free_request', 'kv_cache_config', 'scheduler_config',
          'cache_config', 'max_model_len')),
)


def _resolve(root, dotted):
    value = root
    for part in [part for part in dotted.split('.') if part]:
        value = getattr(value, part)
    return value


def sticky_enabled(environ=None):
    """Whether QWEN_FAST_STICKY_SESSIONS=1 (serving_fast_policy.sticky_sessions_enabled validates the
    value at the fast path's attach; here anything but '1' is off)."""
    environ = os.environ if environ is None else environ
    return environ.get(STICKY_ENV) == '1'


def sticky_speculation(scheduler):
    """Whether the scheduler's speculative config is the one sticky sessions serve: dflash, 15
    proposals."""
    spec = getattr(getattr(scheduler, 'vllm_config', None), 'speculative_config', None)
    return (getattr(spec, 'method', None) == STICKY_METHOD
            and getattr(spec, 'num_speculative_tokens', None) == STICKY_PROPOSALS)


def drops_last_block(coordinator):
    """Whether vLLM's unitary coordinator drops the last matched block of a hit (the EAGLE drop,
    applied to group 0 when use_eagle is set)."""
    groups = getattr(coordinator, 'eagle_group_ids', None)
    if groups is not None:
        return 0 in groups
    singles = getattr(coordinator, 'single_type_managers', None) or ()
    return bool(singles and getattr(singles[0], 'use_eagle', False))


def levern_chunking_problems(scheduler, environ=None):
    """What stands between this scheduler and the merged route's chunked prefill (docs/lever-n-prefix-merged-route.md, R2), [] when
    it may run chunked, else the list of missing pieces. Chunked prefill is accepted only where Lever N's cap is the only thing that can
    split a prefill, and the cap itself is installed: sticky sessions (the plan and the in-flight capture plan are the sticky ones),
    QWEN_FAST_LEVER_N=1, the class's schedule carrying Lever N's wrapper and _schedule_prefill_only the one-fresh-prefill cap's (the
    graft binds scheduler.schedule at construction, so the order matters: Lever N is installed on the class first), an integer
    max_num_scheduled_tokens (the cap's own path: the threshold fallback would let vLLM split without the cap's alignment). The
    threshold and whole-prompt budget refusals stay absolute (install_problems)."""
    environ = os.environ if environ is None else environ
    missing = []
    if not sticky_enabled(environ):
        missing.append('QWEN_FAST_STICKY_SESSIONS=1')
    if environ.get(LEVERN_ENV) != '1':
        missing.append('%s=1' % LEVERN_ENV)
    cls = type(scheduler)
    if not getattr(getattr(cls, 'schedule', None), LEVERN_SCHEDULE_WRAPPED, False):
        missing.append("Lever N's schedule wrapper installed on the class before the graft")
    if not getattr(getattr(cls, '_schedule_prefill_only', None), ADMISSION_WRAPPED, False):
        missing.append("the one-fresh-prefill cap on _schedule_prefill_only")
    if type(getattr(scheduler, 'max_num_scheduled_tokens', None)) is not int:
        missing.append('an integer max_num_scheduled_tokens (the cap would fall back to long_prefill_token_threshold)')
    return missing


def levern_chunked(scheduler, environ=None):
    """Whether this scheduler runs chunked prefill under the merged route (levern_chunking_problems is empty with chunking on)."""
    config = getattr(scheduler, 'scheduler_config', None)
    return bool(getattr(config, 'enable_chunked_prefill', False)) and not levern_chunking_problems(scheduler, environ)


def install_problems(scheduler, environ=None):
    """Every reason prefix reuse would not be exact on this scheduler (F5, F6, F7). All of them are
    reported, so the refusal names the root cause (a hybrid coordinator, say) and not only the
    first internal it lacks. Under sticky sessions the lookahead of DFlash with 15 proposals (16)
    is accepted; any other lookahead is still refused."""
    problems = []
    missing = []
    for owner, names in REQUIRED_INTERNALS:
        try:
            target = _resolve(scheduler, owner)
        except AttributeError:
            missing.append(owner)
            continue
        missing.extend('%s.%s' % (owner, name) if owner else name
                       for name in names if not hasattr(target, name))
    singles = getattr(getattr(getattr(scheduler, 'kv_cache_manager', None), 'coordinator', None),
                      'single_type_managers', None)
    if singles:
        missing.extend('kv_cache_manager.coordinator.single_type_managers[0].%s' % name
                       for name in ('num_cached_block', 'req_to_blocks') if not hasattr(singles[0], name))
    hash_map = getattr(getattr(getattr(scheduler, 'kv_cache_manager', None), 'block_pool', None),
                       'cached_block_hash_to_block', None)
    if hash_map is not None and not hasattr(hash_map, 'get_one_block'):
        missing.append('kv_cache_manager.block_pool.cached_block_hash_to_block.get_one_block')
    try:
        from vllm.v1.core.kv_cache_coordinator import UnitaryKVCacheCoordinator
    except ImportError as error:
        UnitaryKVCacheCoordinator = None
        problems.append('cannot import vLLM\'s UnitaryKVCacheCoordinator: %s' % error)

    config = getattr(scheduler, 'scheduler_config', None)
    max_model_len = getattr(scheduler, 'max_model_len', None)
    if getattr(config, 'async_scheduling', False):
        problems.append('async scheduling is on: blocks hashed at allocation in step t are unwritten '
                        'when step t+1 reads them')
    if getattr(config, 'enable_chunked_prefill', False):
        beside = levern_chunking_problems(scheduler, environ)
        if beside:
            problems.append('chunked prefill is on: start_pos > 0 would also mean "continue my own '
                            'suspended scratch" (Lever N)'
                            + ' [the merged route accepts it only beside: %s]' % '; '.join(beside))
    batched = getattr(config, 'max_num_batched_tokens', None)
    if batched is not None and max_model_len is not None and batched < max_model_len:
        problems.append('max_num_batched_tokens %d < max_model_len %d: a prefill could be split'
                        % (batched, max_model_len))
    threshold = getattr(config, 'long_prefill_token_threshold', None)
    if threshold != 0:
        problems.append('long_prefill_token_threshold is %r, not 0: vLLM splits a prefill there before it '
                        'looks at chunking (scheduler.py:828), and the next piece reaches the model at '
                        'start_pos > 0 with no grant' % (threshold,))
    cache_config = getattr(scheduler, 'cache_config', None)
    if not getattr(cache_config, 'enable_prefix_caching', False):
        problems.append('prefix caching is off')
    algorithm = getattr(cache_config, 'prefix_caching_hash_algo', None)
    if algorithm not in HASH_ALGORITHMS:
        problems.append('prefix_caching_hash_algo is %r, not one of %s: a hit is exact only on a collision-'
                        'resistant hash chain (vLLM config/cache.py:95-110)' % (algorithm, ', '.join(HASH_ALGORITHMS)))
    parallel = getattr(getattr(scheduler, 'vllm_config', None), 'parallel_config', None)
    backend = getattr(parallel, 'distributed_executor_backend', None)
    if backend != EXECUTOR_BACKEND:
        problems.append('the executor backend is %r, not %r: the model would not share this process\'s '
                        'registry, so no checkpoint is ever captured or restored' % (backend, EXECUTOR_BACKEND))
    manager = getattr(scheduler, 'kv_cache_manager', None)
    if not getattr(manager, 'enable_caching', True):
        problems.append('the KV cache manager does not cache blocks')
    coordinator = getattr(manager, 'coordinator', None)
    if (coordinator is not None and UnitaryKVCacheCoordinator is not None
            and not isinstance(coordinator, UnitaryKVCacheCoordinator)):
        problems.append('the KV coordinator is %s, not UnitaryKVCacheCoordinator: the cap would '
                        'be bypassed' % type(coordinator).__name__)
    groups = getattr(getattr(scheduler, 'kv_cache_config', None), 'kv_cache_groups', None)
    if groups is not None and len(groups) != 1:
        problems.append('%d KV cache groups, not one' % len(groups))
    elif groups:
        block_size = groups[0].kv_cache_spec.block_size
        if block_size != BLOCK:
            problems.append('KV block size %d, not %d: the page tables and chunk arithmetic assume %d'
                            % (block_size, BLOCK, BLOCK))
    if getattr(scheduler, 'block_size', BLOCK) != BLOCK:
        problems.append('scheduler block size %s, not %d' % (scheduler.block_size, BLOCK))
    hash_block_size = getattr(getattr(manager, 'block_pool', None), 'hash_block_size', BLOCK)
    if hash_block_size != BLOCK:
        problems.append('hash block size %s, not %d: request.block_hashes would not index 64-token '
                        'blocks' % (hash_block_size, BLOCK))
    if getattr(scheduler, 'has_mamba_layers', False):
        problems.append('the KV config has Mamba layers: vLLM would run its own align-mode split')
    if getattr(scheduler, 'connector', None) is not None:
        problems.append('a KV connector is configured: external tokens would move start_pos past Q')
    lookahead = getattr(scheduler, 'num_lookahead_tokens', 0)
    if lookahead:
        if not sticky_enabled(environ):
            problems.append('speculative lookahead is on')
        elif lookahead != STICKY_LOOKAHEAD or not sticky_speculation(scheduler):
            problems.append('speculative lookahead is on: sticky sessions accept only %s with %d proposals '
                            '(lookahead %d), not lookahead %r' % (STICKY_METHOD, STICKY_PROPOSALS,
                                                                   STICKY_LOOKAHEAD, lookahead))
    resume = prefix_registry.exact_resume_problems(environ)
    problems.extend(resume)
    problems.extend(prefix_registry.telemetry_problems(environ))
    undrop_on = not resume and prefix_registry.undrop_enabled(environ)
    if undrop_on and not sticky_enabled(environ):
        problems.append('%s=1 without sticky sessions: the undrop serves the fast path\'s DFlash hit only' % prefix_registry.ENV_UNDROP)
    elif undrop_on:
        coordinator = getattr(getattr(scheduler, 'kv_cache_manager', None), 'coordinator', None)
        if not isinstance(getattr(coordinator, 'eagle_group_ids', None), set) or not callable(
                getattr(coordinator, 'find_longest_cache_hit', None)):
            problems.append('%s=1: the coordinator has no eagle_group_ids set and find_longest_cache_hit (vLLM 0.25.1\'s unitary '
                            'coordinator), so the drop cannot be lifted' % prefix_registry.ENV_UNDROP)
        elif not drops_last_block(coordinator):
            problems.append('%s=1: vLLM does not drop the last block here, there is nothing to lift' % prefix_registry.ENV_UNDROP)
    if parallel is not None and getattr(parallel, 'pipeline_parallel_size', 1) != 1:
        problems.append('pipeline parallel size %s: a step\'s grants must be read before the next '
                        'schedule()' % parallel.pipeline_parallel_size)
    if missing:
        problems.append('vLLM internals the wrappers bind are missing: %s' % ', '.join(missing))
    return problems


TIER_POOL_INTERNALS = ('get_new_blocks', 'free_blocks', 'touch', 'cache_full_blocks', 'get_num_free_blocks')


def tier_problems(scheduler, registry):
    """What the host KV tier lacks on this scheduler, [] when it is off or has what it needs: the model's IO (attached by the model graft
    at warmup, before the scheduler is built), the block pool's allocation and caching calls, vLLM's block-hash constructor."""
    if getattr(registry, 'tier', None) is None:
        return []
    problems = []
    if registry.tier_io is None:
        problems.append('QWEN_PREFIX_HOST_TIER_GIB is set but the model graft attached no tier IO (an image without the KV region ops, or a '
                        'model tree that predates them)')
    pool = getattr(getattr(scheduler, 'kv_cache_manager', None), 'block_pool', None)
    missing = [name for name in TIER_POOL_INTERNALS if not hasattr(pool, name)]
    if missing:
        problems.append('vLLM internals the host tier binds are missing: %s' % ', '.join('block_pool.' + name for name in missing))
    if not hasattr(scheduler, 'kv_cache_manager') or not hasattr(scheduler, 'num_lookahead_tokens'):
        problems.append('the scheduler has no kv_cache_manager or num_lookahead_tokens')
    try:
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id  # noqa: F401
    except ImportError as error:
        problems.append('cannot import vLLM\'s make_block_hash_with_group_id: %s' % error)
    return problems


QUOTA_POOL_INTERNALS = ('free_blocks', 'get_new_blocks', 'free_block_queue')
QUOTA_QUEUE_INTERNALS = ('remove', 'append', 'prepend_n', 'fake_free_list_head', 'fake_free_list_tail', 'num_free_blocks')


def quota_problems(scheduler, registry):
    """What the device quota lacks on this scheduler, [] when it is off or has what it needs: the pool's free calls and its free queue (the quota moves blocks to the
    queue's front and tail), the manager's allocate_slots, and numbers the pool can honour (a floor above half the pool cannot be guaranteed to two tenants, a cap above
    the whole pool is a typo)."""
    quota = getattr(registry, 'quota', None)
    if quota is None or not quota.device_on:
        return []
    problems = []
    manager = getattr(scheduler, 'kv_cache_manager', None)
    pool = getattr(manager, 'block_pool', None)
    missing = ['block_pool.' + name for name in QUOTA_POOL_INTERNALS if not hasattr(pool, name)]
    queue = getattr(pool, 'free_block_queue', None)
    missing += ['block_pool.free_block_queue.' + name for name in QUOTA_QUEUE_INTERNALS if queue is None or not hasattr(queue, name)]
    if not hasattr(manager, 'allocate_slots'):
        missing.append('kv_cache_manager.allocate_slots')
    if missing:
        problems.append('vLLM internals the device quota binds are missing: %s' % ', '.join(missing))
    blocks = getattr(getattr(scheduler, 'kv_cache_config', None), 'num_blocks', None)
    if isinstance(blocks, int) and blocks > 0:
        if quota.device_floor_blocks * 2 > blocks:
            problems.append('%s=%d is more than half the %d-token pool: two tenants could not both be guaranteed it' % (
                prefix_registry.ENV_QUOTA_DEVICE_FLOOR, quota.device_floor_blocks * BLOCK, blocks * BLOCK))
        if quota.device_cap_blocks > blocks:
            problems.append('%s=%d is more than the whole %d-token pool' % (prefix_registry.ENV_QUOTA_DEVICE_CAP, quota.device_cap_blocks * BLOCK, blocks * BLOCK))
    return problems


def maybe_install(scheduler, environ=None):
    """The hook TTScheduler.__init__ calls last (see INIT_HOOK). Off unless QWEN_PREFIX_REUSE=1."""
    if not prefix_registry.reuse_enabled(environ):
        return None
    if sticky_enabled(environ):
        return install(scheduler, environ=environ)
    return install(scheduler)


def install(scheduler, registry=None, kill_switch_path=KILL_SWITCH_PATH, poll_s=KILL_SWITCH_POLL_S,
            clock=time.monotonic, logger=log, stats=None, environ=None):
    """Wrap scheduler (see the module docstring). stats: the StatsExport (default: from the
    environment, QWEN_PREFIX_STATS_PATH and QWEN_PREFIX_STATS_S). environ: where the sticky-session
    switch is read (default os.environ)."""
    existing = scheduler.__dict__.get('_qwen_prefix')
    if existing is not None:
        return existing
    problems = install_problems(scheduler, environ)
    if problems:
        raise PrefixInstallError('prefix reuse refused: ' + '; '.join(problems))
    registry = registry if registry is not None else shared_registry()
    problems = tier_problems(scheduler, registry) + quota_problems(scheduler, registry)
    if problems:
        raise PrefixInstallError('prefix reuse refused: ' + '; '.join(problems))
    try:
        registry.bind(scheduler)
    except RuntimeError as error:
        raise PrefixInstallError('prefix reuse refused: %s' % error)
    if stats is None:
        stats = StatsExport(clock=clock, logger=logger)
    graft = SchedulerGraft(scheduler, registry, KillSwitch(kill_switch_path, poll_s, clock), logger, stats,
                           sticky=sticky_enabled(environ))
    graft.environ = environ
    if graft.sticky:
        # Exact resume P0 (the module docstring); install_problems has refused every combination that cannot run.
        graft.undrop = prefix_registry.undrop_enabled(environ)
        graft.carry = prefix_registry.window_carry_enabled(environ)
        graft.carry_audit = prefix_registry.window_carry_audit_enabled(environ)
    graft.wrap()
    scheduler._qwen_prefix = graft
    return graft


class _NoStatistics(object):
    """Stands in for KVCacheManager.prefix_cache_stats for the length of one peek, so that asking what a hit would be leaves no trace in
    vLLM's prefix-cache metrics: the in-pass get_computed_blocks records the request's real attempt, once."""

    def record(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return 0


class TierAuditError(AssertionError):
    """QWEN_PREFIX_HOST_TIER_AUDIT=1 read a restored block back and it was not the bytes written: the engine stops."""


class RestoreJob(object):
    """One returning request's restore in progress (docstring item h): the records to write back (one per block from `first` to `need`), the request's own device hit
    (`protected`, touched for the job's length so no allocation of the job can take it), the blocks written so far (`new`, held by the job, not yet hashed), and the cost so far."""

    __slots__ = ('request', 'request_id', 'first', 'need', 'records', 'keys', 'protected', 'new', 'tenant', 'work_ms', 'slices', 'max_ms', 'held', 'step', 'bytes')

    def __init__(self, request, first, need, records, protected, tenant):
        self.request, self.request_id = request, request.request_id
        self.first, self.need = first, need
        self.records, self.keys = records, [record.key for record in records]
        self.protected = protected
        self.new = []
        self.tenant = tenant
        self.work_ms = 0.0
        self.slices = 0
        self.max_ms = 0.0
        self.held = 0
        self.step = None
        self.bytes = 0

    @property
    def total(self):
        return self.need - self.first


class SliceRefused(Exception):
    """A slice could not be written (a record failed its digest, or an NVMe block could not be brought back): the job is abandoned and the request goes on without the tier."""

    def __init__(self, reason='a record failed its digest'):
        Exception.__init__(self, reason)
        self.reason = reason


class IdleCopy(object):
    """A finished request's published prompt blocks waiting for the idle copy: when they are due (ready), the hash of every block of its prompt, how many are published (count),
    the next one to look at, and the chain and tenant they belong to."""

    __slots__ = ('ready', 'hashes', 'count', 'next', 'chain', 'tenant')

    def __init__(self, ready, hashes, count, chain, tenant):
        self.ready, self.hashes, self.count, self.next, self.chain, self.tenant = ready, hashes, count, 0, chain, tenant


IDLE_BACKLOG = 1024        # finished requests remembered for the idle copy


class Pending(object):
    """A block of a sliced restore that is on the NVMe tier and not yet in the host tier: filled (read into a host record) by the slice that needs it."""

    __slots__ = ('key', 'index')

    def __init__(self, key, index):
        self.key, self.index = key, index


# The blocks of the first slice of a job when no cost per block has been measured yet (a probe), and the longest one slice may hold the engine before the gate calls it a defect.
SLICE_PROBE_BLOCKS = 8


class TierHooks(object):
    """The host KV tier's side of the scheduler graft (module docstring, item f). Everything that touches the device goes through
    registry.tier_io (read_blocks, write_blocks), which the model graft attaches; the bytes live in registry.tier (TierStore)."""

    def __init__(self, graft):
        self.graft = graft
        self.registry = graft.registry
        self.tier = graft.registry.tier
        self.io = graft.registry.tier_io
        self.config = graft.registry.tier_config
        # block hash -> (block id, tenant, chain, index): evicted since the last flush, bytes still on the device.
        self.pending = OrderedDict()
        # block id -> (block hash, chain, tenant, index): what the pool's cached blocks are, as the cap or a restore published them.
        self.meta = {}
        self.failures = 0
        self.checked = 0
        self.refused_logged = False
        # The sliced restore (docstring item h): the block cap and the time budget per step, the jobs in progress (request id -> RestoreJob), the requests whose job was abandoned (their tier
        # hit is ignored), whether the admission wrapper that asks for the hold is on the scheduler, and the measured milliseconds per block (EWMA).
        self.slice_blocks = int(getattr(self.config, 'restore_slice', 0) or 0)
        self.slice_ms = float(getattr(self.config, 'restore_slice_ms', prefix_registry.DEFAULT_TIER_SLICE_MS))
        self.jobs = {}
        self.refused = set()
        self.hold_ready = False
        self.rate = None
        self.kill_switch = prefix_registry.KillSwitch(self.config.off_path, prefix_registry.KILL_SWITCH_POLL_S, graft.kill_switch.clock)
        # The upload levers (docstring item i; qwen_tier_upload): threads for the digest checks of a restore. None unless named, and then check_digest runs as before.
        self.verifier = None
        upload = getattr(graft.registry, 'upload_plan', None)
        graft.registry.device_holds = self.device_holds
        graft.registry.tier_slack = 1 if graft.drop_last else 0
        # The NVMe cold tier (docstring item j): None unless the registry opened one on the model's IO.
        self.nvme = getattr(graft.registry, 'nvme', None)
        # Write-through (QWEN_PREFIX_NVME_SPILL=through, the default): finished requests whose still-cached device blocks are copied toward the drive once idle (idle 0: at once), a bounded slice per step.
        self.idle = OrderedDict()
        # The gate reads this line (prefix_markers.TIER_ON): the tier is configured and the model graft's IO is attached.
        graft.log('host tier on gib=%.2f kv_gib=%.2f block_bytes=%d verify=%s audit=%d spill_max_blocks=%d min_tokens=%d slack=%d off_path=%s',
                  self.config.total / float(1 << 30), self.config.kv_bytes / float(1 << 30), int(getattr(self.io, 'block_bytes', 0) or 0),
                  self.config.verify, 1 if self.config.audit else 0, self.config.spill_max, self.config.min_tokens, graft.registry.tier_slack,
                  self.config.off_path)
        if self.nvme is not None:
            # The gate reads this line (prefix_markers.NVME_ON): the NVMe level is configured, opened, and wired to the scheduler graft.
            described = self.nvme.describe()
            graft.log('nvme tier on backend=%s ttl_s=%d max_age_s=%d verify=%s audit=%d queue_bytes=%d capacity_bytes=%d spill=%s through_blocks=%d through_idle_s=%d drain_s=%d fingerprint=%s',
                      described['backend'], described['ttl_s'], described['max_age_s'], described['verify'], described['audit'], described['queue_bytes'],
                      described['capacity_bytes'], described['spill'], described['through_blocks'], described['through_idle_s'], described['drain_s'],
                      described['fingerprint'])
        plan = getattr(graft.registry, 'spill_plan', None)
        if plan is not None:
            # The gate reads this line (prefix_markers.TIER_SPILL_COMPLETE): the cap is lifted to the hard maximum, in batches.
            graft.log('host tier spill complete on cap_blocks=%d hard_max_blocks=%d batch_blocks=%d warn_ms=%d', self.config.spill_max, plan.hard_max, plan.batch, plan.warn_ms)
        if upload is not None:
            if upload.verify_threads:
                import qwen_tier_upload

                self.verifier = qwen_tier_upload.Verifier(upload.verify_threads, graft.registry.stats)
            # The gate reads these lines (c2_smoke_check, the fusion-wp/UP1 rows): the levers are configured, and the tier IO the registry holds is the threaded one when it was asked for.
            import qwen_tier_upload

            wrapped = isinstance(self.io, qwen_tier_upload.ThreadedTierIO)
            graft.log('host tier upload on io_threads=%d io_chunk_blocks=%d verify_threads=%d io_wrapped=%d', upload.io_threads, upload.io_chunk, upload.verify_threads, 1 if wrapped else 0)
            if upload.io_threads and not wrapped:
                graft.log('host tier upload fell back: the tier IO is not the threaded one (io_threads=%d)', upload.io_threads)

    # -- state -----------------------------------------------------------------------------------
    @property
    def active(self):
        return self.registry.tier_on

    def poll(self):
        if self.kill_switch.poll():
            self.registry.tier_latch('kill switch %s present' % self.kill_switch.path)
            self.pending.clear()
        if self.nvme is not None:
            self.nvme.poll()
        return self.active

    def key_of(self, block_hash):
        return self.graft.make_block_hash(block_hash, 0)

    def device_holds(self, block_hash):
        return self.graft.block_pool.cached_block_hash_to_block.get_one_block(self.key_of(block_hash)) is not None

    def eligible(self, request):
        return not self.graft.excluded(request) and self.graft.registry.disabled is None and not self.graft.kill_switch.engaged

    @property
    def slicing(self):
        """Whether restores are sliced: the flag is on, the admission wrapper that holds a request between slices is on the scheduler, and the tier works."""
        return self.slice_blocks > 0 and self.hold_ready and self.active

    # -- publication and eviction (d) --------------------------------------------------------------
    def note_published(self, request, owned, before, after):
        """The cap cached owned[before:after] for request: remember what each block is, so its eviction can tell the tier."""
        hashes = request.block_hashes
        chain = hashes[0] if len(hashes) else None
        tenant = prefix_registry.tenant_of_salt(request.cache_salt)
        for index in range(before, after):
            block = owned[index]
            self.meta[block.block_id] = (hashes[index], chain, tenant, index)

    def on_evict(self, block, keys):
        """vLLM is about to evict `block` (cached under keys). True when its KV is, or will be, in the tier: the checkpoint keyed at it stays."""
        meta = self.meta.pop(block.block_id, None)
        if meta is None or not self.active:
            return False
        raw = [self.graft.get_block_hash(key) for key in keys]
        if meta[0] not in raw:
            return False
        key, chain, tenant, index = meta
        stats = self.registry.stats
        if self.tier.has(key):
            stats['tier_spill_known'] += 1
            return True
        if self.registry._tier_dead(types.SimpleNamespace(chain=chain, index=index)):
            stats['tier_spill_dropped_useless'] += 1
            return False
        self.pending[key] = (block.block_id, tenant, chain, index)
        return True

    def unhold(self, keys):
        """These blocks' checkpoints were kept in the expectation of a spill that did not happen: drop the ones with no KV anywhere."""
        registry = self.registry
        for key in keys:
            if registry.entries.get(key) is not None and not self.tier.has(key) and not self.device_holds(key):
                registry.drop(key, 'coupled')
                registry.stats['tier_ckpt_dropped'] += 1

    def abandon(self):
        keys = list(self.pending)
        self.pending.clear()
        self.unhold(keys)

    def flush(self):
        """Read the blocks evicted since the last flush into the tier. Called at the end of every schedule() call (nothing has written the
        device yet: the model runs afterwards) and before a restore writes. A failure drops what it was keeping and is never raised."""
        if not self.pending:
            return
        items = list(self.pending.items())
        self.pending.clear()
        registry, stats, tier = self.registry, self.registry.stats, self.tier
        if not self.active:
            self.unhold([key for key, _ in items])
            return
        stats['tier_spill_flushes'] += 1
        gauges = prefix_registry.host_gauges()
        available = gauges.get('host_mem_available_bytes')
        if available is not None and available < self.config.min_available:
            stats['tier_spill_dropped_governor'] += len(items)
            self.unhold([key for key, _ in items])
            return
        # A restore needs the run of blocks from the chain's first, so when the cap bites the lowest indexes of each chain are kept.
        items.sort(key=lambda item: (item[1][2] or b'', item[1][3]))
        # COMPLETE SPILLS (QWEN_PREFIX_HOST_TIER_SPILL_MODE=complete; the registry's SpillPlan, None by default): the cap is the plan's hard maximum, and the blocks are
        # read in batches (flush_batches). Without a plan the cap and everything below it are what they always were.
        plan = getattr(registry, 'spill_plan', None)
        limit = self.config.spill_max if plan is None else max(self.config.spill_max, plan.hard_max)
        keep, cut = items[:limit], items[limit:]
        if cut:
            stats['tier_spill_dropped_cap'] += len(cut)
            self.unhold([key for key, _ in cut])
        if plan is not None:
            return self.flush_batches(keep, plan)
        began = time.perf_counter()
        slots = []
        if tier.slot_bytes:
            # Slab mode: the device read goes straight into reserved slots (one copy of each block, into memory already resident).
            slots = tier.reserve(len(keep), [meta[1] for _, meta in keep])
            if len(slots) < len(keep):
                stats['tier_spill_dropped_full'] += len(keep) - len(slots)
                self.unhold([key for key, _ in keep[len(slots):]])
                keep = keep[:len(slots)]
            if not keep:
                return
        try:
            payloads = self.io.read_blocks([meta[0] for _, meta in keep], into=[slot.view for slot in slots] if slots else None)
            if len(payloads) != len(keep):
                raise ValueError('the tier IO read %d payloads for %d blocks' % (len(payloads), len(keep)))
        except Exception as error:
            tier.abort(slots)
            stats['tier_spill_failures'] += 1
            self.failures += 1
            self.graft.log('host tier spill failed (%d so far): %s: %s', self.failures, type(error).__name__, error)
            self.unhold([key for key, _ in keep])
            if self.failures >= 3:
                registry.tier_latch('three device reads in a row failed')
            return
        self.failures = 0
        stored = 0
        for position, (key, (block_id, tenant, chain, index)) in enumerate(keep):
            payload = payloads[position]
            if slots:
                tier.commit(slots[position], key, tenant, chain, index)
                stored += 1
                stats['tier_spill_bytes'] += len(payload)
            elif tier.put(key, payload, tenant, chain, index):
                stored += 1
                stats['tier_spill_bytes'] += len(payload)
            else:
                stats['tier_spill_dropped_full'] += 1
                self.unhold([key])
        elapsed = (time.perf_counter() - began) * 1000.0
        stats['tier_spill_blocks'] += stored
        stats['tier_spill_ms'] += elapsed
        if stored:
            self.graft.log('host tier spill blocks=%d bytes=%d ms=%.1f held=%d', stored, sum(len(p) for p in payloads), elapsed, tier.bytes)

    def flush_batches(self, keep, plan):
        """flush's reads for the complete mode: `keep` (every evicted block up to the plan's hard maximum, lowest indexes of each chain first) is reserved and read in
        batches of plan.batch blocks, so that one device read and one slab reservation never exceed a batch and a read that fails loses its batch and the rest of the flush, not
        what was already stored. Everything else is flush's: the same counters, the same drops (a tier that cannot make room drops the rest of the flush, a failed read
        unholds it), the same latch after three failures in a row; plus the complete mode's own counters and the one line when the flush held the engine past plan.warn_ms."""
        registry, stats, tier = self.registry, self.registry.stats, self.tier
        began = time.perf_counter()
        lifted = max(0, len(keep) - self.config.spill_max)
        if lifted:
            stats['tier_spill_cap_lifted'] += lifted
        stored = batches = 0
        stored_bytes = 0
        low = 0
        while low < len(keep):
            part = keep[low:low + plan.batch]
            slots = []
            if tier.slot_bytes:
                slots = tier.reserve(len(part), [meta[1] for _, meta in part])
                if len(slots) < len(part):
                    stats['tier_spill_dropped_full'] += len(part) - len(slots)
                    self.unhold([key for key, _ in part[len(slots):]])
                    part = part[:len(slots)]
                if not part:
                    # The tier can make no room: the batches that follow meet the same refusal.
                    rest = keep[low + plan.batch:]
                    stats['tier_spill_dropped_full'] += len(rest)
                    self.unhold([key for key, _ in rest])
                    break
            try:
                payloads = self.io.read_blocks([meta[0] for _, meta in part], into=[slot.view for slot in slots] if slots else None)
                if len(payloads) != len(part):
                    raise ValueError('the tier IO read %d payloads for %d blocks' % (len(payloads), len(part)))
            except Exception as error:
                tier.abort(slots)
                stats['tier_spill_failures'] += 1
                self.failures += 1
                self.graft.log('host tier spill failed (%d so far): %s: %s', self.failures, type(error).__name__, error)
                self.unhold([key for key, _ in keep[low:]])
                if self.failures >= 3:
                    registry.tier_latch('three device reads in a row failed')
                break
            self.failures = 0
            batches += 1
            for position, (key, (block_id, tenant, chain, index)) in enumerate(part):
                payload = payloads[position]
                if slots:
                    tier.commit(slots[position], key, tenant, chain, index)
                    stored += 1
                    stats['tier_spill_bytes'] += len(payload)
                    stored_bytes += len(payload)
                elif tier.put(key, payload, tenant, chain, index):
                    stored += 1
                    stats['tier_spill_bytes'] += len(payload)
                    stored_bytes += len(payload)
                else:
                    stats['tier_spill_dropped_full'] += 1
                    self.unhold([key])
            low += plan.batch
        elapsed = (time.perf_counter() - began) * 1000.0
        stats['tier_spill_blocks'] += stored
        stats['tier_spill_ms'] += elapsed
        stats['tier_spill_batches'] += batches
        stats['tier_spill_max_blocks'] = max(stats['tier_spill_max_blocks'], stored)
        stats['tier_spill_max_ms'] = max(stats['tier_spill_max_ms'], elapsed)
        if elapsed > plan.warn_ms:
            stats['tier_spill_slow'] += 1
            self.graft.log('host tier spill slow blocks=%d ms=%.1f warn_ms=%d batches=%d', stored, elapsed, plan.warn_ms, batches)
        if stored:
            self.graft.log('host tier spill blocks=%d bytes=%d ms=%.1f held=%d batches=%d lifted=%d', stored, stored_bytes, elapsed, tier.bytes, batches, lifted)

    # -- the restore (a) ---------------------------------------------------------------------------
    def effective_hit(self, request, h):
        """vLLM's hit length as it would report it if the tier's consecutive blocks after the device hit were cached (h when there are none)."""
        graft = self.graft
        if not self.poll() or not self.eligible(request) or request.request_id in self.refused:
            return h
        hashes = request.block_hashes
        drop = 1 if graft.drop_last and h > 0 else 0
        start = h // BLOCK + drop
        limit = min(len(hashes), floor_chunk(request.num_prompt_tokens) // BLOCK, (request.num_tokens - 1) // BLOCK)
        count = 0
        holds = self.tier.has if self.nvme is None else self.holds
        while start + count < limit and holds(hashes[start + count]):
            count += 1
        effective = h
        if count:
            effective = (start + count) * BLOCK - (BLOCK if graft.drop_last else 0)
        if self.nvme is not None and effective >= CHUNK:
            self.adopt_checkpoint(request, effective)
        return effective

    def holds(self, key):
        """Whether the block is in the host tier or the NVMe tier (the restore can bring it back)."""
        return self.tier.has(key) or self.nvme.has_kv(key)

    def adopt_checkpoint(self, request, h):
        """The NVMe tier's half of the trim: the highest checkpoint the trim could choose for a hit of h tokens, if the store lost it and the NVMe kept it, is read back and
        filed in the store as its own (the trim then finds it, and checks its token ids as for any checkpoint). The candidates are the ones select() walks (its sticky ceilings
        included); the first one the store already holds ends the walk."""
        graft, registry, nvme = self.graft, self.registry, self.nvme
        hashes = request.block_hashes
        k = h // CHUNK
        if graft.sticky:
            window_ceiling = max(0, request.num_prompt_tokens - CHUNK) // CHUNK * CHUNK
            if getattr(graft, 'carry', False):
                k = min(k, max(0, request.num_prompt_tokens - 1) // CHUNK)
            else:
                k = min(k, window_ceiling // CHUNK)
        while k > 0:
            count = k * CHUNK // BLOCK
            k -= 1
            if count > len(hashes):
                continue
            key = hashes[count - 1]
            if registry.entries.get(key) is not None:
                return
            entry = nvme.checkpoint_entry(key)
            if entry is None or entry.index != (k + 1) * CHUNK:
                continue
            image = nvme.load_checkpoint(entry)
            if image is None:
                continue
            if registry.adopt(key, image, chain=hashes[0] if len(hashes) else None, tenant=prefix_registry.tenant_of_salt(request.cache_salt)) is not None:
                return

    def recompute(self, request):
        """vLLM's own hit again, without a trace in its prefix-cache statistics."""
        graft = self.graft
        manager = graft.manager
        statistics = getattr(manager, 'prefix_cache_stats', None)
        if statistics is not None:
            manager.prefix_cache_stats = _NoStatistics()
        try:
            return graft.original['get_computed_blocks'](request)
        finally:
            if statistics is not None:
                manager.prefix_cache_stats = statistics

    def check_digest(self, record):
        mode = self.config.verify
        if mode == 'off':
            return True
        self.checked += 1
        if mode == 'sample' and self.checked % prefix_registry.TIER_VERIFY_EVERY:
            return True
        verdict = self.tier.verify(record)
        if verdict is None:
            return True
        self.registry.stats['tier_digest_checks'] += 1
        if not verdict:
            self.registry.stats['tier_digest_failures'] += 1
        return verdict

    def digest_failures(self, records):
        """The records that fail their digest, in order, lazily: the caller leaves its loop at the first (it drops the record and refuses the restore). Without verify threads
        this is the loop `for record in records: if not self.check_digest(record)` itself. With them the same verdicts come from the worker threads: the same records are
        sampled (the running count advances record by record and is put back to the failing record's own on a failure, as the sequential loop would have left it), the same
        counters move for the records up to and including the failing one."""
        if self.verifier is None:
            for record in records:
                if not self.check_digest(record):
                    yield record
            return
        mode = self.config.verify
        if mode == 'off':
            return
        chosen = []
        for record in records:
            self.checked += 1
            if mode == 'sample' and self.checked % prefix_registry.TIER_VERIFY_EVERY:
                continue
            chosen.append((record, self.checked))
        stats = self.registry.stats
        for (record, checked), verdict in zip(chosen, self.verifier.verdicts(self.tier, [record for record, _ in chosen])):
            if verdict is None:
                continue
            stats['tier_digest_checks'] += 1
            if not verdict:
                stats['tier_digest_failures'] += 1
                self.checked = checked
                yield record
                return

    def load(self, request, blocks, h, q):
        """Make the blocks below Q that the device lacks cached on the device again, from the tier. True when vLLM will now hit them."""
        graft, registry, stats, tier, pool = self.graft, self.registry, self.registry.stats, self.tier, self.graft.block_pool
        job = self.jobs.get(request.request_id)
        if job is not None:
            # The trim reached a request whose sliced restore is unfinished (the admission hold was not asked for it): finish it now, whole.
            state = 'more'
            while state == 'more':
                state = self.advance(job, whole=True)
            return state == 'done'
        if self.slice_blocks:
            stats['tier_slice_inline'] += 1
        drop = 1 if graft.drop_last and h > 0 else 0
        first = h // BLOCK + drop
        need = q // BLOCK + (1 if graft.drop_last else 0)
        if need <= first:
            return False
        hashes = request.block_hashes
        if self.nvme is not None:
            # The NVMe tier's part of the restore (docstring item j): what the host tier lacks and the NVMe holds is read into host records first.
            if self.nvme.backend.direct:
                return self.load_direct(request, blocks, h, first, need, drop)
            absent = [index for index in range(first, need) if not tier.has(hashes[index])]
            if absent:
                lookahead = int(getattr(graft.scheduler, 'num_lookahead_tokens', 0) or 0)
                beyond = -(-(request.num_tokens + lookahead) // BLOCK) - need + 1
                if pool.get_num_free_blocks() < (need - first) + max(0, beyond):
                    stats['tier_restore_refused_room'] += 1
                    return False
                if self.promote(request, first, need, absent) is None:
                    return False
        records = []
        for index in range(first, need):
            record = tier.get(hashes[index])
            if record is None:
                stats['tier_restore_failures'] += 1
                return False
            records.append(record)
        lookahead = int(getattr(graft.scheduler, 'num_lookahead_tokens', 0) or 0)
        beyond = -(-(request.num_tokens + lookahead) // BLOCK) - need + 1
        if pool.get_num_free_blocks() < (need - first) + max(0, beyond):
            stats['tier_restore_refused_room'] += 1
            return False
        for record in self.digest_failures(records):
            stats['tier_restore_refused_digest'] += 1
            graft.log('host tier record failed its digest req=%s block=%d: dropped', request.request_id, record.index)
            tier.remove(record.key)
            self.unhold([record.key])
            return False
        # The request's own device hit must not be among the victims: take it off the free queue for the length of the restore.
        protected = list(blocks.blocks[0])
        if drop:
            extra = pool.cached_block_hash_to_block.get_one_block(self.key_of(hashes[h // BLOCK]))
            if extra is None:
                stats['tier_restore_failures'] += 1
                return False
            protected.append(extra)
        began = time.perf_counter()
        new = []
        ok = False
        pool.touch(protected)
        tier.pin([record.key for record in records])
        tenant = prefix_registry.tenant_of_salt(request.cache_salt)
        quota = graft.quota
        if quota is not None:
            quota.allocating = tenant
        try:
            new = pool.get_new_blocks(need - first)
            # The blocks just taken may have been cached: read them out (they are about to be overwritten).
            self.flush()
            ids = [block.block_id for block in new]
            payloads = [record.payload for record in records]
            self.io.write_blocks(ids, payloads)
            if self.config.audit:
                self.audit(request, ids, payloads)
            pool.cache_full_blocks(request, [None] * first + list(new), first, need, BLOCK, 0)
            chain = hashes[0] if len(hashes) else None
            for offset, block in enumerate(new):
                self.meta[block.block_id] = (hashes[first + offset], chain, tenant, first + offset)
            if quota is not None:
                quota.note_published(new, tenant)
            ok = True
        except TierAuditError:
            raise
        except Exception as error:
            stats['tier_restore_failures'] += 1
            self.failures += 1
            graft.log('host tier restore failed req=%s (%d so far): %s: %s', request.request_id, self.failures, type(error).__name__, error)
            if self.failures >= 3:
                registry.tier_latch('three restores in a row failed')
        finally:
            if quota is not None:
                quota.allocating = None
            tier.unpin([record.key for record in records])
            if new:
                pool.free_blocks(reversed(new))
            pool.free_blocks(reversed(protected))
        if ok:
            self.failures = 0
            elapsed = (time.perf_counter() - began) * 1000.0
            size = sum(len(record.payload) for record in records)
            stats['tier_restore_requests'] += 1
            stats['tier_restore_blocks'] += len(records)
            stats['tier_restore_bytes'] += size
            stats['tier_restore_ms'] += elapsed
            graft.log('host tier restore req=%s blocks=%d bytes=%d ms=%.1f from=%d to=%d', request.request_id, len(records), size,
                      elapsed, first * BLOCK, need * BLOCK)
        return ok

    # -- the NVMe tier's part of a restore (docstring item j) -------------------------------------------------------
    def promote_records(self, request, indices, protect=()):
        """Read the NVMe-resident blocks `indices` of `request` into host-tier records (slab slots reserved by policy, so what that evicts is itself spilled to the NVMe if it
        is worth it), checking the NVMe tier's digests. -> {index: TierRecord}, or None (counted) when any block is gone, there is no room, a digest failed or the device
        failed. `protect`: keys of host records the reservation must not take. A restore that fails after this keeps the records: they are valid host-tier blocks."""
        tier, nvme, stats = self.tier, self.nvme, self.registry.stats
        counters = nvme.c
        hashes = request.block_hashes
        tenant = prefix_registry.tenant_of_salt(request.cache_salt)
        chain = hashes[0] if len(hashes) else None
        keys = [hashes[index] for index in indices]
        # (pins: the host records the reservation must not take are pinned for its length; a pin someone else holds - a sliced job's - is theirs and stays. The new records are
        # not left pinned: the caller pins what it uses, and nothing touches the tier in between.)
        before = set(tier.pinned)
        mine = [key for key in list(protect) if key not in before]
        tier.pin(mine)
        entries = nvme.pin_kv(keys)
        slots = []
        try:
            if any(entry is None for entry in entries):
                counters['nvme_miss_blocks'] += sum(1 for entry in entries if entry is None)
                counters['nvme_restore_failures'] += 1
                stats['tier_restore_failures'] += 1
                return None
            if tier.slot_bytes:
                slots = tier.reserve(len(keys), [tenant] * len(keys))
                if len(slots) < len(keys):
                    tier.abort(slots)
                    slots = []
                    counters['nvme_restore_refused_room'] += 1
                    return None
                views = [slot.view for slot in slots]
            else:
                views = [bytearray(entry.nbytes) for entry in entries]
            results = nvme.backend.read_to_host(nvme, entries, views)
            failed = [index for index, result in zip(indices, results) if result is not None]
            if failed:
                if slots:
                    tier.abort(slots)
                    slots = []
                if any(result == 'digest' for result in results):
                    counters['nvme_restore_refused_digest'] += 1
                    self.graft.log('nvme tier record failed its digest req=%s block=%d: dropped', request.request_id, failed[0])
                else:
                    counters['nvme_restore_failures'] += 1
                return None
            made = {}
            for position, index in enumerate(indices):
                if slots:
                    tier.commit(slots[position], keys[position], tenant, chain, index)
                    record = tier.get(keys[position], touch=False)
                elif tier.put(keys[position], views[position], tenant, chain, index):
                    record = tier.get(keys[position], touch=False)
                else:
                    record = None
                if record is None:
                    counters['nvme_restore_failures'] += 1
                    return None
                made[index] = record
            slots = []
            nvme.touch_kv(keys)
            return made
        except Exception as error:
            counters['nvme_restore_failures'] += 1
            stats['tier_restore_failures'] += 1
            self.graft.log('nvme tier restore failed req=%s: %s: %s', request.request_id, type(error).__name__, error)
            return None
        finally:
            if slots:
                tier.abort(slots)
            tier.unpin(mine)
            nvme.unpin(entries)

    def promote(self, request, first, need, absent):
        """The whole restore's NVMe step: bring `absent` (indexes of the blocks [first, need) the host tier lacks) in as host records. The pins it takes on the new records stay
        for load(), which pins the whole range itself and lets go at its end. -> the records, or None."""
        hashes = request.block_hashes
        gone = set(absent)
        present = [hashes[index] for index in range(first, need) if index not in gone]
        began = time.perf_counter()
        made = self.promote_records(request, absent, protect=present)
        if made is None:
            return None
        size = sum(len(record.payload) for record in made.values())
        self.nvme.note_restore(len(made), size, first, need, request.request_id)
        return made

    def fill(self, job, low, count):
        """The sliced restore's NVMe step: the Pending blocks among job.records[low:low + count] become host records (SliceRefused when they cannot)."""
        wanted = [position for position in range(low, low + count) if isinstance(job.records[position], Pending)]
        if not wanted:
            return
        indices = [job.records[position].index for position in wanted]
        made = self.promote_records(job.request, indices)
        if made is None:
            raise SliceRefused('an NVMe block could not be brought back')
        for position, index in zip(wanted, indices):
            job.records[position] = made[index]
        size = sum(len(record.payload) for record in made.values())
        self.nvme.note_restore(len(made), size, job.first + wanted[0], job.first + wanted[-1] + 1, job.request_id)

    def load_direct(self, request, blocks, h, first, need, drop):
        """load() for a DIRECT backend (RestoreBackend.direct: NVMe -> device with no host copy): the blocks the host tier holds are written as before, the NVMe-resident
        ones are handed to the backend with the device blocks the pool allocated for them. No backend is built; test_qwen_kv_nvme drives this seam with a double. Everything else is
        load()'s, including the miss on any failure."""
        graft, registry, stats, tier, pool, nvme = self.graft, self.registry, self.registry.stats, self.tier, self.graft.block_pool, self.nvme
        hashes = request.block_hashes
        records, cold = {}, []
        for index in range(first, need):
            record = tier.get(hashes[index])
            if record is not None:
                records[index] = record
            elif nvme.has_kv(hashes[index]):
                cold.append(index)
            else:
                stats['tier_restore_failures'] += 1
                return False
        lookahead = int(getattr(graft.scheduler, 'num_lookahead_tokens', 0) or 0)
        beyond = -(-(request.num_tokens + lookahead) // BLOCK) - need + 1
        if pool.get_num_free_blocks() < (need - first) + max(0, beyond):
            stats['tier_restore_refused_room'] += 1
            return False
        for record in self.digest_failures(list(records.values())):
            stats['tier_restore_refused_digest'] += 1
            graft.log('host tier record failed its digest req=%s block=%d: dropped', request.request_id, record.index)
            tier.remove(record.key)
            self.unhold([record.key])
            return False
        entries = nvme.pin_kv([hashes[index] for index in cold])
        if any(entry is None for entry in entries):
            nvme.c['nvme_miss_blocks'] += sum(1 for entry in entries if entry is None)
            nvme.c['nvme_restore_failures'] += 1
            nvme.unpin(entries)
            return False
        protected = list(blocks.blocks[0])
        if drop:
            extra = pool.cached_block_hash_to_block.get_one_block(self.key_of(hashes[h // BLOCK]))
            if extra is None:
                stats['tier_restore_failures'] += 1
                nvme.unpin(entries)
                return False
            protected.append(extra)
        began = time.perf_counter()
        new = []
        ok = False
        pool.touch(protected)
        tier.pin([record.key for record in records.values()])
        tenant = prefix_registry.tenant_of_salt(request.cache_salt)
        quota = graft.quota
        if quota is not None:
            quota.allocating = tenant
        try:
            new = pool.get_new_blocks(need - first)
            self.flush()
            ids = [block.block_id for block in new]
            warm = sorted(records)
            if warm:
                self.io.write_blocks([ids[index - first] for index in warm], [records[index].payload for index in warm])
            if cold:
                nvme.restore_to_device(entries, [ids[index - first] for index in cold])
            pool.cache_full_blocks(request, [None] * first + list(new), first, need, BLOCK, 0)
            chain = hashes[0] if len(hashes) else None
            for offset, block in enumerate(new):
                self.meta[block.block_id] = (hashes[first + offset], chain, tenant, first + offset)
            if quota is not None:
                quota.note_published(new, tenant)
            ok = True
        except Exception as error:
            stats['tier_restore_failures'] += 1
            nvme.c['nvme_restore_failures'] += 1
            self.failures += 1
            graft.log('nvme tier direct restore failed req=%s (%d so far): %s: %s', request.request_id, self.failures, type(error).__name__, error)
            if self.failures >= 3:
                registry.tier_latch('three restores in a row failed')
        finally:
            if quota is not None:
                quota.allocating = None
            tier.unpin([record.key for record in records.values()])
            nvme.unpin(entries)
            if new:
                pool.free_blocks(reversed(new))
            pool.free_blocks(reversed(protected))
        if ok:
            self.failures = 0
            elapsed = (time.perf_counter() - began) * 1000.0
            cold_bytes = sum(entry.nbytes for entry in entries)
            size = sum(len(record.payload) for record in records.values()) + cold_bytes
            stats['tier_restore_requests'] += 1
            stats['tier_restore_blocks'] += need - first
            stats['tier_restore_bytes'] += size
            stats['tier_restore_ms'] += elapsed
            nvme.note_restore(len(cold), cold_bytes, first, need, request.request_id)
            graft.log('host tier restore req=%s blocks=%d bytes=%d ms=%.1f from=%d to=%d', request.request_id, need - first, size, elapsed, first * BLOCK, need * BLOCK)
        return ok

    # -- the sliced restore (docstring item h) -------------------------------------------------------------------
    def hold(self, request):
        """Called by the one-fresh-prefill admission for the candidate it is about to admit: True while the request's blocks are not all back from the tier (this call restored a
        slice of them, or started the job), so the pass must not admit it; False when it may be admitted (nothing to restore, or the last slice landed in this call, or the restore
        was abandoned). Never raises into the scheduler except for a failed audit, which stops the engine as it does unsliced."""
        job = self.jobs.get(request.request_id)
        if not self.slicing or request.request_id in self.refused:
            if job is not None:
                self.abort(job, 'sliced restore off', refuse=False)
            return False
        if not self.poll() or not self.eligible(request):
            if job is not None:
                self.abort(job, 'the tier is off for this request', refuse=False)
            return False
        if job is None:
            job = self.start(request)
            if job is None:
                return False
        if job.step == self.graft.step_no:
            return True
        state = self.advance(job)
        if state == 'more':
            job.held += 1
            self.registry.stats['tier_slice_held'] += 1
            return True
        return False

    def start(self, request):
        """The job for a request that needs the tier's blocks, or None (nothing to restore, a record gone, no room, no slice to give): what load() checks before it allocates, without
        allocating. Takes the pins and touches the request's own device hit; abort() gives them back."""
        graft, registry, stats, tier, pool = self.graft, self.registry, self.registry.stats, self.tier, self.graft.block_pool
        blocks, h = self.recompute(request)
        h_effective = self.effective_hit(request, h)
        if h_effective <= h:
            return None
        q = graft.select(request, blocks, h_effective, graft.peeked.get(request.request_id))[0]
        if q <= h:
            return None
        drop = 1 if graft.drop_last and h > 0 else 0
        first = h // BLOCK + drop
        need = q // BLOCK + (1 if graft.drop_last else 0)
        if need <= first:
            return None
        hashes = request.block_hashes
        records = []
        for index in range(first, need):
            record = tier.get(hashes[index])
            if record is None:
                if self.nvme is not None and not self.nvme.backend.direct and self.nvme.has_kv(hashes[index]):
                    # On the NVMe tier only: the slice that needs it reads it back (fill). (A direct backend has no sliced restore: the trim restores whole.)
                    record = Pending(hashes[index], index)
                else:
                    stats['tier_restore_failures'] += 1
                    return None
            records.append(record)
        lookahead = int(getattr(graft.scheduler, 'num_lookahead_tokens', 0) or 0)
        beyond = -(-(request.num_tokens + lookahead) // BLOCK) - need + 1
        if pool.get_num_free_blocks() < (need - first) + max(0, beyond):
            stats['tier_restore_refused_room'] += 1
            return None
        protected = list(blocks.blocks[0])
        if drop:
            extra = pool.cached_block_hash_to_block.get_one_block(self.key_of(hashes[h // BLOCK]))
            if extra is None:
                stats['tier_restore_failures'] += 1
                return None
            protected.append(extra)
        pool.touch(protected)
        tier.pin([record.key for record in records])
        job = RestoreJob(request, first, need, records, protected, prefix_registry.tenant_of_salt(request.cache_salt))
        self.jobs[request.request_id] = job
        stats['tier_slice_jobs'] += 1
        return job

    def slice_size(self, remaining):
        """How many blocks the next slice restores: the time budget over the measured cost per block (a probe of SLICE_PROBE_BLOCKS before any is measured), at least one, at most the
        block cap and what is left."""
        if self.rate is None:
            blocks = SLICE_PROBE_BLOCKS
        else:
            blocks = int(self.slice_ms / max(self.rate, 1e-6))
        return max(1, min(self.slice_blocks or remaining, blocks, remaining))

    def advance(self, job, whole=False):
        """One slice of `job` (all that is left when `whole`): the digests of its records, an allocation of its blocks (what that evicts is spilled before anything is written), the raw write,
        the read-back audit. -> 'more', 'done' (the job is finished and gone) or 'failed' (abandoned and gone)."""
        graft, registry, stats, tier, pool = self.graft, self.registry, self.registry.stats, self.tier, self.graft.block_pool
        remaining = job.total - len(job.new)
        count = remaining if whole else self.slice_size(remaining)
        low = len(job.new)
        records = job.records[low:low + count]
        if pool.get_num_free_blocks() < count:
            stats['tier_restore_refused_room'] += 1
            self.abort(job, 'no room for %d blocks' % count)
            return 'failed'
        began = time.perf_counter()
        quota = graft.quota
        try:
            if self.nvme is not None:
                self.fill(job, low, count)
                records = job.records[low:low + count]
            for record in self.digest_failures(records):
                stats['tier_restore_refused_digest'] += 1
                graft.log('host tier record failed its digest req=%s block=%d: dropped', job.request_id, record.index)
                tier.remove(record.key)
                self.unhold([record.key])
                raise SliceRefused()
            if quota is not None:
                quota.allocating = job.tenant
            try:
                new = pool.get_new_blocks(count)
            finally:
                if quota is not None:
                    quota.allocating = None
            job.new.extend(new)
            self.flush()
            ids = [block.block_id for block in new]
            payloads = [record.payload for record in records]
            self.io.write_blocks(ids, payloads)
            if self.config.audit:
                self.audit(job.request, ids, payloads)
        except SliceRefused as refused:
            self.abort(job, refused.reason)
            return 'failed'
        except TierAuditError:
            self.abort(job, 'the audit failed', refuse=False)
            raise
        except Exception as error:
            stats['tier_restore_failures'] += 1
            self.failures += 1
            graft.log('host tier restore failed req=%s (%d so far): %s: %s', job.request_id, self.failures, type(error).__name__, error)
            if self.failures >= 3:
                registry.tier_latch('three restores in a row failed')
            self.abort(job, 'a slice failed')
            return 'failed'
        elapsed = (time.perf_counter() - began) * 1000.0
        size = sum(len(record.payload) for record in records)
        job.work_ms += elapsed
        job.bytes += size
        job.slices += 1
        job.max_ms = max(job.max_ms, elapsed)
        job.step = graft.step_no
        per_block = elapsed / count
        self.rate = per_block if self.rate is None else 0.5 * self.rate + 0.5 * per_block
        stats['tier_slice_steps'] += 1
        stats['tier_slice_blocks'] += count
        stats['tier_slice_ms'] += elapsed
        stats['tier_slice_max_ms'] = max(stats['tier_slice_max_ms'], elapsed)
        graft.log('host tier restore slice req=%s blocks=%d done=%d of %d ms=%.1f', job.request_id, count, len(job.new), job.total, elapsed)
        if len(job.new) < job.total:
            return 'more'
        self.finish(job)
        return 'done'

    def finish(self, job):
        """The last slice has landed: hash the blocks into the pool and free them as cached ones (vLLM's own hit logic takes them), let go of the pins and the request's own hit."""
        graft, registry, stats, tier, pool = self.graft, self.registry, self.registry.stats, self.tier, self.graft.block_pool
        request, hashes = job.request, job.request.block_hashes
        try:
            pool.cache_full_blocks(request, [None] * job.first + list(job.new), job.first, job.need, BLOCK, 0)
            chain = hashes[0] if len(hashes) else None
            for offset, block in enumerate(job.new):
                self.meta[block.block_id] = (hashes[job.first + offset], chain, job.tenant, job.first + offset)
            if graft.quota is not None:
                graft.quota.note_published(job.new, job.tenant)
        finally:
            self.release(job)
        self.failures = 0
        stats['tier_restore_requests'] += 1
        stats['tier_restore_blocks'] += job.total
        stats['tier_restore_bytes'] += job.bytes
        stats['tier_restore_ms'] += job.work_ms
        graft.log('host tier restore req=%s blocks=%d bytes=%d ms=%.1f from=%d to=%d slices=%d max_slice_ms=%.1f held_steps=%d', job.request_id, job.total, job.bytes,
                  job.work_ms, job.first * BLOCK, job.need * BLOCK, job.slices, job.max_ms, job.held)

    def release(self, job):
        """Give back everything a job holds: its blocks (cached by now, or plain garbage), the request's own hit, the pins."""
        pool = self.graft.block_pool
        self.jobs.pop(job.request_id, None)
        self.tier.unpin(job.keys)
        try:
            if job.new:
                pool.free_blocks(reversed(job.new))
        finally:
            pool.free_blocks(reversed(job.protected))
        job.new, job.protected = [], []

    def abort(self, job, reason, refuse=True):
        """Abandon a job: the blocks written so far are freed uncached, the pins and the request's own hit given back. `refuse`: the request's tier hit is ignored from now on (a miss,
        not retried every step)."""
        stats = self.registry.stats
        stats['tier_slice_aborted'] += 1
        self.graft.log('host tier restore slice aborted req=%s after %d of %d blocks: %s', job.request_id, len(job.new), job.total, reason)
        self.release(job)
        if refuse:
            self.refused.add(job.request_id)

    # -- write-through: the idle copy (docstring item j) ---------------------------------------------------------
    def note_idle(self, request):
        """A request finished (the free hook): its published prompt blocks stay cached on the device, and under QWEN_PREFIX_NVME_SPILL=through (the default) they are copied toward
        the drive once the request has been idle for QWEN_PREFIX_NVME_THROUGH_IDLE_S (default 0: the next step)."""
        nvme = self.nvme
        if nvme is None or not nvme.through or not nvme.active or not self.active or not self.eligible(request):
            return
        hashes = request.block_hashes
        count = min(len(hashes), floor_chunk(request.num_prompt_tokens) // BLOCK)
        if count * BLOCK < max(self.config.min_tokens, CHUNK):
            return
        clock = self.graft.kill_switch.clock
        chain = hashes[0] if len(hashes) else None
        self.idle.pop(request.request_id, None)
        self.idle[request.request_id] = IdleCopy(clock() + nvme.plan.through_idle_s, hashes, count, chain, prefix_registry.tenant_of_salt(request.cache_salt))
        while len(self.idle) > IDLE_BACKLOG:
            self.idle.popitem(last=False)
        nvme.backlog_blocks = sum(entry.count - entry.next for entry in self.idle.values())

    def copy_idle(self):
        """At the end of a schedule() call (the device idle, nothing yet written for the step - flush's own moment): read up to QWEN_PREFIX_NVME_THROUGH_BLOCKS of the cached
        device blocks of requests that have been idle long enough into the host tier. Each record that enters the tier is queued to the drive by the tier's commit sink, so a warm
        copy is on its way before any eviction. Blocks the host tier or the drive already holds, blocks no longer cached, blocks cached in this very step (not yet written) and blocks
        no checkpoint reaches are skipped. A failing read is counted and, three in a row, latches the host tier as for a spill."""
        nvme, registry, tier, pool = self.nvme, self.registry, self.tier, self.graft.block_pool
        if nvme is not None and self.idle and not nvme.active:
            self.idle.clear()                      # the NVMe level is latched or stopping (a lost drive, a stop): nothing to copy toward
            return
        if not self.idle or not self.active or nvme is None or not nvme.through:
            return
        counters = nvme.c
        clock = self.graft.kill_switch.clock
        now = clock()
        gauges = prefix_registry.host_gauges()
        available = gauges.get('host_mem_available_bytes')
        if available is not None and available < self.config.min_available:
            return
        budget = nvme.plan.through_blocks
        picked = []
        for request_id in list(self.idle):
            entry = self.idle[request_id]
            if entry.ready > now:
                break
            while entry.next < entry.count and len(picked) < budget:
                index = entry.next
                entry.next += 1
                key = entry.hashes[index]
                if tier.has(key) or nvme.has_kv(key) or key in nvme.queued_keys:
                    continue
                block = pool.cached_block_hash_to_block.get_one_block(self.key_of(key))
                if block is None or block.block_id in registry.same_step_blocks:
                    continue
                if registry._tier_dead(types.SimpleNamespace(chain=entry.chain, index=index)):
                    continue
                picked.append((key, block.block_id, entry.tenant, entry.chain, index))
            if entry.next >= entry.count:
                del self.idle[request_id]
            if len(picked) >= budget:
                break
        nvme.backlog_blocks = sum(entry.count - entry.next for entry in self.idle.values())
        if not picked:
            return
        began = time.perf_counter()
        slots = tier.reserve(len(picked), [item[2] for item in picked]) if tier.slot_bytes else []
        if tier.slot_bytes and len(slots) < len(picked):
            tier.abort(slots)
            return
        try:
            payloads = self.io.read_blocks([item[1] for item in picked], into=[slot.view for slot in slots] if slots else None)
            if len(payloads) != len(picked):
                raise ValueError('the tier IO read %d payloads for %d blocks' % (len(payloads), len(picked)))
        except Exception as error:
            tier.abort(slots)
            counters['nvme_through_failures'] += 1
            self.failures += 1
            self.graft.log('nvme tier idle copy failed (%d so far): %s: %s', self.failures, type(error).__name__, error)
            if self.failures >= 3:
                registry.tier_latch('three device reads in a row failed')
            return
        self.failures = 0
        filed = 0
        for position, (key, _block, tenant, chain, index) in enumerate(picked):
            if slots:
                tier.commit(slots[position], key, tenant, chain, index)
                filed += 1
            elif tier.put(key, payloads[position], tenant, chain, index):
                filed += 1
        elapsed = (time.perf_counter() - began) * 1000.0
        counters['nvme_through_copied_blocks'] += filed
        counters['nvme_through_copied_bytes'] += sum(len(payload) for payload in payloads)
        counters['nvme_through_copy_ms'] += elapsed
        counters['nvme_through_copy_steps'] += 1

    def cancel(self, request_id):
        """The request ended (finished, aborted, freed): its job goes with it, and so does the memory of a refusal."""
        job = self.jobs.get(request_id)
        if job is not None:
            self.abort(job, 'the request ended', refuse=False)
        self.refused.discard(request_id)

    def audit(self, request, ids, payloads):
        """QWEN_PREFIX_HOST_TIER_AUDIT=1: read the restored blocks back from the device and compare them with the bytes written."""
        stats = self.registry.stats
        back = self.io.read_blocks(ids)
        stats['tier_audit_reads'] += len(ids)
        bad = [index for index, (a, b) in enumerate(zip(back, payloads)) if bytes(a) != bytes(b)]
        if len(back) != len(payloads) or bad:
            stats['tier_audit_mismatches'] += max(1, len(bad))
            raise TierAuditError('host tier audit: req=%s %d of %d restored blocks read back differently from the bytes written (first '
                                 'at %s)' % (request.request_id, len(bad), len(ids), bad[:4]))


class DeviceQuota(object):
    """The per-tenant quotas on the device pool (module docstring, item g). It keeps, per cached block the cap or a restore published, the tenant it belongs to
    (owner), per tenant the count of such blocks (resident) and the blocks freed and not since touched or evicted, oldest first (idle: entries are validated on use,
    so a block touched back into service is simply skipped). It touches vLLM's free queue in two places only: it moves a tenant's excess idle blocks to the FRONT
    (prepend_n: the next allocation evicts them first), and it sets protected idle blocks aside at the TAIL while an allocation walks the front. Everything else is
    vLLM's own."""

    def __init__(self, graft, config):
        self.graft = graft
        self.registry = graft.registry
        self.pool = graft.block_pool
        self.cap = config.device_cap_blocks
        self.floor = config.device_floor_blocks
        self.owner = {}
        self.resident = {}
        self.idle = {}
        self.demoted = {}
        self.allocating = None
        self.stats = graft.registry.stats
        self.original = {}

    def attach(self, manager):
        """Put the quota's calls on the pool and the manager (instance attributes, like the graft's own wrappers): free_blocks (the idle bookkeeping and the cap's demotion),
        get_new_blocks (the floor's walk; only with a floor) and allocate_slots (who is allocating)."""
        pool, original = self.pool, self.original
        original['free_blocks'] = pool.free_blocks
        original['get_new_blocks'] = pool.get_new_blocks

        def free_blocks(ordered_blocks):
            blocks = list(ordered_blocks)
            original['free_blocks'](blocks)
            self.on_free(blocks)

        def get_new_blocks(num_blocks):
            self.protect(num_blocks)
            return original['get_new_blocks'](num_blocks)

        pool.free_blocks = free_blocks
        if self.floor:
            pool.get_new_blocks = get_new_blocks
        if hasattr(manager, 'allocate_slots'):
            original['allocate_slots'] = manager.allocate_slots

            def allocate_slots(request, *args, **kwargs):
                self.allocating = prefix_registry.tenant_of_salt(request.cache_salt) if request.cache_salt else None
                try:
                    return original['allocate_slots'](request, *args, **kwargs)
                finally:
                    self.allocating = None

            manager.allocate_slots = allocate_slots

    # -- accounting -----------------------------------------------------------------------------
    def note_published(self, blocks, tenant):
        """The cap (or a restore) cached `blocks` for `tenant`: they are its resident blocks now. Over the cap, its excess idle blocks are demoted."""
        for block in blocks:
            if block.block_id not in self.owner:
                self.owner[block.block_id] = tenant
                self.resident[tenant] = self.resident.get(tenant, 0) + 1
        self.enforce(tenant)

    def on_free(self, blocks):
        """vLLM freed `blocks` (after its own free_blocks ran): the cached ones with no reference left are idle, newest last, and their tenants are re-checked."""
        tenants = []
        for block in blocks:
            tenant = self.owner.get(block.block_id)
            if tenant is None or block.ref_cnt != 0 or block.block_hash is None:
                continue
            group = self.idle.setdefault(tenant, OrderedDict())
            group.pop(block.block_id, None)
            group[block.block_id] = block
            self.demoted.pop(block.block_id, None)
            if tenant not in tenants:
                tenants.append(tenant)
        for tenant in tenants:
            self.enforce(tenant)

    def on_evict(self, block):
        """vLLM evicted `block` from the cache: it is no tenant's resident block now. Counted by reason: a block this quota had demoted left because its tenant was over
        its cap (own); any other left by vLLM's own order (global)."""
        tenant = self.owner.pop(block.block_id, None)
        if tenant is None:
            return
        left = self.resident.get(tenant, 0) - 1
        if left > 0:
            self.resident[tenant] = left
        else:
            self.resident.pop(tenant, None)
        group = self.idle.get(tenant)
        if group is not None:
            group.pop(block.block_id, None)
            if not group:
                del self.idle[tenant]
        self.stats['quota_dev_evictions_own' if self.demoted.pop(block.block_id, None) is not None else 'quota_dev_evictions_global'] += 1

    def clear(self):
        self.owner.clear()
        self.resident.clear()
        self.idle.clear()
        self.demoted.clear()

    def _idle_valid(self, tenant, block):
        queue_linked = getattr(block, 'prev_free_block', None) is not None
        return (block.ref_cnt == 0 and block.block_hash is not None and self.owner.get(block.block_id) == tenant and queue_linked)

    # -- the cap ----------------------------------------------------------------------------------
    def enforce(self, tenant):
        """The cap: while `tenant`'s resident blocks exceed it, move its oldest idle blocks (not already moved) to the front of the free queue. A block in use cannot be
        moved: the excess it accounts for is demoted when the request frees it."""
        if not self.cap:
            return 0
        excess = self.resident.get(tenant, 0) - self.cap
        if excess <= 0:
            return 0
        pending = 0
        for block_id in [key for key, (owner, _) in self.demoted.items() if owner == tenant]:
            if self._idle_valid(tenant, self.demoted[block_id][1]):
                pending += 1
            else:
                del self.demoted[block_id]
        need = excess - pending
        if need <= 0:
            return 0
        group = self.idle.get(tenant)
        if not group:
            return 0
        picked, stale = [], []
        for block_id, block in group.items():
            if not self._idle_valid(tenant, block):
                stale.append(block_id)
                continue
            picked.append(block)
            if len(picked) >= need:
                break
        for block_id in stale:
            group.pop(block_id, None)
        queue = self.pool.free_block_queue
        for block in picked:
            queue.remove(block)
            group.pop(block.block_id, None)
            self.demoted[block.block_id] = (tenant, block)
        if picked:
            queue.prepend_n(picked)
            self.stats['quota_dev_demoted_blocks'] += len(picked)
        if group is not None and not group:
            self.idle.pop(tenant, None)
        return len(picked)

    # -- the floor --------------------------------------------------------------------------------
    def protect(self, num_blocks):
        """Before an allocation of `num_blocks`: walk the front of the free queue and put at the tail every idle block that is protected (its tenant would fall below the
        floor by losing it, and the tenant is not the one allocating), until `num_blocks` unprotected blocks head the queue. If there are fewer, the allocation takes
        protected blocks after all (an override: vLLM's allocation never fails for a quota)."""
        if not self.floor or num_blocks <= 0:
            return
        queue = self.pool.free_block_queue
        tail = queue.fake_free_list_tail
        block = queue.fake_free_list_head.next_free_block
        limit = queue.num_free_blocks
        taken, found, examined, moved = {}, 0, 0, 0
        while found < num_blocks and block is not tail and block is not None and examined < limit:
            examined += 1
            following = block.next_free_block
            tenant = self.owner.get(block.block_id) if block.block_hash is not None else None
            if tenant is not None and tenant != self.allocating and self.resident.get(tenant, 0) - taken.get(tenant, 0) <= self.floor:
                queue.remove(block)
                queue.append(block)
                moved += 1
            else:
                found += 1
                if tenant is not None:
                    taken[tenant] = taken.get(tenant, 0) + 1
            block = following
        if moved:
            self.stats['quota_dev_floor_protected'] += moved
        if found < num_blocks:
            self.stats['quota_dev_floor_overrides'] += num_blocks - found

    def gauge(self):
        """The aggregates the stats carry: how many tenants hold resident blocks and how many are at their cap."""
        at_cap = sum(1 for held in self.resident.values() if self.cap and held >= self.cap)
        return dict(quota_dev_tenants=len(self.resident), quota_dev_at_cap=at_cap)


class SchedulerGraft(object):
    # Sticky sessions (the module docstring); off on the class, so a graft built without them - or
    # a stand-in carrying only a registry (the model fixture calls plan() that way) - plans as G1.
    sticky = False
    drop_last = False
    # The host KV tier's hooks (TierHooks), None unless the registry has a tier and the model attached its IO.
    tier_hooks = None
    # The per-tenant device quota (DeviceQuota), None unless QWEN_PREFIX_TENANT_QUOTA_DEVICE_* names one.
    quota = None
    # The serial of the schedule() call in progress (begin_step counts it): a restore job takes one slice per step.
    step_no = 0
    make_block_hash = None
    # Exact resume P0 (the module docstring): set by install() under sticky sessions only, off on the class.
    undrop = False
    carry = False
    carry_audit = False
    # Lever N's peek memo (request id -> the Q the trim would land on), cleared at the start of every schedule() call.
    peeked = {}

    def __init__(self, scheduler, registry, kill_switch, logger, stats=None, sticky=False):
        self.scheduler = scheduler
        self.registry = registry
        self.manager = scheduler.kv_cache_manager
        self.coordinator = self.manager.coordinator
        self.block_pool = self.manager.block_pool
        self.single = self.coordinator.single_type_managers[0]
        self.kill_switch = kill_switch
        self.kill_switch_path = kill_switch.path
        self.log = logger
        self.stats_export = stats
        self.original = {}
        self.get_block_hash = None
        self.sticky = sticky is True
        self.drop_last = self.sticky and drops_last_block(self.coordinator)
        self.peeked = {}
        # Where the merged route's switches are read (install's environ; None: os.environ).
        self.environ = None

    # -- kill switch -----------------------------------------------------------------------------
    @property
    def killed(self):
        return self.kill_switch.engaged or self.registry.disabled is not None

    def restore_hold(self, request):
        """The one-fresh-prefill admission asks, for the candidate it is about to admit, whether to hold it for the next slice of its restore from the tier (TierHooks.hold; docstring
        item h). False when slicing is off, the request may not hit, or there is nothing to restore."""
        hooks = self.tier_hooks
        if hooks is None or not hooks.slicing or self.excluded(request) or self.killed:
            return False
        return hooks.hold(request)

    def kill_switch_engaged(self):
        if self.kill_switch.poll():
            # Checkpoints go. Grants already staged in this schedule() call keep their own
            # reference: vLLM holds their trimmed hits, so they must still commit and restore.
            self.registry.disable('kill switch %s' % self.kill_switch.path)
            self.log('kill switch %s present: no grants, captures or publishing until the engine '
                     'restarts; registry cleared', self.kill_switch.path)
            self.export(force=True)
        return self.killed

    def export(self, force=False):
        if self.stats_export is not None:
            self.stats_export.maybe_export(self.registry, force=force)
            maybe_tier = getattr(self.stats_export, 'maybe_tier', None)
            if maybe_tier is not None:
                maybe_tier(self.registry, force=force)

    def end_boundary(self, request):
        """The boundary this request's own last checkpoint would sit at (plan's prompt candidate): floor2048(P), less one
        chunk under sticky sessions with the coordinator dropping a hit's last block. The telemetry's session marker."""
        boundary = floor_chunk(request.num_prompt_tokens)
        return boundary - CHUNK if self.sticky and self.drop_last else boundary

    def note(self, request, h, q, kind=None):
        """Tell the registry's telemetry what the trim answered for this attempt (it reads, nothing more)."""
        note_attempt = getattr(self.registry, 'note_attempt', None)
        if note_attempt is None:
            return
        try:
            note_attempt(request.request_id, request.block_hashes, request.num_prompt_tokens,
                         self.end_boundary(request), h, q, kind)
        except Exception:
            pass

    @staticmethod
    def excluded(request):
        """The counter to bump when a request may not hit, publish or capture, else None: an
        unsalted request (fail-closed tenancy, S4), or a streaming-input session, whose
        decode-written tokens vLLM folds into its prompt before it re-enters WAITING with
        num_computed_tokens > 0 (scheduler.py:1213-1252; only the realtime speech endpoint
        creates one, async_llm.py:472)."""
        if not request.cache_salt:
            return 'unsalted_denied'
        if getattr(request, 'resumable', False):
            return 'session_denied'
        return None

    # -- a. the trim -----------------------------------------------------------------------------
    def plan(self, request, h, q):
        """([(pos, block hash at pos)] to capture, the mid-loop positions left out, the drain).

        Candidates: the prompt boundary floor2048(P) above Q, and the gap boundary floor2048(h)
        when it is a whole chunk above Q (where a shared system prompt or tool block becomes
        reusable by the next sibling after one miss). The row's chunk loop drains at
        floor2048(num_tokens) - above P for a resumed (preempted) request, whose prefill covers its
        output tokens too. A candidate below the drain needs a mid-loop capture: planned only once
        the model has declared it takes them (registry.enable_mid_loop_capture). Nothing above
        floor2048(P) is ever planned: the cap publishes nothing there, so no request could hit it.

        Sticky sessions, with the coordinator dropping a hit's last block: the prompt's candidate is
        floor2048(P) - 2048 instead. The next turn's hit stops at least one block short of what this
        turn publishes (floor2048(P)), so its trim can land no higher than one chunk below it."""
        prompt_boundary = floor_chunk(request.num_prompt_tokens)
        drain = floor_chunk(request.num_tokens)
        candidates = set()
        undrop = getattr(self, 'undrop', False)
        sticky = getattr(self, 'sticky', False) and getattr(self, 'drop_last', False) and not undrop
        capture_boundary = prompt_boundary - CHUNK if sticky else prompt_boundary
        if capture_boundary > q:
            candidates.add(capture_boundary)
        if undrop and not getattr(self, 'carry', False) and prompt_boundary - CHUNK > q:
            # The undrop without the carry: a next turn whose prompt ends under floor2048(P) + 2048 is held to the window ceiling
            # floor2048(P' - 2048) = C0, so C0 stays planned beside the prompt boundary.
            candidates.add(prompt_boundary - CHUNK)
        gap_boundary = floor_chunk(h)
        if gap_boundary - q >= CHUNK:
            candidates.add(gap_boundary)
        hashes = request.block_hashes
        planned, unplanned = [], []
        for pos in sorted(candidates):
            if pos > prompt_boundary or pos > drain or pos // BLOCK > len(hashes):
                continue
            if pos < drain and not self.registry.mid_loop_capture:
                unplanned.append(pos)
                continue
            planned.append((pos, hashes[pos // BLOCK - 1]))
        return planned, unplanned, drain

    def select(self, request, blocks, h, ceiling=None):
        """(Q, key, checkpoint, rejected_same_step): the trim's choice for vLLM's raw hit h, with no side effect beyond the registry's
        own idempotent token-check memo. `ceiling` (the Lever N peek's answer for this request in this schedule() call) caps Q, so the
        in-pass trim can never land above the Q the step's cap was computed from."""
        registry = self.registry
        group = blocks.blocks[0] if h else ()
        limit = h // BLOCK
        same_step = registry.same_step_blocks
        first_same = limit
        if same_step:
            # (h may reach past the device blocks: the host tier's consecutive blocks, which no step has cached yet.)
            for index in range(min(limit, len(group))):
                if group[index].block_id in same_step:
                    first_same = index
                    break
        hashes = request.block_hashes
        request_id = request.request_id
        q, key, checkpoint = 0, None, None
        rejected_same_step = False
        k = h // CHUNK
        window_ceiling = None
        if self.sticky:
            # The fast path's drafter window [P - 2048, P) must be this prefill's own chunks - unless the checkpoint carries the
            # window rows below it (QWEN_PREFIX_WINDOW_CARRY): then any boundary below P qualifies, checked per candidate below.
            window_ceiling = max(0, request.num_prompt_tokens - CHUNK) // CHUNK * CHUNK
            if getattr(self, 'carry', False):
                k = min(k, max(0, request.num_prompt_tokens - 1) // CHUNK)
            else:
                k = min(k, window_ceiling // CHUNK)
        if ceiling is not None:
            k = min(k, ceiling // CHUNK)
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
            if (window_ceiling is not None and candidate > window_ceiling
                    and not prefix_registry.window_covers(entry, request.num_prompt_tokens, candidate)):
                continue
            if not registry.tokens_match(request_id, candidate, entry, lambda: request.all_token_ids):
                continue
            q, key, checkpoint = candidate, hashes[count - 1], entry
            break
        return q, key, checkpoint, rejected_same_step

    def peek(self, request):
        """The Q this waiting request's admission would be trimmed to (0: no hit), asked BEFORE the pass that would admit it, memoized
        per request for the current schedule() call (begin_step clears it), and without a trace in vLLM's prefix-cache stats or the
        registry's counters beyond the idempotent token-check memo. Lever N's cap needs it: the step budget of a fresh admission
        must be computed from (P, Q), not (P, 0), or a hit at Q = S_last would run [Q, floor2048(P)) as a non-final step and split
        the drafter window across two steps (docs/lever-n-prefix-merged-route.md, D1).

        The in-pass trim (get_computed_blocks) still makes the authoritative call, capped at this answer, so a hit that moved between
        the peek and the pass can only lower Q."""
        request_id = request.request_id
        known = self.peeked.get(request_id)
        if known is not None:
            return known
        q = 0
        if not self.excluded(request) and not self.kill_switch_engaged():
            manager = self.manager
            statistics = getattr(manager, 'prefix_cache_stats', None)
            if statistics is not None:
                manager.prefix_cache_stats = _NoStatistics()
            try:
                blocks, h = self.original['get_computed_blocks'](request)
            finally:
                if statistics is not None:
                    manager.prefix_cache_stats = statistics
            hooks = self.tier_hooks
            q = self.select(request, blocks, h if hooks is None else hooks.effective_hit(request, h))[0]
        self.peeked[request_id] = q
        return q

    def trim(self, request, blocks, h):
        registry = self.registry
        stats = registry.stats
        stats['attempts'] += 1
        empty = self.manager.empty_kv_cache_blocks
        excluded = self.excluded(request)
        if excluded:
            registry.unstage(request.request_id)
            if h:
                stats[excluded] += 1
            self.note(request, h, 0, 'unsalted' if excluded == 'unsalted_denied' else 'denied')
            return empty, 0
        if self.kill_switch_engaged():
            registry.unstage(request.request_id)
            if h:
                stats['killed_denied'] += 1
            self.note(request, h, 0, 'denied')
            return empty, 0
        if registry.turns is not None:
            registry.turns.classify(request, stats)
        hashes = request.block_hashes
        request_id = request.request_id
        hooks = self.tier_hooks
        h_effective = h if hooks is None else hooks.effective_hit(request, h)
        q, key, checkpoint, rejected_same_step = self.select(request, blocks, h_effective, self.peeked.get(request_id))
        if hooks is not None and q > h:
            # The best hit lies beyond the device's: bring the tier's blocks back as cached ones and ask vLLM again.
            if hooks.load(request, blocks, h, q):
                blocks, h = hooks.recompute(request)
            q, key, checkpoint, rejected_same_step = self.select(request, blocks, h, self.peeked.get(request_id))
        group = blocks.blocks[0] if h else ()
        if rejected_same_step:
            stats['same_step_rejects'] += 1
        if registry.entries:
            for boundary in range(floor_chunk(h) + CHUNK, floor_chunk(request.num_tokens - 1) + 1, CHUNK):
                if boundary // BLOCK <= len(hashes):
                    registry.note_orphan(hashes[boundary // BLOCK - 1])
        plan, unplanned, drain = self.plan(request, h, q)
        self.note(request, h, q)
        if q or plan or unplanned:
            gap = floor_chunk(h)
            registry.stage(Grant(request_id, q, h, key, checkpoint, plan, request, drain, unplanned,
                                 chain=hashes[0] if len(hashes) else None, tenant=tenant_of_salt(request.cache_salt),
                                 gap=gap if gap - q >= CHUNK else None))
        else:
            registry.unstage(request_id)
        if q == 0:
            return empty, 0
        return self.manager.create_kv_cache_blocks((list(group[0:q // BLOCK]),)), q

    # -- b. the cap ------------------------------------------------------------------------------
    def cap(self, request, num_computed_tokens):
        if self.excluded(request) or self.killed:
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
            if self.tier_hooks is not None:
                self.tier_hooks.note_published(request, owned, before, after)
            if self.quota is not None:
                self.quota.note_published(owned[before:after], prefix_registry.tenant_of_salt(request.cache_salt))

    # -- c. the per-step commit ------------------------------------------------------------------
    def commit(self, output):
        admitted = {}
        for data in output.scheduled_new_reqs:
            admitted[data.req_id] = data.num_computed_tokens
        cached = output.scheduled_cached_reqs
        resumed = cached.resumed_req_ids
        counts = getattr(output, 'num_scheduled_tokens', None)
        counts = counts if isinstance(counts, dict) else None
        if resumed:
            for index, req_id in enumerate(cached.req_ids):
                if req_id in resumed:
                    admitted[req_id] = cached.num_computed_tokens[index]
        registry = self.registry
        # A request the cap is splitting (Lever N): the steps after its first are CACHED requests this wrapper sees commit nothing for;
        # the registry learns from them when the request's final step is scheduled (its in-flight plan then ends with that step).
        if counts is not None and registry.inflight:
            for index, req_id in enumerate(cached.req_ids):
                if req_id in registry.inflight and req_id not in admitted and req_id in counts:
                    registry.note_scheduled(req_id, int(cached.num_computed_tokens[index]), int(counts[req_id]))
        for grant in registry.commit(admitted, counts):
            self.log('grant %s', grant.describe())
        self.export()

    # -- d. eviction coupling --------------------------------------------------------------------
    def evict(self, block):
        pool = self.block_pool
        keys = []
        if block.block_hash is not None:
            keys.append(block.block_hash)
        extra = pool.cached_block_hashes_by_block.get(block.block_id)
        if extra:
            keys.extend(extra)
        hooks = self.tier_hooks
        # The block's KV goes to the tier at the end of the schedule() call; the checkpoint at it then stays (held).
        held = hooks is not None and bool(keys) and hooks.on_evict(block, keys)
        evicted = self.original['_maybe_evict_cached_block'](block)
        if evicted and self.quota is not None:
            self.quota.on_evict(block)
        if evicted and keys and self.registry.entries:
            for key in keys:
                if pool.cached_block_hash_to_block.get_one_block(key) is None:
                    if held:
                        self.registry.stats['tier_ckpt_kept'] += 1
                    else:
                        self.registry.drop(self.get_block_hash(key), 'coupled')
        return evicted

    # -- installation ----------------------------------------------------------------------------
    def wrap(self):
        from vllm.v1.core.kv_cache_utils import get_block_hash

        self.get_block_hash = get_block_hash
        if self.registry.tier is not None:
            from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

            self.make_block_hash = make_block_hash_with_group_id
            self.tier_hooks = TierHooks(self)
        scheduler, manager, coordinator, pool = self.scheduler, self.manager, self.coordinator, self.block_pool
        registry = self.registry
        original = self.original
        original['get_computed_blocks'] = manager.get_computed_blocks
        original['cache_blocks'] = coordinator.cache_blocks
        original['schedule'] = scheduler.schedule
        original['_maybe_evict_cached_block'] = pool._maybe_evict_cached_block
        original['reset_prefix_cache'] = scheduler.reset_prefix_cache
        original['_free_request'] = scheduler._free_request
        quota = None
        if registry.quota is not None and registry.quota.device_on:
            quota = self.quota = DeviceQuota(self, registry.quota)
            registry.quota_gauge = quota.gauge

        def get_computed_blocks(request):
            blocks, h = original['get_computed_blocks'](request)
            return self.trim(request, blocks, h)

        def begin_step():
            registry.begin_step()
            self.peeked.clear()
            self.step_no += 1

        def cache_blocks(request, num_computed_tokens):
            return self.cap(request, num_computed_tokens)

        def schedule(*args, **kwargs):
            begin_step()
            hooks = self.tier_hooks
            try:
                output = original['schedule'](*args, **kwargs)
            except BaseException:
                if hooks is not None:
                    hooks.abandon()
                raise
            if hooks is not None:
                hooks.flush()
                if hooks.nvme is not None and hooks.nvme.through:
                    hooks.copy_idle()
            self.commit(output)
            return output

        def _maybe_evict_cached_block(block):
            return self.evict(block)

        def reset_prefix_cache(*args, **kwargs):
            result = original['reset_prefix_cache'](*args, **kwargs)
            if result:
                registry.clear()
                if self.tier_hooks is not None:
                    self.tier_hooks.pending.clear()
                    self.tier_hooks.meta.clear()
                if self.quota is not None:
                    self.quota.clear()
            else:
                # vLLM kept every cached block (running requests hold blocks,
                # scheduler.py:2196-2240), so every checkpoint still has its KV: keep them.
                registry.stats['reset_kept'] += 1
            return result

        def _free_request(request, *args, **kwargs):
            if registry.turns is not None:
                # Counting only, never raises (TurnMemory.record skips an unsalted request and a streaming session).
                if not getattr(request, 'resumable', False):
                    registry.turns.record(request, registry.stats)
                registry.turns.forget(request.request_id)
            if self.tier_hooks is not None:
                self.tier_hooks.cancel(request.request_id)
                if self.tier_hooks.nvme is not None:
                    self.tier_hooks.note_idle(request)
            registry.forget_request(request.request_id)
            return original['_free_request'](request, *args, **kwargs)

        if self.undrop:
            original['find_longest_cache_hit'] = coordinator.find_longest_cache_hit

            def find_longest_cache_hit(*args, **kwargs):
                # vLLM 0.25.1's unitary coordinator passes drop_eagle_block=(0 in eagle_group_ids) to its one manager: for the length
                # of this lookup the set is empty, so the hit keeps its last matched block (the module docstring, exact resume P0).
                saved = coordinator.eagle_group_ids
                coordinator.eagle_group_ids = set()
                try:
                    return original['find_longest_cache_hit'](*args, **kwargs)
                finally:
                    coordinator.eagle_group_ids = saved

            coordinator.find_longest_cache_hit = find_longest_cache_hit
        manager.get_computed_blocks = get_computed_blocks
        coordinator.cache_blocks = cache_blocks
        scheduler.schedule = schedule
        pool._maybe_evict_cached_block = _maybe_evict_cached_block
        scheduler.reset_prefix_cache = reset_prefix_cache
        scheduler._free_request = _free_request
        if quota is not None:
            quota.attach(manager)

        spec = scheduler.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        module = sys.modules.get(type(scheduler).__module__)
        try:
            import vllm
            vllm_version = getattr(vllm, '__version__', '?')
        except Exception:
            vllm_version = '?'
        self.log('install scheduler=%s.%s plugin=%s coordinator=%s block_size=%d blocks=%d kv_spec_dtype=%s '
                 'QWEN_SDPA_BF8=%s hash=%s executor=%s store_gib=%.1f kill_switch=%s stats=%s vllm=%s',
                 type(scheduler).__module__, type(scheduler).__name__, getattr(module, '__file__', '?'),
                 type(self.coordinator).__name__, spec.block_size, scheduler.kv_cache_config.num_blocks,
                 getattr(spec, 'dtype', '?'), os.environ.get('QWEN_SDPA_BF8', 'unset'),
                 scheduler.cache_config.prefix_caching_hash_algo,
                 scheduler.vllm_config.parallel_config.distributed_executor_backend,
                 registry.budget_bytes / float(1 << 30), self.kill_switch_path,
                 getattr(self.stats_export, 'path', None), vllm_version)
        self.log('install policy evict=%s supersede=%d telemetry=%d ghost_entries=%d checkpoint_bytes=%d',
                 getattr(registry, 'evict_policy', 'lru'), int(getattr(registry, 'supersede', False)),
                 int(getattr(registry, 'telemetry', False)), getattr(registry, 'ghost_limit', 0),
                 getattr(registry, 'checkpoint_nbytes', prefix_registry.CHECKPOINT_NBYTES))
        if self.tier_hooks is not None and self.tier_hooks.slice_blocks:
            hooks = self.tier_hooks
            hooks.hold_ready = bool(getattr(getattr(type(scheduler), '_schedule_prefill_only', None), ADMISSION_WRAPPED, False))
            if hooks.hold_ready:
                # The gate and the smoke read this line (prefix_markers.TIER_SLICE_ON).
                self.log('host tier restore slice on blocks=%d ms=%.0f', hooks.slice_blocks, hooks.slice_ms)
            else:
                self.log('host tier restore slice fell back: no one-fresh-prefill admission wrapper on this scheduler to hold a request between slices; restores run whole, in the trim')
        if self.tier_hooks is not None:
            config = registry.tier_config
            self.log('install host tier kv_bytes=%d total_bytes=%d policy=%s audit=%d verify=%s spill_max=%d min_tokens=%d min_available=%d '
                     'block_bytes=%s fingerprint=%s', config.kv_bytes, config.total, registry.evict_policy, int(config.audit), config.verify,
                     config.spill_max, config.min_tokens, config.min_available, getattr(registry.tier_io, 'block_bytes', '?'),
                     getattr(registry.tier_io, 'fingerprint', ''))
        if registry.quota is not None:
            # The gate reads this line (prefix_markers.QUOTA_ON): the numbers only, never a tenant.
            self.log('tenant quota on device_cap_tokens=%d device_floor_tokens=%d host_cap_bytes=%d host_floor_bytes=%d device=%d host=%d',
                     registry.quota.device_cap_blocks * BLOCK, registry.quota.device_floor_blocks * BLOCK, registry.quota.host_cap_bytes,
                     registry.quota.host_floor_bytes, int(quota is not None), int(self.tier_hooks is not None and registry.quota.host_on))
        if self.sticky:
            self.log('install sticky=1 lookahead=%d drop_last=%s ceiling=floor2048(P-%d)',
                     getattr(scheduler, 'num_lookahead_tokens', 0), self.drop_last, CHUNK)
            if self.undrop or self.carry:
                self.log('window carry engaged undrop=%d carry=%d audit=%d ceiling=%s', int(self.undrop), int(self.carry),
                         int(self.carry_audit), 'floor2048(P-1) with a covering window, else floor2048(P-%d)' % CHUNK
                         if self.carry else 'floor2048(P-%d)' % CHUNK)
            if levern_chunked(scheduler, self.environ):
                # The merged route (Lever N beside prefix reuse): chunked prefill is on, and the Lever N cap installed on the class is the only thing
                # that splits a prefill (its own line, so the sticky line above is what it always was).
                self.log('install chunked=levern: chunked prefill beside the Lever N cap (max_num_scheduled_tokens=%s)',
                         getattr(scheduler, 'max_num_scheduled_tokens', None))
        self.export(force=True)

    def original_get_computed_blocks(self, request):
        """vLLM's own answer, untrimmed (for probes and audits)."""
        return self.original['get_computed_blocks'](request)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('package', help='the vllm_tt_plugin package directory, e.g. '
                                        '/opt/qwen-fast-plugin/src/vllm_tt_plugin')
    parser.add_argument('--check', action='store_true', help='verify the pins and the patch; write nothing')
    options = parser.parse_args(argv)
    report = stage(options.package, check_only=options.check)
    for name, (before, after) in sorted(report.items()):
        print('%s %s %s -> %s' % ('checked' if options.check else 'staged', name, before or '(new)', after))
    return 0


if __name__ == '__main__':
    sys.exit(main())
