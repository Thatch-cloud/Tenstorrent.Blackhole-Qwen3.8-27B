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

The kill switch (qwen_prefix_registry.KillSwitch, the flag file /models/.qwen-c2/prefix-reuse.off,
polled at most once a second from the trim) disables the registry for the life of the process.

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
# Lever N's scheduler edits (lever_n_scheduler_patch, lever_n_model_patch.patch_scheduler) turn
# vLLM chunking on; prefix reuse is not exact beside them (design F5).
LEVER_N_TAGS = ('Lever N M2', '[PINDIAG] m2 one-in-flight:', '_qwen_saved_waiting')

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


def install_problems(scheduler):
    """Every reason prefix reuse would not be exact on this scheduler (F5, F6, F7). All of them are
    reported, so the refusal names the root cause (a hybrid coordinator, say) and not only the
    first internal it lacks."""
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
        problems.append('chunked prefill is on: start_pos > 0 would also mean "continue my own '
                        'suspended scratch" (Lever N)')
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
    if getattr(scheduler, 'num_lookahead_tokens', 0):
        problems.append('speculative lookahead is on')
    if parallel is not None and getattr(parallel, 'pipeline_parallel_size', 1) != 1:
        problems.append('pipeline parallel size %s: a step\'s grants must be read before the next '
                        'schedule()' % parallel.pipeline_parallel_size)
    if missing:
        problems.append('vLLM internals the wrappers bind are missing: %s' % ', '.join(missing))
    return problems


def maybe_install(scheduler, environ=None):
    """The hook TTScheduler.__init__ calls last (see INIT_HOOK). Off unless QWEN_PREFIX_REUSE=1."""
    if not prefix_registry.reuse_enabled(environ):
        return None
    return install(scheduler)


def install(scheduler, registry=None, kill_switch_path=KILL_SWITCH_PATH, poll_s=KILL_SWITCH_POLL_S,
            clock=time.monotonic, logger=log, stats=None):
    """Wrap scheduler (see the module docstring). stats: the StatsExport (default: from the
    environment, QWEN_PREFIX_STATS_PATH and QWEN_PREFIX_STATS_S)."""
    existing = scheduler.__dict__.get('_qwen_prefix')
    if existing is not None:
        return existing
    problems = install_problems(scheduler)
    if problems:
        raise PrefixInstallError('prefix reuse refused: ' + '; '.join(problems))
    registry = registry if registry is not None else shared_registry()
    try:
        registry.bind(scheduler)
    except RuntimeError as error:
        raise PrefixInstallError('prefix reuse refused: %s' % error)
    if stats is None:
        stats = StatsExport(clock=clock, logger=logger)
    graft = SchedulerGraft(scheduler, registry, KillSwitch(kill_switch_path, poll_s, clock), logger, stats)
    graft.wrap()
    scheduler._qwen_prefix = graft
    return graft


class SchedulerGraft(object):
    def __init__(self, scheduler, registry, kill_switch, logger, stats=None):
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

    # -- kill switch -----------------------------------------------------------------------------
    @property
    def killed(self):
        return self.kill_switch.engaged or self.registry.disabled is not None

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
        floor2048(P) is ever planned: the cap publishes nothing there, so no request could hit it."""
        prompt_boundary = floor_chunk(request.num_prompt_tokens)
        drain = floor_chunk(request.num_tokens)
        candidates = set()
        if prompt_boundary > q:
            candidates.add(prompt_boundary)
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
        request_id = request.request_id
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
            if not registry.tokens_match(request_id, candidate, entry, lambda: request.all_token_ids):
                continue
            q, key, checkpoint = candidate, hashes[count - 1], entry
            break
        if rejected_same_step:
            stats['same_step_rejects'] += 1
        if registry.entries:
            for boundary in range(floor_chunk(h) + CHUNK, floor_chunk(request.num_tokens - 1) + 1, CHUNK):
                if boundary // BLOCK <= len(hashes):
                    registry.note_orphan(hashes[boundary // BLOCK - 1])
        plan, unplanned, drain = self.plan(request, h, q)
        if q or plan or unplanned:
            registry.stage(Grant(request_id, q, h, key, checkpoint, plan, request, drain, unplanned))
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

    # -- c. the per-step commit ------------------------------------------------------------------
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
        evicted = self.original['_maybe_evict_cached_block'](block)
        if evicted and keys and self.registry.entries:
            for key in keys:
                if pool.cached_block_hash_to_block.get_one_block(key) is None:
                    self.registry.drop(self.get_block_hash(key), 'coupled')
        return evicted

    # -- installation ----------------------------------------------------------------------------
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
            if result:
                registry.clear()
            else:
                # vLLM kept every cached block (running requests hold blocks,
                # scheduler.py:2196-2240), so every checkpoint still has its KV: keep them.
                registry.stats['reset_kept'] += 1
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
