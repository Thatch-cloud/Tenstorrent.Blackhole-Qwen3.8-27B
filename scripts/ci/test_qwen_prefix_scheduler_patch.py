"""qwen_prefix_scheduler_patch without vLLM: the AST stage, the hook, and the wrappers on fakes.

Runs in the 3.11 CPU suite, which has no vLLM. Three layers:

1. The stage against fixtures/vllm_tt_plugin_bf77cd63/scheduler.py, the pinned plugin's own bytes
   (git show bf77cd63:src/vllm_tt_plugin/scheduler.py; the image's copy is the same blob, sha256
   a1bd6257..., probe 35665853903): the pin, the exact anchor, refusals, what it writes.
2. The hook executes (memory: graft mounted is not graft executed): the staged scheduler.py is
   imported from a package built in a temp dir and TTScheduler is constructed over a stand-in base
   class; maybe_install runs only when QWEN_PREFIX_REUSE=1.
3. The wrappers (trim, cap, commit, eviction coupling, free, reset, kill switch, install refusals)
   on a small model of vLLM's prefix cache (FakeScheduler and friends below). The same scenarios on
   real vLLM 0.25.1 objects are test_qwen_prefix_scheduler_vllm (the installed-vLLM lane).

FakeGdnModel stands in for the G1 model graft in both: its GDN state is a sha256 chain over the
2048-token chunks its loop actually ran, restored from and captured into the registry, and every
restore is compared with the cold chain of the row's own tokens. A checkpoint filed under the wrong
boundary (a capture taken at the drain for a mid-loop position) is therefore caught at the next
hit, as the real model's output would silently be wrong.
"""

import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qwen_prefix_registry  # noqa: E402
import qwen_prefix_scheduler_patch as patch_module  # noqa: E402
from qwen_prefix_registry import BLOCK, CHUNK, PrefixRegistry  # noqa: E402

FIXTURE = HERE / 'fixtures' / 'vllm_tt_plugin_bf77cd63' / 'scheduler.py'
SALT = 'tenant-a'


def fixture_bytes():
    # .gitattributes checks text out as LF; normalise anyway so a CRLF checkout cannot fake a drift.
    return FIXTURE.read_bytes().replace(b'\r\n', b'\n')


class Quiet(object):
    def __enter__(self):
        self.saved = sys.stderr
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, *exc):
        sys.stderr.close()
        sys.stderr = self.saved


class ModelAssertion(AssertionError):
    """What the G1 model graft raises: the engine would die rather than rewrite shared blocks."""


INITIAL_STATE = b'gdn-initial-state'


def chunk_state(state, chunk):
    return hashlib.sha256(state + array('q', chunk).tobytes()).digest()


def cold_state(tokens, upto):
    """The GDN state a cold prefill of tokens holds after its first upto tokens."""
    state = INITIAL_STATE
    for index in range(upto // CHUNK):
        state = chunk_state(state, tokens[index * CHUNK:(index + 1) * CHUNK])
    return state


class FakeGdnModel(object):
    """The model graft's side of one prefill row (qwen_prefix_registry's model contract).

    mode 'contract': each planned capture when the chunk loop has run exactly pos tokens.
    mode 'drain-honest': every capture after the loop drains, loop_pos the tokens actually run (a
      model built to the old "after the loop drains" wording; capture() refuses the mid-loop ones).
    mode 'drain-lying': the same, but claiming loop_pos=pos - the poison capture() cannot see; the
      restore check below catches it at the next hit."""

    def __init__(self, registry, mode='contract'):
        self.registry = registry
        self.mode = mode
        self.chunks = {}
        self.restored = {}

    def prefill(self, req_id, tokens, start_pos):
        registry = self.registry
        tokens = list(tokens)
        grant = registry.grant_for(req_id)
        if start_pos > 0:
            if grant is None:
                raise ModelAssertion('row %s: start_pos=%d without a committed grant' % (req_id, start_pos))
            if grant.req_id != req_id or grant.q != start_pos:
                raise ModelAssertion('row %s: grant %s does not match start_pos=%d' % (req_id, grant.describe(), start_pos))
            if not grant.checkpoint.matches(tokens[0:start_pos]):
                raise ModelAssertion('row %s: checkpoint tokens differ from the prompt below %d' % (req_id, start_pos))
            state = grant.checkpoint.rec
            if state != cold_state(tokens, start_pos):
                raise ModelAssertion('row %s: the state restored at %d is not the cold state there' % (req_id, start_pos))
            registry.note_restore(1.0)
            self.restored[req_id] = start_pos
        else:
            state = INITIAL_STATE
        drain = len(tokens) // CHUNK * CHUNK
        plan = grant.capture_positions() if grant is not None else []
        if grant is not None and grant.drain != drain:
            raise ModelAssertion('row %s: the scheduler planned for a loop draining at %d, not %d'
                                 % (req_id, grant.drain, drain))
        run = 0
        for index in range(start_pos // CHUNK, drain // CHUNK):
            state = chunk_state(state, tokens[index * CHUNK:(index + 1) * CHUNK])
            run += 1
            done = (index + 1) * CHUNK
            if self.mode == 'contract' and done in plan:
                registry.capture(req_id, done, rec=state, carry=b'carry', nbytes=1, ms=0.5, loop_pos=done)
        if self.mode != 'contract':
            for pos in plan:
                registry.capture(req_id, pos, rec=state, carry=b'carry', nbytes=1,
                                 loop_pos=pos if self.mode == 'drain-lying' else drain)
        self.chunks[req_id] = run
        return state


# ------------------------------------------------------------------------------------------------
# 1. The stage
# ------------------------------------------------------------------------------------------------
class StageTests(unittest.TestCase):
    def package(self):
        directory = tempfile.mkdtemp(prefix='qwen-prefix-stage-')
        self.addCleanup(shutil.rmtree, directory, True)
        (Path(directory) / 'scheduler.py').write_bytes(fixture_bytes())
        return Path(directory)

    def test_the_fixture_is_the_pinned_blob(self):
        self.assertEqual(hashlib.sha256(fixture_bytes()).hexdigest(), patch_module.SCHEDULER_SHA256)

    def test_the_hook_is_appended_to_init_only(self):
        source = fixture_bytes().decode('utf-8')
        patched = patch_module.patch_scheduler(source)
        self.assertEqual(patched.replace(patch_module.INIT_HOOK, '', 1), source)
        start, end = patch_module.method_span(patched, 'TTScheduler', '__init__')
        body = ''.join(patched.splitlines(keepends=True)[start:end])
        self.assertEqual(body, patch_module.INIT_ANCHOR + patch_module.INIT_HOOK)
        for name in ('schedule', '_schedule_prefill_only', '_schedule_decode_only', '_has_pending_prefill'):
            self.assertEqual(self.method(source, name), self.method(patched, name))

    @staticmethod
    def method(source, name):
        start, end = patch_module.method_span(source, 'TTScheduler', name)
        return ''.join(source.splitlines(keepends=True)[start:end])

    def test_the_hook_is_guarded_and_imports_lazily(self):
        hook = patch_module.INIT_HOOK
        guard = hook.index('if _qwen_prefix_os.environ.get("QWEN_PREFIX_REUSE") == "1":')
        self.assertGreater(hook.index('from .qwen_prefix_scheduler_patch import maybe_install'), guard)
        self.assertEqual(hook.count('import '), 2)

    def test_refusals(self):
        source = fixture_bytes().decode('utf-8')
        once = patch_module.patch_scheduler(source)
        with self.assertRaisesRegex(ValueError, 'already carries'):
            patch_module.patch_scheduler(once)
        lever_n = source.replace('        return result\n', '        # Lever N M2: one prefill in flight per lane.\n'
                                 '        return result\n', 1)
        with self.assertRaisesRegex(ValueError, 'Lever N'):
            patch_module.patch_scheduler(lever_n)
        changed = source.replace('self._forced_mode = TTSchedulingMode.DEFAULT',
                                 'self._forced_mode = TTSchedulingMode.DECODE_ONLY')
        with self.assertRaisesRegex(ValueError, 'not the pinned body'):
            patch_module.patch_scheduler(changed)

    def test_stage_writes_the_patch_and_the_runtime(self):
        package = self.package()
        report = patch_module.stage(package)
        self.assertEqual(report['scheduler.py'][0], patch_module.SCHEDULER_SHA256)
        written = (package / 'scheduler.py').read_bytes()
        self.assertEqual(hashlib.sha256(written).hexdigest(), report['scheduler.py'][1])
        self.assertNotIn(b'\r\n', written)
        for name in patch_module.RUNTIME_FILES:
            self.assertEqual((package / name).read_bytes(), (HERE / name).read_bytes())
        with self.assertRaisesRegex(ValueError, 'is not the pinned'):
            patch_module.stage(package)

    def test_check_writes_nothing_and_refusals_write_nothing(self):
        package = self.package()
        patch_module.stage(package, check_only=True)
        self.assertEqual(sorted(p.name for p in package.iterdir()), ['scheduler.py'])
        self.assertEqual((package / 'scheduler.py').read_bytes(), fixture_bytes())
        (package / 'qwen_prefix_registry.py').write_text('# someone else\n')
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            patch_module.stage(package)
        self.assertEqual((package / 'scheduler.py').read_bytes(), fixture_bytes())
        self.assertEqual((package / 'qwen_prefix_registry.py').read_text(), '# someone else\n')
        crlf = self.package()
        (crlf / 'scheduler.py').write_bytes(fixture_bytes().replace(b'\n', b'\r\n'))
        with self.assertRaisesRegex(ValueError, 'is not the pinned'):
            patch_module.stage(crlf)

    def test_a_copy_the_overlay_installed_from_the_same_source_is_accepted(self):
        """The C2 image lists both plugin copies as overlay destinations (so its provenance check
        hashes them); the stage then runs over them and must neither refuse nor rewrite them."""
        package = self.package()
        for name in patch_module.RUNTIME_FILES:
            (package / name).write_bytes((HERE / name).read_bytes())
        report = patch_module.stage(package)
        for name in patch_module.RUNTIME_FILES:
            self.assertEqual(report[name][0], report[name][1], 'reported as already present')
            self.assertEqual((package / name).read_bytes(), (HERE / name).read_bytes())
        self.assertIn(b'qwen_prefix_scheduler_patch', (package / 'scheduler.py').read_bytes())
        stale = self.package()
        (stale / 'qwen_prefix_scheduler_patch.py').write_bytes((HERE / 'qwen_prefix_scheduler_patch.py').read_bytes()
                                                               + b'# an older copy\n')
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite .*qwen_prefix_scheduler_patch.py'):
            patch_module.stage(stale)
        self.assertEqual(sorted(p.name for p in stale.iterdir()), ['qwen_prefix_scheduler_patch.py', 'scheduler.py'])

    def test_cli(self):
        package = self.package()
        with mock.patch('sys.stdout'):
            self.assertEqual(patch_module.main(['--check', str(package)]), 0)
            self.assertEqual(patch_module.main([str(package)]), 0)
        self.assertIn(b'qwen_prefix_scheduler_patch', (package / 'scheduler.py').read_bytes())


# ------------------------------------------------------------------------------------------------
# 2. The hook executes
# ------------------------------------------------------------------------------------------------
def stand_in_vllm_modules():
    """Just enough of vllm for the plugin's scheduler.py to import and construct."""

    class AsyncScheduler(object):
        def __init__(self, *args, **kwargs):
            self.constructed_with = (args, kwargs)

    names = {
        'vllm': {}, 'vllm.v1': {}, 'vllm.v1.core': {}, 'vllm.v1.core.sched': {},
        'vllm.v1.core.sched.async_scheduler': {'AsyncScheduler': AsyncScheduler},
        'vllm.v1.core.sched.output': {'SchedulerOutput': object},
        'vllm.v1.core.sched.request_queue': {'RequestQueue': object, 'create_request_queue': lambda policy: []},
        'vllm.v1.request': {'Request': object},
        'vllm_tt_plugin': {}, 'vllm_tt_plugin.logger': {'init_tt_logger': lambda name: None},
    }
    modules = {}
    for name, attributes in names.items():
        module = types.ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attributes)
        modules[name] = module
    return modules


class HookTests(unittest.TestCase):
    def staged_package(self):
        root = tempfile.mkdtemp(prefix='qwen-prefix-hook-')
        self.addCleanup(shutil.rmtree, root, True)
        name = 'qwen_prefix_hook_pkg_%d' % id(self)
        package = Path(root) / name
        package.mkdir()
        (package / '__init__.py').write_text('')
        (package / 'scheduler.py').write_bytes(fixture_bytes())
        patch_module.stage(package)
        return root, name

    def construct(self, environ):
        root, name = self.staged_package()
        calls = []
        modules = stand_in_vllm_modules()
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, environ, clear=False), \
                mock.patch.object(sys, 'path', [root] + sys.path):
            if 'QWEN_PREFIX_REUSE' not in environ:
                os.environ.pop('QWEN_PREFIX_REUSE', None)
            scheduler_module = importlib.import_module(name + '.scheduler')
            runtime = importlib.import_module(name + '.qwen_prefix_scheduler_patch')
            self.assertEqual(runtime.prefix_registry.__name__, name + '.qwen_prefix_registry',
                             'the package copy imports its sibling registry')
            with mock.patch.object(runtime, 'install', side_effect=lambda s: calls.append(s)):
                scheduler = scheduler_module.TTScheduler('config', kv_cache_config='kv')
        for key in [key for key in sys.modules if key.startswith(name)]:
            del sys.modules[key]
        return scheduler, calls

    def test_the_hook_installs_when_reuse_is_on(self):
        scheduler, calls = self.construct({'QWEN_PREFIX_REUSE': '1'})
        self.assertEqual(calls, [scheduler])
        self.assertEqual(scheduler.constructed_with, (('config',), {'kv_cache_config': 'kv'}))
        self.assertEqual(scheduler._forced_mode.name, 'DEFAULT')

    def test_the_hook_does_nothing_otherwise(self):
        for environ in ({}, {'QWEN_PREFIX_REUSE': '0'}, {'QWEN_PREFIX_REUSE': 'true'}):
            scheduler, calls = self.construct(environ)
            self.assertEqual(calls, [])
            self.assertEqual(scheduler._forced_mode.name, 'DEFAULT')


# ------------------------------------------------------------------------------------------------
# 3. The wrappers on a model of vLLM's prefix cache
# ------------------------------------------------------------------------------------------------
class UnitaryKVCacheCoordinator(object):
    pass


class HybridKVCacheCoordinator(object):
    pass


def fake_vllm_modules():
    modules = {}
    for name in ('vllm', 'vllm.v1', 'vllm.v1.core'):
        module = types.ModuleType(name)
        module.__path__ = []
        modules[name] = module
    modules['vllm'].__version__ = '0.25.1-fake'
    coordinator = types.ModuleType('vllm.v1.core.kv_cache_coordinator')
    coordinator.UnitaryKVCacheCoordinator = UnitaryKVCacheCoordinator
    utils = types.ModuleType('vllm.v1.core.kv_cache_utils')
    utils.get_block_hash = lambda key: key[0]
    modules['vllm.v1.core.kv_cache_coordinator'] = coordinator
    modules['vllm.v1.core.kv_cache_utils'] = utils
    return modules


class Block(object):
    def __init__(self, block_id):
        self.block_id = block_id
        self.block_hash = None


class Blocks(object):
    def __init__(self, blocks):
        self.blocks = blocks


class HashMap(object):
    def __init__(self):
        self.map = {}

    def get_one_block(self, key):
        blocks = self.map.get(key)
        return blocks[0] if blocks else None

    def insert(self, key, block):
        self.map.setdefault(key, []).append(block)

    def pop(self, key, block):
        blocks = self.map.get(key, [])
        if block in blocks:
            blocks.remove(block)
        if not blocks:
            self.map.pop(key, None)

    def __len__(self):
        return len(self.map)


class Pool(object):
    """vLLM's BlockPool, reduced: LRU free list, a hash map, eviction on reuse."""

    def __init__(self, num_blocks):
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free = list(self.blocks)
        self.cached_block_hash_to_block = HashMap()
        self.cached_block_hashes_by_block = {}
        self.hash_block_size = BLOCK
        self.evicted = []

    def take(self, count):
        if count > len(self.free):
            return None
        taken, self.free = self.free[:count], self.free[count:]
        for block in taken:
            self._maybe_evict_cached_block(block)
        return taken

    def _maybe_evict_cached_block(self, block):
        if block.block_hash is None:
            return False
        self.cached_block_hash_to_block.pop(block.block_hash, block)
        block.block_hash = None
        self.evicted.append(block.block_id)
        return True

    def give_back(self, blocks):
        self.free.extend(reversed(blocks))


class Single(object):
    def __init__(self):
        self.num_cached_block = {}
        self.req_to_blocks = {}


class Coordinator(UnitaryKVCacheCoordinator):
    def __init__(self, pool):
        self.pool = pool
        self.single_type_managers = [Single()]
        self.calls = []

    def cache_blocks(self, request, num_computed_tokens):
        self.calls.append((request.request_id, num_computed_tokens))
        single = self.single_type_managers[0]
        owned = single.req_to_blocks[request.request_id]
        done = single.num_cached_block.get(request.request_id, 0)
        full = num_computed_tokens // BLOCK
        for index in range(done, full):
            key = (request.block_hashes[index], 0)
            owned[index].block_hash = key
            self.pool.cached_block_hash_to_block.insert(key, owned[index])
        single.num_cached_block[request.request_id] = max(done, full)


class Manager(object):
    def __init__(self, num_blocks):
        self.block_pool = Pool(num_blocks)
        self.coordinator = Coordinator(self.block_pool)
        self.enable_caching = True
        self.empty_kv_cache_blocks = Blocks(([],))

    def create_kv_cache_blocks(self, blocks):
        return Blocks(blocks)

    def get_computed_blocks(self, request):
        hits = []
        limit = (request.num_tokens - 1) // BLOCK
        for index in range(limit):
            block = self.block_pool.cached_block_hash_to_block.get_one_block((request.block_hashes[index], 0))
            if block is None:
                break
            hits.append(block)
        return Blocks((hits,)), len(hits) * BLOCK


class Request(object):
    def __init__(self, request_id, tokens, salt=SALT, prompt=None, resumable=False):
        self.request_id = request_id
        self.all_token_ids = list(tokens)
        self.num_prompt_tokens = len(tokens) if prompt is None else prompt
        self.cache_salt = salt
        self.resumable = resumable
        self.block_hashes = []
        parent = (salt or '').encode()
        for start in range(0, len(tokens) - len(tokens) % BLOCK, BLOCK):
            parent = hashlib.sha256(parent + repr(tokens[start:start + BLOCK]).encode()).digest()
            self.block_hashes.append(parent)

    @property
    def num_tokens(self):
        return len(self.all_token_ids)


def output(admitted):
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=rid, num_computed_tokens=start) for rid, start in admitted],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], resumed_req_ids=set(), num_computed_tokens=[]))


class FakeScheduler(object):
    """Admits waiting requests FCFS the way vLLM does it: get_computed_blocks, then allocate the rest,
    then cache the full blocks at allocation. `budget` limits admissions per step."""

    def __init__(self, num_blocks=400, max_model_len=65536):
        self.kv_cache_manager = Manager(num_blocks)
        self.scheduler_config = SimpleNamespace(async_scheduling=False, enable_chunked_prefill=False,
                                                max_num_batched_tokens=max_model_len, long_prefill_token_threshold=0)
        self.cache_config = SimpleNamespace(enable_prefix_caching=True, prefix_caching_hash_algo='sha256')
        self.max_model_len = max_model_len
        self.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=BLOCK, dtype='bf16'))],
            num_blocks=num_blocks)
        self.block_size = BLOCK
        self.has_mamba_layers = False
        self.connector = None
        self.num_lookahead_tokens = 0
        self.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=1,
                                                                           distributed_executor_backend='uni'))
        self.waiting = []
        self.requests = {}
        self.freed = []
        self.resets = 0
        self.reset_succeeds = True
        self.budget = None

    def add(self, request):
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def schedule(self):
        admitted = []
        manager = self.kv_cache_manager
        while self.waiting and (self.budget is None or len(admitted) < self.budget):
            request = self.waiting[0]
            blocks, start = manager.get_computed_blocks(request)
            hit = list(blocks.blocks[0]) if start else []
            needed = -(-request.num_tokens // BLOCK) - len(hit)
            pool = manager.block_pool
            free_hits = [block for block in hit if block in pool.free]
            # vLLM's allocate_slots: evictable hit blocks do not count as free, and the hits are
            # touched (taken off the free queue) before the new blocks are allocated.
            if needed > len(pool.free) - len(free_hits):
                break
            for block in free_hits:
                pool.free.remove(block)
            new = pool.take(needed)
            self.waiting.pop(0)
            single = manager.coordinator.single_type_managers[0]
            single.req_to_blocks[request.request_id] = hit + new
            single.num_cached_block[request.request_id] = len(hit)
            manager.coordinator.cache_blocks(request, request.num_tokens)
            admitted.append((request.request_id, start))
        return output(admitted)

    def finish(self, request):
        single = self.kv_cache_manager.coordinator.single_type_managers[0]
        self.kv_cache_manager.block_pool.give_back(single.req_to_blocks.pop(request.request_id))
        single.num_cached_block.pop(request.request_id, None)
        self._free_request(request)

    def _free_request(self, request, delay_free_blocks=False):
        self.freed.append(request.request_id)

    def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
        """vLLM's: False, with every cached block kept, while running requests hold blocks."""
        self.resets += 1
        if not self.reset_succeeds:
            return False
        self.kv_cache_manager.block_pool.cached_block_hash_to_block.map.clear()
        return True


def tokens(count, seed):
    return [(seed * 7919 + index * 104729) % 200000 for index in range(count)]


class GraftTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, fake_vllm_modules())
        patcher.start()
        self.addCleanup(patcher.stop)
        # A graft installed without an explicit StatsExport reads its path from here: no file.
        environment = mock.patch.dict(os.environ, {'QWEN_PREFIX_STATS_PATH': ''})
        environment.start()
        self.addCleanup(environment.stop)
        self.logs = []
        self.model = None

    def logger(self, message, *values):
        self.logs.append(message % values if values else message)

    def make(self, registry=None, mid_loop=False, mode='contract', **kwargs):
        scheduler = FakeScheduler(**kwargs)
        registry = registry or PrefixRegistry(budget_bytes=1 << 40)
        if mid_loop:
            with Quiet():
                registry.enable_mid_loop_capture()
        graft = patch_module.install(scheduler, registry=registry, kill_switch_path=None, logger=self.logger,
                                     stats=patch_module.StatsExport(path='', logger=self.logger))
        self.model = FakeGdnModel(registry, mode)
        return scheduler, graft

    def step(self, scheduler, model=True):
        """schedule, then the model's side (FakeGdnModel): the grant assertions, the restore check
        against the cold chain, the captures. Returns {req_id: (start_pos, grant)}."""
        out = scheduler.schedule()
        registry = scheduler._qwen_prefix.registry
        if getattr(self, 'model', None) is None or self.model.registry is not registry:
            self.model = FakeGdnModel(registry)
        rows = {}
        for data in out.scheduled_new_reqs:
            grant = registry.grant_for(data.req_id)
            rows[data.req_id] = (data.num_computed_tokens, grant)
            if model:
                with Quiet():
                    self.model.prefill(data.req_id, scheduler.requests[data.req_id].all_token_ids,
                                       data.num_computed_tokens)
            elif data.num_computed_tokens:
                self.assertIsNotNone(grant, 'start_pos > 0 without a committed grant')
                self.assertEqual(grant.q, data.num_computed_tokens)
        return rows

    def serve(self, scheduler, request):
        scheduler.add(request)
        rows = self.step(scheduler)
        scheduler.finish(request)
        return rows[request.request_id]

    def test_a_hit_is_trimmed_to_the_last_checkpoint_and_publishing_is_capped(self):
        scheduler, graft = self.make()
        text = tokens(5000, 1)
        start, grant = self.serve(scheduler, Request('a', text))
        self.assertEqual((start, grant.q, grant.capture_positions()), (0, 0, [4096]))
        self.assertEqual(scheduler.kv_cache_manager.coordinator.calls[-1], ('a', 4096),
                         'the cap hands vLLM min(5000, floor2048(5000))')
        self.assertEqual(len(scheduler.kv_cache_manager.block_pool.cached_block_hash_to_block), 64,
                         'only blocks below floor2048(5000) = 4096 are hashed')
        start, grant = self.serve(scheduler, Request('b', text + tokens(3000, 2)))
        self.assertEqual((start, grant.q, grant.h, grant.capture_positions()), (4096, 4096, 4096, [6144]))
        self.assertEqual([line for line in self.logs if line.startswith('grant req=b')],
                         ['grant req=b h=4096 Q=4096 drain=6144 plan=[6144]'])
        self.assertEqual((self.model.restored['b'], self.model.chunks['b']), (4096, 1),
                         'b restored at 4096 and ran one chunk')

    def test_a_kv_hit_without_a_checkpoint_gets_the_older_boundary(self):
        scheduler, graft = self.make()
        registry = graft.registry
        text = tokens(4200, 3)
        self.serve(scheduler, Request('x1', text[0:2100]))
        self.serve(scheduler, Request('x2', text))
        key4096 = Request('probe', text).block_hashes[4096 // BLOCK - 1]
        registry.drop(key4096)
        start, grant = self.serve(scheduler, Request('b', text[0:4096] + tokens(300, 4)))
        self.assertEqual((grant.h, grant.q, start), (4096, 2048, 2048))
        self.assertEqual(registry.stats['kv_hit_without_checkpoint'], 1)
        self.assertEqual(registry.stats['trim_loss_tokens'], 4096 - 2048)

    def test_a_token_mismatch_only_lowers_q(self):
        """P0a check 18 on the fakes."""
        scheduler, graft = self.make()
        registry = graft.registry
        text = tokens(4200, 5)
        self.serve(scheduler, Request('x1', text[0:2100]))
        self.serve(scheduler, Request('x2', text))
        entry = registry.get(Request('probe', text).block_hashes[63])
        entry.token_ids[100] ^= 1
        start, grant = self.serve(scheduler, Request('b', text[0:4096] + tokens(300, 6)))
        self.assertEqual((grant.h, grant.q, start), (4096, 2048, 2048))
        self.assertEqual(registry.stats['token_mismatches'], 1)
        self.assertIsNot(registry.get(entry.key), entry, 'b\'s own capture replaced the bad entry')

    def test_same_step_blocks_are_never_granted(self):
        """P0a check 7(b) on the fakes: a checkpoint at 4096 whose KV chain was broken and re-cached
        by another admission in the same step."""
        scheduler, graft = self.make()
        registry = graft.registry
        text = tokens(4200, 7)
        self.serve(scheduler, Request('x', text))
        pool = scheduler.kv_cache_manager.block_pool
        block10 = pool.cached_block_hash_to_block.get_one_block((Request('p', text).block_hashes[10], 0))
        pool._maybe_evict_cached_block(block10)
        scheduler.add(Request('a', text[0:4096] + tokens(500, 8)))
        scheduler.add(Request('b', text[0:4096] + tokens(600, 9)))
        rows = self.step(scheduler)
        self.assertEqual(rows['a'][1].h, 640)
        self.assertEqual((rows['b'][1].h, rows['b'][1].q, rows['b'][0]), (4096, 0, 0))
        self.assertEqual(registry.stats['same_step_rejects'], 1)
        self.assertTrue(registry.stats['orphans'] >= 1)

    def test_unsalted_requests_get_nothing_and_publish_nothing(self):
        """P0a check 13 on the fakes."""
        scheduler, graft = self.make()
        registry = graft.registry
        pool = scheduler.kv_cache_manager.block_pool
        text = tokens(4200, 10)
        start, grant = self.serve(scheduler, Request('u0', text, salt=None))
        self.assertEqual((start, grant, len(pool.cached_block_hash_to_block)), (0, None, 0))
        self.serve(scheduler, Request('s1', text))
        start, grant = self.serve(scheduler, Request('u1', text + tokens(10, 11), salt=None))
        self.assertEqual((start, grant), (0, None))
        self.assertEqual(registry.stats['unsalted_denied'], 0, 'a different salt never even hashes alike')
        start, grant = self.serve(scheduler, Request('s2', text + tokens(10, 12)))
        self.assertEqual((start, grant.q), (4096, 4096))

    def test_streaming_sessions_get_nothing_and_publish_nothing(self):
        """A resumable request (vLLM's streaming input) is treated like an unsalted one: its prompt
        later absorbs decode-written tokens, which the cap would otherwise publish."""
        scheduler, graft = self.make()
        registry = graft.registry
        pool = scheduler.kv_cache_manager.block_pool
        text = tokens(4200, 30)
        start, grant = self.serve(scheduler, Request('session', text, resumable=True))
        self.assertEqual((start, grant, len(pool.cached_block_hash_to_block)), (0, None, 0))
        self.serve(scheduler, Request('s1', text))
        start, grant = self.serve(scheduler, Request('session2', text + tokens(10, 31), resumable=True))
        self.assertEqual((start, grant), (0, None))
        self.assertEqual(registry.stats['session_denied'], 1)
        self.assertEqual(len(pool.cached_block_hash_to_block), 4096 // BLOCK, 's1 alone published')

    def test_the_plan_takes_mid_loop_boundaries_only_from_a_model_that_declared_them(self):
        """The review's case: P=20000, h=5952, Q=0. The gap boundary 4096 is below the loop's drain
        at 18432, so it is a mid-loop capture; a resumed request (P=5000, 9000 tokens) drains at 8192
        with its prompt boundary at 4096."""
        scheduler, graft = self.make()
        request = Request('p', tokens(20000, 40))
        plan, unplanned, drain = graft.plan(request, 5952, 0)
        self.assertEqual(([pos for pos, _ in plan], unplanned, drain), ([18432], [4096], 18432))
        resumed = Request('r', tokens(9000, 41), prompt=5000)
        self.assertEqual(graft.plan(resumed, 4096, 4096), ([], [], 8192))
        plan, unplanned, drain = graft.plan(resumed, 0, 0)
        self.assertEqual((plan, unplanned, drain), ([], [4096], 8192), 'nothing above the prompt boundary, '
                                                                        'nothing at the resumed drain')
        with Quiet():
            graft.registry.enable_mid_loop_capture()
        plan, unplanned, _ = graft.plan(request, 5952, 0)
        self.assertEqual(([pos for pos, _ in plan], unplanned), ([4096, 18432], []))
        self.assertEqual(plan[0][1], request.block_hashes[4096 // BLOCK - 1])
        self.assertEqual([pos for pos, _ in graft.plan(resumed, 0, 0)[0]], [4096])

    def siblings(self, mid_loop, mode='contract'):
        """Three sibling sub-agents sharing a 6000-token system prompt and tools, one after another."""
        scheduler, graft = self.make(mid_loop=mid_loop, mode=mode)
        shared = tokens(6000, 50)
        rows = {}
        for name, seed, extra in (('a', 51, 3000), ('b', 52, 3500), ('c', 53, 2800)):
            rows[name] = self.serve(scheduler, Request(name, shared + tokens(extra, seed)))
        return graft.registry, rows

    def test_a_sibling_restores_the_gap_boundary_a_previous_sibling_captured_mid_loop(self):
        registry, rows = self.siblings(mid_loop=True)
        self.assertEqual(rows['a'][1].capture_positions(), [8192])
        start, grant = rows['b']
        self.assertEqual((start, grant.h, grant.q, grant.capture_positions(), grant.mid_loop_positions()),
                         (0, 5952, 0, [4096, 8192], [4096]))
        start, grant = rows['c']
        self.assertEqual((start, grant.q), (4096, 4096))
        self.assertEqual(self.model.restored['c'], 4096, 'the restored state was the cold state at 4096')
        self.assertEqual((registry.stats['capture_wrong_position'], registry.stats['capture_failures']), (0, 0))

    def test_without_the_declaration_the_gap_boundary_is_not_planned(self):
        registry, rows = self.siblings(mid_loop=False)
        self.assertEqual((rows['b'][1].capture_positions(), rows['b'][1].unplanned), ([8192], (4096,)))
        self.assertEqual((rows['c'][0], rows['c'][1].q), (0, 0))
        self.assertEqual(registry.stats['mid_loop_unplanned'], 2)

    def test_a_drain_capture_filed_under_a_mid_loop_boundary_is_refused(self):
        """A model built to the old wording ("after the loop drains") reports where its loop really
        was, and capture() refuses: no checkpoint, no hit, nothing inexact."""
        registry, rows = self.siblings(mid_loop=True, mode='drain-honest')
        self.assertEqual(registry.stats['capture_wrong_position'], 2, 'b\'s 4096 and c\'s 4096')
        self.assertEqual((rows['c'][0], rows['c'][1].q), (0, 0))

    def test_the_fake_model_catches_a_poisoned_checkpoint(self):
        """Positive control for the restore check: a model that lies about loop_pos files the state
        at 8192 under the 4096 key; the next sibling restores it and the check fires."""
        with self.assertRaisesRegex(ModelAssertion, 'not the cold state'):
            self.siblings(mid_loop=True, mode='drain-lying')

    def test_a_resumed_request_captures_its_prompt_boundary_mid_loop_only(self):
        """A preempted request re-prefills its output tokens too: P=5000 with 9000 tokens drains at
        8192. Without mid-loop captures nothing is planned; with them, 4096 is captured when the
        loop reaches it and a later turn on the same prompt restores the right state."""
        for mid_loop in (False, True):
            with self.subTest(mid_loop=mid_loop):
                scheduler, graft = self.make(mid_loop=mid_loop)
                registry = graft.registry
                prompt = tokens(5000, 60)
                resumed = Request('r', prompt + tokens(4000, 61), prompt=5000)
                start, grant = self.serve(scheduler, resumed)
                self.assertEqual((start, grant.drain), (0, 8192))
                self.assertEqual(grant.capture_positions(), [4096] if mid_loop else [])
                self.assertEqual(scheduler.kv_cache_manager.coordinator.calls[-1], ('r', 4096),
                                 'the cap stops at the prompt boundary')
                start, grant = self.serve(scheduler, Request('next', prompt + tokens(700, 62)))
                self.assertEqual((start, grant.q), (4096, 4096) if mid_loop else (0, 0))
                if mid_loop:
                    self.assertEqual(self.model.restored['next'], 4096)
                self.assertEqual(registry.stats['capture_wrong_position'], 0)

    def test_the_token_check_runs_once_per_request_and_checkpoint(self):
        """A request blocked at the queue head is re-attempted every step; its token prefix never
        changes, so the 62k-token array is built once per checkpoint, not once per step."""
        scheduler, graft = self.make()
        registry = graft.registry
        text = tokens(4200, 70)
        self.serve(scheduler, Request('x', text))
        pool = scheduler.kv_cache_manager.block_pool
        hoard = [block for block in pool.free if block.block_hash is None][5:]
        for block in hoard:
            pool.free.remove(block)
        checks = registry.stats['token_checks']
        scheduler.add(Request('b', text[0:4096] + tokens(900, 71)))
        for _ in range(4):
            self.step(scheduler)
        self.assertEqual(registry.stats['token_checks'] - checks, 1)
        key = Request('p', text).block_hashes[4096 // BLOCK - 1]
        entry = registry.get(key)
        with Quiet():
            registry.put(key, 4096, list(entry.token_ids), rec=entry.rec, nbytes=1)
        self.step(scheduler)
        self.assertEqual(registry.stats['token_checks'] - checks, 2, 'a replaced checkpoint is checked again')
        pool.give_back(hoard)
        rows = self.step(scheduler)
        self.assertEqual(rows['b'][1].q, 4096)
        scheduler.finish(scheduler.requests['b'])
        self.assertNotIn('b', registry.token_checks, 'forgotten with the request')

    def test_an_orphan_is_counted_once(self):
        """x's checkpoint at 4096 loses block 40 of its chain. Three admissions see it (resumed
        requests whose prompt ends at 2100 re-publish only up to 2048, so the chain stays broken):
        one orphan, not three. A request that re-caches the chain and re-captures 4096 retires it."""
        scheduler, graft = self.make()
        registry = graft.registry
        text = tokens(4200, 80)
        self.serve(scheduler, Request('x', text))
        pool = scheduler.kv_cache_manager.block_pool
        pool._maybe_evict_cached_block(
            pool.cached_block_hash_to_block.get_one_block((Request('p', text).block_hashes[40], 0)))
        for index in range(3):
            start, grant = self.serve(scheduler, Request('o%d' % index, text, prompt=2100))
            self.assertEqual((start, grant.h), (0, 2560))
        self.assertEqual((registry.stats['orphans'], registry.snapshot()['orphans_now']), (1, 1))
        self.serve(scheduler, Request('heal', text[0:4096] + tokens(300, 84)))
        self.assertEqual((registry.stats['orphans'], registry.snapshot()['orphans_now']), (1, 0))

    def test_an_attempt_that_is_not_admitted_leaves_no_grant(self):
        """P0a check 10 on the fakes: the budget break."""
        scheduler, graft = self.make()
        registry = graft.registry
        text = tokens(4200, 13)
        self.serve(scheduler, Request('x', text))
        scheduler.budget = 0
        scheduler.add(Request('b', text[0:4096] + tokens(900, 14)))
        self.step(scheduler)
        self.assertIsNone(registry.grant_for('b'))
        self.assertEqual(registry.stats['dropped_attempts'], 0, 'budget 0 never attempts')
        scheduler.budget = None
        pool = scheduler.kv_cache_manager.block_pool
        hoard = [block for block in pool.free if block.block_hash is None][5:]
        for block in hoard:
            pool.free.remove(block)
        for _ in range(3):
            self.step(scheduler)
        self.assertEqual((registry.stats['staged'] - 1, registry.stats['dropped_attempts']), (3, 3))
        self.assertIsNone(registry.grant_for('b'))
        self.assertEqual(registry.pins(), 0)
        pool.give_back(hoard)
        rows = self.step(scheduler)
        self.assertEqual((rows['b'][0], rows['b'][1].q), (4096, 4096))

    def test_eviction_drops_the_checkpoint_with_its_last_boundary_block(self):
        """P0a check 12 on the fakes."""
        scheduler, graft = self.make()
        registry = graft.registry
        pool = scheduler.kv_cache_manager.block_pool
        text = tokens(4200, 15)
        scheduler.add(Request('x', text))
        scheduler.add(Request('x2', text))
        self.step(scheduler)
        key = Request('p', text).block_hashes[63]
        first = pool.cached_block_hash_to_block.get_one_block((key, 0))
        pool._maybe_evict_cached_block(first)
        self.assertIsNotNone(registry.get(key), 'another block still holds the boundary hash')
        second = pool.cached_block_hash_to_block.get_one_block((key, 0))
        pool._maybe_evict_cached_block(second)
        self.assertIsNone(registry.get(key))
        self.assertEqual(registry.stats['evicted_coupled'], 1)

    def test_free_and_reset(self):
        """P0a checks 11 and 17 on the fakes."""
        scheduler, graft = self.make()
        registry = graft.registry
        text = tokens(4200, 16)
        self.serve(scheduler, Request('x', text))
        scheduler.add(Request('c', text[0:4096] + tokens(2904, 17)))
        rows = self.step(scheduler, model=False)
        self.assertEqual(registry.get(rows['c'][1].key).pins, 1)
        scheduler.finish(SimpleNamespace(request_id='c'))
        self.assertIsNone(registry.grant_for('c'))
        self.assertEqual(registry.pins(), 0)
        self.assertEqual(scheduler.freed[-1], 'c', 'the original _free_request still runs')
        self.serve(scheduler, Request('y', tokens(4200, 90)))
        kept = len(registry.entries)
        scheduler.reset_succeeds = False
        self.assertFalse(scheduler.reset_prefix_cache())
        self.assertEqual((len(registry.entries), registry.stats['reset_kept']), (kept, 1),
                         'vLLM kept the cached blocks, so the checkpoints stay')
        scheduler.reset_succeeds = True
        self.assertTrue(scheduler.reset_prefix_cache())
        self.assertEqual((scheduler.resets, len(registry.entries)), (2, 0))

    def test_the_kill_switch_disables_everything_and_latches(self):
        """P0a check 16 on the fakes."""
        flag = Path(tempfile.mkdtemp(prefix='qwen-prefix-kill-')) / 'prefix-reuse.off'
        self.addCleanup(shutil.rmtree, str(flag.parent), True)
        now = [0.0]
        scheduler = FakeScheduler()
        graft = patch_module.install(scheduler, registry=PrefixRegistry(budget_bytes=1 << 40),
                                     kill_switch_path=str(flag), clock=lambda: now[0], logger=self.logger)
        registry = graft.registry
        pool = scheduler.kv_cache_manager.block_pool
        text = tokens(4200, 18)
        self.serve(scheduler, Request('x', text))
        self.assertEqual(self.serve(scheduler, Request('a', text[0:4096] + tokens(9, 1)))[0], 4096)
        flag.write_text('')
        now[0] += 0.5
        self.assertEqual(self.serve(scheduler, Request('b', text[0:4096] + tokens(9, 2)))[0], 4096)
        now[0] += 1.0
        before = len(pool.cached_block_hash_to_block)
        self.assertEqual(self.serve(scheduler, Request('c', text[0:4096] + tokens(9, 3)))[0], 0)
        self.assertTrue(graft.killed)
        self.assertFalse(registry.entries)
        self.assertEqual(len(pool.cached_block_hash_to_block), before, 'publishing is off')
        self.serve(scheduler, Request('n', tokens(6000, 19)))
        self.assertEqual(len(pool.cached_block_hash_to_block), before)
        flag.unlink()
        now[0] += 5.0
        self.assertEqual(self.serve(scheduler, Request('d', text[0:4096] + tokens(9, 4)))[0], 0, 'latched')
        self.assertTrue(any('kill switch' in line for line in self.logs))

    def test_install_refuses_configurations_where_reuse_is_not_exact(self):
        """P0a check 14 on the fakes (and the checks the productionised graft adds)."""
        cases = [
            ('async scheduling is on', lambda s: setattr(s.scheduler_config, 'async_scheduling', True)),
            ('chunked prefill is on', lambda s: setattr(s.scheduler_config, 'enable_chunked_prefill', True)),
            ('max_num_batched_tokens', lambda s: setattr(s.scheduler_config, 'max_num_batched_tokens', 2048)),
            ('long_prefill_token_threshold is 2048', lambda s: setattr(s.scheduler_config,
                                                                       'long_prefill_token_threshold', 2048)),
            ("prefix_caching_hash_algo is 'xxhash'", lambda s: setattr(s.cache_config, 'prefix_caching_hash_algo',
                                                                       'xxhash')),
            ("executor backend is 'mp'", lambda s: setattr(s.vllm_config.parallel_config,
                                                           'distributed_executor_backend', 'mp')),
            ('prefix caching is off', lambda s: setattr(s.cache_config, 'enable_prefix_caching', False)),
            ('does not cache blocks', lambda s: setattr(s.kv_cache_manager, 'enable_caching', False)),
            ('not UnitaryKVCacheCoordinator', self.make_hybrid),
            # An uninitialised hybrid coordinator (P0a check 14's object.__new__) still names the root cause.
            ('not UnitaryKVCacheCoordinator', lambda s: setattr(s.kv_cache_manager, 'coordinator',
                                                                HybridKVCacheCoordinator())),
            ('KV block size 128', lambda s: setattr(s.kv_cache_config.kv_cache_groups[0].kv_cache_spec,
                                                    'block_size', 128)),
            ('2 KV cache groups', lambda s: s.kv_cache_config.kv_cache_groups.append(
                s.kv_cache_config.kv_cache_groups[0])),
            ('scheduler block size 128', lambda s: setattr(s, 'block_size', 128)),
            ('hash block size 16', lambda s: setattr(s.kv_cache_manager.block_pool, 'hash_block_size', 16)),
            ('Mamba layers', lambda s: setattr(s, 'has_mamba_layers', True)),
            ('KV connector', lambda s: setattr(s, 'connector', object())),
            ('speculative lookahead', lambda s: setattr(s, 'num_lookahead_tokens', 16)),
            ('pipeline parallel size 2', lambda s: setattr(s.vllm_config.parallel_config,
                                                           'pipeline_parallel_size', 2)),
            ('internals the wrappers bind are missing', lambda s: delattr(s.kv_cache_manager.block_pool,
                                                                          'cached_block_hashes_by_block')),
            ('single_type_managers[0].req_to_blocks', lambda s: delattr(
                s.kv_cache_manager.coordinator.single_type_managers[0], 'req_to_blocks')),
        ]
        for expected, mutate in cases:
            scheduler = FakeScheduler()
            mutate(scheduler)
            with self.assertRaises(patch_module.PrefixInstallError) as caught:
                patch_module.install(scheduler, registry=PrefixRegistry(budget_bytes=1), kill_switch_path=None,
                                     logger=self.logger)
            self.assertIn(expected, str(caught.exception))
            self.assertNotIn('schedule', vars(scheduler), 'refused before any wrapper went on')
            self.assertIsNone(scheduler.__dict__.get('_qwen_prefix'))
        scheduler, graft = self.make()
        self.assertIs(patch_module.install(scheduler), graft, 'a second install is the first')
        self.assertTrue(self.logs[0].startswith('install scheduler='))
        for field in ('block_size=64', 'hash=sha256', 'executor=uni'):
            self.assertIn(field, self.logs[0])
        for algorithm in ('sha256', 'sha256_cbor'):
            scheduler = FakeScheduler()
            scheduler.cache_config.prefix_caching_hash_algo = algorithm
            self.assertEqual(patch_module.install_problems(scheduler), [])

    @staticmethod
    def make_hybrid(scheduler):
        scheduler.kv_cache_manager.coordinator = HybridKVCacheCoordinator()
        scheduler.kv_cache_manager.coordinator.cache_blocks = None
        scheduler.kv_cache_manager.coordinator.single_type_managers = []

    def test_the_commit_exports_the_counters(self):
        directory = tempfile.mkdtemp(prefix='qwen-prefix-stats-')
        self.addCleanup(shutil.rmtree, directory, True)
        path = os.path.join(directory, 'stats.json')
        now = [0.0]
        scheduler = FakeScheduler()
        registry = PrefixRegistry(budget_bytes=1 << 40)
        graft = patch_module.install(scheduler, registry=registry, kill_switch_path=None, clock=lambda: now[0],
                                     logger=self.logger,
                                     stats=patch_module.StatsExport(path=path, interval_s=30, clock=lambda: now[0],
                                                                    logger=self.logger))
        self.model = FakeGdnModel(registry)
        with open(path, encoding='utf-8') as handle:
            self.assertEqual(json.load(handle)['captures'], 0, 'written at install')
        text = tokens(4200, 95)
        self.serve(scheduler, Request('x', text))
        now[0] += 31.0
        self.serve(scheduler, Request('b', text[0:4096] + tokens(300, 96)))
        with open(path, encoding='utf-8') as handle:
            exported = json.load(handle)
        self.assertEqual((exported['grants'], exported['entries'], exported['pins']), (1, 1, 1))
        for name in ('trim_loss_tokens', 'orphans', 'capture_failures', 'restore_ms', 'capture_ms', 'bytes',
                     'orphans_now', 'mid_loop_capture', 'pid', 'time'):
            self.assertIn(name, exported)
        lines = [line for line in self.logs if line.startswith('stats ')]
        self.assertEqual(len(lines), 2, 'install, then once in 30 s')
        self.assertEqual(json.loads(lines[-1][len('stats '):])['grants'], 1)
        graft.export()
        self.assertEqual(len([line for line in self.logs if line.startswith('stats ')]), 2, 'rate-limited')

    def test_one_live_scheduler_per_shared_registry(self):
        registry = PrefixRegistry(budget_bytes=1 << 30)
        first = FakeScheduler()
        patch_module.install(first, registry=registry, kill_switch_path=None, logger=self.logger)
        with self.assertRaisesRegex(patch_module.PrefixInstallError, 'another live scheduler'):
            patch_module.install(FakeScheduler(), registry=registry, kill_switch_path=None, logger=self.logger)

    def test_maybe_install_follows_the_environment(self):
        saved = sys.modules.pop(qwen_prefix_registry.REGISTRY_KEY, None)
        try:
            self.assertIsNone(patch_module.maybe_install(FakeScheduler(), environ={}))
            with Quiet():
                scheduler = FakeScheduler()
                graft = patch_module.maybe_install(scheduler, environ={'QWEN_PREFIX_REUSE': '1'})
            self.assertIs(graft.registry, qwen_prefix_registry.shared_registry())
            self.assertIs(scheduler._qwen_prefix, graft)
        finally:
            sys.modules.pop(qwen_prefix_registry.REGISTRY_KEY, None)
            if saved is not None:
                sys.modules[qwen_prefix_registry.REGISTRY_KEY] = saved


if __name__ == '__main__':
    unittest.main()
