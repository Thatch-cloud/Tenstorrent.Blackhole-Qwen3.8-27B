"""The G1 scheduler graft on real vLLM 0.25.1 objects: the P0a probe's checks 5-14 and 16-18.

The P0a probe (scripts/ci/prefix_p0a_probe.py, 18/18 locally and as the c2 serving job's probe
action) drove the P0a prototype through vLLM's real Scheduler, KVCacheManager, BlockPool, Request,
SchedulerOutput and ModelRunnerOutput. These are the same checks as unit tests, against the
productionised graft AS STAGED: fixtures/vllm_tt_plugin_bf77cd63/scheduler.py (the pinned plugin
blob, which is the image's copy) is patched by qwen_prefix_scheduler_patch.stage into a package
built in a temp dir, and every scheduler here is that package's TTScheduler. One test constructs it
with QWEN_PREFIX_REUSE=1 so the hook itself installs the wrappers.

The config is the `general` profile's scheduler shape (max_model_len 65536, max_num_seqs 4, a
whole-prompt 65536-token budget, block 64, prefix caching on, chunked prefill and async scheduling
off) on a tiny stand-in HF config: no weights, no device, no TT platform. What depends on the TT
platform and the served model config - the probe's checks 1-4 (the align assertion, the platform
turning chunking back off, validate_block_size, validate_mamba_block_size) and check 5's TT worker
KV spec - stays with the probe, which runs inside the serving image.

A fake TT model stands in for the device: it asserts what the G1 model graft asserts (a prefill row
with start_pos > 0 has a committed grant whose Q equals start_pos and whose checkpoint's token ids
match the row) and takes the planned captures.

Skipped where vLLM is not importable (the 3.11 CPU suite). Runs in qwen-fast-vllm-cpu.yml and
locally on the P0a python 3.10 environment with the vLLM 0.25.1 source on PYTHONPATH.
"""

import collections
import copy
import dataclasses
import importlib
import os
import random
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qwen_prefix_scheduler_patch as source_patch  # noqa: E402

try:
    import torch
    from transformers import GPT2Config
    from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash, make_block_hash_with_group_id
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager
    VLLM_ERROR = None
except Exception as error:  # the 3.11 CPU suite has no vLLM
    VLLM_ERROR = '%s: %s' % (type(error).__name__, error)

FIXTURE = HERE / 'fixtures' / 'vllm_tt_plugin_bf77cd63' / 'scheduler.py'
MAX_MODEL_LEN = 65536
BLOCK = source_patch.BLOCK
CHUNK = source_patch.CHUNK
SALT = 'tenant-a'
DEFAULT_BLOCKS = 4100
Row = collections.namedtuple('Row', 'step rid start q h')
STATE = {}


class ModelAssertion(AssertionError):
    """What the G1 model graft raises: the engine would die rather than rewrite shared blocks."""


def setUpModule():
    if VLLM_ERROR is not None:
        return
    os.environ.setdefault('VLLM_USE_V2_MODEL_RUNNER', '0')
    saved = {name: sys.modules.get(name) for name in ('vllm_tt_plugin', 'vllm_tt_plugin.logger')}
    try:
        import vllm_tt_plugin.logger  # noqa: F401  (installed in the vLLM lane: stdlib only)
        STATE['plugin'] = 'installed vllm_tt_plugin.logger'
    except ImportError:
        import logging

        package = types.ModuleType('vllm_tt_plugin')
        package.__path__ = []
        logger = types.ModuleType('vllm_tt_plugin.logger')
        logger.init_tt_logger = logging.getLogger
        sys.modules['vllm_tt_plugin'] = package
        sys.modules['vllm_tt_plugin.logger'] = logger
        STATE['plugin'] = 'stand-in vllm_tt_plugin.logger'
    STATE['saved_modules'] = saved
    root = tempfile.mkdtemp(prefix='qwen-prefix-vllm-')
    STATE['root'] = root
    name = 'qwen_prefix_vllm_pkg'
    package_dir = Path(root) / name
    package_dir.mkdir()
    (package_dir / '__init__.py').write_text('')
    (package_dir / 'scheduler.py').write_bytes(FIXTURE.read_bytes().replace(b'\r\n', b'\n'))
    source_patch.stage(package_dir)
    sys.path.insert(0, root)
    STATE['scheduler_module'] = importlib.import_module(name + '.scheduler')
    STATE['graft'] = importlib.import_module(name + '.qwen_prefix_scheduler_patch')
    STATE['package'] = name
    model_dir = Path(root) / 'model'
    GPT2Config(n_positions=MAX_MODEL_LEN, n_embd=256, n_layer=1, n_head=4,
               architectures=['GPT2LMHeadModel']).save_pretrained(str(model_dir))
    STATE['model_dir'] = str(model_dir)


def tearDownModule():
    if VLLM_ERROR is not None or 'root' not in STATE:
        return
    for key in [key for key in sys.modules if key.startswith(STATE['package'])]:
        del sys.modules[key]
    for name, module in STATE['saved_modules'].items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
    if STATE['root'] in sys.path:
        sys.path.remove(STATE['root'])
    shutil.rmtree(STATE['root'], True)


def tokens(count, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, 200000) for _ in range(count)]


class Env(object):
    """The general profile's scheduler shape on a stand-in model config."""

    def __init__(self):
        model = ModelConfig(model=STATE['model_dir'], dtype='float32', max_model_len=MAX_MODEL_LEN,
                            skip_tokenizer_init=True, seed=0)
        self.vllm_config = VllmConfig(
            model_config=model, device_config=DeviceConfig(device='cpu'),
            scheduler_config=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=MAX_MODEL_LEN,
                                             max_model_len=MAX_MODEL_LEN, is_encoder_decoder=False,
                                             enable_chunked_prefill=False, async_scheduling=False),
            cache_config=CacheConfig(block_size=BLOCK, enable_prefix_caching=True),
            parallel_config=ParallelConfig())
        self.vllm_config.cache_config.num_gpu_blocks = DEFAULT_BLOCKS
        register_all_kvcache_specs(self.vllm_config)
        self.kv_cache_config = KVCacheConfig(num_blocks=DEFAULT_BLOCKS, kv_cache_tensors=[], kv_cache_groups=[
            KVCacheGroupSpec(['foo'], FullAttentionSpec(block_size=BLOCK, num_kv_heads=2, head_size=256,
                                                        dtype=torch.bfloat16))])
        hash_fn = get_hash_fn_by_name(self.vllm_config.cache_config.prefix_caching_hash_algo)
        init_none_hash(hash_fn)
        self.hasher = get_request_block_hasher(BLOCK, hash_fn)
        self.structured = StructuredOutputManager(self.vllm_config)
        self.graft = STATE['graft']
        self.scheduler_cls = STATE['scheduler_module'].TTScheduler
        self.logs = []

    def request(self, rid, prompt, max_tokens, salt):
        params = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
        return Request(request_id=rid, prompt_token_ids=list(prompt), sampling_params=params, pooling_params=None,
                       cache_salt=salt, block_hasher=self.hasher)

    def make(self, num_blocks=None, install=True, registry=None, kill_switch_path=None, clock=None, ledger=None,
             vllm_config=None):
        kv_cache_config = self.kv_cache_config
        if num_blocks is not None:
            kv_cache_config = dataclasses.replace(kv_cache_config, num_blocks=num_blocks)
        scheduler = self.scheduler_cls(vllm_config=vllm_config or self.vllm_config, kv_cache_config=kv_cache_config,
                                       structured_output_manager=self.structured, block_size=BLOCK,
                                       hash_block_size=BLOCK, include_finished_set=False, log_stats=True)
        scheduler.use_v2_model_runner = False
        if ledger is not None:
            attach_ledger(scheduler, ledger)
        state = None
        if install:
            extra = {} if clock is None else {'clock': clock}
            state = self.graft.install(scheduler, registry=registry or self.graft.PrefixRegistry(budget_bytes=1 << 40),
                                       kill_switch_path=kill_switch_path, logger=self.log, **extra)
        return scheduler, state

    def log(self, message, *values):
        self.logs.append(message % values if values else message)


def attach_ledger(scheduler, ledger):
    """Who published each cached block, at which index, from what prompt length."""
    pool = scheduler.kv_cache_manager.block_pool
    original = pool.cache_full_blocks

    def cache_full_blocks(request, blocks, num_cached_blocks, num_full_blocks, block_size, kv_cache_group_id,
                          block_mask=None):
        for index in range(num_cached_blocks, num_full_blocks):
            block = blocks[index]
            if not block.is_null and (block_mask is None or block_mask[index - num_cached_blocks]):
                ledger[block.block_id] = (request.request_id, index, request.num_prompt_tokens)
        return original(request=request, blocks=blocks, num_cached_blocks=num_cached_blocks,
                        num_full_blocks=num_full_blocks, block_size=block_size,
                        kv_cache_group_id=kv_cache_group_id, block_mask=block_mask)

    pool.cache_full_blocks = cache_full_blocks


class Drive(object):
    """schedule -> fake TT model -> update_from_output, as EngineCore.step does it (sync)."""

    def __init__(self, env, scheduler, state, name):
        self.env, self.scheduler, self.state = env, scheduler, state
        self.registry = state.registry if state is not None else None
        self.requests = {}
        self.rows = []
        self.steps = 0
        self.history = []
        self.rng = random.Random(name)

    def add(self, rid, prompt, max_tokens=1, salt=SALT):
        request = self.env.request(rid, prompt, max_tokens, salt)
        self.requests[rid] = request
        self.scheduler.add_request(request)
        return request

    def row(self, rid):
        found = [row for row in self.rows if row.rid == rid]
        return found[-1] if found else None

    def prefill_row(self, rid, start):
        request = self.requests[rid]
        grant = self.registry.grant_for(rid) if self.registry is not None else None
        self.rows.append(Row(self.steps, rid, start, grant.q if grant else None, grant.h if grant else None))
        if self.registry is None:
            return
        if start > 0:
            if grant is None:
                raise ModelAssertion('row %s: start_pos=%d without a committed grant' % (rid, start))
            if grant.req_id != rid or grant.q != start:
                raise ModelAssertion('row %s: grant %s does not match start_pos=%d' % (rid, grant.describe(), start))
            if not grant.checkpoint.matches(request.all_token_ids[0:start]):
                raise ModelAssertion('row %s: checkpoint tokens differ from the prompt below %d' % (rid, start))
        if grant is not None:
            for pos in grant.capture_positions():
                self.registry.capture(rid, pos, rec='rec@%d' % pos, carry='carry@%d' % pos, nbytes=1)

    def execute(self, output):
        req_ids = list(output.num_scheduled_tokens)
        if not req_ids:
            return EMPTY_MODEL_RUNNER_OUTPUT
        for data in output.scheduled_new_reqs:
            self.prefill_row(data.req_id, data.num_computed_tokens)
        cached = output.scheduled_cached_reqs
        for index, rid in enumerate(cached.req_ids):
            if rid in cached.resumed_req_ids:
                self.prefill_row(rid, cached.num_computed_tokens[index])
        return ModelRunnerOutput(req_ids=req_ids, req_id_to_index=dict((rid, i) for i, rid in enumerate(req_ids)),
                                 sampled_token_ids=[[self.rng.randrange(1000, 200000)] for _ in req_ids])

    def step(self):
        output = self.scheduler.schedule()
        self.steps += 1
        self.history.append((self.steps, sorted(output.num_scheduled_tokens)))
        self.scheduler.update_from_output(output, self.execute(output))
        return output

    def run(self, limit=6000):
        count = 0
        while self.scheduler.get_num_unfinished_requests() > 0:
            self.step()
            count += 1
            if count > limit:
                raise AssertionError('the scheduler did not drain in %d steps' % limit)
        return count


def cached_key(pool, block_hash):
    return pool.cached_block_hash_to_block.get_one_block(make_block_hash_with_group_id(block_hash, 0))


def full_chunk_written(entry, index):
    return entry is not None and entry[1] == index and (index + 1) * BLOCK <= source_patch.floor_chunk(entry[2])


@unittest.skipIf(VLLM_ERROR is not None, 'vLLM is not importable here (%s)' % VLLM_ERROR)
class GraftOnRealVllmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = Env()

    def setUp(self):
        self.env.logs[:] = []

    # -- the hook -------------------------------------------------------------------------------
    def test_the_staged_hook_installs_on_construction(self):
        graft = STATE['graft']
        key = graft.prefix_registry.REGISTRY_KEY
        saved = sys.modules.pop(key, None)
        previous = os.environ.get('QWEN_PREFIX_REUSE')
        try:
            os.environ['QWEN_PREFIX_REUSE'] = '1'
            scheduler, _ = self.env.make(install=False)
            state = scheduler.__dict__.get('_qwen_prefix')
            self.assertIsNotNone(state, 'TTScheduler.__init__ did not install the graft')
            self.assertIs(state.registry, sys.modules[key].registry)
            for name in ('schedule', 'reset_prefix_cache', '_free_request'):
                self.assertIn(name, vars(scheduler))
            self.assertIn('get_computed_blocks', vars(scheduler.kv_cache_manager))
            self.assertIn('cache_blocks', vars(scheduler.kv_cache_manager.coordinator))
            self.assertIn('_maybe_evict_cached_block', vars(scheduler.kv_cache_manager.block_pool))
            with self.assertRaises(graft.PrefixInstallError):
                self.env.make(install=False)  # a second live scheduler on the shared registry
            del scheduler, state
            os.environ.pop('QWEN_PREFIX_REUSE')
            plain, _ = self.env.make(install=False)
            self.assertNotIn('schedule', vars(plain))
            self.assertNotIn('_qwen_prefix', vars(plain))
        finally:
            if previous is None:
                os.environ.pop('QWEN_PREFIX_REUSE', None)
            else:
                os.environ['QWEN_PREFIX_REUSE'] = previous
            sys.modules.pop(key, None)
            if saved is not None:
                sys.modules[key] = saved

    # -- P0a check 5 ----------------------------------------------------------------------------
    def test_check_5_the_tt_scheduler_is_unitary_without_mamba(self):
        scheduler, _ = self.env.make(install=False)
        coordinator = scheduler.kv_cache_manager.coordinator
        self.assertEqual(type(scheduler).__name__, 'TTScheduler')
        self.assertFalse(scheduler.has_mamba_layers)
        self.assertEqual(type(coordinator).__name__, 'UnitaryKVCacheCoordinator')
        self.assertFalse(scheduler.need_mamba_block_aligned_split)
        self.assertTrue(scheduler.kv_cache_manager.enable_caching)
        self.assertEqual(scheduler.kv_cache_manager.block_pool.hash_block_size, BLOCK)
        self.assertEqual(STATE['graft'].install_problems(scheduler), [])

    # -- P0a check 6 ----------------------------------------------------------------------------
    def test_check_6_get_computed_blocks_returns_hits(self):
        prompt = tokens(4200, 'c6-a')
        follow = prompt + tokens(500, 'c6-b')
        scheduler, _ = self.env.make(install=False)
        raw = Drive(self.env, scheduler, None, 'c6-raw')
        raw.add('a', prompt)
        raw.run()
        _, h = scheduler.kv_cache_manager.get_computed_blocks(self.env.request('b-probe', follow, 1, SALT))
        self.assertTrue(h > 0 and h % BLOCK == 0)
        scheduler, state = self.env.make()
        drive = Drive(self.env, scheduler, state, 'c6-graft')
        drive.add('a', prompt)
        drive.run()
        drive.add('b', follow)
        drive.run()
        row = drive.row('b')
        self.assertEqual((row.start, row.q), (4096, 4096))
        self.assertIn('grant req=b h=4096 Q=4096 plan=[]', self.env.logs)
        self.assertTrue(any(line.startswith('install scheduler=%s.scheduler.TTScheduler' % STATE['package'])
                            for line in self.env.logs))

    # -- P0a check 7 ----------------------------------------------------------------------------
    def test_check_7_the_same_step_rule(self):
        env = self.env
        # (a) two fresh arrivals sharing a 4096-token prefix in one step: B's raw hit is on blocks
        # A was only allocated this step; no checkpoint exists yet either.
        scheduler, state = env.make()
        drive = Drive(env, scheduler, state, 'c7a')
        shared = tokens(4096, 'c7a-shared')
        drive.add('a', shared + tokens(300, 'c7a-a'))
        drive.add('b', shared + tokens(400, 'c7a-b'))
        drive.step()
        a, b = drive.row('a'), drive.row('b')
        self.assertEqual((a.step, b.h, b.q, b.start), (b.step, 4096, 0, 0))
        drive.run()
        # (b) the rule itself: a checkpoint at 4096 whose KV chain below it was broken (block 10
        # evicted); A re-caches block 10 this step, so B's raw hit reaches 4096 through it.
        scheduler, state = env.make()
        registry = state.registry
        drive = Drive(env, scheduler, state, 'c7b')
        prefix = tokens(4200, 'c7b-x')
        x = drive.add('x', prefix)
        drive.run()
        key = x.block_hashes[4096 // BLOCK - 1]
        pool = scheduler.kv_cache_manager.block_pool
        pool.evict_blocks({cached_key(pool, x.block_hashes[10]).block_id})
        self.assertIsNotNone(registry.get(key))
        rejects = registry.stats['same_step_rejects']
        drive.add('a', prefix[0:4096] + tokens(500, 'c7b-a'))
        drive.add('b', prefix[0:4096] + tokens(600, 'c7b-b'))
        drive.step()
        a, b = drive.row('a'), drive.row('b')
        self.assertEqual((a.step, a.h, b.h, b.q, b.start), (b.step, 640, 4096, 0, 0))
        self.assertGreater(registry.stats['same_step_rejects'], rejects)
        drive.run()
        # (c) the rule holds only within a step.
        drive.add('c', prefix[0:4096] + tokens(700, 'c7b-c'))
        drive.run()
        self.assertEqual((drive.row('c').q, drive.row('c').start), (4096, 4096))
        # (d) not over-eager: two same-step arrivals on a prefix cached a step earlier both hit.
        scheduler, state = env.make()
        drive = Drive(env, scheduler, state, 'c7d')
        base = tokens(4200, 'c7d-x')
        drive.add('x', base)
        drive.run()
        drive.add('s1', base[0:4096] + tokens(500, 'c7d-s1'))
        drive.add('s2', base[0:4096] + tokens(600, 'c7d-s2'))
        drive.step()
        s1, s2 = drive.row('s1'), drive.row('s2')
        self.assertEqual((s1.step, s1.q, s2.q, s1.start, s2.start), (s2.step, 4096, 4096, 4096, 4096))
        drive.run()

    # -- P0a check 8 ----------------------------------------------------------------------------
    def test_check_8_a_turn_containing_the_answer_hits_only_full_chunk_blocks(self):
        def arm(install):
            ledger = {}
            scheduler, state = self.env.make(install=install, ledger=ledger)
            drive = Drive(self.env, scheduler, state, 'c8')
            first = drive.add('t1', tokens(5000, 'c8-t1'), 1500)
            drive.run()
            second = list(first.prompt_token_ids) + list(first.output_token_ids) + tokens(700, 'c8-new')
            probe = self.env.request('t2-probe', second, 1, SALT)
            lookup = state.original_get_computed_blocks if state else scheduler.kv_cache_manager.get_computed_blocks
            blocks, h = lookup(probe)
            ids = [block.block_id for block in blocks.blocks[0]] if h else []
            bad = [(index, ledger.get(block_id)) for index, block_id in enumerate(ids)
                   if not full_chunk_written(ledger.get(block_id), index)]
            row = None
            if install:
                drive.add('t2', second)
                drive.run()
                row = drive.row('t2')
            return h, bad, row

        h, bad, row = arm(True)
        self.assertEqual((h, bad, row.q, row.start), (4096, [], 4096, 4096))
        h_control, bad_control, _ = arm(False)
        self.assertGreater(h_control, 4096, 'control: vLLM alone serves tail and decode blocks')
        self.assertTrue(bad_control)

    # -- P0a check 9 ----------------------------------------------------------------------------
    def test_check_9_decode_publishes_nothing_beyond_the_cap(self):
        def arm(install):
            scheduler, state = self.env.make(install=install)
            drive = Drive(self.env, scheduler, state, 'c9')
            request = drive.add('d', tokens(5000, 'c9'), 2100)
            pool = scheduler.kv_cache_manager.block_pool
            allowed = 4096 // BLOCK
            worst = 0
            while not request.is_finished():
                drive.step()
                worst = max(worst, sum(1 for block_hash in request.block_hashes[allowed:]
                                       if cached_key(pool, block_hash) is not None))
                self.assertLess(drive.steps, 2300)
            kept = sum(1 for block_hash in request.block_hashes[0:allowed] if cached_key(pool, block_hash) is not None)
            return len(request.output_token_ids) - 1, worst, kept

        decoded, worst, kept = arm(True)
        self.assertGreaterEqual(decoded, 2048)
        self.assertEqual((worst, kept), (0, 4096 // BLOCK))
        _, worst_control, _ = arm(False)
        self.assertGreater(worst_control, 0)

    # -- P0a check 10 ---------------------------------------------------------------------------
    def test_check_10_a_discarded_attempt_commits_nothing(self):
        env = self.env
        scheduler, state = env.make(num_blocks=81)
        registry = state.registry
        stats = registry.stats
        pool = scheduler.kv_cache_manager.block_pool
        drive = Drive(env, scheduler, state, 'c10')
        drive.add('r', tokens(580, 'c10-r'), 1000, salt='tenant-r')
        drive.step()
        text = tokens(4200, 'c10-x')
        drive.add('x1', text[0:2100])
        drive.step()
        x2 = drive.add('x2', text[0:4200])
        drive.step()
        boundary = x2.block_hashes[4096 // BLOCK - 1]
        self.assertEqual((drive.row('x1').start, drive.row('x2').q), (0, 2048))
        self.assertIsNotNone(registry.get(boundary))
        before = dict(stats)
        b = drive.add('b', text[0:4096] + tokens(904, 'c10-b'))
        for _ in range(3):
            drive.step()
            self.assertEqual(drive.history[-1][1], ['r'], 'the TT decode fallback ran r')
        self.assertIsNone(registry.grant_for('b'))
        self.assertIsNone(drive.row('b'))
        self.assertEqual(b.status, RequestStatus.WAITING)
        self.assertEqual(registry.get(boundary).pins, 0)
        self.assertEqual((stats['staged'] - before['staged'], stats['dropped_attempts'] - before['dropped_attempts']),
                         (3, 3))
        for index in range(2048 // BLOCK, 4096 // BLOCK):
            block = cached_key(pool, x2.block_hashes[index])
            if block is not None:
                pool.evict_blocks({block.block_id})
        self.assertIsNone(registry.get(boundary), 'coupled out with its boundary block')
        scheduler.finish_requests('r', RequestStatus.FINISHED_ABORTED)
        drive.step()
        row = drive.row('b')
        self.assertEqual((row.q, row.start), (2048, 2048))
        self.assertEqual((stats['grants'] - before['grants'], stats['grant_tokens'] - before['grant_tokens']), (1, 2048))
        drive.step()
        self.assertEqual(registry.pins(), 0)
        drive.run()
        # (b) a budget break (scheduler.py:833-840) after the hit was taken.
        scheduler, state = env.make()
        registry = state.registry
        drive = Drive(env, scheduler, state, 'c10b')
        text = tokens(4200, 'c10b-x')
        drive.add('x', text)
        drive.run()
        before = dict(registry.stats)
        drive.add('big', tokens(65000, 'c10b-big'), 1, salt='tenant-z')
        drive.add('b', text[0:4096] + tokens(904, 'c10b-b'))
        output = drive.step()
        self.assertIn('big', output.num_scheduled_tokens)
        self.assertNotIn('b', output.num_scheduled_tokens)
        self.assertIsNone(registry.grant_for('b'))
        self.assertEqual(registry.stats['dropped_attempts'] - before['dropped_attempts'], 1)
        drive.run()
        self.assertEqual((drive.row('b').q, drive.row('b').start), (4096, 4096))
        self.assertEqual(registry.stats['grants'] - before['grants'], 1)

    # -- P0a check 11 ---------------------------------------------------------------------------
    def test_check_11_abort_clears_plan_grant_and_pin(self):
        env = self.env
        scheduler, state = env.make()
        registry = state.registry
        drive = Drive(env, scheduler, state, 'c11')
        text = tokens(4200, 'c11-x')
        x = drive.add('x', text)
        drive.run()
        key = x.block_hashes[4096 // BLOCK - 1]
        drive.add('big', tokens(65000, 'c11-big'), 1, salt='tenant-z')
        drive.add('w', text[0:4096] + tokens(904, 'c11-w'))
        drive.step()
        self.assertIsNone(drive.row('w'))
        scheduler.finish_requests('w', RequestStatus.FINISHED_ABORTED)
        self.assertIsNone(registry.grant_for('w'))
        self.assertNotIn('w', registry.staged)
        self.assertEqual(registry.pins(), 0)
        drive.run()
        self.assertIsNone(drive.row('w'))
        freed = registry.stats['freed_requests']
        drive.add('c', text[0:4096] + tokens(2904, 'c11-c'))
        output = scheduler.schedule()
        grant = registry.grant_for('c')
        self.assertEqual((grant.q, grant.capture_positions(), registry.get(key).pins), (4096, [6144], 1))
        scheduler.finish_requests('c', RequestStatus.FINISHED_ABORTED)
        self.assertIsNone(registry.grant_for('c'))
        self.assertEqual((registry.pins(), registry.stats['freed_requests']), (0, freed + 1))
        scheduler.update_from_output(output, EMPTY_MODEL_RUNNER_OUTPUT)
        drive.add('d', text[0:4096] + tokens(100, 'c11-d'))
        drive.run()
        self.assertEqual(drive.row('d').q, 4096)

    # -- P0a check 12 ---------------------------------------------------------------------------
    def test_check_12_eviction_coupling(self):
        env = self.env
        # (a) natural LRU pressure: a finished request frees tail-first, so its boundary block is
        # the first of its cached blocks an allocation evicts.
        scheduler, state = env.make(num_blocks=71)
        registry = state.registry
        pool = scheduler.kv_cache_manager.block_pool
        drive = Drive(env, scheduler, state, 'c12a')
        x = drive.add('x', tokens(4200, 'c12a-x'))
        drive.run()
        key = x.block_hashes[4096 // BLOCK - 1]
        self.assertIsNotNone(registry.get(key))
        drive.add('y', tokens(438, 'c12a-y'), 1, salt='tenant-y')
        drive.run()
        self.assertIsNone(cached_key(pool, key))
        self.assertIsNotNone(cached_key(pool, x.block_hashes[4096 // BLOCK - 2]))
        self.assertIsNone(registry.get(key))
        self.assertEqual(registry.stats['evicted_coupled'], 1)
        # (b) the hash still maps to another block: two same-step copies of one prompt.
        scheduler, state = env.make()
        registry = state.registry
        pool = scheduler.kv_cache_manager.block_pool
        drive = Drive(env, scheduler, state, 'c12b')
        text = tokens(4200, 'c12b-x')
        x = drive.add('x', text)
        drive.add('x2', text)
        drive.run()
        key = x.block_hashes[4096 // BLOCK - 1]
        pool.evict_blocks({cached_key(pool, key).block_id})
        self.assertIsNotNone(registry.get(key))
        self.assertIsNotNone(cached_key(pool, key))
        pool.evict_blocks({cached_key(pool, key).block_id})
        self.assertIsNone(registry.get(key))

    # -- P0a check 13 ---------------------------------------------------------------------------
    def test_check_13_fail_closed_salt(self):
        env = self.env
        graft = STATE['graft']
        scheduler, _ = env.make(install=False)
        pool = scheduler.kv_cache_manager.block_pool
        raw = Drive(env, scheduler, None, 'c13-raw')
        text = tokens(4200, 'c13-u')
        raw.add('u0', text, 1, salt=None)
        raw.run()
        state = graft.install(scheduler, registry=graft.PrefixRegistry(budget_bytes=1 << 40), kill_switch_path=None,
                              logger=env.log)
        registry = state.registry
        drive = Drive(env, scheduler, state, 'c13')
        _, h_raw = state.original_get_computed_blocks(env.request('u1-probe', text + tokens(300, 'c13-u1'), 1, None))
        size = len(pool.cached_block_hash_to_block)
        denied = registry.stats['unsalted_denied']
        drive.add('u1', text + tokens(300, 'c13-u1'), 1, salt=None)
        drive.run()
        u1 = drive.row('u1')
        self.assertGreater(h_raw, 0)
        self.assertEqual((u1.start, u1.q), (0, None))
        self.assertEqual(registry.stats['unsalted_denied'], denied + 1)
        self.assertEqual(len(pool.cached_block_hash_to_block), size)
        self.assertFalse(registry.entries)
        drive.add('s1', text + tokens(300, 'c13-s1'), 1, salt='tenant-a')
        drive.run()
        self.assertEqual(len(pool.cached_block_hash_to_block) - size, 4096 // BLOCK)
        drive.add('s2', text + tokens(400, 'c13-s2'), 1, salt='tenant-a')
        drive.add('s3', text + tokens(400, 'c13-s3'), 1, salt='tenant-b')
        drive.run()
        self.assertEqual((drive.row('s1').start, drive.row('s2').q, drive.row('s2').start, drive.row('s3').start),
                         (0, 4096, 4096, 0))

    # -- P0a check 14 ---------------------------------------------------------------------------
    def test_check_14_install_refuses(self):
        graft = STATE['graft']

        def with_config(field, value, target='scheduler_config'):
            def mutate(scheduler):
                config = copy.copy(getattr(scheduler, target))
                object.__setattr__(config, field, value)
                setattr(scheduler, target, config)
            return mutate

        def hybrid(scheduler):
            unitary = scheduler.kv_cache_manager.coordinator
            coordinator = object.__new__(HybridKVCacheCoordinator)
            coordinator.single_type_managers = unitary.single_type_managers
            scheduler.kv_cache_manager.coordinator = coordinator

        def block_size(scheduler):
            scheduler.block_size = 128

        cases = [
            ('async scheduling', with_config('async_scheduling', True), 'async scheduling is on'),
            ('chunked prefill (Lever N)', with_config('enable_chunked_prefill', True), 'chunked prefill is on'),
            ('split budget', with_config('max_num_batched_tokens', MAX_MODEL_LEN - 2048), 'max_num_batched_tokens'),
            ('hybrid coordinator', hybrid, 'not UnitaryKVCacheCoordinator'),
            ('bare hybrid coordinator (the probe\'s)',
             lambda s: setattr(s.kv_cache_manager, 'coordinator', object.__new__(HybridKVCacheCoordinator)),
             'not UnitaryKVCacheCoordinator'),
            ('prefix caching off', with_config('enable_prefix_caching', False, 'cache_config'), 'prefix caching is off'),
            ('block size 128', block_size, 'scheduler block size 128'),
            ('lookahead', lambda s: setattr(s, 'num_lookahead_tokens', 16), 'speculative lookahead'),
        ]
        for name, mutate, expected in cases:
            scheduler, _ = self.env.make(install=False)
            mutate(scheduler)
            with self.assertRaises(graft.PrefixInstallError, msg=name) as caught:
                graft.install(scheduler, registry=graft.PrefixRegistry(budget_bytes=1), kill_switch_path=None,
                              logger=self.env.log)
            self.assertIn(expected, str(caught.exception), name)
            self.assertNotIn('schedule', vars(scheduler), name)
        scheduler, state = self.env.make()
        self.assertIn('schedule', vars(scheduler))

    # -- P0a check 16 ---------------------------------------------------------------------------
    def test_check_16_the_kill_switch(self):
        now = [0.0]
        root = tempfile.mkdtemp(prefix='qwen-prefix-kill-')
        self.addCleanup(shutil.rmtree, root, True)
        flag = os.path.join(root, 'prefix-reuse.off')
        scheduler, state = self.env.make(kill_switch_path=flag, clock=lambda: now[0])
        registry = state.registry
        pool = scheduler.kv_cache_manager.block_pool
        drive = Drive(self.env, scheduler, state, 'c16')
        text = tokens(4200, 'c16-x')
        drive.add('x', text)
        drive.run()
        drive.add('a', text[0:4096] + tokens(300, 'c16-a'))
        drive.run()
        self.assertEqual(drive.row('a').q, 4096)
        open(flag, 'w').close()
        now[0] += 0.5
        drive.add('b', text[0:4096] + tokens(300, 'c16-b'))
        drive.run()
        self.assertEqual(drive.row('b').q, 4096, 'within the poll interval')
        now[0] += 1.0
        size = len(pool.cached_block_hash_to_block)
        drive.add('c', text[0:4096] + tokens(300, 'c16-c'))
        drive.run()
        self.assertEqual(drive.row('c').start, 0)
        self.assertFalse(registry.entries)
        self.assertTrue(state.killed)
        self.assertEqual(len(pool.cached_block_hash_to_block), size, 'publishing is off')
        os.remove(flag)
        now[0] += 5.0
        drive.add('d', text[0:4096] + tokens(300, 'c16-d'))
        drive.run()
        self.assertEqual(drive.row('d').start, 0, 'latched until the engine restarts')

    # -- P0a check 17 ---------------------------------------------------------------------------
    def test_check_17_reset_prefix_cache_clears_the_registry(self):
        scheduler, state = self.env.make()
        registry = state.registry
        pool = scheduler.kv_cache_manager.block_pool
        drive = Drive(self.env, scheduler, state, 'c17')
        text = tokens(4200, 'c17-x')
        drive.add('x', text)
        drive.run()
        self.assertEqual(len(registry.entries), 1)
        self.assertTrue(scheduler.reset_prefix_cache())
        self.assertFalse(registry.entries)
        self.assertEqual(len(pool.cached_block_hash_to_block), 0)
        drive.add('b', text[0:4096] + tokens(300, 'c17-b'))
        drive.run()
        self.assertEqual(drive.row('b').start, 0)

    # -- P0a check 18 ---------------------------------------------------------------------------
    def test_check_18_a_token_mismatch_only_lowers_q(self):
        scheduler, state = self.env.make()
        registry = state.registry
        drive = Drive(self.env, scheduler, state, 'c18')
        text = tokens(4200, 'c18-x')
        drive.add('x1', text[0:2100])
        drive.run()
        x2 = drive.add('x2', text[0:4200])
        drive.run()
        key = x2.block_hashes[4096 // BLOCK - 1]
        entry = registry.get(key)
        self.assertIsNotNone(registry.get(x2.block_hashes[2048 // BLOCK - 1]))
        entry.token_ids[100] ^= 1
        mismatches = registry.stats['token_mismatches']
        drive.add('b', text[0:4096] + tokens(300, 'c18-b'))
        drive.run()
        b = drive.row('b')
        self.assertEqual((b.h, b.q, b.start), (4096, 2048, 2048))
        self.assertEqual(registry.stats['token_mismatches'], mismatches + 1)
        self.assertIsNot(registry.get(key), entry, 'b\'s own capture replaced the bad entry')
        drive.add('c', text[0:4096] + tokens(400, 'c18-c'))
        drive.run()
        self.assertEqual((drive.row('c').q, drive.row('c').start), (4096, 4096))

    # -- beyond P0a: a chained conversation and preemption -------------------------------------
    def test_a_chained_conversation_hits_every_turn(self):
        """Six turns, each prompt = the previous prompt + its answer + a new message: every turn
        after the first restores the previous turn's prompt boundary (the design's data flow)."""
        scheduler, state = self.env.make()
        drive = Drive(self.env, scheduler, state, 'chain')
        prompt = tokens(3000, 'chain-0')
        expected = []
        for turn in range(6):
            request = drive.add('t%d' % turn, prompt, 300)
            drive.run()
            expected.append(source_patch.floor_chunk(len(prompt)))
            prompt = list(request.prompt_token_ids) + list(request.output_token_ids) + tokens(900, 'chain-%d' % turn)
        starts = [drive.row('t%d' % turn).start for turn in range(6)]
        self.assertEqual(starts, [0] + expected[:-1])
        self.assertEqual(state.registry.stats['token_mismatches'], 0)

    def test_preemption_resumes_through_a_fresh_grant(self):
        """A decode preempted for KV re-enters through get_computed_blocks; the fake model asserts
        the resumed row's start_pos is its committed grant's Q."""
        scheduler, state = self.env.make(num_blocks=150)
        drive = Drive(self.env, scheduler, state, 'preempt')
        text = tokens(4200, 'pre-x')
        drive.add('a', text, 2600)
        drive.add('b', text[0:4096] + tokens(200, 'pre-b'), 2600)
        drive.run(limit=9000)
        preempted = [request for request in drive.requests.values() if request.num_preemptions > 0]
        self.assertTrue(preempted, 'the pool never ran out; the test did not exercise preemption')
        resumed = [row for row in drive.rows if row.rid in {request.request_id for request in preempted}]
        self.assertGreaterEqual(len(resumed), 2)
        self.assertEqual(state.registry.pins(), 0)


if __name__ == '__main__':
    unittest.main()
