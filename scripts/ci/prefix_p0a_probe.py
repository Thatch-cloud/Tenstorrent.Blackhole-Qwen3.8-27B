#!/usr/bin/env python3
"""P0a: the config and scheduler probe for conversation prefix reuse (the TT prefix-reuse
design, revision 2 of 2026-09-26, section 2.1).

Runs INSIDE the C2 serving image (vLLM 0.25.1 and the TT plugin it installs), with no Tenstorrent
device attached - it opens none - and the Qwen3.8-27B snapshot mounted read-only at /models:

    docker run --rm --network none -v /home/thatch/hf-cache/hub:/models:ro -v <checkout>:/c2:ro \
        --entrypoint python3 <C2 image> /c2/scripts/ci/prefix_p0a_probe.py

Checks 1-4 build the engine config of a general-prefix profile - the image's `general` profile with
prefix caching and chunked prefill on, exactly as the design's section 2.0.1 item 5 has it - through
vLLM's own argument parser and create_engine_config, which runs the TT platform's
check_and_update_config. The model graft that makes Qwen36ForCausalLM report
supports_prefix_caching when QWEN_PREFIX_REUSE=1 does not exist yet, so the probe sets that one
capability on the class, and says so. Two controls run the same build in child processes: chunked
prefill off (the align assertion must fire) and the capability left as the image has it (the
platform must drop prefix caching and the mamba-block-size validator must refuse).

Checks 5-14 mirror EngineCore._initialize_kv_caches on that config with the installed TT worker's
own functions (the single "foo" FullAttentionSpec, the TT block count), build the real TTScheduler
on it, and drive it with real Requests, SchedulerOutputs and ModelRunnerOutputs. A fake TT model
stands in for the device: it asserts what the G1 model graft will assert (a prefill row with
start_pos > 0 has a committed grant whose Q equals start_pos and whose checkpoint tokens match),
and takes the planned captures. The wrappers under test - cap, trim, per-step commit, eviction
coupling, fail-closed salt, install assertions - are prefix_scheduler_graft.py, which becomes G1's
scheduler graft; these checks are its unit tests. Checks 15-17 are extra unit tests of the same
module (registry LRU, kill switch, reset).

Prints PASS or FAIL per check with its evidence and exits 1 if any check fails.
"""

import argparse
import collections
import copy
import dataclasses
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
# `python3 /c2/scripts/ci/prefix_p0a_probe.py` puts this directory first on sys.path, after the
# image's .pth boot has already laid out the served path. Take it off again so every module the
# engine imports resolves exactly as it does when the image serves.
sys.path[:] = [entry for entry in sys.path if not entry or os.path.abspath(entry) != HERE]


def load_local(name):
    """A module of this checkout, by path. The checkout is NOT put on sys.path: in the image its
    scripts/ci would shadow the baked evidence tree (/experiment-scripts/ci) the model code imports."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


graft = load_local('prefix_scheduler_graft')

PROFILES_IMAGE = '/opt/qwen-c2/profiles.json'
PROFILES_CHECKOUT = os.path.join(HERE, 'qwen_c2_profiles.json')
TT_ARCH = 'TTQwen3_5ForConditionalGeneration'
TT_MODEL_FALLBACK = ('models.demos.blackhole.qwen36.tt.qwen36_vllm', 'Qwen36ForCausalLM')
VARIANT_TAG = 'P0A-VARIANT '
SALT = 'tenant-a'
CHUNK, BLOCK = graft.CHUNK, graft.BLOCK

RESULTS = []
Row = collections.namedtuple('Row', 'step rid start q h')


class ProbeError(RuntimeError):
    pass


class ModelAssertion(AssertionError):
    """What the G1 model graft raises: the engine would die rather than rewrite shared blocks."""


def say(text=''):
    sys.stdout.write(text + '\n')
    sys.stdout.flush()


def record(check, ok, title, details):
    RESULTS.append((check, bool(ok), title, details))
    say('%s %-4s %s' % ('PASS' if ok else 'FAIL', check, title))
    for line in details.splitlines():
        say('          ' + line)


def run_check(check, title, function, *args):
    try:
        ok, details = function(*args)
    except Exception as error:
        ok, details = False, 'raised %s: %s\n%s' % (type(error).__name__, error,
                                                   ''.join(traceback.format_exc()[-2500:]))
    record(check, ok, title, details)


def tokens(count, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, 200000) for _ in range(count)]


def digest(path, algorithm='sha256', length=16):
    try:
        with open(path, 'rb') as handle:
            return hashlib.new(algorithm, handle.read()).hexdigest()[:length]
    except OSError as error:
        return 'unreadable (%s)' % error.strerror


# --------------------------------------------------------------------------------------------
# Engine config (checks 1-4)
# --------------------------------------------------------------------------------------------
def load_contract():
    # The image's .pth hook imported its own copy at interpreter start (QWEN_C2_SERVING=1); the
    # checkout's copy is only the fallback outside the image.
    if 'serving_c2_contract' in sys.modules:
        return sys.modules['serving_c2_contract']
    try:
        import serving_c2_contract
        return serving_c2_contract
    except ImportError:
        return load_local('serving_c2_contract')


def variant_argv(variant):
    contract = load_contract()
    path = PROFILES_IMAGE if os.path.exists(PROFILES_IMAGE) else PROFILES_CHECKOUT
    profile = contract.load_profile(path, 'general')
    snapshot = contract.resolve_snapshot(profile)
    engine = dict(profile['engine'])
    for flag in ('no-enable-prefix-caching', 'enable-prefix-caching', 'no-enable-chunked-prefill',
                 'enable-chunked-prefill'):
        engine.pop(flag, None)
    if variant == 'fallback-general':
        engine['no-enable-prefix-caching'] = True
        engine['no-enable-chunked-prefill'] = True
    else:
        engine['enable-prefix-caching'] = True
        engine['no-enable-chunked-prefill' if variant == 'no-chunking' else 'enable-chunked-prefill'] = True
    argv = contract.engine_arguments(dict(profile, engine=engine), snapshot)
    return path, contract.__file__, snapshot, argv


def tt_model_class():
    from vllm_tt_plugin.platform import register_tt_models
    from vllm.model_executor.models.registry import ModelRegistry

    register_tt_models(False)
    entry = getattr(ModelRegistry, 'models', {}).get(TT_ARCH)
    module_name = getattr(entry, 'module_name', None) or TT_MODEL_FALLBACK[0]
    class_name = getattr(entry, 'class_name', None) or TT_MODEL_FALLBACK[1]
    module = importlib.import_module(module_name)
    return module, getattr(module, class_name)


def simulate_capability(enable):
    module, cls = tt_model_class()
    before = dict(getattr(cls, 'model_capabilities', None) or {})
    if enable:
        cls.model_capabilities = dict(before, supports_prefix_caching=True)
    return dict(module=module.__name__, cls=cls.__name__, file=getattr(module, '__file__', '?'),
                image_capabilities=before, used_capabilities=dict(getattr(cls, 'model_capabilities', {}) or {}),
                simulated=bool(enable))


def config_view(vllm_config):
    cache, sched = vllm_config.cache_config, vllm_config.scheduler_config
    model = vllm_config.model_config
    return dict(
        enable_prefix_caching=cache.enable_prefix_caching, mamba_cache_mode=cache.mamba_cache_mode,
        mamba_block_size=cache.mamba_block_size, block_size=cache.block_size,
        enable_chunked_prefill=sched.enable_chunked_prefill,
        long_prefill_token_threshold=sched.long_prefill_token_threshold,
        max_num_batched_tokens=sched.max_num_batched_tokens, max_model_len=model.max_model_len,
        disable_chunked_mm_input=sched.disable_chunked_mm_input, async_scheduling=sched.async_scheduling,
        scheduler_cls=str(sched.scheduler_cls), architecture=model.architecture,
        architectures=list(model.architectures), is_hybrid=model.is_hybrid,
        worker_cls=str(vllm_config.parallel_config.worker_cls),
        executor=str(vllm_config.parallel_config.distributed_executor_backend))


def install_config_hooks(evidence):
    """Record what the align assertion and the platform hook saw, without changing either."""
    from vllm.model_executor.models import config as model_configs
    from vllm.platforms import current_platform

    mamba = model_configs.MambaModelConfig
    original_mamba = mamba.verify_and_update_config

    def mamba_hook(cls, vllm_config):
        cache, sched = vllm_config.cache_config, vllm_config.scheduler_config
        evidence['mamba_entry'] = dict(prefix=cache.enable_prefix_caching, chunked=sched.enable_chunked_prefill,
                                       mode=cache.mamba_cache_mode)
        try:
            original_mamba(vllm_config)
        except BaseException as error:
            evidence['mamba_error'] = '%s: %s' % (type(error).__name__, error)
            raise
        evidence['mamba_exit'] = dict(mode=cache.mamba_cache_mode, mamba_block_size=cache.mamba_block_size,
                                      chunked=sched.enable_chunked_prefill)

    mamba.verify_and_update_config = classmethod(mamba_hook)

    platform_cls = type(current_platform)
    original_platform = platform_cls.check_and_update_config

    def view(vllm_config):
        sched = vllm_config.scheduler_config
        return dict(chunked=sched.enable_chunked_prefill, threshold=sched.long_prefill_token_threshold,
                    batched=sched.max_num_batched_tokens, prefix=vllm_config.cache_config.enable_prefix_caching,
                    async_scheduling=sched.async_scheduling, scheduler_cls=str(sched.scheduler_cls))

    def platform_hook(cls, vllm_config):
        evidence['platform_before'] = view(vllm_config)
        try:
            original_platform(vllm_config)
        except BaseException as error:
            evidence['platform_error'] = '%s: %s' % (type(error).__name__, error)
            raise
        evidence['platform_after'] = view(vllm_config)

    platform_cls.check_and_update_config = classmethod(platform_hook)


def build_config(variant):
    os.environ['QWEN_PREFIX_REUSE'] = '1'
    evidence = dict(variant=variant)
    path, contract_file, snapshot, argv = variant_argv(variant)
    evidence.update(profiles=path, contract=contract_file, snapshot=snapshot, argv=argv)
    from vllm.platforms import current_platform

    evidence['platform'] = '%s.%s' % (type(current_platform).__module__, type(current_platform).__name__)
    if type(current_platform).__name__ != 'TTPlatform':
        evidence['error'] = 'current platform is %s, not TTPlatform (did import ttnn fail?)' % evidence['platform']
        evidence['error_type'] = 'ProbeError'
        return None, evidence
    evidence['capability'] = simulate_capability(variant in ('general-prefix', 'no-chunking'))
    install_config_hooks(evidence)
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = AsyncEngineArgs.add_cli_args(FlexibleArgumentParser())
    arguments = AsyncEngineArgs.from_cli_args(parser.parse_args(argv))
    try:
        vllm_config = arguments.create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)
    except BaseException as error:
        evidence['error_type'] = type(error).__name__
        evidence['error'] = str(error)[:1500]
        return None, evidence
    evidence['final'] = config_view(vllm_config)
    return vllm_config, evidence


def variant_main(variant):
    try:
        _, evidence = build_config(variant)
    except BaseException as error:
        evidence = dict(variant=variant, error_type=type(error).__name__, error=str(error)[:1500],
                        traceback=traceback.format_exc()[-2000:])
    say(VARIANT_TAG + json.dumps(evidence, default=str))
    return 0


def run_variant(variant):
    command = [sys.executable, os.path.abspath(__file__), '--variant', variant]
    try:
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1200,
                                 universal_newlines=True)
    except subprocess.TimeoutExpired as error:
        return None, 'timed out after %ss' % error.timeout
    for line in process.stdout.splitlines():
        if line.startswith(VARIANT_TAG):
            return json.loads(line[len(VARIANT_TAG):]), ''
    return None, 'exit %s, no result line; tail:\n%s' % (process.returncode, '\n'.join(process.stdout.splitlines()[-25:]))


def show(value):
    return json.dumps(value, default=str, sort_keys=True)


def check_1(evidence, control, control_note):
    entry, exit_ = evidence.get('mamba_entry'), evidence.get('mamba_exit')
    ok = (evidence.get('final') is not None and entry is not None and entry['prefix'] and entry['chunked']
          and exit_ is not None and exit_['mode'] == 'align' and 'mamba_error' not in evidence)
    lines = ['general-prefix argv: %s' % ' '.join(evidence.get('argv', [])),
             'MambaModelConfig.verify_and_update_config (config.py:545-603) entry=%s exit=%s error=%s'
             % (show(entry), show(exit_), evidence.get('mamba_error')),
             'capability: %s' % show(evidence.get('capability'))]
    if evidence.get('error'):
        lines.append('config build failed: %s: %s' % (evidence.get('error_type'), evidence.get('error')))
    control_ok = (control is not None and 'Chunked prefill is required' in str(control.get('error', ''))
                  and (control.get('mamba_entry') or {}).get('chunked') is False)
    lines.append('control, chunked prefill off: %s' % (
        '%s: %s' % (control.get('error_type'), str(control.get('error'))[:300]) if control else control_note))
    lines.append('control verdict: %s (the assertion must fire when chunking is off)'
                 % ('as expected' if control_ok else 'NOT as expected'))
    return ok and control_ok, '\n'.join(lines)


def check_2(evidence):
    before, after, final = evidence.get('platform_before'), evidence.get('platform_after'), evidence.get('final')
    ok = (before is not None and after is not None and final is not None and before['chunked'] is True
          and after['chunked'] is False and final['enable_chunked_prefill'] is False
          and final['long_prefill_token_threshold'] == 0
          and final['max_num_batched_tokens'] >= final['max_model_len'] and final['enable_prefix_caching'] is True
          and final['async_scheduling'] is False and final['scheduler_cls'].endswith('TTScheduler'))
    return ok, ('TTPlatform.check_and_update_config before=%s\nafter=%s\nfinal=%s'
                % (show(before), show(after), show(final)))


def check_3(evidence, env):
    final = evidence.get('final') or {}
    ok = (env is not None and env.validate_block_size_error is None and final.get('mamba_cache_mode') == 'align'
          and env.vllm_config.scheduler_config.long_prefill_token_threshold == 0
          and env.vllm_config.scheduler_config.disable_chunked_mm_input is False
          and env.vllm_config.cache_config.block_size <= env.vllm_config.scheduler_config.max_num_batched_tokens)
    if env is None:
        return False, 'no KV config was built'
    return ok, ('VllmConfig.validate_block_size() (vllm.py:2154-2199, called at engine/core.py:318) after the KV '
                'config set block_size=%d: %s\nmamba_cache_mode=%s long_prefill_token_threshold=%s '
                'disable_chunked_mm_input=%s max_num_batched_tokens=%s'
                % (env.vllm_config.cache_config.block_size, env.validate_block_size_error or 'passed',
                   final.get('mamba_cache_mode'), env.vllm_config.scheduler_config.long_prefill_token_threshold,
                   env.vllm_config.scheduler_config.disable_chunked_mm_input,
                   env.vllm_config.scheduler_config.max_num_batched_tokens))


def check_4(evidence, vllm_config, control, control_note):
    final = evidence.get('final') or {}
    explicit = 'not run'
    ok = False
    if vllm_config is not None:
        # Informational: the validator already ran inside construction; calling it again only
        # works if pydantic left the plain function on the class.
        try:
            from vllm.config import VllmConfig

            method = getattr(VllmConfig, 'validate_mamba_block_size', None)
            method = getattr(method, '__func__', method)
            method = getattr(method, 'wrapped', method)
            if callable(method):
                method(vllm_config)
                explicit = 'passed when called again'
            else:
                explicit = 'not callable here (it ran inside construction)'
        except ValueError as error:
            explicit = 'REFUSED when called again: %s' % error
        except Exception as error:
            explicit = 'could not be called again (%s: %s)' % (type(error).__name__, error)
        cache = vllm_config.cache_config
        ok = (cache.enable_prefix_caching and cache.mamba_block_size == cache.block_size == BLOCK
              and not explicit.startswith('REFUSED'))
    control_error = str((control or {}).get('error', ''))
    control_ok = (control is not None and '--mamba-block-size can only be set with --enable-prefix-caching' in control_error
                  and (control.get('platform_after') or {}).get('prefix') is False)
    lines = ['construction passed the after-validator validate_mamba_block_size (vllm.py:2213-2225): %s; '
             'mamba_block_size=%s block_size=%s prefix caching=%s; explicit call: %s'
             % (vllm_config is not None, final.get('mamba_block_size'), final.get('block_size'),
                final.get('enable_prefix_caching'), explicit),
             'control, capability as the image ships it (%s): %s' % (
                 show(((control or {}).get('capability') or {}).get('image_capabilities')),
                 '%s: %s' % (control.get('error_type'), control_error[:300]) if control else control_note),
             'control platform_after=%s' % show((control or {}).get('platform_after')),
             'control verdict: %s (flag absent => platform drops prefix caching => validator refuses; '
             'the capability flag and the profile must ship together)' % ('as expected' if control_ok else 'NOT as expected')]
    return ok and control_ok, '\n'.join(lines)


# --------------------------------------------------------------------------------------------
# The TT scheduler on the TT KV config (checks 5-17)
# --------------------------------------------------------------------------------------------
class SchedEnv(object):
    """EngineCore._initialize_kv_caches (engine/core.py:240-318) and scheduler construction
    (:128-146), with the TT worker's own spec and block-count functions in place of a device."""

    def __init__(self, vllm_config, logs):
        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core.kv_cache_utils import (generate_scheduler_kv_cache_config, get_kv_cache_configs,
                                                 get_request_block_hasher, init_none_hash,
                                                 resolve_kv_cache_block_sizes)
        from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
        from vllm.v1.structured_output import StructuredOutputManager
        from vllm_tt_plugin import worker as tt_worker

        self.vllm_config = vllm_config
        self.logs = logs
        view = SimpleNamespace(model_config=vllm_config.model_config, parallel_config=vllm_config.parallel_config,
                               cache_config=vllm_config.cache_config, vllm_config=vllm_config)
        try:
            self.hook_spec = tt_worker.TTWorker._try_get_spec_from_model_hook(view)
        except Exception as error:
            self.hook_spec = 'raised %s: %s' % (type(error).__name__, error)
        self.spec = tt_worker.TTWorker._build_default_kv_cache_spec(view)
        self.tt_blocks = tt_worker.get_num_available_blocks_tt(vllm_config, 2)
        vllm_config.cache_config.num_gpu_blocks_override = self.tt_blocks
        memory = tt_worker._available_kv_cache_memory_bytes_for_num_blocks(vllm_config, self.spec, self.tt_blocks)
        register_all_kvcache_specs(vllm_config)
        configs = get_kv_cache_configs(vllm_config, [self.spec], [memory])
        self.kv_cache_config = generate_scheduler_kv_cache_config(configs)
        vllm_config.cache_config.num_gpu_blocks = self.kv_cache_config.num_blocks
        vllm_config.cache_config.block_size = min(
            group.kv_cache_spec.block_size for group in self.kv_cache_config.kv_cache_groups)
        try:
            vllm_config.validate_block_size()
            self.validate_block_size_error = None
        except Exception as error:
            self.validate_block_size_error = '%s: %s' % (type(error).__name__, error)
        self.scheduler_block, self.hash_block = resolve_kv_cache_block_sizes(self.kv_cache_config, vllm_config)
        hash_fn = get_hash_fn_by_name(vllm_config.cache_config.prefix_caching_hash_algo)
        init_none_hash(hash_fn)
        self.hasher = get_request_block_hasher(self.hash_block, hash_fn)
        self.structured = StructuredOutputManager(vllm_config)
        self.scheduler_cls = vllm_config.scheduler_config.get_scheduler_cls()

    def request(self, rid, prompt, max_tokens, salt):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        params = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
        return Request(request_id=rid, prompt_token_ids=list(prompt), sampling_params=params, pooling_params=None,
                       cache_salt=salt, block_hasher=self.hasher)

    def make(self, num_blocks=None, install=True, registry=None, kill_switch_path=None, clock=time.monotonic,
             ledger=None):
        kv_cache_config = self.kv_cache_config
        if num_blocks is not None:
            kv_cache_config = dataclasses.replace(kv_cache_config, num_blocks=num_blocks)
        scheduler = self.scheduler_cls(vllm_config=self.vllm_config, kv_cache_config=kv_cache_config,
                                       structured_output_manager=self.structured, block_size=self.scheduler_block,
                                       hash_block_size=self.hash_block, include_finished_set=False, log_stats=True)
        if ledger is not None:
            attach_ledger(scheduler, ledger)
        state = None
        if install:
            state = graft.install(scheduler, registry=registry or graft.PrefixRegistry(),
                                  kill_switch_path=kill_switch_path, clock=clock, logger=self.log)
        return scheduler, state

    def log(self, message, *values):
        self.logs.append('[PINDIAG] prefix: ' + (message % values if values else message))


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
        self.history = []
        self.steps = 0
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
            for pos, _ in grant.plan:
                self.registry.capture(rid, pos)

    def execute(self, output):
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

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
                raise ProbeError('the scheduler did not drain in %d steps' % limit)
        return count


def cached_key(pool, block_hash):
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    return pool.cached_block_hash_to_block.get_one_block(make_block_hash_with_group_id(block_hash, 0))


def check_5(env):
    scheduler, _ = env.make(install=False)
    coordinator = scheduler.kv_cache_manager.coordinator
    groups = env.kv_cache_config.kv_cache_groups
    spec = groups[0].kv_cache_spec if groups else None
    module = sys.modules.get(type(scheduler).__module__)
    ok = (type(scheduler).__name__ == 'TTScheduler' and scheduler.has_mamba_layers is False
          and env.kv_cache_config.has_mamba_layers is False and type(coordinator).__name__ == 'UnitaryKVCacheCoordinator'
          and not scheduler.need_mamba_block_aligned_split and len(groups) == 1
          and type(spec).__name__ == 'FullAttentionSpec' and spec.block_size == BLOCK
          and env.scheduler_block == env.hash_block == BLOCK and scheduler.kv_cache_manager.enable_caching)
    return ok, ('scheduler=%s.%s (%s)\ncoordinator=%s has_mamba_layers=%s need_mamba_block_aligned_split=%s '
                'enable_caching=%s\nmodel get_kv_cache_spec hook=%s; spec=%r\nTT blocks=%d (get_num_available_blocks_tt), '
                'KV config num_blocks=%d groups=%d scheduler/hash block=%d/%d'
                % (type(scheduler).__module__, type(scheduler).__name__, getattr(module, '__file__', '?'),
                   type(coordinator).__name__, scheduler.has_mamba_layers, scheduler.need_mamba_block_aligned_split,
                   scheduler.kv_cache_manager.enable_caching, env.hook_spec, spec, env.tt_blocks,
                   env.kv_cache_config.num_blocks, len(groups), env.scheduler_block, env.hash_block))


def check_6(env):
    prompt = tokens(4200, 'c6-a')
    follow = prompt + tokens(500, 'c6-b')
    scheduler, _ = env.make(install=False)
    raw = Drive(env, scheduler, None, 'c6-raw')
    raw.add('a', prompt)
    raw.run()
    _, h = scheduler.kv_cache_manager.get_computed_blocks(env.request('b-probe', follow, 1, SALT))
    scheduler, state = env.make()
    drive = Drive(env, scheduler, state, 'c6-graft')
    drive.add('a', prompt)
    drive.run()
    drive.add('b', follow)
    drive.run()
    row = drive.row('b')
    ok = h > 0 and h % BLOCK == 0 and row is not None and row.start == 4096 and row.q == 4096
    return ok, ('vLLM alone on the TT config: a %d-token prompt extending a finished %d-token one hits h=%d\n'
                'with the graft: row b start_pos=%s Q=%s h=%s' % (len(follow), len(prompt), h, row.start, row.q, row.h))


def check_7(env):
    lines = []
    # (a) two fresh arrivals sharing a 4096-token prefix, one step: vLLM hands B blocks A has only been
    # allocated this step.
    scheduler, state = env.make()
    drive = Drive(env, scheduler, state, 'c7a')
    shared = tokens(4096, 'c7a-shared')
    drive.add('a', shared + tokens(300, 'c7a-a'))
    drive.add('b', shared + tokens(400, 'c7a-b'))
    drive.step()
    a, b = drive.row('a'), drive.row('b')
    ok_a = a is not None and b is not None and a.step == b.step and b.h == 4096 and b.q == 0 and b.start == 0
    lines.append('(a) same step=%s; raw hit for b h=%s (blocks a allocated this step) -> Q=%s start_pos=%s'
                 % (a and b and a.step == b.step, b and b.h, b and b.q, b and b.start))
    drive.run()
    # (b) the rule itself: a checkpoint exists at 4096 but the KV chain below it was broken (block 10
    # evicted, an orphan); A re-caches block 10 this step, so B's raw hit reaches 4096 through it.
    scheduler, state = env.make()
    registry = state.registry
    drive = Drive(env, scheduler, state, 'c7b')
    prefix = tokens(4200, 'c7b-x')
    x = drive.add('x', prefix)
    drive.run()
    key = x.block_hashes[4096 // BLOCK - 1]
    pool = scheduler.kv_cache_manager.block_pool
    block10 = cached_key(pool, x.block_hashes[10])
    pool.evict_blocks({block10.block_id})
    entry_before = registry.get(key) is not None
    rejects = registry.stats['same_step_rejects']
    orphans = registry.stats['orphans']
    drive.add('a', prefix[0:4096] + tokens(500, 'c7b-a'))
    drive.add('b', prefix[0:4096] + tokens(600, 'c7b-b'))
    drive.step()
    a, b = drive.row('a'), drive.row('b')
    ok_b = (entry_before and a.step == b.step and a.h == 640 and b.h == 4096 and b.q == 0 and b.start == 0
            and registry.stats['same_step_rejects'] > rejects)
    lines.append('(b) checkpoint at 4096 present=%s; block 10 evicted; a: h=%s Q=%s; b (same step): h=%s Q=%s '
                 'start_pos=%s; same-step rejects +%d; orphans counted +%d'
                 % (entry_before, a.h, a.q, b.h, b.q, b.start, registry.stats['same_step_rejects'] - rejects,
                    registry.stats['orphans'] - orphans))
    drive.run()
    drive.add('c', prefix[0:4096] + tokens(700, 'c7b-c'))
    drive.run()
    c = drive.row('c')
    ok_c = c.q == 4096 and c.start == 4096
    lines.append('(c) the next step, same prefix: c Q=%s start_pos=%s (the rule holds only within a step)' % (c.q, c.start))
    return ok_a and ok_b and ok_c, '\n'.join(lines)


def full_chunk_written(entry, index):
    return entry is not None and entry[1] == index and (index + 1) * BLOCK <= graft.floor_chunk(entry[2])


def check_8(env):
    def arm(install):
        ledger = {}
        scheduler, state = env.make(install=install, ledger=ledger)
        drive = Drive(env, scheduler, state, 'c8')
        first = drive.add('t1', tokens(5000, 'c8-t1'), 1500)
        drive.run()
        answer = list(first.output_token_ids)
        second = list(first.prompt_token_ids) + answer + tokens(700, 'c8-new')
        probe = env.request('t2-probe', second, 1, SALT)
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
        return len(answer), len(second), h, bad, row

    answer, second, h, bad, row = arm(True)
    ok = h == 4096 and not bad and row.q == 4096 and row.start == 4096
    lines = ['graft: turn 1 prompt 5000 + %d-token answer; turn 2 (%d tokens) contains the answer verbatim; '
             'raw hit h=%d; blocks not written inside a full prefill chunk: %d; turn 2 Q=%s start_pos=%s'
             % (answer, second, h, len(bad), row.q, row.start)]
    _, _, h_control, bad_control, _ = arm(False)
    control_ok = h_control > 4096 and len(bad_control) > 0
    lines.append('control, no graft: raw hit h=%d; %d hit blocks were written by the prompt tail or decode '
                 '(first: index %s publisher %s) - %s'
                 % (h_control, len(bad_control), bad_control[0][0] if bad_control else '-',
                    bad_control[0][1] if bad_control else '-',
                    'the hazard the cap removes' if control_ok else 'NOT as expected'))
    return ok and control_ok, '\n'.join(lines)


def check_9(env):
    def arm(install):
        scheduler, state = env.make(install=install)
        drive = Drive(env, scheduler, state, 'c9')
        request = drive.add('d', tokens(5000, 'c9'), 2100)
        pool = scheduler.kv_cache_manager.block_pool
        allowed = 4096 // BLOCK
        worst, first_bad = 0, None
        while not request.is_finished():
            drive.step()
            present = sum(1 for block_hash in request.block_hashes[allowed:] if cached_key(pool, block_hash) is not None)
            if present and first_bad is None:
                first_bad = drive.steps
            worst = max(worst, present)
            if drive.steps > 2300:
                raise ProbeError('decode did not finish')
        kept = sum(1 for block_hash in request.block_hashes[0:allowed] if cached_key(pool, block_hash) is not None)
        return len(request.output_token_ids) - 1, len(request.block_hashes), worst, first_bad, kept, \
            (state.registry.stats['publish_capped'] if state else 0)

    decode, hashes, worst, first_bad, kept, capped = arm(True)
    ok = decode >= 2048 and worst == 0 and kept == 4096 // BLOCK
    lines = ['graft: %d decode tokens through update_from_output; %d block hashes on the request; hashes beyond '
             'floor2048(5000)/64=64 ever in the pool map: %d; prompt blocks below the cap cached: %d; '
             'capped publish calls: %d' % (decode, hashes, worst, kept, capped)]
    decode_c, _, worst_c, first_bad_c, _, _ = arm(False)
    control_ok = worst_c > 0
    lines.append('control, no graft: %d decode tokens; up to %d hashes beyond the cap in the map, from step %s - %s'
                 % (decode_c, worst_c, first_bad_c, 'as expected' if control_ok else 'NOT as expected'))
    return ok and control_ok, '\n'.join(lines)


def check_10(env):
    from vllm.v1.request import RequestStatus

    lines = []
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
    setup = (drive.row('x1').start == 0 and drive.row('x2').q == 2048 and registry.get(boundary) is not None)
    lines.append('setup: pool 81 blocks; r decoding in 10 blocks; x1 (2100) then x2 (4200, Q=%s) leave checkpoints '
                 'at 2048 and 4096: %s; free blocks %d' % (drive.row('x2').q, setup, pool.get_num_free_blocks()))
    before = dict(stats)
    b = drive.add('b', text[0:4096] + tokens(904, 'c10-b'))
    failed = []
    for _ in range(3):
        drive.step()
        failed.append(drive.history[-1][1])
    stale = (registry.grant_for('b') is None and drive.row('b') is None and b.status == RequestStatus.WAITING
             and registry.get(boundary).pins == 0)
    attempts = stats['staged'] - before['staged']
    dropped = stats['dropped_attempts'] - before['dropped_attempts']
    fallback = all(scheduled == ['r'] for scheduled in failed)
    lines.append('3 steps: b staged Q=4096 each time (staged +%d, dropped +%d), allocation needs 79 > %d free; '
                 'TT decode fallback ran r each step: %s; no committed grant, no pin, b waiting: %s'
                 % (attempts, dropped, pool.get_num_free_blocks(), fallback, stale))
    for index in range(2048 // BLOCK, 4096 // BLOCK):
        block = cached_key(pool, x2.block_hashes[index])
        if block is not None:
            pool.evict_blocks({block.block_id})
    coupled = registry.get(boundary) is None
    scheduler.finish_requests('r', RequestStatus.FINISHED_ABORTED)
    drive.step()
    row = drive.row('b')
    grants = stats['grants'] - before['grants']
    grant_tokens = stats['grant_tokens'] - before['grant_tokens']
    fresh = row is not None and row.q == 2048 and row.start == 2048
    lines.append('blocks 32-63 evicted (checkpoint 4096 coupled out: %s), r aborted; next step b admitted with a '
                 'fresh grant Q=%s start_pos=%s; grants +%d grant tokens +%d'
                 % (coupled, row and row.q, row and row.start, grants, grant_tokens))
    drive.step()
    pins = registry.pins()
    lines.append('after the next step began: pins=%d' % pins)
    ok_a = (setup and attempts == 3 and dropped == 3 and fallback and stale and coupled and fresh and grants == 1
            and grant_tokens == 2048 and pins == 0)
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
    broke = ('big' in output.num_scheduled_tokens and 'b' not in output.num_scheduled_tokens
             and registry.grant_for('b') is None
             and registry.stats['dropped_attempts'] - before['dropped_attempts'] == 1)
    drive.run()
    row = drive.row('b')
    ok_b = broke and row.q == 4096 and row.start == 4096 and registry.stats['grants'] - before['grants'] == 1
    lines.append('(b) budget break: big (65000) took the budget, b staged Q=4096 then broke out: dropped, no grant: '
                 '%s; next step b Q=%s start_pos=%s' % (broke, row.q, row.start))
    return ok_a and ok_b, '\n'.join(lines)


def check_11(env):
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
    from vllm.v1.request import RequestStatus

    lines = []
    scheduler, state = env.make()
    registry = state.registry
    drive = Drive(env, scheduler, state, 'c11')
    text = tokens(4200, 'c11-x')
    x = drive.add('x', text)
    drive.run()
    key = x.block_hashes[4096 // BLOCK - 1]
    # (a) waiting: big takes the step's budget, w waits; abort w before the next step.
    drive.add('big', tokens(65000, 'c11-big'), 1, salt='tenant-z')
    drive.add('w', text[0:4096] + tokens(904, 'c11-w'))
    drive.step()
    waiting = drive.row('w') is None
    scheduler.finish_requests('w', RequestStatus.FINISHED_ABORTED)
    clean_w = registry.grant_for('w') is None and 'w' not in registry.staged and registry.pins() == 0
    drive.run()
    lines.append('(a) w waited behind a budget break (%s), aborted: no staged or committed grant, no pin: %s; '
                 'w never ran: %s' % (waiting, clean_w, drive.row('w') is None))
    # (b) admitted with a committed grant, a pin and a planned capture, aborted before the model ran.
    freed = registry.stats['freed_requests']
    drive.add('c', text[0:4096] + tokens(2904, 'c11-c'))
    output = scheduler.schedule()
    grant = registry.grant_for('c')
    held = grant is not None and grant.q == 4096 and [pos for pos, _ in grant.plan] == [6144] \
        and registry.get(key).pins == 1
    scheduler.finish_requests('c', RequestStatus.FINISHED_ABORTED)
    released = registry.grant_for('c') is None and registry.pins() == 0 \
        and registry.stats['freed_requests'] == freed + 1
    scheduler.update_from_output(output, EMPTY_MODEL_RUNNER_OUTPUT)
    drive.add('d', text[0:4096] + tokens(100, 'c11-d'))
    drive.run()
    after = drive.row('d')
    lines.append('(b) c admitted (grant Q=4096, plan [6144], pin 1: %s), aborted before its row ran: grant, plan '
                 'and pin gone: %s; a later d still gets Q=%s' % (held, released, after.q))
    return waiting and clean_w and drive.row('w') is None and held and released and after.q == 4096, '\n'.join(lines)


def check_12(env):
    lines = []
    # (a) natural LRU pressure. A finished request frees tail-first (single_type_kv_cache_manager.py:402-410),
    # so its boundary block is the first of its cached blocks an allocation evicts.
    scheduler, state = env.make(num_blocks=71)
    registry = state.registry
    pool = scheduler.kv_cache_manager.block_pool
    drive = Drive(env, scheduler, state, 'c12a')
    x = drive.add('x', tokens(4200, 'c12a-x'))
    drive.run()
    key = x.block_hashes[4096 // BLOCK - 1]
    present = registry.get(key) is not None
    coupled = registry.stats['evicted_coupled']
    drive.add('y', tokens(438, 'c12a-y'), 1, salt='tenant-y')
    drive.run()
    dropped = registry.get(key) is None and registry.stats['evicted_coupled'] == coupled + 1
    boundary_gone = cached_key(pool, key) is None
    below_kept = cached_key(pool, x.block_hashes[4096 // BLOCK - 2]) is not None
    lines.append('(a) pool 71: x (4200) leaves 64 cached blocks and a checkpoint at 4096 (%s); y needs 7 blocks, '
                 '6 are uncached, so it evicts x\'s boundary block (%s) and nothing below it (%s); '
                 'checkpoint dropped with it: %s' % (present, boundary_gone, below_kept, dropped))
    ok_a = present and dropped and boundary_gone and below_kept
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
    first = cached_key(pool, key)
    pool.evict_blocks({first.block_id})
    kept = registry.get(key) is not None and cached_key(pool, key) is not None
    second = cached_key(pool, key)
    pool.evict_blocks({second.block_id})
    gone = registry.get(key) is None
    lines.append('(b) x and x2 (same prompt, same step; x2 recomputed) cache the boundary hash twice (x2 Q=%s); '
                 'evicting one copy keeps the checkpoint: %s; evicting the last drops it: %s'
                 % (drive.row('x2').q, kept, gone))
    return ok_a and kept and gone, '\n'.join(lines)


def check_13(env):
    lines = []
    scheduler, _ = env.make(install=False)
    pool = scheduler.kv_cache_manager.block_pool
    raw = Drive(env, scheduler, None, 'c13-raw')
    text = tokens(4200, 'c13-u')
    raw.add('u0', text, 1, salt=None)
    raw.run()
    state = graft.install(scheduler, registry=graft.PrefixRegistry(), kill_switch_path=None, logger=env.log)
    registry = state.registry
    drive = Drive(env, scheduler, state, 'c13')
    _, h_raw = state.original_get_computed_blocks(env.request('u1-probe', text + tokens(300, 'c13-u1'), 1, None))
    size = len(pool.cached_block_hash_to_block)
    denied = registry.stats['unsalted_denied']
    drive.add('u1', text + tokens(300, 'c13-u1'), 1, salt=None)
    drive.run()
    u1 = drive.row('u1')
    unsalted_ok = (h_raw > 0 and u1.start == 0 and u1.q is None and registry.stats['unsalted_denied'] == denied + 1
                   and len(pool.cached_block_hash_to_block) == size and not registry.entries)
    lines.append('unsalted: vLLM alone would hit h=%d (u0 published before the graft); u1 got start_pos=%d, no grant, '
                 'published %d blocks, registry entries %d' % (h_raw, u1.start, len(pool.cached_block_hash_to_block) - size,
                                                              len(registry.entries)))
    drive.add('s1', text + tokens(300, 'c13-s1'), 1, salt='tenant-a')
    drive.run()
    grown = len(pool.cached_block_hash_to_block) - size
    drive.add('s2', text + tokens(400, 'c13-s2'), 1, salt='tenant-a')
    drive.add('s3', text + tokens(400, 'c13-s3'), 1, salt='tenant-b')
    drive.run()
    s1, s2, s3 = drive.row('s1'), drive.row('s2'), drive.row('s3')
    salted_ok = s1.start == 0 and grown == 4096 // BLOCK and s2.q == 4096 and s2.start == 4096 and s3.start == 0
    lines.append('salted: s1 (tenant-a) start_pos=%d published %d blocks; s2 (tenant-a) Q=%s start_pos=%d; '
                 's3 (tenant-b, same text) start_pos=%d' % (s1.start, grown, s2.q, s2.start, s3.start))
    return unsalted_ok and salted_ok, '\n'.join(lines)


def check_14(env):
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

    def with_config(field, value, target='scheduler_config'):
        def mutate(scheduler):
            config = copy.copy(getattr(scheduler, target))
            object.__setattr__(config, field, value)
            setattr(scheduler, target, config)
        return mutate

    def hybrid(scheduler):
        scheduler.kv_cache_manager.coordinator = object.__new__(HybridKVCacheCoordinator)

    def block_size(scheduler):
        scheduler.block_size = 128

    max_model_len = env.vllm_config.model_config.max_model_len
    cases = [
        ('async scheduling', with_config('async_scheduling', True), 'async scheduling is on'),
        ('chunked prefill (Lever N)', with_config('enable_chunked_prefill', True), 'chunked prefill is on'),
        ('split budget', with_config('max_num_batched_tokens', max_model_len - 2048), 'max_num_batched_tokens'),
        ('hybrid coordinator', hybrid, 'not UnitaryKVCacheCoordinator'),
        ('prefix caching off', with_config('enable_prefix_caching', False, 'cache_config'), 'prefix caching is off'),
        ('block size 128', block_size, 'scheduler block size 128'),
    ]
    lines, ok = [], True
    for name, mutate, expected in cases:
        scheduler, _ = env.make(install=False)
        mutate(scheduler)
        try:
            graft.install(scheduler, registry=graft.PrefixRegistry(), kill_switch_path=None, logger=env.log)
            outcome, good = 'installed (NOT refused)', False
        except graft.PrefixInstallError as error:
            good = expected in str(error) and 'schedule' not in scheduler.__dict__
            outcome = 'refused: %s' % str(error)[len('prefix reuse refused: '):][:160]
        ok = ok and good
        lines.append('%-26s %s' % (name, outcome))
    scheduler, state = env.make()
    installed = state is not None and 'schedule' in scheduler.__dict__
    lines.append('%-26s %s' % ('the served config', 'installed' if installed else 'NOT installed'))
    return ok and installed, '\n'.join(lines)


def check_15():
    registry = graft.PrefixRegistry(budget_bytes=300)
    ids = list(range(CHUNK))
    for index in range(3):
        registry.put(b'k%d' % index, CHUNK, ids, nbytes=100)
    registry.entries[b'k0'].pins = 1
    registry.put(b'k3', CHUNK, ids, nbytes=100)
    kept = list(registry.entries)
    skipped = registry.put(b'k4', CHUNK, ids, nbytes=301) is None
    registry.entries[b'k0'].pins = 0
    refused = 0
    for pos, size in ((CHUNK + 64, CHUNK + 64), (CHUNK, CHUNK - 1)):
        try:
            registry.put(b'bad', pos, list(range(size)), nbytes=1)
        except ValueError:
            refused += 1
    ok = kept == [b'k0', b'k2', b'k3'] and skipped and registry.bytes == 300 and refused == 2 \
        and registry.stats['evicted_lru'] == 1
    return ok, ('budget 300 B, three 100 B checkpoints, the oldest pinned; a fourth evicts the oldest unpinned: '
                'entries %s; an oversize capture skipped: %s; off-boundary puts refused: %d'
                % ([key.decode() for key in kept], skipped, refused))


def check_16(env):
    now = [0.0]
    flag = os.path.join(tempfile.mkdtemp(prefix='p0a-'), 'prefix-reuse.off')
    scheduler, state = env.make(kill_switch_path=flag, clock=lambda: now[0])
    registry = state.registry
    pool = scheduler.kv_cache_manager.block_pool
    drive = Drive(env, scheduler, state, 'c16')
    text = tokens(4200, 'c16-x')
    drive.add('x', text)
    drive.run()
    drive.add('a', text[0:4096] + tokens(300, 'c16-a'))
    drive.run()
    before = drive.row('a').q
    open(flag, 'w').close()
    now[0] += 0.5
    drive.add('b', text[0:4096] + tokens(300, 'c16-b'))
    drive.run()
    within_poll = drive.row('b').q
    now[0] += 1.0
    size = len(pool.cached_block_hash_to_block)
    drive.add('c', text[0:4096] + tokens(300, 'c16-c'))
    drive.run()
    killed = drive.row('c').start == 0 and not registry.entries and state.killed
    published = len(pool.cached_block_hash_to_block) - size
    os.remove(flag)
    now[0] += 5.0
    drive.add('d', text[0:4096] + tokens(300, 'c16-d'))
    drive.run()
    latched = drive.row('d').start == 0
    ok = before == 4096 and within_poll == 4096 and killed and published == 0 and latched
    return ok, ('before the flag: Q=%s; flag written, 0.5 s later (poll interval 1 s): Q=%s; after the poll: '
                'start_pos=%d, registry emptied, publishing off (%d new blocks): %s; flag removed: start_pos=%d '
                '(latched until restart)' % (before, within_poll, drive.row('c').start, published, killed,
                                             drive.row('d').start))


def check_17(env):
    scheduler, state = env.make()
    registry = state.registry
    pool = scheduler.kv_cache_manager.block_pool
    drive = Drive(env, scheduler, state, 'c17')
    text = tokens(4200, 'c17-x')
    drive.add('x', text)
    drive.run()
    had = len(registry.entries)
    reset = scheduler.reset_prefix_cache()
    cleared = not registry.entries and len(pool.cached_block_hash_to_block) == 0
    drive.add('b', text[0:4096] + tokens(300, 'c17-b'))
    drive.run()
    row = drive.row('b')
    ok = had == 1 and reset and cleared and row.start == 0
    return ok, ('checkpoints before %d; reset_prefix_cache()=%s; registry and pool map empty: %s; the same prefix '
                'then gets start_pos=%d' % (had, reset, cleared, row.start))


# --------------------------------------------------------------------------------------------
def environment_report():
    say('INFO python %s' % sys.version.split()[0])
    try:
        import vllm

        say('INFO vllm %s at %s' % (getattr(vllm, '__version__', '?'), os.path.dirname(vllm.__file__)))
    except Exception as error:
        say('INFO vllm import failed: %s' % error)
    try:
        import vllm_tt_plugin

        root = os.path.dirname(vllm_tt_plugin.__file__)
        say('INFO TT plugin at %s' % root)
        for name in ('scheduler.py', 'platform.py', 'worker.py', 'model_runner.py'):
            say('INFO   %-15s sha256 %s' % (name, digest(os.path.join(root, name))))
    except Exception as error:
        say('INFO TT plugin import failed: %s' % error)
    for key in ('QWEN_C2_SERVING', 'QWEN_C2_PROFILE', 'QWEN_SDPA_BF8', 'VLLM_USE_V2_MODEL_RUNNER', 'VLLM_PLUGINS',
                'HF_HUB_OFFLINE', 'TT_METAL_CACHE'):
        say('INFO env %s=%s' % (key, os.environ.get(key, '<unset>')))
    say('INFO /dev/tenstorrent present: %s (the probe opens no device)' % os.path.exists('/dev/tenstorrent'))
    say('INFO sys.path head: %s' % sys.path[0:6])


def model_tree_report():
    try:
        module, cls = tt_model_class()
        root = os.path.dirname(module.__file__)
        say('INFO TT model %s.%s at %s' % (module.__name__, cls.__name__, root))
        say('INFO   model_capabilities as shipped: %s' % show(getattr(cls, 'model_capabilities', None)))
        for name in ('qwen36_vllm.py', 'model.py'):
            path = os.path.join(root, name)
            say('INFO   %-15s md5 %s sha256 %s' % (name, digest(path, 'md5', 8), digest(path)))
        say('INFO   (the design\'s IMG copies: model.py md5 e4ba08d9, qwen36_vllm.py md5 b5230935)')
    except Exception as error:
        say('INFO TT model import failed: %s: %s' % (type(error).__name__, error))


def main():
    say('=== P0a probe: prefix reuse on the TT general path, config and scheduler (design section 2.1)')
    environment_report()
    model_tree_report()
    started = time.time()
    try:
        vllm_config, evidence = build_config('general-prefix')
    except Exception as error:
        vllm_config, evidence = None, dict(error_type=type(error).__name__, error=str(error),
                                           traceback=traceback.format_exc()[-2500:])
    say('INFO general-prefix config built in %.1f s: %s' % (time.time() - started, vllm_config is not None))
    if evidence.get('traceback'):
        say(evidence['traceback'])
    say('INFO profiles %s, contract %s, snapshot %s' % (evidence.get('profiles'), evidence.get('contract'),
                                                         evidence.get('snapshot')))
    controls = {}
    for variant in ('no-chunking', 'no-capability'):
        began = time.time()
        controls[variant] = run_variant(variant)
        say('INFO control %s ran in %.1f s' % (variant, time.time() - began))

    run_check('1', 'the mamba align-mode assertion passes with chunked prefill on (config.py:579-582)',
              check_1, evidence, controls['no-chunking'][0], controls['no-chunking'][1])
    run_check('2', 'the TT platform then turns chunked prefill off, whole-prompt budget (platform.py:67-105)',
              check_2, evidence)

    config_for_scheduler, note = vllm_config, 'the general-prefix config'
    if config_for_scheduler is None:
        # Keep the scheduler checks informative when the config fails: the stock general config
        # with prefix caching forced on in the scheduler's copy.
        try:
            config_for_scheduler, _ = build_config('fallback-general')
            if config_for_scheduler is not None:
                config_for_scheduler.cache_config.enable_prefix_caching = True
                note = 'FALLBACK: the stock general config with enable_prefix_caching forced on after the fact'
        except Exception:
            config_for_scheduler = None
    env, env_error = None, None
    logs = []
    if config_for_scheduler is not None:
        try:
            env = SchedEnv(config_for_scheduler, logs)
        except Exception:
            env_error = traceback.format_exc()[-2500:]
    say('INFO scheduler checks run on %s%s' % (note, '' if env else ' - NOT RUN: %s' % (env_error or 'no config')))
    run_check('3', 'validate_block_size passes: threshold 0, chunked MM input allowed (vllm.py:2186-2199)',
              check_3, evidence if vllm_config is not None else {}, env if vllm_config is not None else None)
    run_check('4', 'validate_mamba_block_size passes (vllm.py:2213-2225)', check_4, evidence, vllm_config,
              controls['no-capability'][0], controls['no-capability'][1])

    scheduler_checks = [
        ('5', 'a TTScheduler on the TT single-spec KV config: no Mamba layers, unitary coordinator', check_5),
        ('6', 'get_computed_blocks returns hits', check_6),
        ('7', 'same-step rule: two same-step arrivals sharing a prefix', check_7),
        ('8', 'cap, positive control: a turn containing the previous answer hits only full-chunk blocks', check_8),
        ('9', 'decode drive: 2k+ decode tokens publish no hash beyond floor2048(P) (F6)', check_9),
        ('10', 'allocation failure / budget break after a grant: nothing committed, a fresh grant next (F2, S6)', check_10),
        ('11', 'abort while waiting, and abort after admission, clear plan, grant and pin (S6)', check_11),
        ('12', 'eviction coupling: evicting a boundary block drops its checkpoint (F8)', check_12),
        ('13', 'fail-closed salt: unsalted gets no hit and publishes nothing; salted does (S4)', check_13),
        ('14', 'install assertions refuse async, chunked prefill, a hybrid coordinator (F5, F6)', check_14),
    ]
    for check, title, function in scheduler_checks:
        if env is None:
            record(check, False, title, 'not run: no scheduler environment')
        else:
            run_check(check, title, function, env)
    run_check('15', 'extra: registry LRU by bytes never evicts a pinned checkpoint', check_15)
    for check, title, function in (('16', 'extra: kill switch polls at most once a second and latches', check_16),
                                   ('17', 'extra: reset_prefix_cache clears the registry', check_17)):
        if env is None:
            record(check, False, title, 'not run: no scheduler environment')
        else:
            run_check(check, title, function, env)

    if logs:
        say('INFO graft markers (%d; first 12):' % len(logs))
        for line in logs[0:12]:
            say('          ' + line)
    failed = [check for check, ok, _, _ in RESULTS if not ok]
    say('SUMMARY %d PASS %d FAIL%s' % (len(RESULTS) - len(failed), len(failed),
                                       (' (' + ' '.join(failed) + ')') if failed else ''))
    say('VERDICT %s' % ('PASS' if not failed else 'FAIL'))
    return 1 if failed else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--variant', help='build one engine config variant and print its evidence (internal)')
    options = parser.parse_args()
    if options.variant:
        sys.exit(variant_main(options.variant))
    sys.exit(main())
