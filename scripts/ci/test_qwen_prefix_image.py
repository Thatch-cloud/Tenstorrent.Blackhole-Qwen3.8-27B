"""G1 of the TT prefix-reuse design, the image track: the general-prefix profiles, the contract's
prefix guard, the build's prefix stage (qwen_prefix_stage.py) and its plumbing into the C2 image.

Reads the checkout (the Dockerfile, the overlay manifest, the workflows), so it is not overlaid into
the image; test_qwen_prefix_metrics is. The pins are held to the plugin and model sources where a
local copy exists (QWEN_TT_PLUGIN_CHECKOUT, QWEN_IMG_TREE, or the job directory's), else skipped - and,
always, to what the repository records elsewhere about the same files (PinCorroborationTests).
"""

import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_overlay  # noqa: E402
import prefix_scheduler_graft as graft  # noqa: E402
import qwen_prefix_stage as stage  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

PROFILES = HERE / 'qwen_c2_profiles.json'
DOCKERFILE = ROOT / 'docker' / 'qwen-c2-serving.Dockerfile'
MANIFEST = ROOT / 'docker' / 'qwen-c2-overlay.txt'
CPU_WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml'
PREFIX_PROFILES = ('general-prefix', 'general-prefix-eager')
JOB = Path('C:/Users/liamb/.claude/jobs/8376c877/tmp')
PLUGIN_CHECKOUT = Path(os.environ.get('QWEN_TT_PLUGIN_CHECKOUT') or JOB / 'risks' / 'vllm-tt-plugin')
IMG_TREE = Path(os.environ.get('QWEN_IMG_TREE') or JOB / 'matmul-attr' / 'img')
PLUGIN_COMMIT = 'bf77cd63756fc891b8fb7f7cb3f5c1420f0e044c'


def profiles():
    with open(str(PROFILES), encoding='utf-8') as handle:
        return json.load(handle)


def load(name):
    return contract.load_profile(str(PROFILES), name)


def code_lines(path):
    return '\n'.join(line for line in path.read_text(encoding='utf-8').splitlines()
                     if not line.lstrip().startswith('#'))


def manifest_sources():
    return {entry.source for entry in c2_overlay.read_manifest(MANIFEST)}


def p0a_general_prefix_engine(general):
    """The engine the P0a probe built and passed (prefix_p0a_probe.variant_argv, variant 'general-prefix'):
    general's, the four prefix and chunking flags removed, then enable-prefix-caching and
    enable-chunked-prefill appended."""
    engine = dict(general['engine'])
    for flag in ('no-enable-prefix-caching', 'enable-prefix-caching', 'no-enable-chunked-prefill',
                 'enable-chunked-prefill'):
        engine.pop(flag, None)
    engine['enable-prefix-caching'] = True
    engine['enable-chunked-prefill'] = True
    return engine


def local_imports(source):
    """Top-level names of every absolute import in a python source, wherever it sits in the file."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split('.')[0])
    return names


def script_closure(modules, directory=HERE):
    """The scripts/ci modules `modules` are, plus every scripts/ci module they import, transitively (by
    import statement; qwen_prefix_stage records what a stage really loaded at build, and provenance (f)
    holds that record to the overlay)."""
    seen, queue = set(), list(modules)
    while queue:
        name = queue.pop()
        path = Path(directory) / (name + '.py')
        if name in seen or not path.is_file():
            continue
        seen.add(name)
        queue.extend(local_imports(path.read_text(encoding='utf-8')))
    return sorted(seen)


def g1_modules(stages=None, directory=HERE):
    """The G1 modules the image must carry at this commit: every stage module, every qwen_prefix_* runtime
    module (not the stage tool, which is a build tool), and the scheduler graft."""
    stages = stage.STAGES if stages is None else stages
    names = {module for _, module, _ in stages} | {'prefix_scheduler_graft'}
    names |= {path.stem for path in Path(directory).glob('qwen_prefix_*.py') if path.stem != 'qwen_prefix_stage'}
    return sorted(names)


def stage_test_problems(stages, directory, workflow_text):
    """A stage module must ship a CPU test, allowlisted, that holds the patched code with the switch off
    to the original's behaviour: scripts/ci/test_<module>.py with a test whose name says switch_off."""
    problems = []
    for module in sorted({module for _, module, _ in stages}):
        test = 'test_' + module
        path = Path(directory) / (test + '.py')
        if not path.is_file():
            problems.append('stage module %s has no scripts/ci/%s.py' % (module, test))
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test')
                   and 'switch_off' in node.name for node in ast.walk(tree)):
            problems.append('%s.py has no test whose name says switch_off: nothing holds the patched code with '
                            'QWEN_PREFIX_REUSE unset to the original\'s behaviour' % test)
        if not re.search(r'python -B -m unittest [^\n]*\b%s\b' % test, workflow_text):
            problems.append('%s is not allowlisted in .github/workflows/qwen-integration-cpu.yml' % test)
    return problems


def platform_scheduler(profile):
    """A stand-in for the TTScheduler vLLM builds under `profile`, with the engine flags as the TT platform
    leaves them for qwen3_5 (PLG/platform.py:67-105: chunked prefill off again, max-num-batched-tokens
    raised to max-model-len; what P0a checks 1-2 measured in the image) and the single 64-token spec the
    TT worker builds. Returns (scheduler, the fake coordinator module install_problems imports)."""
    engine = profile['engine']
    coordinator = types.ModuleType('vllm.v1.core.kv_cache_coordinator')
    coordinator.UnitaryKVCacheCoordinator = type('UnitaryKVCacheCoordinator', (), {})
    context = engine['max-model-len']
    speculative = engine.get('speculative-config') or {}
    scheduler = types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(
            async_scheduling=bool(engine.get('async-scheduling')) and not engine.get('no-async-scheduling'),
            enable_chunked_prefill=False, max_num_batched_tokens=max(engine['max-num-batched-tokens'], context)),
        cache_config=types.SimpleNamespace(enable_prefix_caching=engine.get('enable-prefix-caching') is True),
        max_model_len=context,
        kv_cache_manager=types.SimpleNamespace(coordinator=coordinator.UnitaryKVCacheCoordinator()),
        kv_cache_config=types.SimpleNamespace(kv_cache_groups=[types.SimpleNamespace(
            kv_cache_spec=types.SimpleNamespace(block_size=engine['block-size']))]),
        block_size=engine['block-size'], has_mamba_layers=False, connector=None,
        num_lookahead_tokens=int(speculative.get('num_speculative_tokens', 0)))
    return scheduler, coordinator


def boot_api_server(name, environ_extra=None, platform_argv=()):
    """contract.boot as the .pth hook runs it in the API server under profile `name`, with the process state
    it touches restored afterwards. Returns (profile or the exception, environ, log lines, the post-import
    hooks it left, the launched argv)."""
    environ = {'QWEN_C2_SERVING': '1', 'QWEN_C2_PROFILES': str(PROFILES)}
    environ.update(environ_extra or {})
    logged, metrics_calls = [], []
    saved = list(sys.argv), list(sys.meta_path), list(sys.path)
    try:
        with mock.patch.object(contract, 'install_prefix_metrics', lambda api_server: metrics_calls.append(api_server)), \
                mock.patch.object(contract, 'install_teardown_skip', lambda: None), \
                mock.patch.object(contract, 'resolve_snapshot', lambda profile: profile['snapshots'][0]), \
                mock.patch.object(contract, 'log', lambda *a: logged.append(a[0] % a[1:] if len(a) > 1 else a[0])), \
                mock.patch.dict(os.environ, {'QWEN_C2_PROFILE': name}):
            sys.argv[:] = ['-m', '--port', '8001'] + list(platform_argv)
            try:
                result = contract.boot(environ=environ, orig_argv=['python3', '-m', contract.API_SERVER])
            except ValueError as error:
                result = error
            hooks = [hook for hook in sys.meta_path if isinstance(hook, contract.PostImportHook)]
            launched = list(sys.argv[1:])
    finally:
        sys.argv[:], sys.meta_path[:], sys.path[:] = saved
    return result, environ, logged + ['metrics api_server=%s' % call for call in metrics_calls], hooks, launched


class ProfileTests(unittest.TestCase):
    def test_general_prefix_is_the_argv_p0a_passed(self):
        """P0a's argv plus --prefix-caching-hash-algo sha256, which is vLLM's default (config/cache.py:95), so
        the engine config is the one P0a built; the profile pins it so a platform value cannot weaken it."""
        general, prefix = load('general'), load('general-prefix')
        wanted = dict(p0a_general_prefix_engine(general), **{'prefix-caching-hash-algo': contract.PREFIX_HASH_ALGO})
        self.assertEqual(prefix['engine'], wanted)
        self.assertEqual(contract.engine_arguments(prefix, '/snap'),
                         contract.engine_arguments(dict(general, engine=wanted), '/snap'))
        self.assertEqual(contract.PREFIX_HASH_ALGO, 'sha256')
        self.assertTrue(contract.OWNED_FLAGS['prefix-caching-hash-algo'], 'owned, and takes a value')
        cache = JOB / 'risks' / 'vllm-0.25.1' / 'vllm-0.25.1' / 'vllm' / 'config' / 'cache.py'
        if cache.is_file():
            self.assertIn('prefix_caching_hash_algo: PrefixCachingHashAlgo = "sha256"', cache.read_text(encoding='utf-8'))
        # the P0a probe reads general and rewrites exactly those flags: keep its rule where it is
        probe = (HERE / 'prefix_p0a_probe.py').read_text(encoding='utf-8')
        self.assertIn("profile = contract.load_profile(path, 'general')", probe)
        self.assertIn("engine['enable-prefix-caching'] = True", probe)
        self.assertIn("engine['no-enable-chunked-prefill' if variant == 'no-chunking' else 'enable-chunked-prefill'] = True",
                      probe)

    def test_general_prefix_is_general_plus_the_switch(self):
        general, prefix = load('general'), load('general-prefix')
        self.assertEqual(prefix['env'], dict(general['env'], QWEN_PREFIX_REUSE='1', QWEN_PREFIX_STORE_GIB='8'))
        for key in ('eos_ids', 'snapshots', 'mesh_graph_descriptor', 'request_contract', 'drop_batched_decode_mode'):
            self.assertEqual(prefix.get(key), general.get(key), key)
        self.assertEqual(sorted(set(prefix) - set(general)), [])
        self.assertNotIn('gate_only', prefix)
        # general's default trace mode ("all", PLG/worker.py:198): prefill takes the traced chunk loop
        self.assertNotIn('trace_mode', prefix['engine']['additional-config']['tt'])
        self.assertEqual(float(prefix['env']['QWEN_PREFIX_STORE_GIB']), graft.DEFAULT_STORE_GIB)

    def test_the_eager_gate_profile_is_general_prefix_in_decode_only_trace_mode(self):
        prefix, eager = load('general-prefix'), load('general-prefix-eager')
        self.assertEqual(eager['env'], prefix['env'])
        engine = json.loads(json.dumps(eager['engine']))
        self.assertEqual(engine['additional-config']['tt'].pop('trace_mode'), 'decode_only')
        self.assertEqual(engine, prefix['engine'])
        self.assertTrue(eager['description'].startswith('GATE ONLY'))
        self.assertIs(eager['gate_only'], True)
        self.assertEqual(sorted(name for name, profile in profiles()['profiles'].items() if profile.get('gate_only')),
                         ['general-prefix-eager'])

    def test_the_prefix_profiles_launch_with_prefix_caching_and_chunking_on(self):
        for name in PREFIX_PROFILES:
            with self.subTest(profile=name):
                argv = contract.engine_arguments(load(name), '/snap')
                for flag in ('--enable-prefix-caching', '--enable-chunked-prefill', '--no-async-scheduling'):
                    self.assertIn(flag, argv)
                for flag in ('--no-enable-prefix-caching', '--no-enable-chunked-prefill', '--speculative-config'):
                    self.assertNotIn(flag, argv)
                self.assertEqual(argv[argv.index('--block-size') + 1], '64')
                self.assertEqual(argv[argv.index('--prefix-caching-hash-algo') + 1], 'sha256')
                self.assertTrue(contract.prefix_reuse(load(name)))

    def test_every_other_profile_is_untouched(self):
        """exact, c2, c2-gate, coding and general: no switch, prefix caching off in the argv, and the
        guard has nothing to say - their boots are what they were."""
        for name in sorted(set(profiles()['profiles']) - set(PREFIX_PROFILES)):
            with self.subTest(profile=name):
                profile = load(name)
                self.assertNotIn(contract.PREFIX_SWITCH, profile['env'])
                self.assertFalse(contract.prefix_reuse(profile))
                self.assertEqual(contract.prefix_reuse_problems(profile), [])
                self.assertIn('--no-enable-prefix-caching', contract.engine_arguments(profile, '/snap'))
                self.assertNotIn('--prefix-caching-hash-algo', contract.engine_arguments(profile, '/snap'))
                self.assertNotIn('gate_only', profile)
        self.assertEqual(profiles()['default'], 'general', 'general-prefix becomes the default only at release')

    def test_the_prefix_profiles_pass_the_guard_and_the_scheduler_graft_s_install_rules(self):
        """prefix_scheduler_graft.install_problems itself, on the scheduler vLLM builds under each prefix
        profile as the TT platform rewrites it - and it refuses the same scheduler with a flag broken."""
        for name in PREFIX_PROFILES:
            with self.subTest(profile=name):
                profile = load(name)
                self.assertEqual(contract.prefix_reuse_problems(profile), [])
                scheduler, coordinator = platform_scheduler(profile)
                with mock.patch.dict(sys.modules, {coordinator.__name__: coordinator}):
                    self.assertEqual(graft.install_problems(scheduler), [])
                self.assertEqual(profile['engine']['max-model-len'] % graft.CHUNK, 0)
        broken = json.loads(json.dumps(load('general-prefix')))
        broken['engine'].update({'no-async-scheduling': False, 'async-scheduling': True, 'block-size': 128,
                                 'enable-prefix-caching': False,
                                 'speculative-config': {'method': 'dflash', 'num_speculative_tokens': 4}})
        scheduler, coordinator = platform_scheduler(broken)
        with mock.patch.dict(sys.modules, {coordinator.__name__: coordinator}):
            problems = graft.install_problems(scheduler)
        for words in ('async scheduling is on', 'prefix caching is off', 'KV block size 128', 'lookahead'):
            self.assertTrue(any(words in problem for problem in problems), (words, problems))


class SaltTests(unittest.TestCase):
    """A cache_salt partitions the prefix cache: the API server keeps only one the platform minted."""

    KEY = b'k' * 32

    def test_verdicts(self):
        salt = contract.mint_salt(self.KEY, 'tenant_0123-abc')
        self.assertRegex(salt, r'^qps1\.tenant_0123-abc\.[0-9a-f]{64}$')
        cases = ((salt, self.KEY, 'verified'), (salt, b'j' * 32, 'dropped-unverified'),
                 (salt.replace('tenant_0123', 'tenant_0124'), self.KEY, 'dropped-unverified'),
                 (salt[:-1] + ('0' if salt[-1] != '0' else '1'), self.KEY, 'dropped-unverified'),
                 (salt.upper(), self.KEY, 'dropped-unverified'), ('my-sdk-constant', self.KEY, 'dropped-unverified'),
                 ('qps1.short.' + '0' * 64, self.KEY, 'dropped-unverified'), (7, self.KEY, 'dropped-unverified'),
                 (salt, None, 'dropped-no-key'), (None, self.KEY, 'unset'), ('', None, 'unset'))
        for value, key, verdict in cases:
            with self.subTest(salt=value, key=key):
                self.assertEqual(contract.salt_verdict(value, key), verdict)
        for tag in ('short', 'has.dot', 'x' * 129, None):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                contract.mint_salt(self.KEY, tag)

    def test_the_key(self):
        tmp = Path(tempfile.mkdtemp(prefix='qwen-prefix-salt-'))
        try:
            path = tmp / 'key'
            path.write_bytes(b'x' * 40 + b'\n')
            self.assertEqual(contract.read_salt_key({contract.SALT_KEY_ENV: str(path)}), (b'x' * 40, str(path)))
            path.write_bytes(b'x' * 31)
            key, why = contract.read_salt_key({contract.SALT_KEY_ENV: str(path)})
            self.assertIsNone(key)
            self.assertIn('fewer than 32', why)
            key, why = contract.read_salt_key({contract.SALT_KEY_ENV: str(tmp / 'absent')})
            self.assertIsNone(key)
            self.assertIn(str(tmp / 'absent'), why)
        finally:
            shutil.rmtree(str(tmp), ignore_errors=True)
        self.assertEqual(contract.SALT_KEY_FILE, '/models/.qwen-c2/prefix-salt.key')

    def test_the_policy_on_every_request(self):
        class InputProcessor(object):
            def process_inputs(self, request_id, prompt, params, **kwargs):
                return types.SimpleNamespace(request_id=request_id, cache_salt=prompt.get('cache_salt'))

        module = types.ModuleType('vllm.v1.engine.input_processor')
        module.InputProcessor = InputProcessor
        verdicts = []
        with mock.patch.object(contract, 'log', lambda *a: None):
            self.assertTrue(contract.install_salt_policy(module, self.KEY, verdicts.append))
            self.assertFalse(contract.install_salt_policy(module, self.KEY, verdicts.append))
        good = contract.mint_salt(self.KEY, 'tenant_aaaa')
        processor = InputProcessor()
        self.assertEqual(processor.process_inputs('a', {'cache_salt': good}, None).cache_salt, good)
        self.assertIsNone(processor.process_inputs('b', {'cache_salt': 'sdk-default'}, None).cache_salt)
        self.assertIsNone(processor.process_inputs('c', {}, None).cache_salt)
        self.assertEqual(verdicts, ['verified', 'dropped-unverified', 'unset'])

        def exploding(verdict):
            raise RuntimeError('metrics down')

        class Fresh(object):
            def process_inputs(self, request_id, prompt, params, **kwargs):
                return types.SimpleNamespace(request_id=request_id, cache_salt=prompt.get('cache_salt'))

        module.InputProcessor = Fresh
        with mock.patch.object(contract, 'log', lambda *a: None):
            contract.install_salt_policy(module, None, exploding)
        self.assertIsNone(module.InputProcessor().process_inputs('d', {'cache_salt': good}, None).cache_salt,
                          'no key: every salt is dropped, and a failing counter never fails the request')

    def test_the_boot_hooks_the_policy_with_the_key_it_read(self):
        tmp = Path(tempfile.mkdtemp(prefix='qwen-prefix-salt-'))
        try:
            (tmp / 'key').write_bytes(self.KEY)
            _, _, logged, hooks, _ = boot_api_server('general-prefix', {contract.SALT_KEY_ENV: str(tmp / 'key')})
        finally:
            shutil.rmtree(str(tmp), ignore_errors=True)
        self.assertTrue(any(line.endswith('salt key %s' % (tmp / 'key')) for line in logged), logged)
        hook = [hook for hook in hooks if hook.name == contract.INPUT_PROCESSOR][0]
        module = types.ModuleType(contract.INPUT_PROCESSOR)
        module.InputProcessor = type('InputProcessor', (object,), {
            'process_inputs': lambda self, *a, **k: types.SimpleNamespace(cache_salt=a[1])})
        with mock.patch.object(contract, 'log', lambda *a: None):
            hook.callback(module)
        self.assertEqual(module.InputProcessor().process_inputs('r', contract.mint_salt(self.KEY, 'tenant_bbbb')).cache_salt,
                         contract.mint_salt(self.KEY, 'tenant_bbbb'))
        self.assertIsNone(module.InputProcessor().process_inputs('r', 'client-chosen').cache_salt)


class GuardTests(unittest.TestCase):
    def broken(self, **changes):
        profile = json.loads(json.dumps(load('general-prefix')))
        for key, value in changes.items():
            if value is None:
                profile['engine'].pop(key, None)
            else:
                profile['engine'][key] = value
        return profile

    def test_each_rule(self):
        cases = (
            (dict(**{'enable-prefix-caching': None}), 'needs enable-prefix-caching'),
            (dict(**{'no-enable-prefix-caching': True}), 'both enable-prefix-caching and no-enable-prefix-caching'),
            (dict(**{'enable-chunked-prefill': None}), 'enable-chunked-prefill is required'),
            (dict(**{'no-enable-chunked-prefill': True}), 'enable-chunked-prefill is required'),
            (dict(**{'no-async-scheduling': None}), 'no-async-scheduling is required'),
            (dict(**{'block-size': 128}), 'block-size must be 64'),
            (dict(**{'max-num-batched-tokens': 2048}), 'must cover max-model-len'),
            (dict(**{'additional-config': {'qwen_fast_t16': True}}), 'cannot admit a prefix hit yet'),
            (dict(**{'speculative-config': {'method': 'dflash'}}), 'refuses lookahead'),
            (dict(**{'prefix-caching-hash-algo': 'xxhash'}), 'prefix-caching-hash-algo must be sha256'),
            (dict(**{'prefix-caching-hash-algo': None}), 'prefix-caching-hash-algo must be sha256'),
        )
        for changes, words in cases:
            with self.subTest(changes=changes):
                problems = contract.prefix_reuse_problems(self.broken(**changes))
                self.assertTrue(any(words in problem for problem in problems), problems)

    def test_prefix_caching_without_the_switch_and_a_bad_switch(self):
        general = json.loads(json.dumps(load('general')))
        general['engine'].pop('no-enable-prefix-caching')
        general['engine']['enable-prefix-caching'] = True
        self.assertIn('without QWEN_PREFIX_REUSE=1', contract.prefix_reuse_problems(general)[0])
        odd = json.loads(json.dumps(load('general-prefix')))
        odd['env'][contract.PREFIX_SWITCH] = 'yes'
        self.assertTrue(any('neither 1 nor 0' in problem for problem in contract.prefix_reuse_problems(odd)))

    def test_boot_refuses_a_broken_prefix_profile_before_it_touches_anything(self):
        path = Path(tempfile.mkdtemp(prefix='qwen-prefix-profiles-'))
        try:
            data = profiles()
            data['profiles']['broken'] = self.broken(**{'no-async-scheduling': None})
            (path / 'profiles.json').write_text(json.dumps(data), encoding='utf-8')
            environ = {'QWEN_C2_SERVING': '1', 'QWEN_C2_PROFILES': str(path / 'profiles.json')}
            sys_path = list(sys.path)
            try:
                with mock.patch.dict(os.environ, {'QWEN_C2_PROFILE': 'broken'}):
                    with self.assertRaisesRegex(ValueError, 'profile broken cannot serve prefix reuse exactly'):
                        contract.boot(environ=environ, orig_argv=['python3', '-m', contract.API_SERVER])
            finally:
                sys.path[:] = sys_path
            self.assertNotIn(contract.PREFIX_SWITCH, environ)
        finally:
            shutil.rmtree(str(path), ignore_errors=True)

    def test_only_the_profile_turns_the_switch_on(self):
        environ = contract.apply_environment(load('general'), {contract.PREFIX_SWITCH: '1'})
        self.assertNotIn(contract.PREFIX_SWITCH, environ)
        environ = contract.apply_environment(load('general-prefix'), {'QWEN36_BATCHED_DECODE_MODE': 'host'})
        self.assertEqual(environ[contract.PREFIX_SWITCH], '1')
        self.assertEqual(environ['QWEN36_BATCHED_DECODE_MODE'], 'host', "general's stock decode keeps it")
        for name in ('exact', 'c2'):
            self.assertNotIn(contract.PREFIX_SWITCH, contract.apply_environment(load(name), {contract.PREFIX_SWITCH: '1'}))

    def test_a_platform_value_for_the_hash_algorithm_is_replaced(self):
        platform = ['-m', '--port', '8001', '--prefix-caching-hash-algo', 'xxhash', '--prefix_caching_hash_algo=xxhash_cbor']
        argv = contract.rewrite_argv(platform, load('general-prefix'), '/snap')
        self.assertEqual(contract.launched_values(argv, 'prefix-caching-hash-algo'), ['sha256'])
        # under a profile without prefix caching the platform's value goes too, and nothing replaces it
        self.assertEqual(contract.launched_values(contract.rewrite_argv(platform, load('general'), '/s'),
                                                  'prefix-caching-hash-algo'), [])

    def test_launch_flags_the_contract_does_not_own(self):
        self.assertEqual(contract.launched_values(['--a', '1', '--b=2', '--a_x', '3', '--a'], 'a'), ['1', ''])
        self.assertEqual(contract.launched_values(['--kv_transfer_config={}'], 'kv-transfer-config'), ['{}'])
        refusals, warnings = contract.prefix_launch_problems(
            ['--kv-transfer-config', '{"kv_connector": "x"}',
             '--default-chat-template-kwargs', '{"enable_thinking": false, "preserve_thinking": false}'])
        self.assertEqual(len(refusals), 1)
        self.assertIn('refuses a KV connector', refusals[0])
        self.assertEqual(len(warnings), 1)
        self.assertIn('preserve_thinking=false', warnings[0])
        for kwargs, count in (('{"enable_thinking": false, "preserve_thinking": true}', 0), ('{}', 0),
                              ('{"preserve_thinking": "true"}', 1), ('{"preserve_thinking": null}', 1), ('nope', 1)):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(len(contract.prefix_launch_problems(['--default-chat-template-kwargs', kwargs])[1]),
                                 count)
        self.assertEqual(contract.prefix_launch_problems(['--port', '8001']), ([], []))

    def test_the_api_server_boot_under_a_prefix_profile(self):
        """The launched argv pins the hash, the salt policy and the model-tree log are hooked, and a launched
        KV connector refuses the boot; a stripping chat-template default is only warned about."""
        profile, environ, logged, hooks, launched = boot_api_server('general-prefix')
        self.assertEqual(profile['name'], 'general-prefix')
        self.assertEqual(contract.launched_values(launched, 'prefix-caching-hash-algo'), ['sha256'])
        self.assertEqual(sorted(hook.name for hook in hooks), sorted([contract.INPUT_PROCESSOR, contract.MODEL_ENTRY]))
        self.assertIn('metrics api_server=True', logged)
        self.assertTrue(any('salt key' in line for line in logged), logged)
        refused, _, _, _, _ = boot_api_server('general-prefix', platform_argv=['--kv-transfer-config', '{}'])
        self.assertIsInstance(refused, ValueError)
        self.assertIn('cannot serve prefix reuse with this launch: --kv-transfer-config', str(refused))
        profile, _, logged, _, _ = boot_api_server(
            'general-prefix', platform_argv=['--default-chat-template-kwargs', '{"preserve_thinking":false}'])
        self.assertEqual(profile['name'], 'general-prefix')
        self.assertTrue(any(line.startswith('prefix: WARNING --default-chat-template-kwargs sets preserve_thinking=false')
                            for line in logged), logged)

    def test_the_other_profiles_boot_as_before(self):
        """No prefix hook, no salt policy, no metrics and no launch check outside the prefix profiles."""
        for name in sorted(set(profiles()['profiles']) - set(PREFIX_PROFILES)):
            with self.subTest(profile=name):
                profile, environ, logged, hooks, launched = boot_api_server(
                    name, platform_argv=['--kv-transfer-config', '{}'])
                self.assertEqual(profile['name'], name)
                # only the request contract's hook, where the profile has one - as before G1
                self.assertEqual([hook.name for hook in hooks],
                                 [contract.INPUT_PROCESSOR] if profile.get('request_contract', True) else [])
                self.assertFalse(any(line.startswith(('prefix:', 'metrics')) or 'prefix reuse' in line or 'salt' in line
                                     for line in logged), logged)
                self.assertNotIn(contract.PREFIX_SWITCH, environ)

    def test_a_gate_only_profile_boots_only_in_a_gate(self):
        refused, environ, _, hooks, _ = boot_api_server('general-prefix-eager')
        self.assertIsInstance(refused, ValueError)
        self.assertIn('profile general-prefix-eager is gate only', str(refused))
        self.assertNotIn(contract.PREFIX_SWITCH, environ, 'refused before the environment is applied')
        self.assertEqual(hooks, [])
        profile, environ, _, _, _ = boot_api_server('general-prefix-eager', {contract.GATE_SWITCH: '1'})
        self.assertEqual(profile['name'], 'general-prefix-eager')
        self.assertEqual(environ[contract.PREFIX_SWITCH], '1')
        self.assertEqual(contract.gate_problems(load('general-prefix'), {}), [])

    def test_the_model_tree_line_is_the_bring_up_check_s(self):
        module = types.ModuleType(contract.MODEL_ENTRY)
        module.__file__ = stage.MODEL_ROOT + '/qwen36_vllm.py'
        err = io.StringIO()
        with mock.patch.object(sys, 'stderr', err), mock.patch.object(contract.os.path, 'realpath', lambda path: path):
            contract.log_model_tree(module)
            contract.log_model_tree(object())
        self.assertEqual(stage.MODEL_LINE.findall(err.getvalue()), [module.__file__, '?'])

    def test_the_metrics_install_never_raises(self):
        logged = []
        with mock.patch.dict(sys.modules, {'qwen_prefix_metrics': None}), \
                mock.patch.object(contract, 'log', lambda *a: logged.append(a[0] % a[1:])):
            self.assertFalse(contract.install_prefix_metrics(True))
        self.assertTrue(logged and logged[0].startswith('prefix metrics not installed'))


def stage_tree(root, targets):
    """A staging tree standing in for the image: each target holds `text` at its path."""
    rows = []
    for name, text in targets:
        path = '/opt/fake/%s' % name.replace('/', '_')
        real = Path(root) / path.lstrip('/')
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_bytes(text.encode('utf-8'))
        rows.append((name, path, hashlib.sha256(text.encode('utf-8')).hexdigest(), 'test pin'))
    return tuple(rows)


STAGE_MODULE = '''
import fake_prefix_helper


def patch_a(source):
    if 'ANCHOR_A' not in source:
        raise ValueError('anchor ANCHOR_A missing')
    return source.replace('ANCHOR_A', 'PATCHED_A')


def patch_b(source):
    return source + 'PATCHED_B = 1\\n'


def patch_nothing(source):
    return source


def patch_broken(source):
    return source + 'def (:\\n'
'''
HELPER_MODULE = 'HELPER = 1\n'


class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='qwen-prefix-stage-'))
        self.modules = self.tmp / 'modules'
        self.modules.mkdir()
        (self.modules / 'fake_prefix_stage.py').write_text(STAGE_MODULE, encoding='utf-8')
        (self.modules / 'fake_prefix_helper.py').write_text(HELPER_MODULE, encoding='utf-8')
        self.image = self.tmp / 'image'
        self.targets = stage_tree(self.image, (('t/a.py', 'ANCHOR_A = 1\n'), ('t/b.py', 'B = 2\n')))
        self.record = str(self.tmp / 'record.json')
        self.logs = []

    def tearDown(self):
        sys.modules.pop('fake_prefix_stage', None)
        sys.modules.pop('fake_prefix_helper', None)
        if str(self.modules) in sys.path:
            sys.path.remove(str(self.modules))
        shutil.rmtree(str(self.tmp), ignore_errors=True)

    def apply(self, stages, targets=None):
        with mock.patch.object(stage, 'log', self.logs.append):
            return stage.apply(str(self.modules), self.record, root=str(self.image), targets=targets or self.targets,
                               stages=stages, check_resolution=False)

    def read(self, name):
        path = dict((row[0], row[1]) for row in self.targets)[name]
        return (self.image / path.lstrip('/')).read_text(encoding='utf-8')

    def test_the_stages_patch_and_record(self):
        stages = (('t/a.py', 'fake_prefix_stage', 'patch_a'), ('t/b.py', 'fake_prefix_stage', 'patch_b'),
                  ('t/a.py', 'fake_prefix_stage', 'patch_b'))
        result = self.apply(stages)
        self.assertEqual(self.read('t/a.py'), 'PATCHED_A = 1\nPATCHED_B = 1\n')
        self.assertEqual(self.read('t/b.py'), 'B = 2\nPATCHED_B = 1\n')
        self.assertTrue(result['complete'])
        row = result['targets']['t/a.py']
        self.assertEqual(row['patched_by'], ['fake_prefix_stage.patch_a', 'fake_prefix_stage.patch_b'])
        self.assertEqual(row['before'], self.targets[0][2])
        self.assertEqual(row['after'], hashlib.sha256(b'PATCHED_A = 1\nPATCHED_B = 1\n').hexdigest())
        module_sha = hashlib.sha256((self.modules / 'fake_prefix_stage.py').read_bytes()).hexdigest()
        self.assertEqual({entry['sha256'] for entry in result['stages']}, {module_sha})
        # the stage module and what it imported from the overlaid tree, for provenance (f) to hold to the overlay
        self.assertEqual(result['imported'], [
            dict(module='fake_prefix_helper', path=str(self.modules) + '/fake_prefix_helper.py',
                 sha256=hashlib.sha256((self.modules / 'fake_prefix_helper.py').read_bytes()).hexdigest()),
            dict(module='fake_prefix_stage', path=str(self.modules) + '/fake_prefix_stage.py', sha256=module_sha)])
        self.assertEqual(result['warnings'], [])
        with open(self.record, encoding='utf-8') as handle:
            self.assertEqual(json.load(handle), json.loads(json.dumps(result)))

    def test_no_stages_checks_the_anchors_and_records_the_originals(self):
        result = self.apply(())
        self.assertFalse(result['complete'])
        self.assertEqual([row['after'] == row['before'] == row['pin'] for row in result['targets'].values()],
                         [True, True])
        self.assertEqual((result['imported'], result['warnings']), ([], []))
        self.assertTrue(any('NOT grafted' in line for line in self.logs))

    def test_an_empty_table_records_a_moved_base_instead_of_refusing(self):
        """With nothing to write, a base off its pins (or missing a target) cannot hurt exact, c2 or general:
        the build goes on and the record says what it found."""
        target = self.image / self.targets[1][1].lstrip('/')
        target.write_bytes(b'B = 3\n')
        (self.image / self.targets[0][1].lstrip('/')).unlink()
        result = self.apply(())
        self.assertEqual(target.read_bytes(), b'B = 3\n')
        rows = result['targets']
        self.assertEqual((rows['t/a.py']['before'], rows['t/a.py']['after']), (None, None))
        moved = hashlib.sha256(b'B = 3\n').hexdigest()
        self.assertEqual((rows['t/b.py']['before'], rows['t/b.py']['after']), (moved, moved))
        self.assertEqual(len(result['warnings']), 2, result['warnings'])
        self.assertTrue(any(line.startswith('WARNING (not fatal') for line in self.logs))

    def test_resolution_is_fatal_only_with_stages(self):
        with mock.patch.object(stage, 'log', self.logs.append):
            result = stage.apply(str(self.modules), self.record, root=str(self.image), targets=self.targets, stages=(),
                                 find_spec=lambda name: None, check_resolution=True)
            self.assertEqual(len(result['warnings']), 2)
            self.assertEqual(result['resolved']['plugin'], None)
            with self.assertRaisesRegex(stage.StageError, 'nothing runs'):
                stage.apply(str(self.modules), self.record, root=str(self.image), targets=self.targets,
                            stages=(('t/a.py', 'fake_prefix_stage', 'patch_a'), ('t/b.py', 'fake_prefix_stage', 'patch_b')),
                            find_spec=lambda name: None, check_resolution=True)

    def test_a_target_off_its_pin_refuses_and_writes_nothing(self):
        (self.image / self.targets[1][1].lstrip('/')).write_text('B = 3\n', encoding='utf-8')
        with self.assertRaisesRegex(stage.StageError, 'the anchors .nothing was written.: t/b.py'):
            self.apply((('t/a.py', 'fake_prefix_stage', 'patch_a'), ('t/b.py', 'fake_prefix_stage', 'patch_b')))
        self.assertEqual(self.read('t/a.py'), 'ANCHOR_A = 1\n')
        self.assertFalse(os.path.exists(self.record))

    def test_a_stage_that_misses_or_breaks_refuses_and_writes_nothing(self):
        for function, words in (('patch_nothing', 'left t/b.py unchanged'), ('patch_broken', 'uncompilable'),
                                ('patch_absent', 'has no function patch_absent')):
            with self.subTest(function=function):
                with self.assertRaisesRegex(stage.StageError, words):
                    self.apply((('t/a.py', 'fake_prefix_stage', 'patch_a'), ('t/b.py', 'fake_prefix_stage', function)))
                self.assertEqual(self.read('t/a.py'), 'ANCHOR_A = 1\n')
        (self.image / self.targets[0][1].lstrip('/')).write_text('A = 1\n', encoding='utf-8')
        targets = stage_tree(self.image, (('t/a.py', 'A = 1\n'), ('t/b.py', 'B = 2\n')))
        with self.assertRaisesRegex(stage.StageError, 'refused t/a.py: ValueError: anchor ANCHOR_A missing'):
            self.apply((('t/a.py', 'fake_prefix_stage', 'patch_a'), ('t/b.py', 'fake_prefix_stage', 'patch_b')),
                       targets)

    def test_a_missing_stage_module_names_the_manifest(self):
        with self.assertRaisesRegex(stage.StageError, 'add scripts/ci/absent_stage.py to docker/qwen-c2-overlay.txt'):
            self.apply((('t/a.py', 'absent_stage', 'patch_a'), ('t/b.py', 'absent_stage', 'patch_b')))

    def test_the_table_is_all_or_nothing(self):
        problems = stage.table_problems(self.targets, (('t/a.py', 'fake_prefix_stage', 'patch_a'),))
        self.assertTrue(any('prefix reuse is all or nothing' in problem for problem in problems), problems)
        with self.assertRaisesRegex(stage.StageError, 'all or nothing'):
            self.apply((('t/a.py', 'fake_prefix_stage', 'patch_a'),))
        self.assertTrue(stage.table_problems(self.targets, (('t/zzz.py', 'm', 'f'), ('t/a.py', 'm', 'f'),
                                                            ('t/b.py', 'm', 'f'))))
        self.assertTrue(stage.table_problems(self.targets, (('t/a.py', 'dir/m.py', 'f'), ('t/b.py', 'm', 'f'))))
        self.assertTrue(stage.table_problems(self.targets, (('t/a.py', 'm'),)))
        self.assertTrue(stage.table_problems(self.targets, (('t/a.py', 'm', 'f'), ('t/a.py', 'm', 'f'),
                                                            ('t/b.py', 'm', 'f'))))
        self.assertEqual(stage.table_problems(self.targets, ()), [])

    def test_the_checkout_s_table(self):
        self.assertEqual(stage.table_problems(), [])
        self.assertEqual(stage.target_names(), ['plugin/scheduler.py', 'plugin/model_runner.py', 'plugin/worker.py',
                                                'model/model.py', 'model/qwen36_vllm.py'])
        for name, path, pin, _ in stage.TARGETS:
            with self.subTest(target=name):
                self.assertRegex(pin, '^[0-9a-f]{64}$')
                root = stage.PLUGIN_ROOT if name.startswith('plugin/') else stage.MODEL_ROOT
                self.assertEqual(path, root + '/' + name.split('/', 1)[1])

    def test_resolution(self):
        def spec(*locations):
            return types.SimpleNamespace(submodule_search_locations=list(locations), origin=None)

        models = stage.MODEL_TREE + '/models'
        holds = {models + '/' + stage.QWEN_ENTRY}
        isfile = lambda path: path.replace(os.sep, '/') in holds  # noqa: E731
        good = {stage.PLUGIN_PACKAGE: spec(stage.PLUGIN_ROOT), stage.MODEL_PACKAGE: spec(models)}
        with mock.patch.object(stage.os.path, 'realpath', lambda path: path), \
                mock.patch.object(stage.os.path, 'join', lambda *parts: '/'.join(parts)):
            problems, found = stage.resolution_problems(good.get, isfile)
            self.assertEqual(problems, [])
            self.assertEqual((found['plugin'], found['models']), (stage.PLUGIN_ROOT, models))
            # a namespace `models` spanning another tree first still imports qwen36 from /opt/tt-metal
            spanning = dict(good, **{stage.MODEL_PACKAGE: spec('/experiment-scripts/ci/models', models)})
            self.assertEqual(stage.resolution_problems(spanning.get, isfile)[0], [])
            other = dict(good, **{stage.PLUGIN_PACKAGE: spec('/opt/vllm-tt-plugin/src/vllm_tt_plugin')})
            problems, _ = stage.resolution_problems(other.get, isfile)
            self.assertTrue(problems and 'nothing runs' in problems[0], problems)
            elsewhere = dict(good, **{stage.MODEL_PACKAGE: spec('/usr/lib/python3/dist-packages/models')})
            problems, _ = stage.resolution_problems(elsewhere.get, isfile)
            self.assertTrue(problems and 'model tree nothing runs' in problems[0], problems)
            problems, _ = stage.resolution_problems({}.get, isfile)
            self.assertEqual(len(problems), 2)

            def raising(name):
                raise ImportError(name)

            self.assertIsNone(stage.package_directory('x', raising))
            module = types.SimpleNamespace(submodule_search_locations=None, origin='/a/b/mod.py')
            self.assertEqual(stage.package_locations('mod', {'mod': module}.get), ['/a/b'])

    def test_the_record_check(self):
        """record_problems is what c2_image_provenance (f) runs on the built image."""
        rows = {name: dict(path=path, pin=pin, before=pin, after=pin, patched_by=[])
                for name, path, pin, _ in stage.TARGETS}
        record = dict(schema=stage.SCHEMA, targets=rows, stages=[], imported=[], warnings=[], complete=False,
                      resolved=dict(plugin=stage.PLUGIN_ROOT, models=stage.MODEL_TREE + '/models'))
        shas = {path: pin for _, path, pin, _ in stage.TARGETS}
        problems, lines = stage.record_problems(record, shas, {})
        self.assertEqual(problems, [])
        self.assertTrue(lines[0].startswith('(f) prefix-reuse stage: 0 stage(s), no target grafted'))
        self.assertEqual(stage.record_problems(None, shas, {})[0],
                         ['(f) the image has no /opt/qwen-c2/prefix-stage.json: the prefix-reuse stage never ran'])
        model = stage.MODEL_ROOT + '/model.py'
        problems, _ = stage.record_problems(record, dict(shas, **{model: '0' * 64}), {})
        self.assertEqual(len(problems), 1)
        self.assertIn('not the %s the stage wrote' % rows['model/model.py']['after'], problems[0])
        foreign = json.loads(json.dumps(record))
        foreign['targets']['plugin/worker.py'].update(path='/elsewhere/worker.py')
        foreign['stages'] = [dict(module='qwen_prefix_scheduler_patch', sha256='2' * 64)]
        problems, _ = stage.record_problems(foreign, shas, {'qwen_prefix_scheduler_patch': '3' * 64})
        self.assertEqual(len(problems), 2, problems)
        self.assertIn('plugin/worker.py: the record has path', problems[0])
        self.assertIn('ran at %s; the context overlays %s' % ('2' * 64, '3' * 64), problems[1])

    def test_the_record_check_holds_every_module_the_stages_ran_to_the_overlay(self):
        rows = {name: dict(path=path, pin=pin, before=pin, after=pin, patched_by=[])
                for name, path, pin, _ in stage.TARGETS}
        record = dict(schema=stage.SCHEMA, targets=rows, stages=[], warnings=[], complete=False,
                      resolved=dict(plugin=stage.PLUGIN_ROOT, models=stage.MODEL_TREE + '/models'),
                      imported=[dict(module='lever_n_model_patch', path='/experiment-scripts/ci/lever_n_model_patch.py',
                                     sha256='4' * 64)])
        shas = {path: pin for _, path, pin, _ in stage.TARGETS}
        problems, _ = stage.record_problems(record, shas, {}, {})
        self.assertEqual(len(problems), 1, problems)
        self.assertIn('which the overlay does not lay', problems[0])
        self.assertIn('name scripts/ci/lever_n_model_patch.py in docker/qwen-c2-overlay.txt', problems[0])
        overlaid = {'/experiment-scripts/ci/lever_n_model_patch.py': '5' * 64}
        problems, _ = stage.record_problems(record, shas, {}, overlaid)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn('at %s; the context overlays %s' % ('4' * 64, '5' * 64), problems[0])
        overlaid['/experiment-scripts/ci/lever_n_model_patch.py'] = '4' * 64
        problems, lines = stage.record_problems(record, shas, {}, overlaid)
        self.assertEqual(problems, [])
        self.assertIn('(f) the stages ran /experiment-scripts/ci/lever_n_model_patch.py at %s' % ('4' * 16), lines)

    def test_the_record_check_is_strict_once_the_table_patches(self):
        rows = {name: dict(path=path, pin=pin, before=pin, after=pin, patched_by=[])
                for name, path, pin, _ in stage.TARGETS}
        rows['model/model.py'].update(before='6' * 64, after='6' * 64)
        record = dict(schema=stage.SCHEMA, targets=rows, stages=[], imported=[], complete=False, resolved={},
                      warnings=['model/model.py moved'])
        shas = {path: row['after'] for path, row in ((row['path'], row) for row in rows.values())}
        problems, lines = stage.record_problems(record, shas, {}, {})
        self.assertEqual(problems, [])
        self.assertTrue(any('the build found %s' % ('6' * 64) in line and 'not fatal' in line for line in lines), lines)
        self.assertIn('(f) the prefix stage warned: model/model.py moved', lines)
        table = tuple((name, 'm', 'f') for name in stage.target_names())
        problems, _ = stage.record_problems(record, shas, {}, {}, stages=table)
        self.assertTrue(any('the build found %s' % ('6' * 64) in problem for problem in problems), problems)
        self.assertTrue(any('does not say which plugin and model trees' in problem for problem in problems), problems)
        self.assertTrue(any('complete=False' in problem for problem in problems), problems)
        rows['plugin/worker.py'].update(after='7' * 64)
        problems, _ = stage.record_problems(record, dict(shas, **{rows['plugin/worker.py']['path']: '7' * 64}), {}, {})
        self.assertEqual(len(problems), 1, problems)
        self.assertIn('no stage patched it, yet the stage left %s' % ('7' * 64), problems[0])

    def patched_record(self):
        return dict(complete=True, targets={name: dict(path=path, patched_by=['m.f']) for name, path, _, _ in stage.TARGETS})

    def runner(self, calls, fail=None, elsewhere=None):
        names = {module: name for name, module in stage.TARGET_MODULES.items()}
        paths = {name: path for name, path, _, _ in stage.TARGETS}

        def run(argv, env, cwd):
            calls.append(dict(argv=argv, switch=env.get(stage.PREFIX_SWITCH), cwd=cwd, plugins=env.get('VLLM_PLUGINS')))
            if argv[2] == '-c':
                name = names[argv[-1]]
                if fail == (name, env.get(stage.PREFIX_SWITCH)):
                    return 1, 'Traceback\nImportError: boom\n'
                path = elsewhere if elsewhere and name == 'plugin/scheduler.py' else paths[name]
                return 0, 'noise\n%s%s\n' % (stage.IMPORT_MARK, os.path.realpath(path))
            if fail == 'tests' and 'test_serving_scheduler' in argv:
                return 1, 'FAILED (failures=1)\n'
            return 0, 'OK\n'

        return run

    def test_check_imports_every_patched_module_both_ways_and_reruns_the_tests(self):
        calls = []
        with mock.patch.object(stage, 'log', self.logs.append), mock.patch('sys.stdout', new_callable=io.StringIO):
            problems = stage.check(self.patched_record(), ['test_serving_c2_contract', 'test_qwen_prefix_metrics'],
                                   runner=self.runner(calls), environ={stage.PREFIX_SWITCH: '1', 'PATH': '/bin'},
                                   python='python3')
        self.assertEqual(problems, [])
        imports = [call for call in calls if call['argv'][2] == '-c']
        self.assertEqual(sorted((call['argv'][-1], str(call['switch'])) for call in imports),
                         sorted((module, switch) for module in stage.TARGET_MODULES.values() for switch in ('None', '1')))
        self.assertTrue(all(call['cwd'] == stage.MODEL_TREE for call in imports))
        tests = [call for call in calls if call['argv'][2] == '-m']
        self.assertEqual([call['argv'][4:] for call in tests],
                         [list(stage.P8_INSTALLED_TESTS), ['test_serving_c2_contract', 'test_qwen_prefix_metrics']])
        self.assertEqual([call['cwd'] for call in tests], [stage.MODEL_TREE, stage.MODULES])
        self.assertEqual([call['switch'] for call in tests], [None, None], 'the tests run with the switch unset')
        self.assertTrue(all(call['plugins'] == '' for call in calls))

    def test_check_refuses_a_failed_import_a_foreign_file_and_a_failing_test(self):
        for kwargs, words in ((dict(fail=('model/model.py', '1')), 'model/model.py: importing '
                               'models.demos.blackhole.qwen36.tt.model with QWEN_PREFIX_REUSE 1 failed (exit 1)'),
                              (dict(elsewhere='/opt/other/vllm_tt_plugin/scheduler.py'), 'graft-mounted-is-not'),
                              (dict(fail='tests'), 'P8\'s installed-plugin tests (test_serving_scheduler')):
            with self.subTest(**kwargs):
                calls = []
                with mock.patch.object(stage, 'log', self.logs.append), mock.patch('sys.stdout', new_callable=io.StringIO):
                    problems = stage.check(self.patched_record(), (), runner=self.runner(calls, **kwargs), environ={})
                self.assertTrue(any(words in problem for problem in problems), problems)

    def test_check_with_nothing_patched_runs_nothing(self):
        calls = []
        record = dict(complete=False, targets={name: dict(path=path, patched_by=[]) for name, path, _, _ in stage.TARGETS})
        with mock.patch.object(stage, 'log', self.logs.append):
            self.assertEqual(stage.check(record, ['test_x'], runner=self.runner(calls)), [])
        self.assertEqual(calls, [])
        self.assertTrue(any('nothing to exercise' in line for line in self.logs))
        with open(self.record, 'w', encoding='utf-8') as handle:
            json.dump(record, handle)
        with mock.patch.object(stage, 'log', self.logs.append):
            self.assertEqual(stage.main(['check', '--record', self.record, '--tests', 'test_x']), 0)
        with mock.patch('sys.stderr') as err:
            self.assertEqual(stage.main(['check', '--record', str(self.tmp / 'absent.json')]), 1)
        self.assertIn('no readable stage record', ''.join(call.args[0] for call in err.write.call_args_list))

    def test_the_target_modules_are_the_target_files(self):
        self.assertEqual(sorted(stage.TARGET_MODULES), sorted(stage.target_names()))
        for name, path, _, _ in stage.TARGETS:
            with self.subTest(target=name):
                module = stage.TARGET_MODULES[name]
                root = stage.PLUGIN_ROOT[:-len(stage.PLUGIN_PACKAGE)] if name.startswith('plugin/') else stage.MODEL_TREE + '/'
                self.assertEqual(root + module.replace('.', '/') + '.py', path)

    def test_bring_up_holds_the_served_trees_to_the_record(self):
        record = dict(complete=True, targets={name: dict(path=path) for name, path, _, _ in stage.TARGETS})
        scheduler, entry = stage.PLUGIN_ROOT + '/scheduler.py', stage.MODEL_ROOT + '/qwen36_vllm.py'
        text = ('[PINDIAG] prefix: install scheduler=vllm_tt_plugin.scheduler.TTScheduler plugin=%s '
                'coordinator=UnitaryKVCacheCoordinator block_size=64\n[QWEN-C2] prefix: model tree %s\n'
                '[QWEN-C2] prefix: cache_salt kept only when it verifies against the salt key (present)\n'
                % (scheduler, entry))
        problems, lines = stage.bringup_problems(text, record)
        self.assertEqual(problems, [])
        self.assertEqual(len(lines), 3)
        self.assertIn('bring-up: the salt policy installed, salt key present', lines)
        thin = text.replace(stage.PLUGIN_ROOT, '/opt/thatch/site-packages/vllm_tt_plugin')
        problems, _ = stage.bringup_problems(thin, record)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn('installed from /opt/thatch/site-packages/vllm_tt_plugin/scheduler.py, not the patched', problems[0])
        problems, _ = stage.bringup_problems('', dict(record, complete=False))
        self.assertEqual(len(problems), 4, problems)
        self.assertEqual(stage.bringup_problems(text, None)[0], ['bring-up: no prefix-stage record to hold the log to'])
        # the lines are the ones the scheduler graft and the contract write
        source = (HERE / 'prefix_scheduler_graft.py').read_text(encoding='utf-8')
        self.assertIn("'[PINDIAG] prefix: '", source)
        self.assertIn("'install scheduler=%s.%s plugin=%s ", source)
        module = types.ModuleType('vllm.v1.engine.input_processor')
        module.InputProcessor = type('InputProcessor', (object,), {'process_inputs': lambda self, *a: None})
        err = io.StringIO()
        with mock.patch.object(sys, 'stderr', err):
            contract.install_salt_policy(module, None)
        self.assertEqual(stage.SALT_LINE.findall(err.getvalue()), ['ABSENT'])
        path = self.tmp / 'serve.log'
        path.write_text(text, encoding='utf-8')
        with open(self.record, 'w', encoding='utf-8') as handle:
            json.dump(record, handle)
        with mock.patch.object(stage, 'log', self.logs.append):
            self.assertEqual(stage.main(['bringup', '--log', str(path), '--record', self.record]), 0)

    def test_the_anchors_command(self):
        stdout = []
        with mock.patch.object(stage, 'TARGETS', self.targets), mock.patch('builtins.print', stdout.append):
            self.assertEqual(stage.main(['anchors', '--root', str(self.image)]), 0)
            (self.image / self.targets[0][1].lstrip('/')).write_text('changed\n', encoding='utf-8')
            self.assertEqual(stage.main(['anchors', '--root', str(self.image)]), 1)
        self.assertTrue(stdout[-2].endswith('NOT the pin'), stdout)

    def test_the_apply_command_reports_a_refusal(self):
        with mock.patch.object(stage, 'apply', side_effect=stage.StageError('x')), mock.patch('sys.stderr') as err:
            self.assertEqual(stage.main(['apply', '--modules', 'm', '--record', 'r']), 1)
        self.assertIn('REFUSED: x', ''.join(call.args[0] for call in err.write.call_args_list))


@unittest.skipUnless((PLUGIN_CHECKOUT / '.git').exists(), 'no local vllm-tt-plugin checkout at bf77cd63')
class PluginPinTests(unittest.TestCase):
    """The plugin pins are the git blobs at bf77cd63 (the image clones it on Linux: no CRLF), and
    worker.py's is that blob after P8's serving_plugin_patch.patch_worker, P8's own copy of it."""

    def blob(self, name):
        return subprocess.run(['git', '-C', str(PLUGIN_CHECKOUT), 'show', '%s:src/vllm_tt_plugin/%s' % (PLUGIN_COMMIT, name)],
                              stdout=subprocess.PIPE, check=True).stdout.decode('utf-8')

    def test_the_plugin_pins(self):
        pins = {name: pin for name, _, pin, _ in stage.TARGETS}
        for name in ('scheduler.py', 'model_runner.py'):
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256(self.blob(name).encode('utf-8')).hexdigest(), pins['plugin/' + name])
        dockerfile = code_lines(DOCKERFILE)
        p8 = re.search(r'ARG BASE=qwen-fast-serving:ci-([0-9a-f]{40})', dockerfile).group(1)
        patch_source = subprocess.run(['git', '-C', str(ROOT), 'show', '%s:scripts/ci/serving_plugin_patch.py' % p8],
                                      stdout=subprocess.PIPE, check=True).stdout.decode('utf-8')
        module = types.ModuleType('serving_plugin_patch_p8')
        exec(compile(patch_source, 'serving_plugin_patch_p8', 'exec'), module.__dict__)
        worker = module.patch_worker(self.blob('worker.py'))
        self.assertEqual(hashlib.sha256(worker.encode('utf-8')).hexdigest(), pins['plugin/worker.py'])
        p8_dockerfile = subprocess.run(['git', '-C', str(ROOT), 'show', '%s:docker/qwen-fast-serving.Dockerfile' % p8],
                                       stdout=subprocess.PIPE, check=True).stdout.decode('utf-8')
        self.assertIn('checkout --detach ' + PLUGIN_COMMIT, p8_dockerfile)
        self.assertIn('serving_plugin_patch.py /opt/qwen-fast-plugin', p8_dockerfile)
        self.assertIn('pip install --no-deps -e /opt/qwen-fast-plugin', p8_dockerfile)


@unittest.skipUnless((IMG_TREE / 'model.py').is_file(), 'no local copy of the image model tree (IMG)')
class ModelPinTests(unittest.TestCase):
    def test_the_model_pins_are_img_whose_graft_originals_match(self):
        pins = {name: pin for name, _, pin, _ in stage.TARGETS}
        for name in ('model.py', 'qwen36_vllm.py'):
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((IMG_TREE / name).read_bytes()).hexdigest(), pins['model/' + name])
        # IMG is the C2 base tree for the five files the v235 graft pins as its originals
        for line in (ROOT / 'docker' / 'qwen-c2-graft' / 'source.sha256').read_text().splitlines():
            digest, name = line.split()
            name = name[len('graft/'):-len('.orig')]
            self.assertEqual(hashlib.sha256((IMG_TREE / name).read_bytes()).hexdigest(), digest, name)


class PinCorroborationTests(unittest.TestCase):
    """What CI can check about the pins without the plugin checkout or IMG (PluginPinTests and
    ModelPinTests skip there): the repository's other records of the same files agree with them."""

    PIN = re.compile(r'[\'"]models/demos/blackhole/qwen36/tt/(model\.py|qwen36_vllm\.py|gdn/tp\.py|attention/tp\.py)'
                     r'[\'"]\s*:\s*[\'"]([0-9a-f]{64})[\'"]')

    def test_the_model_pins_are_the_ones_pinned_beside_the_graft_originals(self):
        """dspark-target-hardware.py and full-prefix.py pin model.py and qwen36_vllm.py beside gdn/tp.py and
        attention/tp.py, and those two are the graft's originals, which the C2 Dockerfile checks against P8
        on every build (source.sha256): so the model pins are a tree P8 served."""
        pins = {name.split('/', 1)[1]: pin for name, _, pin, _ in stage.TARGETS if name.startswith('model/')}
        originals = {}
        for line in (ROOT / 'docker' / 'qwen-c2-graft' / 'source.sha256').read_text().splitlines():
            digest, name = line.split()
            originals[name[len('graft/'):-len('.orig')]] = digest
        for script in ('dspark-target-hardware.py', 'full-prefix.py'):
            with self.subTest(script=script):
                found = dict(self.PIN.findall((HERE / script).read_text(encoding='utf-8')))
                self.assertEqual({name: found.get(name) for name in pins}, pins)
                self.assertEqual({name: found.get(name) for name in ('gdn/tp.py', 'attention/tp.py')},
                                 {name: originals[name] for name in ('gdn/tp.py', 'attention/tp.py')})

    def test_the_scheduler_pin_is_the_one_measured_in_the_image(self):
        """cpu-probe run 35665853903 hashed the image's scheduler.py; serving_one_in_flight records it.
        model_runner.py and worker.py have no such record: the build's anchor check is their first."""
        measured = re.findall(r'scheduler\.py sha256 ([0-9a-f]{16})', (HERE / 'serving_one_in_flight.py').read_text(
            encoding='utf-8'))
        self.assertTrue(measured)
        pin = dict((name, pin) for name, _, pin, _ in stage.TARGETS)['plugin/scheduler.py']
        self.assertEqual({pin[:16]}, set(measured))


class PlumbingTests(unittest.TestCase):
    def test_the_dockerfile_runs_the_stage_after_the_overlay_and_before_the_environment(self):
        text = code_lines(DOCKERFILE)
        copy = text.index('COPY qwen_prefix_stage.py /opt/qwen-c2/')
        run = text.index('/opt/qwen-c2/qwen_prefix_stage.py apply')
        self.assertLess(text.index('c2_overlay.py install --manifest'), copy)
        self.assertLess(text.index('c2_overlay.py tests --manifest'), copy)
        self.assertLess(copy, run)
        self.assertLess(run, text.index('ENV QWEN_ATTN_PREP=1'))
        command = text[run - 200:run + 200]
        self.assertIn('--modules /experiment-scripts/ci', command)
        self.assertIn('--record ' + stage.RECORD, command)
        self.assertIn("VLLM_PLUGINS=''", command)
        self.assertEqual(stage.MODULES, '/experiment-scripts/ci')
        self.assertNotIn(contract.PREFIX_SWITCH, text, 'only a profile sets the switch; never the image ENV')

    def test_the_dockerfile_exercises_what_the_stage_patched_in_the_same_run(self):
        """check follows apply in the same RUN (a failure fails the layer), after the overlay's own test run,
        with the overlay's in-image tests; P8's installed-plugin tests are check's own."""
        text = code_lines(DOCKERFILE)
        run = text.index('/opt/qwen-c2/qwen_prefix_stage.py apply')
        block = text[run:text.index('COPY draft-config/')]
        self.assertEqual(block.count('\nRUN '), 0, 'apply and check share one RUN')
        tests = block.index('tests=$(python3 -B /opt/qwen-c2/c2_overlay.py tests --manifest /opt/qwen-c2/qwen-c2-overlay.txt)')
        check = block.index('/opt/qwen-c2/qwen_prefix_stage.py check --record %s --tests $tests' % stage.RECORD)
        self.assertLess(tests, check)

    def test_check_reruns_what_p8_ran_as_p8_ran_it(self):
        """P8_INSTALLED_TESTS and P8_TEST_ENV are the last unittest RUN of the P8 base's Dockerfile, which runs
        from WORKDIR /opt/tt-metal (MODEL_TREE)."""
        p8 = re.search(r'ARG BASE=qwen-fast-serving:ci-([0-9a-f]{40})', code_lines(DOCKERFILE)).group(1)
        dockerfile = subprocess.run(['git', '-C', str(ROOT), 'show', '%s:docker/qwen-fast-serving.Dockerfile' % p8],
                                    stdout=subprocess.PIPE, check=True).stdout.decode('utf-8')
        runs = [line for line in dockerfile.splitlines() if line.startswith('RUN ') and '-m unittest' in line]
        last = runs[-1]
        self.assertEqual(tuple(last.split('-m unittest', 1)[1].split()), stage.P8_INSTALLED_TESTS)
        for key, value in stage.P8_TEST_ENV:
            with self.subTest(key=key):
                self.assertTrue(re.search(r'\b%s=%s(\s|$)' % (key, re.escape(value) if value else "''"), last), last)
        self.assertLess(dockerfile.index('WORKDIR %s' % stage.MODEL_TREE), dockerfile.index(last))

    def test_the_stage_is_a_build_tool_not_an_overlay_file(self):
        self.assertIn(c2_overlay.PREFIX_STAGE, c2_overlay.TOOLS)
        self.assertEqual(c2_overlay.PREFIX_STAGE, 'scripts/ci/qwen_prefix_stage.py')
        self.assertNotIn(c2_overlay.PREFIX_STAGE, manifest_sources())
        self.assertIn(c2_overlay.PREFIX_STAGE, c2_overlay.staged_paths(c2_overlay.read_manifest(MANIFEST)))

    def test_every_g1_module_is_overlaid(self):
        """A G1 runtime or stage module that is not in the manifest never reaches the image (memory
        serving-image-bundle-provenance), and qwen_prefix_stage refuses a stage module it cannot find. So is
        every scripts/ci module one of them imports, transitively: otherwise the image runs the bundle's or
        P8's version of it (lever_n_model_patch at 77d6995a patches only the traced loop)."""
        sources = manifest_sources()
        closure = script_closure(g1_modules())
        wanted = ['scripts/ci/%s.py' % name for name in closure]
        self.assertIn('scripts/ci/qwen_prefix_metrics.py', wanted)
        self.assertIn('scripts/ci/prefix_scheduler_graft.py', wanted)
        self.assertEqual(sorted(set(wanted) - sources), [])
        self.assertNotIn(c2_overlay.PREFIX_STAGE, wanted)
        self.assertIn('scripts/ci/test_qwen_prefix_metrics.py', sources, 'runs inside the image at build')
        self.assertNotIn('scripts/ci/test_qwen_prefix_image.py', sources, 'reads the checkout: CPU suite only')
        self.assertNotIn('scripts/ci/prefix_p0a_probe.py', sources, 'the probe mounts the checkout')

    def test_the_closure_follows_a_stage_module_s_imports(self):
        """The model stage reuses lever_n_model_patch: the closure reaches it, and it is overlaid."""
        tmp = Path(tempfile.mkdtemp(prefix='qwen-prefix-closure-'))
        try:
            (tmp / 'qwen_prefix_model_patch.py').write_text(
                'import os\nimport lever_n_model_patch as m1\n\ndef patch_model(source):\n'
                '    from helper_two import x\n    return m1.patch_tp_replay(source)\n', encoding='utf-8')
            (tmp / 'lever_n_model_patch.py').write_text('import ast\nimport helper_one\n', encoding='utf-8')
            (tmp / 'helper_one.py').write_text('X = 1\n', encoding='utf-8')
            (tmp / 'helper_two.py').write_text('x = 2\n', encoding='utf-8')
            (tmp / 'qwen_prefix_stage.py').write_text('import stage_only\n', encoding='utf-8')
            table = (('model/model.py', 'qwen_prefix_model_patch', 'patch_model'),)
            modules = g1_modules(table, tmp)
            self.assertEqual(modules, ['prefix_scheduler_graft', 'qwen_prefix_model_patch'])
            self.assertEqual(script_closure(modules, tmp),
                             ['helper_one', 'helper_two', 'lever_n_model_patch', 'qwen_prefix_model_patch'])
        finally:
            shutil.rmtree(str(tmp), ignore_errors=True)
        self.assertIn('lever_n_model_patch', script_closure(['lever_n_model_patch']))
        self.assertIn('scripts/ci/lever_n_model_patch.py', manifest_sources())

    def test_every_stage_module_ships_a_switch_off_test(self):
        workflow = CPU_WORKFLOW.read_text(encoding='utf-8')
        self.assertEqual(stage_test_problems(stage.STAGES, HERE, workflow), [])
        tmp = Path(tempfile.mkdtemp(prefix='qwen-prefix-stage-tests-'))
        try:
            table = (('plugin/scheduler.py', 'qwen_prefix_scheduler_patch', 'patch_scheduler'),
                     ('model/model.py', 'qwen_prefix_model_patch', 'patch_model'),
                     ('plugin/worker.py', 'qwen_prefix_runner_patch', 'patch_worker'))
            (tmp / 'test_qwen_prefix_scheduler_patch.py').write_text(
                'class T:\n    def test_the_switch_off_scheduler_is_the_original(self):\n        pass\n', encoding='utf-8')
            (tmp / 'test_qwen_prefix_model_patch.py').write_text(
                'class T:\n    def test_patch(self):\n        pass\n', encoding='utf-8')
            text = '          python -B -m unittest test_qwen_prefix_scheduler_patch test_other\n'
            problems = stage_test_problems(table, tmp, text)
        finally:
            shutil.rmtree(str(tmp), ignore_errors=True)
        self.assertEqual(len(problems), 3, problems)
        self.assertIn('test_qwen_prefix_model_patch.py has no test whose name says switch_off', problems[0])
        self.assertIn('test_qwen_prefix_model_patch is not allowlisted', problems[1])
        self.assertIn('stage module qwen_prefix_runner_patch has no scripts/ci/test_qwen_prefix_runner_patch.py',
                      problems[2])

    def test_the_new_tests_are_allowlisted(self):
        text = CPU_WORKFLOW.read_text(encoding='utf-8')
        for module in ('test_qwen_prefix_metrics', 'test_qwen_prefix_image'):
            with self.subTest(module=module):
                self.assertRegex(text, r'python -B -m unittest [^\n]*\b%s\b' % module)

    def test_the_contract_boots_the_metrics_it_overlays(self):
        text = (HERE / 'serving_c2_contract.py').read_text(encoding='utf-8')
        self.assertIn('import qwen_prefix_metrics', text)
        self.assertIn('/experiment-scripts/ci', contract.FAST_PATHS)


if __name__ == '__main__':
    unittest.main()
