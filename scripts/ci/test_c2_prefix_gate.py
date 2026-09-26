"""c2_prefix_gate: the prefix-reuse (G1) gates in the node agent's container shape, held on CPU.

Nothing here opens a device or runs docker. The arms' `docker run` argv (the S1 gate's agent shape,
detached, the platform's argv, a derived profile mounted where one is needed), what each plan
refuses before any container, and the whole Runner against test_prefix_replay.FakeEngine - one per
arm, built from the profile the arm's docker argv serves (the derived file when it mounts one), so
the pool, the loop path, audit, dev mode and the store are what that profile would give; grants come
from the real scheduler graft - with faults switched on to see each verdict. The workflow step, the
job keys and the CPU allowlist are read from the files."""

import copy
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_prefix_gate as gate  # noqa: E402
import c2_serving_gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
import prefix_judge as judge  # noqa: E402
import prefix_markers as pm  # noqa: E402
import prefix_replay as replay  # noqa: E402
import qwen_prefix_model_patch as model_patch  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import test_prefix_markers as marker_fixture  # noqa: E402
import test_prefix_replay as fakes  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml')
CPU_WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-integration-cpu.yml')
JOB_FILE = os.path.join(ROOT, '.github', 'c2-serving-job.env')
NEW_TESTS = ('test_prefix_agent_corpus', 'test_prefix_markers', 'test_prefix_judge', 'test_prefix_report',
             'test_prefix_replay', 'test_prefix_oracle_check', 'test_c2_prefix_gate')
CURRENT = dict(engine=None)


def profiles():
    """The checkout's profiles plus general-prefix (design 2.0.1 item 5; the image track adds it)."""
    document = copy.deepcopy(marker_fixture.PROFILES)
    document['profiles']['general-prefix'] = marker_fixture.prefix_profile()
    return document


SHA = dict(a='a' * 64, b='b' * 64)
GOOD_ANCHOR = dict(files=dict(gate.STAGE_PINS), pins={'mlp.py': SHA['a']}, actual={'mlp.py': SHA['a']}, mismatched=[],
                   stage_pins=dict(gate.STAGE_PINS), stage_mismatched=[], marker_files=['model.py'],
                   prefix_marker_in=['model.py'], unpinned=[])


class Harness(object):
    """The Runner's docker, client, log and container, faked; a FakeEngine per arm, built from the
    profile the arm's docker argv serves. faults: {kind or profile name or 'all': {fault: True,
    '_env': {KEY: value or None}, '_engine': FakeEngine subclass}}."""

    def __init__(self, engine_class=None, **faults):
        self.engine_class = engine_class or fakes.FakeEngine
        self.faults = faults
        self.engine = None
        self.engines = []
        self.calls = []

    def docker(self, arguments, timeout=600):
        self.calls.append(arguments)
        if arguments[:3] == ['docker', 'run', '-d']:
            name = [a.split('=', 1)[1] for a in arguments if a.startswith('QWEN_C2_PROFILE=')][0]
            derived = [a for a in arguments if a.startswith('type=bind,src=') and gate.DERIVED_MOUNT in a]
            if derived:
                path = derived[0].split('src=', 1)[1].split(',', 1)[0]
                with open(path, encoding='utf-8') as handle:
                    document = json.load(handle)
            else:
                document = profiles()
            profile = copy.deepcopy(document['profiles'][name])
            options = {}
            for key in (name.split('+')[0], name.split('+')[-1] if '+' in name else None, 'all'):
                options.update(self.faults.get(key) or {})
            for key, value in (options.pop('_env', None) or {}).items():
                if value is None:
                    profile['env'].pop(key, None)
                else:
                    profile['env'][key] = value
            engine_class = options.pop('_engine', None) or self.engine_class
            self.engine = engine_class(profile=profile, name=name, **options)
            CURRENT['engine'] = self.engine
            self.engines.append(self.engine)
            return 0, 'container-id'
        if arguments[:3] == ['docker', 'logs', '--timestamps']:
            return 0, '\n'.join(self.engine.lines) if self.engine else ''
        return 0, ''

    def proxy(self):
        harness = self

        class Proxy(object):
            def __getattr__(self, name):
                return getattr(harness.engine, name)

        return Proxy()

    def runner(self, results):
        return gate.Runner('img', results, ROOT, ['/dev/tenstorrent/1', '/dev/tenstorrent/0'], docker=self.docker,
                           make_client=self.proxy, make_log=lambda name, path: fakes.FakeLog(self.proxy()),
                           make_container=lambda name: fakes.FakeContainer(self.proxy()),
                           containers=lambda: [], log=lambda text: None, sleep=lambda seconds: None,
                           corpus=fakes.CORPUS, agents=(1, 2), turns=2)


def run_plan(plan, results, anchor=GOOD_ANCHOR, baseline='general', engine_class=None, **faults):
    harness = Harness(engine_class, **faults)
    runner = harness.runner(results)
    arms = gate.plan_arms(plan, 'general-prefix', baseline, profiles())
    return gate.run_plan(plan, arms, runner, anchor), harness, runner


def arm_of(plan, name):
    return [arm for arm in gate.plan_arms(plan, 'general-prefix', 'general', profiles()) if arm['arm'] == name][0]


class ArmTests(unittest.TestCase):
    def test_the_plans_arms_and_their_profiles(self):
        document = profiles()
        names = dict((plan, [(a['arm'], a['served']) for a in gate.plan_arms(plan, 'general-prefix', 'general', document)])
                     for plan in gate.PLANS)
        self.assertEqual(names['bringup'], [('bringup-reference', 'general'), ('bringup-prefix', 'general-prefix')])
        self.assertEqual(names['exactness'], [('exactness-traced', 'general-prefix'),
                                              ('exactness-audit', 'general-prefix+audit'),
                                              ('exactness-eager', 'general-prefix+eager')])
        self.assertEqual(names['lifecycle'], [('lifecycle-evict', 'general-prefix+dev'),
                                              ('lifecycle-store', 'general-prefix+store'),
                                              ('lifecycle-tiny', 'general-prefix+tiny')])
        self.assertEqual(names['timing'], [('timing-prefix', 'general-prefix'), ('timing-baseline', 'general')])
        timing = gate.plan_arms('timing', 'general-prefix', 'none', document)
        self.assertEqual([a['arm'] for a in timing], ['timing-prefix'])

    def test_derived_profiles_change_one_thing_each(self):
        document = profiles()
        base = document['profiles']['general-prefix']
        for kind in gate.DERIVED:
            name, derived = gate.derive(document, 'general-prefix', kind)
            self.assertEqual((name, derived['default'], list(derived['profiles'])), ('general-prefix+' + kind, name, [name]))
            changed = derived['profiles'][name]
            self.assertEqual(changed['eos_ids'], base['eos_ids'])
            limits = contract.request_limits(dict(changed, name=name))
            self.assertEqual(limits['budget'], 256)
        eager = gate.derive(document, 'general-prefix', 'eager')[1]['profiles']['general-prefix+eager']
        self.assertEqual(eager['engine']['additional-config']['tt']['trace_mode'], 'decode_only')
        self.assertEqual(eager['engine']['additional-config']['tt']['l1_small_size'], 24576, 'the rest of tt kept')
        store = gate.derive(document, 'general-prefix', 'store')[1]['profiles']['general-prefix+store']
        self.assertEqual(store['env']['QWEN_PREFIX_STORE_GIB'], '0.5')
        self.assertEqual(judge.store_entries(gate.SMALL_STORE_GIB), 3)
        self.assertNotIn('+', json.dumps(document['profiles']['general-prefix']), 'the image profile is untouched')

    def test_the_tiny_pool_is_set_where_the_tt_worker_reads_it(self):
        """The TT worker overwrites num_gpu_blocks_override (plugin worker.py:388-390); the image's
        model reads QWEN36_MAX_TOKENS_ALL_USERS (qwen36_vllm.py:96-98) and the worker adds a block
        per sequence (worker.py:538-539): 81664 + 4 x 64 = 81920 tokens, 1280 blocks."""
        tiny = gate.derive(profiles(), 'general-prefix', 'tiny')[1]['profiles']['general-prefix+tiny']
        self.assertNotIn('num-gpu-blocks-override', tiny['engine'])
        self.assertEqual(tiny['env']['QWEN36_MAX_TOKENS_ALL_USERS'], '81664')
        self.assertEqual(fakes.pool_blocks(tiny), gate.TINY_BLOCKS)
        self.assertEqual(gate.TINY_BLOCKS * judge.BLOCK, replay.TINY_POOL_TOKENS)
        self.assertGreaterEqual(replay.TINY_POOL_TOKENS, tiny['engine']['max-model-len'],
                                'one request of the full context still fits the tiny pool')
        overridden = copy.deepcopy(profiles()['profiles']['general-prefix'])
        overridden['engine']['num-gpu-blocks-override'] = gate.TINY_BLOCKS
        self.assertEqual(fakes.pool_blocks(overridden), 4100, 'the override alone leaves the default pool')

    def test_what_is_refused_before_any_container(self):
        document = profiles()
        for plan, profile, baseline in (('bringup', 'general', 'general'), ('bringup', 'nope', 'general'),
                                        ('bringup', 'general-prefix', 'none'), ('timing', 'general-prefix', 'nope'),
                                        ('timing', 'general-prefix', 'general-prefix')):
            with self.subTest(plan=plan, profile=profile, baseline=baseline), self.assertRaises(gate.PlanError):
                gate.plan_arms(plan, profile, baseline, document)
        with self.assertRaises(ValueError):
            gate.plan_arms('soak', 'general-prefix', 'general', document)

    def test_every_exactness_and_lifecycle_arm_is_a_plan_of_its_own(self):
        """G1 v47 (run 36246961161): the eager arm alone, without the traced and audit arms' hour."""
        document = profiles()
        self.assertEqual(gate.PLANS, job.PREFIX_PLANS + tuple(arm for arm, _ in job.PREFIX_ARM_PLANS))
        self.assertEqual(sorted(arm for arm, _ in job.PREFIX_ARM_PLANS),
                         sorted(entry[0] for plan in ('exactness', 'lifecycle') for entry in gate.PLAN_ARMS[plan]))
        for arm, plan in job.PREFIX_ARM_PLANS:
            with self.subTest(arm=arm):
                alone = gate.plan_arms(arm, 'general-prefix', 'general', document)
                self.assertEqual(alone, [a for a in gate.plan_arms(plan, 'general-prefix', 'general', document)
                                         if a['arm'] == arm])
        eager, = gate.plan_arms('exactness-eager', 'general-prefix', 'none', document)
        self.assertEqual((eager['served'], eager['kind'], eager['scenario'], eager['timeout']),
                         ('general-prefix+eager', 'eager', 'exactness_eager', 5400))
        self.assertEqual(gate.worst_case_seconds({'exactness-eager': [eager]}), 5400 + gate.ARM_OVERHEAD_SECONDS)
        for plan in ('bringup-prefix', 'timing-prefix'):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                gate.plan_arms(plan, 'general-prefix', 'general', document)

    def test_the_worst_case_fits_one_plan_per_step(self):
        document = profiles()
        for plan in gate.PLANS:
            arms = gate.plan_arms(plan, 'general-prefix', 'general', document)
            self.assertLessEqual(gate.worst_case_seconds({plan: arms}), 380 * 60 - 600, plan)


class ShapeTests(unittest.TestCase):
    def test_the_server_is_the_s1_gates_agent_shape_detached_on_the_platform_argv(self):
        arguments = gate.server_run('img', 'qwen-c2-prefix-x', 'general-prefix', ['/dev/tenstorrent/3', '/dev/tenstorrent/1'])
        agent = c2_serving_gate.agent_shape('img', 'qwen-c2-prefix-x', 'general-prefix',
                                            ['/dev/tenstorrent/3', '/dev/tenstorrent/1'])
        self.assertEqual(arguments[:3], ['docker', 'run', '-d'])
        self.assertNotIn('--rm', arguments, 'the reload drill stops and starts the same container')
        self.assertEqual(arguments[3:len(agent)], agent[3:])
        self.assertIn('--read-only', arguments)
        self.assertEqual(arguments[arguments.index('-p') + 1], '127.0.0.1:%d:8000' % gate.PORT)
        entry = arguments.index('--entrypoint')
        self.assertEqual(arguments[entry:entry + 5], ['--entrypoint', 'python3', 'img', '-m',
                                                      'vllm.entrypoints.openai.api_server'])
        self.assertIn('--no-enable-prefix-caching', arguments, 'the platform\'s flag, which the contract drops')
        self.assertNotIn('QWEN_C2_PROFILES=%s' % gate.DERIVED_MOUNT, arguments)
        derived = gate.server_run('img', 'n', 'general-prefix+eager', ['/a', '/b'], derived_path='/r/x/profiles.json')
        self.assertIn('type=bind,src=/r/x/profiles.json,dst=%s,readonly' % gate.DERIVED_MOUNT, derived)
        self.assertIn('QWEN_C2_PROFILES=%s' % gate.DERIVED_MOUNT, derived)
        self.assertIn('QWEN_C2_PROFILE=general-prefix+eager', derived)
        self.assertNotIn(gate.DIGESTS_ENV, arguments)
        self.assertNotIn(gate.STATS_NOW_ENV, arguments)
        now = gate.server_run('img', 'n', 'general-prefix', ['/a', '/b'], stats_now=True)
        self.assertEqual(now[now.index(gate.STATS_NOW_ENV) - 1], '-e')
        import qwen_prefix_registry
        self.assertEqual(gate.STATS_NOW_ENV.split('=')[0], qwen_prefix_registry.ENV_STATS_S)
        digests = gate.server_run('img', 'n', 'general-prefix', ['/a', '/b'], digests=True)
        self.assertEqual(digests[digests.index(gate.DIGESTS_ENV) - 1], '-e')
        self.assertLess(digests.index(gate.DIGESTS_ENV), digests.index('--entrypoint'))

    def test_a_prefix_arm_mounts_its_own_salt_key_and_mints_salts_the_contract_keeps(self):
        """The image keeps a cache_salt only when it verifies against its salt key (serving_c2_contract's
        salt policy); a gate that sent raw salts would get no hit at all. Each prefix arm writes a key,
        mounts it where QWEN_PREFIX_SALT_KEY_FILE points, and its driver mints every salt under it."""
        import serving_c2_contract as contract

        arguments = gate.server_run('img', 'n', 'general-prefix', ['/a', '/b'], salt_key_path='/r/arm/salt.key')
        self.assertIn('type=bind,src=/r/arm/salt.key,dst=%s,readonly' % gate.SALT_KEY_MOUNT, arguments)
        self.assertIn('%s=%s' % (contract.SALT_KEY_ENV, gate.SALT_KEY_MOUNT), arguments)
        self.assertEqual(gate.SALT_KEY_ENV, contract.SALT_KEY_ENV)
        self.assertLess(arguments.index('%s=%s' % (contract.SALT_KEY_ENV, gate.SALT_KEY_MOUNT)),
                        arguments.index('--entrypoint'))
        directory = tempfile.mkdtemp()
        try:
            path = os.path.join(directory, 'salt.key')
            key = gate.write_salt_key(path)
            read, where = contract.read_salt_key({contract.SALT_KEY_ENV: path})
            self.assertEqual((read, where), (key, path))
        finally:
            shutil.rmtree(directory, ignore_errors=True)
        driver = replay.Driver(None, 'exactness-traced', None, salt_key=key)
        for salt in (driver.salt('chain'), driver.fresh_salt(), driver.fresh_salt('rerun')):
            self.assertEqual(contract.salt_verdict(salt, read), 'verified', salt)
            self.assertEqual(contract.salt_verdict(salt, b'k' * 32), 'dropped-unverified')
        self.assertEqual(driver.salt('chain'), driver.salt('chain'))
        self.assertNotEqual(driver.salt('chain'), driver.salt('shared'))

    def test_the_row_digests_go_to_every_prefix_arm_but_timing(self):
        """The model prints slot_sha / logits_sha only with QWEN_PREFIX_DIGESTS=1: the exactness, bring-up
        and lifecycle arms compare them; the timing arm measures TTFT without the hashing, and the
        baseline (general) has no prefix route."""
        document = profiles()
        wanted = {}
        for plan in gate.PLANS:
            for arm in gate.plan_arms(plan, 'general-prefix', 'general', document):
                wanted[arm['arm']] = gate.wants_digests(arm)
        self.assertEqual(sorted(name for name, on in wanted.items() if not on),
                         ['bringup-reference', 'timing-baseline', 'timing-prefix'])

    def test_the_contract_reads_the_derived_file(self):
        """serving_c2_contract.boot loads QWEN_C2_PROFILES: the derived file is what serves, and its
        env reaches every process (apply_environment)."""
        for kind in ('eager', 'tiny'):
            name, document = gate.derive(profiles(), 'general-prefix', kind)
            directory = tempfile.mkdtemp()
            try:
                path = os.path.join(directory, 'profiles.json')
                with open(path, 'w', encoding='utf-8') as handle:
                    json.dump(document, handle)
                profile = contract.load_profile(path, name)
                argv = contract.rewrite_argv(['api_server.py'] + gate.PLATFORM_ARGS, profile, profile['snapshots'][0])
                self.assertEqual(pm.prefix_argv_problems(argv[1:]), [])
                environ = contract.apply_environment(profile, {})
                if kind == 'eager':
                    tt = json.loads(argv[argv.index('--additional-config') + 1])['tt']
                    self.assertEqual(tt['trace_mode'], 'decode_only')
                else:
                    self.assertEqual(environ['QWEN36_MAX_TOKENS_ALL_USERS'], '81664')
                    self.assertEqual(pm.argv_flags(argv[1:])['max-num-seqs'], '4', 'the padding the pool assumes')
            finally:
                shutil.rmtree(directory, ignore_errors=True)

    def test_the_platform_argv_is_the_smoke_steps(self):
        with open(WORKFLOW, encoding='utf-8') as handle:
            text = handle.read()
        smoke = text[text.index('- name: Smoke on cards M+A'):text.index('- name: Run the gate')]
        for flag in ('--reasoning-parser qwen3', '--tool-call-parser qwen3_xml', '--enable-auto-tool-choice',
                     '--max-num-seqs 2', '--no-enable-prefix-caching', '--served-model-name Qwen/Qwen3.8-27B'):
            self.assertIn(flag, smoke)
            self.assertIn(flag, ' '.join(gate.PLATFORM_ARGS))

    def test_the_anchor_probe_holds_every_pinned_file(self):
        with open(os.path.join(ROOT, 'docker', 'qwen-c2-graft', 'graft.sha256'), encoding='utf-8') as handle:
            pinned = handle.read()
        paths = [line.split()[1][len('graft/'):] for line in pinned.splitlines() if line.strip()]
        self.assertGreater(len(paths), 5, 'more than the five S1 model files')
        installed = ['%s  %s' % (line.split()[0], line.split()[1][len('graft/'):]) for line in pinned.splitlines()
                     if line.strip()]
        installed[-1] = '%s  %s' % (SHA['b'], paths[-1])
        text = '\n'.join(['%s  model.py' % SHA['a'], '%s  qwen36_vllm.py' % SHA['a'], '==pins', pinned.rstrip('\n'),
                          '%s  graft/model.py.orig' % SHA['b'], '==pinned'] + installed +
                         ['missing  gdn/tp.py', '==markers', './model.py', './stale_copy.py'])
        anchor = gate.parse_anchor(text)
        self.assertEqual(sorted(anchor['pins']), sorted(paths), 'every non-.orig pin, not a fixed list')
        self.assertEqual(anchor['mismatched'], sorted(['gdn/tp.py', paths[-1]]))
        self.assertEqual((anchor['prefix_marker_in'], anchor['unpinned']), (['model.py'], []))
        # The anchor files are the prefix stage's (graft.sha256 pins neither): held to PATCHED_SHA256.
        self.assertEqual(anchor['stage_pins'], dict(model_patch.PATCHED_SHA256))
        self.assertEqual(anchor['stage_mismatched'], ['model.py', 'qwen36_vllm.py'])
        staged = gate.parse_anchor(text.replace('%s  model.py\n' % SHA['a'], '%s  model.py\n' % gate.STAGE_PINS['model.py'])
                                   .replace('%s  qwen36_vllm.py' % SHA['a'], '%s  qwen36_vllm.py' % gate.STAGE_PINS['qwen36_vllm.py']))
        self.assertEqual((staged['stage_mismatched'], staged['unpinned']), ([], []))
        missing = gate.parse_anchor(text.replace('%s  qwen36_vllm.py\n' % SHA['a'], ''))
        self.assertIn('qwen36_vllm.py', missing['stage_mismatched'], 'an anchor file the probe could not hash')
        pinned_by_graft = gate.parse_anchor(text.replace('==pinned\n', '%s  graft/model.py\n==pinned\n' % SHA['a'], 1))
        self.assertEqual((sorted(pinned_by_graft['stage_pins']), pinned_by_graft['stage_mismatched']),
                         (['qwen36_vllm.py'], ['qwen36_vllm.py']), 'graft.sha256 takes over a file it pins')
        with mock.patch.object(gate, 'STAGE_PINS', {}):
            self.assertEqual(gate.parse_anchor(text)['unpinned'], ['model.py', 'qwen36_vllm.py'])
        stray = gate.parse_anchor(text.replace('./model.py\n', ''))
        self.assertEqual((stray['marker_files'], stray['prefix_marker_in']), (['stale_copy.py'], []),
                         'a marker in an unpinned stray file vouches for nothing')
        script = gate.anchor_script()
        self.assertIn('cat /opt/qwen-c2/graft.sha256', script)
        self.assertIn('p !~ /[.]orig$/', script)
        self.assertNotIn('\\', script)

        class Result(object):
            stdout, returncode = text.encode(), 0

        seen = []
        probe = gate.anchor_probe('img', run=lambda arguments, **k: seen.append(arguments) or Result())
        self.assertEqual(probe['mismatched'], anchor['mismatched'])
        self.assertEqual(seen[0][:6], ['docker', 'run', '--rm', '--network', 'none', '--entrypoint'])
        self.assertNotIn('--device', seen[0])


class GenericTests(unittest.TestCase):
    ARM = dict(arm='lifecycle-evict', prefix=False, strict=False, kind='dev')

    def test_every_failure_line_is_scanned_for_fatal_signatures(self):
        lines = ['Traceback (most recent call last)'] * 20 + ['AssertionError: grant Q 4096 != start_pos 2048']
        problems, notes, _ = gate.generic_problems(self.ARM, pm.scan(lines), [], None, 'x')
        self.assertTrue(any('AssertionError' in p for p in problems), problems)
        self.assertEqual(len([n for n in notes if 'traceback at line' in n]), gate.MAX_LISTED)
        self.assertTrue(any('4 more tracebacks' in n for n in notes))

    def test_failure_lines_of_the_restart_drill_are_recorded_not_failed(self):
        lines = ['ok', 'EngineDeadError: engine stopped', 'ok', 'EngineDeadError: later']
        problems, notes, _ = gate.generic_problems(self.ARM, pm.scan(lines), [], None, 'x', windows=[(0, 2)])
        self.assertEqual([p for p in problems if 'EngineDead' in p], ['server log line 3: EngineDeadError: later'])
        self.assertTrue(any('during the restart drill' in n for n in notes))
        self.assertEqual(gate.quiet_windows(dict(reload=dict(log_window=[4, 9]), other={})), [(4, 9)])

    def test_the_raw_hit_where_publishing_is_the_claim(self):
        arm = dict(arm='bringup-prefix', prefix=True, strict=True, kind=None)
        record = dict(tag='u', role='unsalted', ok=True, prompt_tokens=4000, expected_raw_h=0,
                      markers=dict(rows=[dict(q=0, l=4000, captured=[])], row=dict(q=0, l=4000, captured=[]), q=0,
                                   l=4000, grants=[], grant=None, skipped=[], admissions=1),
                      counters=dict(before={'vllm:prefix_cache_hits': 0.0, 'vllm:prefix_cache_queries': 0.0},
                                    after={'vllm:prefix_cache_hits': 2048.0, 'vllm:prefix_cache_queries': 4000.0}))
        problems, _, _ = gate.generic_problems(arm, pm.scan([]), [record], None, 'x')
        self.assertTrue(any('found 2048 cached tokens' in p for p in problems), problems)
        record['counters'] = dict(before={}, after={})
        _, _, missing = gate.generic_problems(arm, pm.scan([]), [record], None, 'x')
        self.assertTrue(any('no vllm:prefix_cache_hits reading' in m for m in missing), missing)


def hit(tag, case, prompt, q, captured=(), role='hit'):
    row = dict(q=q, l=prompt, captured=list(captured), tag=tag)
    return dict(tag=tag, case=case, role=role, prompt_tokens=prompt, ok=True,
                markers=dict(q=q, l=prompt, row=row, rows=[row], admissions=1))


class ExactnessCaseTests(unittest.TestCase):
    ARM = dict(arm='exactness-traced', prefix=True, strict=True)

    def test_a_boundary_case_served_off_its_target_is_not_judged(self):
        records = [hit('a', 'boundary-2048', 2050, 0), hit('b', 'boundary-2048', 2400, 2048)]
        events = {'boundary-2047': dict(fitted=True, tokens=2047, served=2047),
                  'boundary-2048': dict(fitted=True, tokens=2048, served=2050),
                  'boundary-2049': dict(fitted=True, tokens=2049, served=2049)}
        missing, problems, _ = gate.exercised_exactness(dict(arm='exactness-audit'), records, events)
        self.assertTrue(any('boundary-2048: /tokenize fitted 2048 tokens but the chat endpoint served 2050' in m
                            for m in missing), missing)
        self.assertEqual(problems, [])
        events['boundary-2047'] = dict(fitted=True, tokens=2047, served=2049)
        records = [hit('c', 'boundary-2047', 2049, 0), hit('d', 'boundary-2047', 2300, 2048)]
        missing, problems, _ = gate.exercised_exactness(dict(arm='exactness-audit'), records, events)
        self.assertEqual(problems, [], 'the probe\'s false FAIL: a 2047 case served as 2049')

    def test_the_gap_capture_and_its_restore(self):
        shared = [hit('s0', 'shared-system', 6900, 0, [6144]), hit('s1', 'shared-system', 6800, 0, [4096, 6144]),
                  hit('s2', 'shared-system', 6850, 4096, [6144])]
        missing, problems, lines = gate.shared_gap(shared)
        self.assertEqual((missing, problems), ([], []))
        self.assertTrue(any('gap boundary 4096 captured by s1, restored by s2' in line for line in lines))
        no_gap = [hit('s0', 'shared-system', 4300, 0, [4096]), hit('s1', 'shared-system', 4280, 4096, []),
                  hit('s2', 'shared-system', 4290, 4096, [])]
        missing, problems, _ = gate.shared_gap(no_gap)
        self.assertTrue(any('captured no gap boundary' in m for m in missing), 'the reviewer\'s 4.3k case')
        wrong = shared[:2] + [hit('s2', 'shared-system', 6850, 2048, [6144])]
        self.assertTrue(gate.shared_gap(wrong)[0])
        past = shared[:2] + [hit('s2', 'shared-system', 6850, 6144, [])]
        self.assertTrue(gate.shared_gap(past)[1])


class LifecycleCaseTests(unittest.TestCase):
    def tiny_events(self, **grant):
        base = dict(ok=True, filler_running=True, waited_for_filler=True, tag='x2', first_token_s=9.0, filler_end_s=8.9)
        base.update(grant)
        return {'tiny-grant': base, 'tiny': dict(ok=True, preemptions=1.0)}

    def records(self, q=16384):
        waited = hit('x2', 'tiny-grant', 22000, q)
        resumed = hit('t1', 'tiny', 26000, 24576)
        resumed['markers']['admissions'] = 2
        return [waited, resumed]

    def test_the_dropped_grant_evidence(self):
        problems, missing, lines = gate.tiny_findings(self.records(), self.tiny_events(), dict(dropped_hits=3))
        self.assertEqual((problems, missing), ([], []))
        self.assertTrue(any('t1 (2 admissions)' in line for line in lines), lines)
        problems, _, _ = gate.tiny_findings(self.records(), self.tiny_events(), dict(dropped_hits=0))
        self.assertTrue(any('dropped_hits counter' in p for p in problems), 'the counter contradicts what happened')
        self.assertEqual(gate.tiny_findings(self.records(), self.tiny_events(), dict(pins=0))[1], [],
                         'no dropped_hits counter yet: the constructed evidence stands')
        _, missing, _ = gate.tiny_findings(self.records(), self.tiny_events(waited_for_filler=False), None)
        self.assertTrue(any('did not wait for the filler' in m for m in missing))
        _, missing, _ = gate.tiny_findings(self.records(q=0), self.tiny_events(), None)
        self.assertTrue(any('admitted at Q=0' in m for m in missing))

    def test_a_skipped_pool_is_not_exercised(self):
        events = dict(tiny=dict(skipped=True, pool_tokens=262400, expected_pool=81920, reason='wrong pool'))
        problems, missing, _ = gate.tiny_findings([], events, None)
        self.assertEqual((problems, missing), ([], ['the tiny pool did not run: wrong pool']))

    def test_required_counters(self):
        self.assertEqual(gate.required_stats(None, ('evicted_lru',)), [])
        self.assertIn('no evicted_lru counter', gate.required_stats({}, ('evicted_lru',))[0])
        self.assertIn('LRU', gate.required_stats(dict(evicted_lru=0), ('evicted_lru',))[0])
        self.assertEqual(gate.required_stats(dict(evicted_lru=2), ('evicted_lru',)), [])


class RunnerTests(unittest.TestCase):
    """Whole arms against the fake engine. The chain and the KV flood are shortened (the fake's
    'tokenizer' is slow); every case still runs: the variants fork after turn 3 of 5."""

    def setUp(self):
        self.results = tempfile.mkdtemp()
        for name, value in (('CHAIN_HITS', (4200, 9000, 16500, 24500)), ('EVICT_LENGTHS', (12000, 24000))):
            patcher = mock.patch.object(replay, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = fakes.burst_aware(lambda: CURRENT['engine'])
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        shutil.rmtree(self.results, ignore_errors=True)

    def plan(self, plan, name=None, **kwargs):
        return run_plan(plan, os.path.join(self.results, name or plan), **kwargs)

    def test_bringup_passes_on_a_good_engine_and_keeps_its_files(self):
        result, harness, _ = self.plan('bringup')
        self.assertEqual(result['verdict'], 'PASS', result['lines'])
        self.assertEqual([a['verdict'] for a in result['arms'].values()], ['PASS', 'PASS'])
        for label in ('grants disabled (unsalted) vs baseline', 'capturing (fresh salt) vs baseline'):
            self.assertEqual(len([line for line in result['lines'] if label in line and 'IDENTICAL' in line]), 3, label)
        prefix = result['arms']['bringup-prefix']
        detail = [line for line in prefix['lines'] if line.startswith('program cache across')][0]
        self.assertIn('"capture_compiled": 0', detail)
        self.assertIn('"hits_measured": 2', detail)
        for name in ('server-final.log', 'records.jsonl', 'pairs.json', 'events.json', 'arm.json', 'docker-run.json'):
            self.assertTrue(os.path.exists(os.path.join(self.results, 'bringup', 'bringup-prefix', name)), name)
        removed = [c for c in harness.calls if c[:3] == ['docker', 'rm', '-f']]
        self.assertEqual(len(removed), 4, 'each container removed before and after its arm')
        dram = [line for line in result['lines'] if line.startswith('dram')]
        self.assertEqual(dram[:2], ['dram after registry: ' + fakes.FakeEngine.DRAM_TEXT,
                                    'dram after first capture: ' + fakes.FakeEngine.DRAM_TEXT])
        self.assertIn('dram: beside a KV pool of', dram[2])
        self.assertEqual(result['cross_not_exercised'], [])
        for line in ('anchor: model.py at the prefix stage\'s pin %s' % gate.STAGE_PINS['model.py'],
                     'anchor: qwen36_vllm.py at the prefix stage\'s pin %s' % gate.STAGE_PINS['qwen36_vllm.py']):
            self.assertTrue(any(text.startswith(line) for text in result['lines']), line)
        self.assertEqual([p for p in result['arms']['bringup-prefix']['dram_readings'] if p['chips']][0]['point'],
                         pm.DRAM_REGISTRY)

    def test_bringup_without_gs_dram_reading_is_not_exercised(self):
        cases = dict(
            none=dict(fault='no_dram', count=2, text=('no "[PINDIAG] dram after registry" reading',
                                                      'dram: none logged')),
            unavailable=dict(fault='dram_unavailable', count=2,
                             text=('no "[PINDIAG] dram after registry" reading with per-chip figures on the prefix arm '
                                   '(the model warm and the registry created; logged: unavailable (RuntimeError: no '
                                   'allocator on this device))',)),
            capture=dict(fault='no_capture_dram', count=1, text=('no "[PINDIAG] dram after first capture" reading',
                                                        'the arm stored')))
        for name, case in cases.items():
            with self.subTest(name=name):
                result, _, _ = self.plan('bringup', 'dram-' + name, **{'general-prefix': {case['fault']: True}})
                self.assertEqual(result['verdict'], 'NOT_EXERCISED', result['lines'])
                self.assertEqual([a['verdict'] for a in result['arms'].values()], ['PASS', 'PASS'])
                for text in case['text']:
                    self.assertTrue(any(text in line for line in result['lines']), (text, result['lines']))
                self.assertEqual(len(result['cross_not_exercised']), case['count'], result['cross_not_exercised'])
                self.assertTrue(all('G2 has no DRAM reading' in m for m in result['cross_not_exercised']))

    def test_the_dram_findings_read_only_the_parsed_readings(self):
        reading = lambda point, chips=True: dict(point=point, chips=[dict(chip=0)] if chips else [],  # noqa: E731
                                                 text='chip0 ...' if chips else 'unavailable (x)')
        missing, lines = gate.dram_findings(dict(dram_readings=[reading('registry')], stats=dict(captures=0)))
        self.assertEqual((missing, lines), ([], ['dram after registry: chip0 ...']))
        missing, _ = gate.dram_findings(dict(dram_readings=[reading('registry')], stats=dict(captures=3)))
        self.assertEqual(len(missing), 1)
        self.assertIn('first capture', missing[0])
        missing, _ = gate.dram_findings(dict(dram_readings=[reading('attach'), reading('registry', chips=False)]))
        self.assertEqual(len(missing), 1, 'another point does not stand in for the registry reading')
        self.assertIn('logged: unavailable (x)', missing[0])
        missing, lines = gate.dram_findings(dict(dram=['[PINDIAG] dram after kv: 7 GB'], kv_tokens=262400))
        self.assertEqual(len(missing), 1, 'a raw line is not a reading')
        self.assertEqual(lines, ['dram: none logged', 'dram: beside a KV pool of 262400 tokens (vLLM\'s GPU KV cache size)'])

    def test_bringup_fails_on_each_thing_it_checks(self):
        cases = dict(
            programs=dict(faults={'general-prefix': dict(grow_programs=True)}, text='a restore compiled'),
            capture_compiles=dict(faults={'general-prefix': dict(grow_on_capture=True)}, text='a capture compiled'),
            capture_differs=dict(faults={'general-prefix': dict(capture_differs=True)},
                                 text='capturing (fresh salt) is not byte-identical'),
            unsalted=dict(faults={'general-prefix': dict(unsalted_differs=True)}, text='grants disabled (unsalted) is not'),
            publishes=dict(faults={'general-prefix': dict(publish_unsalted=True)}, text='cached tokens'),
            rows=dict(faults={'general-prefix': dict(drop_rows=True)}, text='no [PREFIX] row'),
            anchor=dict(anchor=dict(GOOD_ANCHOR, mismatched=['gdn/tp.py']), text='not the image\'s pinned graft'),
            graft=dict(anchor=dict(GOOD_ANCHOR, prefix_marker_in=[]), text='prefix model graft is not in the served'),
            no_pins=dict(anchor=dict(GOOD_ANCHOR, pins={}), text='read no pins'),
            stage=dict(anchor=dict(GOOD_ANCHOR, files=dict(gate.STAGE_PINS, **{'model.py': SHA['b']}),
                                   stage_mismatched=['model.py']), text='model.py is not the prefix stage\'s graft'),
            unpinned=dict(anchor=dict(GOOD_ANCHOR, unpinned=['qwen36_vllm.py']), text='pinned by neither'))
        for name, case in cases.items():
            with self.subTest(name=name):
                result, _, _ = self.plan('bringup', name, anchor=case.get('anchor', GOOD_ANCHOR), **case.get('faults', {}))
                self.assertEqual(result['verdict'], 'FAIL', name)
                self.assertTrue(any(case['text'] in line for line in result['lines']) or any(
                    case['text'] in p for a in result['arms'].values() for p in a.get('problems') or ()),
                    (name, result['lines']))

    def test_the_capture_path_fault_passes_every_cold_hit_pair(self):
        """The reviewer's false PASS: a capture that disturbs the prefill makes cold equal hit."""
        result, _, _ = self.plan('bringup', 'capture-pairs', **{'general-prefix': dict(capture_differs=True)})
        prefix = result['arms']['bringup-prefix']
        self.assertGreater(prefix['identical'], 0, 'pairs where both runs captured still match')
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual(len([p for p in result['cross_problems'] if 'capturing (fresh salt) is not byte-identical' in p]),
                         3, 'the baseline comparison catches every capturing turn')

    def test_a_missing_install_line_or_a_bad_argv_fails_the_arm(self):
        class NoInstall(fakes.FakeEngine):
            def boot(self):
                super(NoInstall, self).boot()
                self.lines[:] = [line for line in self.lines if 'prefix: install' not in line
                                 and 'Automatic prefix caching' not in line]

        harness = Harness(NoInstall)
        runner = harness.runner(self.results)
        _, result = runner.run(arm_of('bringup', 'bringup-prefix'))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('prefix: install' in p for p in result['problems']))
        self.assertTrue(any('Automatic prefix caching is enabled' in p for p in result['problems']))

    def test_exactness_passes_and_every_arm_runs_its_path(self):
        result, harness, _ = self.plan('exactness')
        self.assertEqual(result['verdict'], 'PASS', result['lines'])
        self.assertEqual([e.path for e in harness.engines], ['traced', 'traced', 'eager'])
        self.assertTrue(harness.engines[1].audit)
        traced = result['arms']['exactness-traced']
        self.assertTrue(any('boundary-2048: a tail-only hit' in line for line in traced['lines']), traced['lines'])
        self.assertTrue(any(line.startswith('chain hits (L, Q)') for line in traced['lines']))
        self.assertTrue(any('gap boundary' in line and 'restored by' in line for line in traced['lines']))

    def test_exactness_fails_on_a_divergent_hit_a_bad_digest_or_a_bad_audit(self):
        for faults, text in ((dict(all=dict(diverge_hits=True)), 'DIVERGED'),
                             (dict(all=dict(diverge_once=True)), 'did not reproduce'),
                             (dict(all=dict(bad_slot_on_hit=True)), 'GDN state after prefill differs')):
            with self.subTest(faults=faults):
                result, _, _ = self.plan('exactness', text.split()[0], **faults)
                traced = result['arms']['exactness-traced']
                self.assertEqual(traced['verdict'], 'FAIL')
                self.assertTrue(any(text in p for p in traced['problems']), traced['problems'][:5])

        class BadAudit(fakes.FakeEngine):
            def say(self, line):
                if '[PREFIX-AUDIT]' in line and ' Q=0 ' not in line:
                    line = line.replace('slot_sha=', 'slot_sha=ff')
                super(BadAudit, self).say(line)

        harness = Harness(BadAudit)
        runner = harness.runner(os.path.join(self.results, 'audit'))
        _, arm_result = runner.run(arm_of('exactness', 'exactness-audit'))
        self.assertEqual(arm_result['verdict'], 'FAIL')
        self.assertTrue(any('GDN slot bytes differ' in p for p in arm_result['problems']))

    def test_rows_without_digests_leave_exactness_not_exercised(self):
        result, _, _ = self.plan('exactness', 'nodigest', all=dict(no_digests=True))
        traced = result['arms']['exactness-traced']
        self.assertEqual(traced['verdict'], 'NOT_EXERCISED')
        self.assertTrue(any('no [PREFIX] row carries slot_sha and logits_sha' in m for m in traced['not_exercised']))

    def test_the_eager_arm_alone_passes_and_names_the_eager_warm(self):
        result, harness, _ = self.plan('exactness-eager')
        self.assertEqual(result['verdict'], 'PASS', result['lines'])
        self.assertEqual(list(result['arms']), ['exactness-eager'])
        self.assertEqual([e.path for e in harness.engines], ['eager'])
        self.assertTrue(any(note.startswith('eager warm: ') and '"page_table_blocks": 4128' in note
                            for note in result['arms']['exactness-eager']['notes']))

    def test_an_eager_arm_without_the_eager_warm_fails(self):
        """G1 v47 (run 36246961161): an image that compiles the eager prefill after the decode trace is parked."""
        result, _, _ = self.plan('exactness-eager', 'nowarm', eager=dict(no_eager_warm=True))
        eager = result['arms']['exactness-eager']
        self.assertEqual(eager['verdict'], 'FAIL')
        self.assertTrue(any('no "%s" line' % pm.EAGER_WARM in p for p in eager['problems']), eager['problems'])

    def test_a_row_that_compiles_fails_an_exactness_arm(self):
        """F3 on every row, not only across a hit: v47's eager arm compiled 133 programs in its first (cold)
        row, and its next prefill hung the device."""
        class Grows(fakes.FakeEngine):
            def say(self, line):
                if line.startswith('[PREFIX] req=') and '-0002-cold' in line:
                    line = line.replace('programs=%d' % self.programs, 'programs=%d' % (self.programs + 133))
                super(Grows, self).say(line)

        result, _, _ = self.plan('exactness-eager', 'grows', engine_class=Grows)
        eager = result['arms']['exactness-eager']
        self.assertEqual(eager['verdict'], 'FAIL')
        self.assertTrue(any('pfx-exactness-eager-0002-cold (path eager, Q=0' in p and 'compiled 133 programs' in p
                            for p in eager['problems']), eager['problems'])

    def test_the_mmio_timeout_line_is_a_named_failure(self):
        line = ('(EngineCore pid=67) ERROR 09-26 15:04:40 [core.py:1233] RuntimeError: MMIO per-op timeout: 4B load '
                'took 49571 us (budget=2 ms), 4 of 4 bytes remaining.')
        scanned = pm.scan([line])
        problems, _, _ = gate.generic_problems(arm_of('exactness', 'exactness-eager'), scanned, [], None,
                                               'general-prefix+eager')
        self.assertTrue(any('MMIO per-op timeout' in p for p in problems), problems)

    def test_an_eager_arm_that_ran_the_traced_loop_fails(self):
        result, _, _ = self.plan('exactness', eager=dict(path='traced'))
        eager = result['arms']['exactness-eager']
        self.assertEqual(eager['verdict'], 'FAIL')
        self.assertTrue(any('path traced, the arm serves eager' in p for p in eager['problems']))

    def test_lifecycle_passes_with_every_event_exercised(self):
        result, harness, _ = self.plan('lifecycle')
        self.assertEqual(result['verdict'], 'PASS', result['lines'])
        evict = harness.engines[0]
        self.assertFalse(os.path.exists(evict.kill_path), 'the kill switch file is removed')
        tiny = result['arms']['lifecycle-tiny']
        self.assertTrue(any('waited for the filler' in line for line in tiny['lines']), tiny['lines'])
        self.assertTrue(any(line.startswith('preempted and resumed: pfx-') for line in tiny['lines']))
        evicted = result['arms']['lifecycle-evict']
        self.assertTrue(any(line.startswith('registry stats before the restart') for line in evicted['lines']))

    def test_a_preempted_hit_that_diverges_after_it_resumes_is_named_not_failed(self):
        harness = Harness(tiny=dict(diverge_after_resume=True))
        _, tiny = harness.runner(self.results).run(arm_of('lifecycle', 'lifecycle-tiny'))
        self.assertEqual(tiny['verdict'], 'NOT_COMPARABLE', tiny['lines'])
        self.assertTrue(all('admissions)' in text for text in tiny['not_comparable']), tiny['not_comparable'])

    def test_the_tiny_arm_on_the_default_pool_is_not_exercised_not_failed(self):
        harness = Harness(tiny={'_env': dict(QWEN36_MAX_TOKENS_ALL_USERS=None)})
        _, tiny = harness.runner(self.results).run(arm_of('lifecycle', 'lifecycle-tiny'))
        self.assertEqual(tiny['verdict'], 'NOT_EXERCISED', tiny['lines'])
        self.assertTrue(any('262,400' in m or '262400' in m for m in tiny['not_exercised']), tiny['not_exercised'])

    def test_lifecycle_says_not_exercised_when_an_event_did_not_happen(self):
        result, _, _ = self.plan('lifecycle', 'events', dev={'_env': dict(VLLM_SERVER_DEV_MODE=None)},
                                 all=dict(no_stats=True))
        evict = result['arms']['lifecycle-evict']
        self.assertEqual(evict['verdict'], 'NOT_EXERCISED')
        self.assertTrue(any('reset_prefix_cache answered 404' in t for t in evict['not_exercised']), evict['not_exercised'])
        for arm in result['arms'].values():
            self.assertTrue(any('no registry stats export' in t for t in arm['not_exercised']), arm['not_exercised'])

    def test_a_kill_switch_that_grants_or_publishes_anyway_fails(self):
        class Leaky(fakes.FakeEngine):
            """Never sees the kill switch file: grants go on."""

            def kill_switch(self, on):
                pass

        harness = Harness(Leaky)
        _, result = harness.runner(self.results).run(arm_of('lifecycle', 'lifecycle-evict'))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('kill switch engaged' in p or 'no "[PINDIAG] prefix: kill switch" line' in p
                            for p in result['problems']), result['problems'])
        harness = Harness(dev=dict(publish_when_killed=True))
        _, result = harness.runner(os.path.join(self.results, 'publish')).run(arm_of('lifecycle', 'lifecycle-evict'))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('cached tokens' in p and 'kill switch' in p for p in result['problems']), result['problems'])

    def test_a_failure_line_outside_the_restart_drill_fails_the_evict_arm(self):
        class Noisy(fakes.FakeEngine):
            def restart(self):
                self.say('ERROR EngineDeadError: the engine core died (docker stop)')
                super(Noisy, self).restart()

        _, result = Harness(Noisy).runner(self.results).run(arm_of('lifecycle', 'lifecycle-evict'))
        self.assertEqual(result['verdict'], 'PASS', result['problems'])
        self.assertTrue(any('during the restart drill' in n for n in result['notes']))

        class Dying(fakes.FakeEngine):
            def reset_prefix_cache(self):
                self.say('ERROR EngineDeadError: the engine core died')
                return super(Dying, self).reset_prefix_cache()

        _, result = Harness(Dying).runner(os.path.join(self.results, 'dying')).run(arm_of('lifecycle', 'lifecycle-evict'))
        self.assertEqual(result['verdict'], 'FAIL')

    def test_timing_records_phases_and_compares_with_the_baseline(self):
        result, _, _ = self.plan('timing')
        self.assertEqual(result['verdict'], 'PASS', result['lines'])
        prefix = result['arms']['timing-prefix']['phases']
        self.assertEqual(sorted(prefix), ['agents-1', 'agents-2'])
        self.assertEqual(prefix['agents-2']['turns'], 4)
        self.assertIn('agents-2', result['comparison'])
        self.assertTrue(any('with reuse vs without' in line for line in result['lines']))

    def test_a_platform_container_is_infra_and_skips_the_arm(self):
        harness = Harness()
        runner = harness.runner(self.results)
        runner.containers = lambda: ['thatch-inference-Qwen-Qwen3.8-27B']
        arms = gate.plan_arms('bringup', 'general-prefix', 'general', profiles())
        result = gate.run_plan('bringup', arms, runner, GOOD_ANCHOR)
        self.assertEqual(result['verdict'], 'INFRA')
        self.assertFalse([c for c in harness.calls if c[:3] == ['docker', 'run', '-d']])

    def test_the_wedge_is_infra(self):
        class Wedged(fakes.FakeEngine):
            def boot(self):
                super(Wedged, self).boot()
                self.say('ERROR llrt.cpp:594 %s 31-25' % pm.WEDGE)

        harness = Harness(Wedged)
        runner = harness.runner(self.results)
        arms = gate.plan_arms('bringup', 'general-prefix', 'general', profiles())
        result = gate.run_plan('bringup', arms, runner, GOOD_ANCHOR)
        self.assertEqual(result['verdict'], 'INFRA')
        self.assertEqual(result['arms']['bringup-prefix']['verdict'], 'INFRA')

    def test_a_scenario_that_raises_is_a_recorded_failure_and_the_container_still_goes(self):
        harness = Harness()
        runner = harness.runner(self.results)
        arm = dict(arm_of('bringup', 'bringup-prefix'))
        arm['scenario'] = 'boom'
        replay.SCENARIOS['boom'] = lambda driver, **kwargs: 1 / 0
        try:
            _, result = runner.run(arm)
        finally:
            replay.SCENARIOS.pop('boom')
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('ZeroDivisionError', result['error'])
        self.assertEqual(harness.calls[-1][:3], ['docker', 'rm', '-f'])

    def test_the_restart_waits_no_longer_than_the_arm_has(self):
        harness = Harness()
        runner = harness.runner(self.results)
        engine = fakes.FakeEngine()
        waits = []
        runner.wait_ready = lambda client, container, seconds: waits.append(seconds)

        class Driver(object):
            def __init__(self, left):
                self.left = left

            def remaining(self):
                return self.left

        follower, container = fakes.FakeLog(engine), fakes.FakeContainer(engine)
        info = runner.restart(container, engine, follower, Driver(400))
        self.assertEqual(waits, [340])
        self.assertEqual(len(info['log_window']), 2)
        with self.assertRaises(replay.OutOfTime):
            runner.restart(container, engine, follower, Driver(30))


class MainTests(unittest.TestCase):
    def setUp(self):
        self.results = tempfile.mkdtemp()
        self.profiles = os.path.join(self.results, 'profiles.json')
        with open(self.profiles, 'w', encoding='utf-8') as handle:
            json.dump(profiles(), handle)

    def tearDown(self):
        shutil.rmtree(self.results, ignore_errors=True)

    def main(self, *extra, **kwargs):
        lines = []
        code = gate.main(['--image', 'img', '--profiles', self.profiles, '--results', os.path.join(self.results, 'out')]
                         + list(extra), devices=['/a', '/b'], log=lines.append, **kwargs)
        return code, lines

    def test_dry_run_prints_every_arm(self):
        code, lines = self.main('--plan', 'bringup,exactness', '--dry-run')
        self.assertEqual(code, 0)
        arms = [json.loads(line) for line in lines[1:]]
        self.assertEqual([a['arm'] for a in arms], ['bringup-reference', 'bringup-prefix', 'exactness-traced',
                                                    'exactness-audit', 'exactness-eager'])
        self.assertIn('QWEN_C2_PROFILES=%s' % gate.DERIVED_MOUNT, arms[3]['docker'])

    def test_the_eager_arm_alone(self):
        code, lines = self.main('--plan', 'exactness-eager', '--dry-run')
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(lines[0])['worst_case_seconds'], 5400 + gate.ARM_OVERHEAD_SECONDS)
        arm, = [json.loads(line) for line in lines[1:]]
        self.assertEqual((arm['plan'], arm['arm'], arm['served']), ('exactness-eager', 'exactness-eager',
                                                                   'general-prefix+eager'))
        for plans in ('exactness,exactness-eager', 'exactness-eager,exactness-eager', 'lifecycle-tiny,lifecycle'):
            with self.subTest(plans=plans):
                code, lines = self.main('--plan', plans, '--dry-run')
                self.assertEqual(code, 2)
                self.assertIn('would run twice', lines[-1])

    def test_refusals(self):
        self.assertEqual(self.main('--plan', 'soak')[0], 2)
        self.assertEqual(self.main('--agents', '1,x')[0], 2)
        self.assertEqual(self.main('--profile', 'general')[0], 2)
        code, lines = self.main('--plan', 'exactness,lifecycle', '--budget-seconds', '3600')
        self.assertEqual(code, 2)
        self.assertIn('run fewer plans per tag', lines[-1])

    def test_a_run_writes_the_summary_and_exits_by_the_verdict(self):
        harness = Harness()

        def factory(*args, **kwargs):
            runner = harness.runner(args[1])
            runner.log = kwargs.get('log', runner.log)
            return runner

        with fakes.burst_aware(lambda: CURRENT['engine']):
            code, lines = self.main('--plan', 'bringup', runner_factory=factory, anchor=GOOD_ANCHOR)
            self.assertEqual(code, 0, lines[-5:])
            with open(os.path.join(self.results, 'out', 'c2-prefix-summary.json'), encoding='utf-8') as handle:
                summary = json.load(handle)
            self.assertEqual((summary['passed'], summary['results']['bringup']['verdict']), (True, 'PASS'))
            self.assertIn('C2_PREFIX profile=general-prefix plans=bringup passed=True', lines[-1])
            harness = Harness(**{'general-prefix': dict(grow_programs=True)})
            code, _ = self.main('--plan', 'bringup', runner_factory=factory, anchor=GOOD_ANCHOR)
            self.assertEqual(code, 1)


class JobTests(unittest.TestCase):
    PROFILES = ['coding', 'exact', 'general', 'general-prefix']

    def read(self, **values):
        base = dict(C2_IMAGE_TAG='v7-prefix', C2_ACTIONS='prefix')
        base.update(values)
        return job.read_job(base, self.PROFILES)

    def test_defaults(self):
        outputs = self.read()
        self.assertEqual((outputs['prefix_plan'], outputs['prefix_profile'], outputs['prefix_baseline'],
                          outputs['prefix_agents']), ('bringup', 'general-prefix', 'general', '1,4,5,6'))
        outputs = job.read_job(dict(C2_IMAGE_TAG='v7-prefix'), ['general'])
        self.assertEqual(outputs['prefix_profile'], 'general-prefix', 'unchecked until the action runs')

    def test_the_keys(self):
        outputs = self.read(C2_ACTIONS='reset prefix', C2_PREFIX_PLAN='exactness, timing',
                            C2_PREFIX_BASELINE='none', C2_PREFIX_AGENTS='1, 4')
        self.assertEqual(outputs['actions'], 'reset prefix')
        self.assertEqual((outputs['prefix_plan'], outputs['prefix_baseline'], outputs['prefix_agents']),
                         ('exactness,timing', 'none', '1,4'))
        outputs = self.read(C2_PREFIX_PLAN='bringup lifecycle')
        self.assertEqual(outputs['prefix_plan'], 'bringup,lifecycle')

    def test_what_is_refused_when_the_action_runs(self):
        for values in (dict(C2_PREFIX_PLAN='soak'), dict(C2_PREFIX_PROFILE='nope'), dict(C2_PREFIX_BASELINE='nope'),
                       dict(C2_PREFIX_AGENTS='0'), dict(C2_PREFIX_AGENTS='x'), dict(C2_PREFIX_BASELINE='none')):
            with self.subTest(values=values), self.assertRaises(job.JobError):
                self.read(**values)
        with self.assertRaises(job.JobError):
            job.read_job(dict(C2_IMAGE_TAG='v7-prefix', C2_ACTIONS='prefix'), ['general'])
        self.assertEqual(job.PREFIX_PLANS + tuple(arm for arm, _ in job.PREFIX_ARM_PLANS), gate.PLANS)

    def test_one_arm_as_its_own_plan(self):
        outputs = self.read(C2_ACTIONS='status reset build prefix', C2_PREFIX_PLAN='exactness-eager')
        self.assertEqual((outputs['actions'], outputs['prefix_plan']), ('status reset build prefix', 'exactness-eager'))
        outputs = self.read(C2_PREFIX_PLAN='exactness-eager, lifecycle-tiny', C2_PREFIX_BASELINE='none')
        self.assertEqual(outputs['prefix_plan'], 'exactness-eager,lifecycle-tiny')
        for plan in ('exactness exactness-eager', 'exactness-eager exactness-eager', 'lifecycle-store lifecycle',
                     'exactness exactness', 'bringup-prefix', 'timing-baseline'):
            with self.subTest(plan=plan), self.assertRaises(job.JobError):
                self.read(C2_PREFIX_PLAN=plan)

    def test_a_job_that_does_not_run_prefix_ignores_its_keys(self):
        """An exact or c2 run is never refused over a prefix key it does not use."""
        for values in (dict(C2_PREFIX_PLAN='soak'), dict(C2_PREFIX_AGENTS='x'), dict(C2_PREFIX_PROFILE='nope')):
            with self.subTest(values=values):
                outputs = self.read(C2_ACTIONS='status smoke gate', **values)
                self.assertEqual((outputs['prefix_plan'], outputs['prefix_agents']), ('bringup', '1,4,5,6'))


class WorkflowTests(unittest.TestCase):
    @staticmethod
    def text():
        with open(WORKFLOW, encoding='utf-8') as handle:
            return handle.read()

    def step(self):
        text = self.text()
        return text[text.index('- name: Prefix-reuse gates'):text.index('- name: Replay the node agent')]

    def test_the_prefix_step_runs_the_driver_with_the_jobs_choices(self):
        step = self.step()
        self.assertIn("if: contains(steps.job.outputs.actions, 'prefix')", step)
        for key in ('tag', 'prefix_profile', 'prefix_plan', 'prefix_baseline', 'prefix_agents'):
            self.assertRegex(step, re.escape(key.upper()) + r': \$\{\{ steps\.job\.outputs\.' + key + r' \}\}')
        command = step[step.index('python3 scripts/ci/c2_prefix_gate.py'):]
        for option in ('--image "$image"', '--profile "$PREFIX_PROFILE"', '--plan "$PREFIX_PLAN"',
                       '--baseline "$PREFIX_BASELINE"', '--agents "$PREFIX_AGENTS"', '--budget-seconds "$budget"',
                       '--results "$results"'):
            self.assertIn(option, command)
        self.assertIn("grep -F '[QWEN-C2]'", step)
        self.assertIn("grep -F '[PINDIAG] prefix: install'", step)

    def test_every_prefix_container_and_only_the_gates_flag_go_with_the_step(self):
        step = self.step()
        self.assertIn("grep '^%s' | xargs -r docker rm -f" % gate.CONTAINER_PREFIX, step)
        self.assertIn('trap cleanup_prefix EXIT', step)
        self.assertLess(step.index('trap cleanup_prefix EXIT'), step.index('python3 scripts/ci/c2_prefix_gate.py'))
        self.assertIn('flag=/home/thatch/hf-cache/hub/.qwen-c2/prefix-reuse.off', step)
        self.assertIn('= %s ]; then sudo -n rm -f "$flag"' % gate.KILL_SWITCH_OWNER, step)
        self.assertEqual(replay.KILL_SWITCH_PATH.replace('/models/', '/home/thatch/hf-cache/hub/'),
                         '/home/thatch/hf-cache/hub/.qwen-c2/prefix-reuse.off')
        self.assertIn('refusing to open M+A under it', step)
        self.assertIn('sudo -n fuser -v "$m" "$a"', step)

    def test_the_budget_is_the_steps_and_the_jobs_own_timeouts(self):
        step = self.step()
        self.assertIn('timeout-minutes: 380', step)
        self.assertIn('step_left=$(( 380 * 60 - 600 ))', step)
        self.assertIn('job_left=$(( 600 * 60 - ($(date +%s) - ${C2_JOB_STARTED:?}) - 900 ))', step)

    def test_the_probe_step_runs_the_oracle_against_the_real_graft(self):
        text = self.text()
        probe = text[text.index('- name: Prefix-reuse P0a probe'):text.index('- name: Smoke on cards M+A')]
        self.assertIn('/c2/scripts/ci/prefix_oracle_check.py', probe)
        self.assertIn('/c2/scripts/ci/prefix_p0a_probe.py', probe)
        for line in probe.splitlines():
            if 'docker run' in line:
                self.assertNotIn('--device', line)
        self.assertIn('exit "$status"', probe)

    def test_the_job_file_names_the_prefix_action_above_the_replay_key(self):
        with open(JOB_FILE, encoding='utf-8') as handle:
            text = handle.read()
        self.assertIn('gate prefix replay push', text)
        for key in ('C2_PREFIX_PLAN', 'C2_PREFIX_PROFILE', 'C2_PREFIX_BASELINE', 'C2_PREFIX_AGENTS'):
            self.assertIn(key, text)
        lines = text.splitlines()
        replay_key = [i for i, line in enumerate(lines) if line.startswith('C2_REPLAY_PROFILE=')][0]
        self.assertTrue(lines[replay_key - 1].startswith('# C2_REPLAY_PROFILE:'), 'the key follows its own comment')
        self.assertLess(max(i for i, line in enumerate(lines) if 'C2_PREFIX_' in line), replay_key - 1)

    def test_every_new_test_module_is_allowlisted(self):
        with open(CPU_WORKFLOW, encoding='utf-8') as handle:
            text = handle.read()
        named = set(re.findall(r'python -B -m unittest ([^\n]+)', text))
        modules = set(token for line in named for token in line.split())
        for module in NEW_TESTS:
            self.assertIn(module, modules, '%s is not run by qwen-integration-cpu.yml' % module)


if __name__ == '__main__':
    unittest.main()
