"""c2_prefix_gate: the prefix-reuse (G1) gates in the node agent's container shape, held on CPU.

Nothing here opens a device or runs docker. The arms' `docker run` argv (the S1 gate's agent shape,
detached, the platform's argv, a derived profile mounted where one is needed), what each plan
refuses before any container, and the whole Runner against test_prefix_replay.FakeEngine - one per
arm, built from the docker argv the Runner passes - with faults switched on to see each verdict. The
workflow step, the job keys and the CPU allowlist are read from the files."""

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


def profiles():
    """The checkout's profiles plus general-prefix (design 2.0.1 item 5; the image track adds it)."""
    document = copy.deepcopy(marker_fixture.PROFILES)
    document['profiles']['general-prefix'] = marker_fixture.prefix_profile()
    return document


GOOD_ANCHOR = dict(files={}, pins={}, mismatched=[], marker_files=['model.py'])


class Harness(object):
    """The Runner's docker, client, log and container, faked; a FakeEngine per arm, shaped by the
    arm's own docker argv (its profile, its derived profile file)."""

    def __init__(self, engine_class=None, **faults):
        self.engine_class = engine_class or FloodEngine
        self.faults = faults
        self.engine = None
        self.engines = []
        self.calls = []

    def docker(self, arguments, timeout=600):
        self.calls.append(arguments)
        if arguments[:3] == ['docker', 'run', '-d']:
            profile = [a.split('=', 1)[1] for a in arguments if a.startswith('QWEN_C2_PROFILE=')][0]
            derived = [a for a in arguments if a.startswith('type=bind,src=') and gate.DERIVED_MOUNT in a]
            if derived:
                path = derived[0].split('src=', 1)[1].split(',', 1)[0]
                with open(path, encoding='utf-8') as handle:
                    doc = json.load(handle)
                assert profile in doc['profiles'], (profile, list(doc['profiles']))
            prefix = profile != 'general'
            options = dict(prefix=prefix, profile=profile, path='eager' if profile.endswith('+eager') else 'traced',
                           audit=profile.endswith('+audit'), store=3 if profile.endswith('+store') else None,
                           store_gib=gate.SMALL_STORE_GIB if profile.endswith('+store') else None,
                           kv_tokens=gate.TINY_BLOCKS * 64 if profile.endswith('+tiny') else 262144,
                           preempt_on_ignore_eos=1 if profile.endswith('+tiny') else 0,
                           dev_mode=profile.endswith(('+dev', '+tiny')))
            options.update(self.faults.get(profile.split('+')[-1] if '+' in profile else profile, {}))
            options.update(self.faults.get('all', {}))
            self.engine = self.engine_class(**options)
            self.engines.append(self.engine)
            return 0, 'container-id'
        if arguments[:3] == ['docker', 'logs', '--timestamps']:
            return 0, '\n'.join(self.engine.lines) if self.engine else ''
        return 0, ''

    def client(self):
        harness = self

        class Proxy(object):
            def __getattr__(self, name):
                return getattr(harness.engine, name)

        return Proxy()

    def runner(self, results):
        return gate.Runner('img', results, ROOT, ['/dev/tenstorrent/1', '/dev/tenstorrent/0'], docker=self.docker,
                           make_client=self.client, make_log=lambda name, path: fakes.FakeLog(self.engine_proxy()),
                           make_container=lambda name: fakes.FakeContainer(self.engine_proxy()),
                           containers=lambda: [], log=lambda text: None, sleep=lambda seconds: None,
                           corpus=fakes.CORPUS, agents=(1, 2), turns=2)

    def engine_proxy(self):
        harness = self

        class Proxy(object):
            def __getattr__(self, name):
                return getattr(harness.engine, name)

            def __setattr__(self, name, value):
                setattr(harness.engine, name, value)

        return Proxy()


class FloodEngine(fakes.FakeEngine):
    """FakeEngine whose pool evicts: a flood request drops the first two evict- conversations'
    published blocks and their checkpoints (coupled), as a full pool pops its LRU head."""

    def chat(self, body, tag, salt=None, **kwargs):
        if tag.endswith('-flood'):
            with self.lock:
                for conv in ('evict-0', 'evict-1'):
                    for key in [k for k in self.oracle.published if k.endswith('-' + conv)]:
                        self.oracle.published.pop(key)
                    for key in [k for k in self.oracle.checkpoints if k[0].endswith('-' + conv)]:
                        self.oracle.checkpoints.pop(key)
        return super(FloodEngine, self).chat(body, tag, salt, **kwargs)


def run_plan(plan, results, anchor=GOOD_ANCHOR, baseline='general', **faults):
    harness = Harness(**faults)
    runner = harness.runner(results)
    arms = gate.plan_arms(plan, 'general-prefix', baseline, profiles())
    return gate.run_plan(plan, arms, runner, anchor), harness, runner


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
        tiny = gate.derive(document, 'general-prefix', 'tiny')[1]['profiles']['general-prefix+tiny']
        self.assertEqual(tiny['engine']['num-gpu-blocks-override'], gate.TINY_BLOCKS)
        self.assertGreaterEqual(gate.TINY_BLOCKS * 64, tiny['engine']['max-model-len'],
                                'one request of the full context still fits the tiny pool')
        store = gate.derive(document, 'general-prefix', 'store')[1]['profiles']['general-prefix+store']
        self.assertEqual(store['env']['QWEN_PREFIX_STORE_GIB'], '0.5')
        self.assertEqual(judge.store_entries(gate.SMALL_STORE_GIB), 3)
        self.assertNotIn('+', json.dumps(document['profiles']['general-prefix']), 'the image profile is untouched')

    def test_what_is_refused_before_any_container(self):
        document = profiles()
        for plan, profile, baseline in (('bringup', 'general', 'general'), ('bringup', 'nope', 'general'),
                                        ('bringup', 'general-prefix', 'none'), ('timing', 'general-prefix', 'nope'),
                                        ('timing', 'general-prefix', 'general-prefix')):
            with self.subTest(plan=plan, profile=profile, baseline=baseline), self.assertRaises(gate.PlanError):
                gate.plan_arms(plan, profile, baseline, document)
        with self.assertRaises(ValueError):
            gate.plan_arms('soak', 'general-prefix', 'general', document)

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

    def test_the_contract_reads_the_derived_file(self):
        """serving_c2_contract.boot loads QWEN_C2_PROFILES: the derived file is what serves."""
        name, document = gate.derive(profiles(), 'general-prefix', 'eager')
        directory = tempfile.mkdtemp()
        try:
            path = os.path.join(directory, 'profiles.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(document, handle)
            profile = contract.load_profile(path, name)
            argv = contract.rewrite_argv(['api_server.py'] + gate.PLATFORM_ARGS, profile, profile['snapshots'][0])
            self.assertEqual(pm.prefix_argv_problems(argv[1:]), [])
            tt = json.loads(argv[argv.index('--additional-config') + 1])['tt']
            self.assertEqual(tt['trace_mode'], 'decode_only')
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

    def test_the_anchor_probe_reads_shas_pins_and_markers(self):
        sha = lambda c: c * 64  # noqa: E731
        text = '\n'.join(['%s  %s' % (sha('a'), path) for path in gate.ANCHOR_FILES] + [
            '==pins', '%s  graft/attention/tp.py.orig' % sha('0')] +
            ['%s  graft/%s' % (sha('a'), path) for path in gate.GRAFTED if path != 'mlp.py'] +
            ['%s  graft/mlp.py' % sha('b'), '==markers', './model.py', './prefix_restore.py'])
        anchor = gate.parse_anchor(text)
        self.assertEqual(anchor['mismatched'], ['mlp.py'])
        self.assertEqual(anchor['marker_files'], ['model.py', 'prefix_restore.py'])
        self.assertEqual(anchor['files']['qwen36_vllm.py'], sha('a'))
        self.assertEqual(anchor['pins']['attention/tp.py'], sha('a'), '.orig pins are the sources, not the graft')

        class Result(object):
            stdout, returncode = text.encode(), 0

        seen = []
        probe = gate.anchor_probe('img', run=lambda arguments, **k: seen.append(arguments) or Result())
        self.assertEqual(probe['mismatched'], ['mlp.py'])
        self.assertEqual(seen[0][:6], ['docker', 'run', '--rm', '--network', 'none', '--entrypoint'])
        self.assertNotIn('--device', seen[0])


class RunnerTests(unittest.TestCase):
    """Whole arms against the fake engine. The chain and the KV flood are shortened (the fake's
    'tokenizer' is slow); every case still runs: the variants fork after turn 3 of 5."""

    def setUp(self):
        self.results = tempfile.mkdtemp()
        for name, value in (('CHAIN_HITS', (4200, 9000, 16500, 24500)), ('EVICT_LENGTHS', (12000, 24000))):
            patcher = mock.patch.object(replay, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        shutil.rmtree(self.results, ignore_errors=True)

    def test_bringup_passes_on_a_good_engine_and_keeps_its_files(self):
        result, harness, _ = run_plan('bringup', self.results)
        self.assertEqual(result['verdict'], 'PASS', result['lines'])
        self.assertEqual([a['verdict'] for a in result['arms'].values()], ['PASS', 'PASS'])
        self.assertTrue(any('grants disabled vs baseline, turn 1' in line and 'IDENTICAL' in line for line in result['lines']))
        for name in ('server-final.log', 'records.jsonl', 'pairs.json', 'events.json', 'arm.json', 'docker-run.json'):
            self.assertTrue(os.path.exists(os.path.join(self.results, 'bringup-prefix', name)), name)
        removed = [c for c in harness.calls if c[:3] == ['docker', 'rm', '-f']]
        self.assertEqual(len(removed), 4, 'each container removed before and after its arm')

    def test_bringup_fails_on_each_thing_it_checks(self):
        cases = dict(
            programs=dict(faults={'general-prefix': dict(grow_programs=True)}, text='program cache grew'),
            unsalted=dict(faults={'general-prefix': dict(unsalted_differs=True)}, text='not byte-identical'),
            rows=dict(faults={'general-prefix': dict(drop_rows=True)}, text='no [PREFIX] row'),
            anchor=dict(anchor=dict(GOOD_ANCHOR, mismatched=['gdn/tp.py']), text='not the image\'s pinned graft'),
            graft=dict(anchor=dict(GOOD_ANCHOR, marker_files=[]), text='prefix model graft is not in the image'))
        for name, case in cases.items():
            with self.subTest(name=name):
                result, _, _ = run_plan('bringup', os.path.join(self.results, name), anchor=case.get('anchor', GOOD_ANCHOR),
                                        **case.get('faults', {}))
                self.assertEqual(result['verdict'], 'FAIL', name)
                self.assertTrue(any(case['text'] in line for line in result['lines']) or any(
                    case['text'] in p for a in result['arms'].values() for p in a.get('problems') or ()),
                    (name, result['lines']))

    def test_a_missing_install_line_or_a_bad_argv_fails_the_arm(self):
        harness = Harness()
        runner = harness.runner(self.results)
        arms = gate.plan_arms('bringup', 'general-prefix', 'general', profiles())
        original = FloodEngine.boot

        def boot(engine):
            original(engine)
            engine.lines[:] = [line for line in engine.lines if 'prefix: install' not in line
                               and 'Automatic prefix caching' not in line]

        FloodEngine.boot = boot
        try:
            _, result = runner.run(arms[1])
        finally:
            FloodEngine.boot = original
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('prefix: install' in p for p in result['problems']))
        self.assertTrue(any('Automatic prefix caching is enabled' in p for p in result['problems']))

    def test_exactness_passes_and_every_arm_runs_its_path(self):
        result, harness, _ = run_plan('exactness', self.results)
        self.assertEqual(result['verdict'], 'PASS', result['lines'])
        paths = [e.path for e in harness.engines]
        self.assertEqual(paths, ['traced', 'traced', 'eager'])
        traced = result['arms']['exactness-traced']
        self.assertTrue(any('boundary-2048: a tail-only hit' in line for line in traced['lines']), traced['lines'])
        self.assertTrue(any(line.startswith('chain hits (L, Q)') for line in traced['lines']))

    def test_exactness_fails_on_a_divergent_hit_or_a_bad_audit(self):
        result, _, _ = run_plan('exactness', self.results, all=dict(diverge_hits=True))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(all(a['verdict'] == 'FAIL' for a in result['arms'].values()))

        class BadAudit(FloodEngine):
            def say(self, line):
                if '[PREFIX-AUDIT]' in line and ' Q=0 ' not in line:
                    line = line.replace('slot_sha=', 'slot_sha=ff')
                super(BadAudit, self).say(line)

        harness = Harness(engine_class=BadAudit)
        runner = harness.runner(os.path.join(self.results, 'audit'))
        arm = gate.plan_arms('exactness', 'general-prefix', 'general', profiles())[1]
        _, arm_result = runner.run(arm)
        self.assertEqual(arm_result['verdict'], 'FAIL')
        self.assertTrue(any('GDN slot bytes differ' in p for p in arm_result['problems']))

    def test_an_eager_arm_that_ran_the_traced_loop_fails(self):
        result, _, _ = run_plan('exactness', self.results, eager=dict(path='traced'))
        eager = result['arms']['exactness-eager']
        self.assertEqual(eager['verdict'], 'FAIL')
        self.assertTrue(any('path traced, the arm serves eager' in p for p in eager['problems']))

    def test_lifecycle_passes_with_every_event_exercised(self):
        result, harness, _ = run_plan('lifecycle', self.results)
        self.assertEqual(result['verdict'], 'PASS', result['lines'])
        evict = harness.engines[0]
        self.assertFalse(evict.flag, 'the kill switch file is removed')
        tiny = result['arms']['lifecycle-tiny']
        self.assertTrue(any('tiny pool' in line for line in tiny['lines']))

    def test_lifecycle_says_not_exercised_when_an_event_did_not_happen(self):
        result, _, _ = run_plan('lifecycle', self.results, tiny=dict(preempt_on_ignore_eos=0),
                                dev=dict(dev_mode=False))
        self.assertEqual(result['arms']['lifecycle-tiny']['verdict'], 'NOT_EXERCISED')
        self.assertTrue(any('no preemption' in text for text in result['arms']['lifecycle-tiny']['not_exercised']))
        evict = result['arms']['lifecycle-evict']
        self.assertEqual(evict['verdict'], 'NOT_EXERCISED')
        self.assertTrue(any('reset_prefix_cache answered 404' in t for t in evict['not_exercised']), evict['not_exercised'])

    def test_a_kill_switch_that_grants_anyway_fails(self):
        class Leaky(FloodEngine):
            """Ignores the kill switch file: grants go on."""

            def chat(self, body, tag, salt=None, **kwargs):
                self.flag = False
                return super(Leaky, self).chat(body, tag, salt, **kwargs)

        harness = Harness(engine_class=Leaky)
        runner = harness.runner(self.results)
        arm = gate.plan_arms('lifecycle', 'general-prefix', 'general', profiles())[0]
        _, result = runner.run(arm)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('kill switch engaged' in p or 'no "[PINDIAG] prefix: kill switch" line' in p
                            for p in result['problems']), result['problems'])

    def test_timing_records_phases_and_compares_with_the_baseline(self):
        result, _, _ = run_plan('timing', self.results)
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
        class Wedged(FloodEngine):
            def boot(self):
                super(Wedged, self).boot()
                self.say('ERROR llrt.cpp:594 %s 31-25' % pm.WEDGE)

        harness = Harness(engine_class=Wedged)
        runner = harness.runner(self.results)
        arms = gate.plan_arms('bringup', 'general-prefix', 'general', profiles())
        result = gate.run_plan('bringup', arms, runner, GOOD_ANCHOR)
        self.assertEqual(result['verdict'], 'INFRA')
        self.assertEqual(result['arms']['bringup-prefix']['verdict'], 'INFRA')

    def test_a_scenario_that_raises_is_a_recorded_failure_and_the_container_still_goes(self):
        harness = Harness()
        runner = harness.runner(self.results)
        arm = dict(gate.plan_arms('bringup', 'general-prefix', 'general', profiles())[1])
        arm['scenario'] = 'boom'
        replay.SCENARIOS['boom'] = lambda driver, **kwargs: 1 / 0
        try:
            _, result = runner.run(arm)
        finally:
            replay.SCENARIOS.pop('boom')
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('ZeroDivisionError', result['error'])
        self.assertEqual(harness.calls[-1][:3], ['docker', 'rm', '-f'])


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
        base = dict(C2_IMAGE_TAG='v7-prefix')
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
        outputs = self.read(C2_ACTIONS='prefix', C2_PREFIX_PLAN='bringup lifecycle')
        self.assertEqual(outputs['prefix_plan'], 'bringup,lifecycle')

    def test_what_is_refused(self):
        for values in (dict(C2_PREFIX_PLAN='soak'), dict(C2_PREFIX_PROFILE='nope'), dict(C2_PREFIX_BASELINE='nope'),
                       dict(C2_PREFIX_AGENTS='0'), dict(C2_PREFIX_AGENTS='x'), dict(C2_PREFIX_BASELINE='none')):
            with self.subTest(values=values), self.assertRaises(job.JobError):
                self.read(**values)
        with self.assertRaises(job.JobError):
            job.read_job(dict(C2_IMAGE_TAG='v7-prefix', C2_ACTIONS='prefix'), ['general'])
        self.assertEqual(job.PREFIX_PLANS, gate.PLANS)


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

    def test_the_job_file_names_the_prefix_action(self):
        with open(JOB_FILE, encoding='utf-8') as handle:
            text = handle.read()
        self.assertIn('gate prefix replay push', text)
        for key in ('C2_PREFIX_PLAN', 'C2_PREFIX_PROFILE', 'C2_PREFIX_BASELINE', 'C2_PREFIX_AGENTS'):
            self.assertIn(key, text)

    def test_every_new_test_module_is_allowlisted(self):
        with open(CPU_WORKFLOW, encoding='utf-8') as handle:
            text = handle.read()
        named = set(re.findall(r'python -B -m unittest ([^\n]+)', text))
        modules = set(token for line in named for token in line.split())
        for module in NEW_TESTS:
            self.assertIn(module, modules, '%s is not run by qwen-integration-cpu.yml' % module)


if __name__ == '__main__':
    unittest.main()
