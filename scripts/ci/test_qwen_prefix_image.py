"""G1 of the TT prefix-reuse design, the image track: the general-prefix profiles, the contract's
prefix guard, the build's prefix stage (qwen_prefix_stage.py) and its plumbing into the C2 image.

Reads the checkout (the Dockerfile, the overlay manifest, the workflows), so it is not overlaid into
the image; test_qwen_prefix_metrics is. The pins are held to the plugin and model sources where a
local copy exists (QWEN_TT_PLUGIN_CHECKOUT, QWEN_IMG_TREE, or the job directory's), else skipped.
"""

import hashlib
import importlib.util
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


class ProfileTests(unittest.TestCase):
    def test_general_prefix_is_the_argv_p0a_passed(self):
        general, prefix = load('general'), load('general-prefix')
        self.assertEqual(prefix['engine'], p0a_general_prefix_engine(general))
        self.assertEqual(contract.engine_arguments(prefix, '/snap'),
                         contract.engine_arguments(dict(general, engine=p0a_general_prefix_engine(general)), '/snap'))
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

    def test_the_prefix_profiles_launch_with_prefix_caching_and_chunking_on(self):
        for name in PREFIX_PROFILES:
            with self.subTest(profile=name):
                argv = contract.engine_arguments(load(name), '/snap')
                for flag in ('--enable-prefix-caching', '--enable-chunked-prefill', '--no-async-scheduling'):
                    self.assertIn(flag, argv)
                for flag in ('--no-enable-prefix-caching', '--no-enable-chunked-prefill', '--speculative-config'):
                    self.assertNotIn(flag, argv)
                self.assertEqual(argv[argv.index('--block-size') + 1], '64')
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
        self.assertEqual(profiles()['default'], 'general', 'general-prefix becomes the default only at release')

    def test_the_prefix_profiles_pass_the_guard_and_the_scheduler_graft_s_install_rules(self):
        for name in PREFIX_PROFILES:
            with self.subTest(profile=name):
                profile = load(name)
                self.assertEqual(contract.prefix_reuse_problems(profile), [])
                engine = profile['engine']
                # prefix_scheduler_graft.install_problems: whole-prompt budget, block 64, no lookahead
                self.assertGreaterEqual(engine['max-num-batched-tokens'], engine['max-model-len'])
                self.assertEqual(engine['block-size'], graft.BLOCK)
                self.assertEqual(engine['max-model-len'] % graft.CHUNK, 0)


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


class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='qwen-prefix-stage-'))
        self.modules = self.tmp / 'modules'
        self.modules.mkdir()
        (self.modules / 'fake_prefix_stage.py').write_text(STAGE_MODULE, encoding='utf-8')
        self.image = self.tmp / 'image'
        self.targets = stage_tree(self.image, (('t/a.py', 'ANCHOR_A = 1\n'), ('t/b.py', 'B = 2\n')))
        self.record = str(self.tmp / 'record.json')
        self.logs = []

    def tearDown(self):
        sys.modules.pop('fake_prefix_stage', None)
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
        with open(self.record, encoding='utf-8') as handle:
            self.assertEqual(json.load(handle), json.loads(json.dumps(result)))

    def test_no_stages_checks_the_anchors_and_records_the_originals(self):
        result = self.apply(())
        self.assertFalse(result['complete'])
        self.assertEqual([row['after'] == row['before'] == row['pin'] for row in result['targets'].values()],
                         [True, True])
        self.assertTrue(any('NOT grafted' in line for line in self.logs))

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
        record = dict(schema=stage.SCHEMA, targets=rows, stages=[], complete=False,
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
        foreign['targets']['plugin/worker.py']['before'] = '1' * 64
        foreign['stages'] = [dict(module='qwen_prefix_scheduler_patch', sha256='2' * 64)]
        problems, _ = stage.record_problems(foreign, shas, {'qwen_prefix_scheduler_patch': '3' * 64})
        self.assertEqual(len(problems), 2, problems)
        self.assertIn('plugin/worker.py: the record has path', problems[0])
        self.assertIn('ran at %s; the context overlays %s' % ('2' * 64, '3' * 64), problems[1])

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

    def test_the_stage_is_a_build_tool_not_an_overlay_file(self):
        self.assertIn(c2_overlay.PREFIX_STAGE, c2_overlay.TOOLS)
        self.assertEqual(c2_overlay.PREFIX_STAGE, 'scripts/ci/qwen_prefix_stage.py')
        self.assertNotIn(c2_overlay.PREFIX_STAGE, manifest_sources())
        self.assertIn(c2_overlay.PREFIX_STAGE, c2_overlay.staged_paths(c2_overlay.read_manifest(MANIFEST)))

    def test_every_g1_module_is_overlaid(self):
        """A G1 runtime or stage module that is not in the manifest never reaches the image (memory
        serving-image-bundle-provenance), and qwen_prefix_stage refuses a stage module it cannot find."""
        sources = manifest_sources()
        names = sorted(path.name for path in HERE.glob('qwen_prefix_*.py')) + ['prefix_scheduler_graft.py']
        wanted = ['scripts/ci/' + name for name in names if name != 'qwen_prefix_stage.py']
        wanted += ['scripts/ci/%s.py' % module for _, module, _ in stage.STAGES]
        self.assertIn('scripts/ci/qwen_prefix_metrics.py', wanted)
        self.assertEqual(sorted(set(wanted) - sources), [])
        self.assertIn('scripts/ci/test_qwen_prefix_metrics.py', sources, 'runs inside the image at build')
        self.assertNotIn('scripts/ci/test_qwen_prefix_image.py', sources, 'reads the checkout: CPU suite only')
        self.assertNotIn('scripts/ci/prefix_p0a_probe.py', sources, 'the probe mounts the checkout')

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
