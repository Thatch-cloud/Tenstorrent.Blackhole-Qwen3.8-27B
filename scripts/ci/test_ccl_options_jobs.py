"""WP5's card templates and manifest on the CPU: the pack (references/fusion-jobs/WP5) is what optimisation/ttnn-op/ccl_sweep/make_pack.py generates, every template parses
through c2_serving_job.py with rc 0 (the twin profiles the integrator's generator makes are built here from the manifest when the profiles file does not have them yet, and the
fabric-probe patch is applied to a copy of the job reader when the tree has not taken it yet), the manifest names real files and a lever value the strict parser accepts."""

import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'optimisation' / 'ttnn-op' / 'ccl_sweep'))

import ccl_options_tp  # noqa: E402
import make_pack  # noqa: E402

PACK = HERE / 'references' / 'fusion-jobs' / 'WP5'
MANIFEST = HERE / 'fusion-wp' / 'WP5.json'
PATCH = HERE / 'fusion-wp' / 'WP5-fabric-probe.patch'
PROFILES = HERE / 'qwen_c2_profiles.json'
WORKFLOW = REPO / '.github' / 'workflows' / 'qwen-c2-serving.yml'
JOB_READER = HERE / 'c2_serving_job.py'


def manifest():
    with open(str(MANIFEST), encoding='utf-8') as handle:
        return json.load(handle)


def profiles_with_twins():
    """The profiles file with the WP5 twins the integrator's generator makes (a gate-only copy of the production profile, without its owner traffic waiver, plus the
    manifest's env); the real file when it already has them."""
    with open(str(PROFILES), encoding='utf-8') as handle:
        data = json.load(handle)
    plan = manifest()
    wanted = {}
    for lever in plan['levers']:
        env = dict(lever['env'], **{lever['flag']: lever['value']})
        wanted['%s-fx-%s' % (make_pack.PRODUCTION, lever['id'])] = env
        wanted['%s-fx-%s-audit' % (make_pack.PRODUCTION, lever['id'])] = dict(env, **dict(lever['audit_env'], **{lever['audit_flag']: '1'}))
    for item in plan['profiles']:
        wanted['%s-fx-%s' % (make_pack.PRODUCTION, item['name'])] = item['env']
    for name, env in wanted.items():
        if name not in data['profiles']:
            twin = copy.deepcopy(data['profiles'][make_pack.PRODUCTION])
            twin['env'].update(env)
            twin.pop('owner_traffic_waiver', None)
            twin['gate_only'] = True
            data['profiles'][name] = twin
    return data, wanted


def job_tree(test):
    """(module, tree): c2_serving_job with the fabric-probe patch in and the profiles file with the WP5 twins, both in a scratch tree (scripts/ci/ and .github/workflows/ as in the repo),
    loaded from there. The tree's own files are copied as they are when it has taken the patch or has the twins; the patch is applied to the copies when it has not."""
    tree = Path(tempfile.mkdtemp(prefix='wp5-tree-'))
    test.addCleanup(shutil.rmtree, str(tree), True)
    for relative in ('scripts/ci/c2_serving_job.py', '.github/workflows/qwen-c2-serving.yml'):
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(REPO / relative), str(target))
    if "'ccl-sweep'" not in JOB_READER.read_text(encoding='utf-8'):
        result = subprocess.run(['git', 'apply', '--unsafe-paths', str(PATCH)], cwd=str(tree), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode:
            raise AssertionError('the fabric-probe patch does not apply: %s' % result.stderr)
    data, _ = profiles_with_twins()
    with open(str(tree / 'scripts' / 'ci' / 'qwen_c2_profiles.json'), 'w', encoding='utf-8') as handle:
        json.dump(data, handle)
    spec = importlib.util.spec_from_file_location('c2_serving_job_wp5', str(tree / 'scripts' / 'ci' / 'c2_serving_job.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, tree


class ThePack(unittest.TestCase):
    def test_the_checked_in_pack_is_what_the_generator_makes(self):
        wanted = make_pack.generate()
        self.assertEqual(make_pack.stale(PACK, wanted), [])
        self.assertEqual(sorted(path.name for path in PACK.glob('*')), sorted(wanted))

    def test_the_pack_is_a_subfolder_the_pack_generator_never_reads(self):
        self.assertTrue(PACK.is_dir())
        self.assertEqual(PACK.parent.name, 'fusion-jobs')
        self.assertEqual(PACK.name, 'WP5')

    def test_order_lists_every_template_once_with_a_mode_and_minutes(self):
        lines = [line.split() for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line and not line.startswith('#')]
        names = [line[0] for line in lines]
        self.assertEqual(sorted(names), sorted(path.stem for path in PACK.glob('*.env')))
        for name, mode, image, minutes in lines:
            self.assertIn(mode, ('stop', 'soft'))
            self.assertEqual(image, make_pack.IMAGE)
            self.assertTrue(int(minutes) > 0)
        self.assertEqual([name for name in names if name.startswith('CCLH')], ['CCLH%d-ccl-hang-shapes' % index for index in range(1, 6)])

    def test_every_template_parses_through_the_job_reader_with_rc_0(self):
        module, tree = job_tree(self)
        profiles = str(tree / 'scripts' / 'ci' / 'qwen_c2_profiles.json')
        count = 0
        for path in sorted(PACK.glob('*.env')):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                status = module.main([str(path), profiles])
            self.assertEqual(status, 0, '%s: %s' % (path.name, err.getvalue()))
            count += 1
        self.assertEqual(count, 5 + 1 + 5)

    def test_the_probe_templates_run_one_fabric_config_and_the_probe_names_the_workflow_knows(self):
        module, tree = job_tree(self)
        workflow = (tree / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
        seen = []
        for path in sorted(PACK.glob('P*.env')):
            values = module.parse_env(path.read_text(encoding='utf-8'))
            self.assertEqual(values['C2_ACTIONS'], 'reset fabric')
            self.assertEqual(values['C2_CARDS'], 'quad')
            self.assertIn(values['C2_FABRIC'], module.FABRIC_CONFIGS)
            self.assertIn(values['C2_FABRIC_PROBE'], module.FABRIC_PROBES)
            self.assertIn('\n            %s) script=tp4_ccl_sweep_probe.py;' % values['C2_FABRIC_PROBE'], workflow)
            self.assertLessEqual(int(values['C2_BOX_MINUTES']), 45)
            seen.append((values['C2_FABRIC'], values['C2_FABRIC_PROBE']))
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(sorted(set(fabric for fabric, _ in seen)), ['FABRIC_1D', 'FABRIC_1D_RING'])
        self.assertIn('${probe_args}', workflow)
        self.assertIn('CCL_SWEEP', workflow)
        for name in ('ccl-sweep', 'ccl-sweep-safe', 'ccl-sweep-p8192', 'ccl-sweep-quick'):
            self.assertIn(name, module.FABRIC_PROBES)

    def test_the_lever_jobs_name_the_profiles_of_the_manifest(self):
        data, wanted = profiles_with_twins()
        for path in sorted(PACK.glob('CCL*.env')):
            text = path.read_text(encoding='utf-8')
            profile = [line.split('=', 1)[1] for line in text.splitlines() if line.startswith('C2_PROFILE=')][0]
            self.assertIn(profile, wanted, path.name)
            self.assertIn(profile, data['profiles'])
        hang = [line.split('=', 1)[1] for line in (PACK / 'CCLH1-ccl-hang-shapes.env').read_text(encoding='utf-8').splitlines() if line.startswith('C2_PROFILE=')]
        self.assertEqual(hang, [make_pack.LEVER])
        self.assertIn('concurrent8_steady', (PACK / 'CCLH1-ccl-hang-shapes.env').read_text(encoding='utf-8'))

    def test_the_production_profile_is_the_fusion_generators_when_it_is_there(self):
        try:
            import make_fusion_profiles
        except ImportError:
            self.skipTest('the integration scaffolding is not in this tree')
        self.assertEqual(make_pack.PRODUCTION, make_fusion_profiles.PARENT)
        self.assertEqual(make_fusion_profiles.NAMESPACE, make_pack.PRODUCTION + '-fx-')


class TheManifest(unittest.TestCase):
    def test_the_lever_is_one_strict_set_with_the_audit_pair_and_the_marker_the_flags_log(self):
        plan = manifest()
        self.assertEqual(plan['wp'], 'WP5')
        self.assertEqual(len(plan['levers']), 1)
        lever = plan['levers'][0]
        self.assertEqual((lever['id'], lever['flag'], lever['audit_flag']), ('ccl', ccl_options_tp.OPTIONS_FLAG, ccl_options_tp.AUDIT_FLAG))
        selection = ccl_options_tp.parse_set(lever['value'])
        self.assertEqual(selection.name, lever['value'], 'the manifest carries the canonical form')
        self.assertTrue(selection.rs and selection.ag)
        self.assertEqual(lever['marker'], 'tp4 ccl options')
        for marker in (ccl_options_tp.ENGAGED_MARKER, ccl_options_tp.FALLBACK_MARKER, ccl_options_tp.AUDIT_MARKER):
            self.assertTrue(marker.startswith('[PINDIAG] ' + lever['marker'] + ' '), marker)

    def test_every_profile_env_is_accepted_by_the_strict_flag_parser(self):
        data, wanted = profiles_with_twins()
        for name, env in wanted.items():
            full = dict(data['profiles'][name]['env'])
            self.assertEqual(full['QWEN_FAST_TP'], '4', name)
            configured = ccl_options_tp.settings(full)
            self.assertIsNotNone(configured, name)
            self.assertEqual(bool(configured.audit_calls), full.get(ccl_options_tp.AUDIT_FLAG) == '1', name)
            self.assertEqual(full[ccl_options_tp.UNIT_MAJOR_FLAG], '1', name)

    def test_the_parent_profile_does_not_already_name_the_flags(self):
        with open(str(PROFILES), encoding='utf-8') as handle:
            parent = json.load(handle)['profiles'][make_pack.PRODUCTION]
        for flag in (ccl_options_tp.OPTIONS_FLAG, ccl_options_tp.AUDIT_FLAG, ccl_options_tp.AUDIT_CALLS_FLAG):
            self.assertNotIn(flag, parent['env'])

    def test_the_manifest_names_files_and_tests_that_exist(self):
        plan = manifest()
        for item in plan['image_files']:
            self.assertTrue((REPO / item['path']).is_file(), item['path'])
            self.assertTrue(item['path'].startswith('scripts/ci/'))
        for item in plan['tests']:
            if isinstance(item, dict):
                self.assertTrue((REPO / item['discover']).is_dir())
                self.assertTrue(list((REPO / item['discover']).glob('test_*.py')))
            else:
                self.assertTrue((HERE / (item + '.py')).is_file(), item)

    def test_the_manifest_keys_are_the_schemas(self):
        self.assertEqual(sorted(set(manifest()) - {'wp', 'branch', 'levers', 'profiles', 'smoke', 'image_files', 'tests', 'tp_addresses', 'notes'}), [])

    def test_the_overlay_does_not_already_carry_the_new_modules_or_lack_the_collective_one(self):
        overlay = (REPO / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8')
        self.assertIn('scripts/ci/tile_collective_tp.py', overlay.splitlines())


class TheProbePatch(unittest.TestCase):
    def test_the_patch_applies_to_the_trees_files_or_the_tree_has_taken_it(self):
        text = JOB_READER.read_text(encoding='utf-8')
        if "'ccl-sweep'" in text:
            self.assertIn('ccl-sweep-quick', text)
            self.assertIn('ccl-sweep-quick) script=tp4_ccl_sweep_probe.py', WORKFLOW.read_text(encoding='utf-8'))
            return
        directory = tempfile.mkdtemp(prefix='wp5-patch-')
        self.addCleanup(shutil.rmtree, directory, True)
        for relative in ('scripts/ci/c2_serving_job.py', '.github/workflows/qwen-c2-serving.yml'):
            target = Path(directory) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(REPO / relative), str(target))
        result = subprocess.run(['git', 'apply', '--check', '--unsafe-paths', str(PATCH)], cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_probe_names_in_the_patch_are_the_arguments_the_wrapper_takes(self):
        import tp4_ccl_sweep_probe
        parser = tp4_ccl_sweep_probe.build_parser()
        text = PATCH.read_text(encoding='utf-8')
        for line in text.splitlines():
            if line.startswith('+') and 'probe_args="' in line:
                arguments = line.split('probe_args="')[1].split('"')[0].split()
                options = parser.parse_args(['--output', 'x'] + arguments)
                self.assertTrue(options.skip_probe_only or options.quick or not arguments)


if __name__ == '__main__':
    unittest.main()
