"""Tests of vllm030_suites.py (the manifest-driven runner of qwen-vllm030-cpu.yml) and of the workflow that calls it.

The runner is exercised on small throwaway trees with real unittest modules, through the real required_tests.py in a child
interpreter: a skip fails, an import error fails, a missing suite fails unless the per-branch manifest excuses it, an excuse that
outlives its reason fails (whatever branch it names), a test file the suite would never run fails, a module that shares its name with
one of scripts/ci fails to be shadowed by it, a hung suite times out with its stacks in the log, and a run in which nothing ran
fails. The workflow checks read .github/workflows (not present inside the serving image, where they skip): triggers, runner label,
Python 3.10, pins that agree with the manifest, action pinning like the repo's other CPU workflows, the CI scaffolding checked out
apart from the tree under test, and that this file is on the integration workflow's allowlist in the form the generator writes.

    python -B -m unittest test_vllm030_suites          (PYTHONPATH=scripts/ci; PyYAML for the workflow checks)
"""

import contextlib
import fnmatch
import io
import json
import os
import re
import sys
import tempfile
import textwrap
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import vllm030_suites as suites  # noqa: E402

WORKFLOWS = os.path.join(ROOT, '.github', 'workflows')
NEW_WORKFLOW = os.path.join(WORKFLOWS, 'qwen-vllm030-cpu.yml')
INTEGRATION_WORKFLOW = os.path.join(WORKFLOWS, 'qwen-integration-cpu.yml')

PASSING = '''
import unittest
class T(unittest.TestCase):
    def test_a(self):
        pass
    def test_b(self):
        pass
'''
SKIPPING = '''
import unittest
class T(unittest.TestCase):
    def test_a(self):
        pass
    @unittest.skip('no device')
    def test_b(self):
        pass
'''
BROKEN_IMPORT = '''
import module_that_does_not_exist
'''
SHARED_HELPER = '''
VALUE = 7
'''
SHADOWS_A_CI_MODULE = '''
import unittest
import vllm030_suites
class T(unittest.TestCase):
    def test_the_suite_s_own_module_wins_over_scripts_ci(self):
        self.assertEqual(vllm030_suites.VALUE, 7)
'''
HANGS = '''
import threading
import unittest
class T(unittest.TestCase):
    def test_waits_forever(self):
        threading.Event().wait()
'''
LOADED_FROM_ELSEWHERE = '''
import sys
import types
import unittest
elsewhere = types.ModuleType('test_x')
elsewhere.__file__ = '/nowhere/test_x.py'
class T(unittest.TestCase):
    def test_a(self):
        pass
elsewhere.T = T
sys.modules['test_x'] = elsewhere
'''
USES_HELPER = '''
import unittest
import helper_mod
class T(unittest.TestCase):
    def test_helper_is_importable_by_plain_name(self):
        self.assertEqual(helper_mod.VALUE, 7)
'''
USES_EXTRA_PATH = '''
import unittest
import extra_pkg
class T(unittest.TestCase):
    def test_extra_path(self):
        self.assertEqual(extra_pkg.NAME, 'extra')
'''


def tree(files):
    """A temporary directory holding {relative path: text}; -> (TemporaryDirectory, root)."""
    holder = tempfile.TemporaryDirectory(prefix='vllm030_')
    for relative, text in files.items():
        path = os.path.join(holder.name, *relative.split('/'))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(textwrap.dedent(text))
    return holder, holder.name


def manifest(suite_overrides=None, absent_ok=None):
    base = {'one': {'dir': 'one/tests', 'glob': 'test_*.py', 'pythonpath': [], 'require': ['test_x'], 'min_tests': 2}}
    if suite_overrides:
        base = suite_overrides
    return {'schema': 1, 'pins': {}, 'suites': base, 'absent_ok': absent_ok or {}}


def quiet_run(spec, root, branch, environment=None):
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        code, results = suites.run(spec, root, branch, sys.executable, os.environ if environment is None else environment)
    return code, {r['suite']: r for r in results}, sink.getvalue()


class RunnerTests(unittest.TestCase):
    def test_every_test_ran_passes(self):
        holder, root = tree({'one/tests/test_x.py': PASSING})
        with holder:
            code, results, output = quiet_run(manifest(), root, 'any')
        self.assertEqual(code, 0, output)
        self.assertEqual(results['one']['status'], suites.PASS)
        self.assertEqual(results['one']['ran'], 2)

    def test_a_skipped_test_fails_the_suite(self):
        holder, root = tree({'one/tests/test_x.py': SKIPPING})
        with holder:
            code, results, output = quiet_run(manifest(), root, 'any')
        self.assertEqual(code, 1)
        self.assertEqual(results['one']['status'], suites.FAIL)
        self.assertIn('skipped', results['one']['detail'])

    def test_a_module_that_does_not_import_fails_the_suite(self):
        holder, root = tree({'one/tests/test_x.py': PASSING, 'one/tests/test_y.py': BROKEN_IMPORT})
        with holder:
            code, results, output = quiet_run(manifest(), root, 'any')
        self.assertEqual(code, 1)
        self.assertEqual(results['one']['status'], suites.FAIL)

    def test_fewer_tests_than_the_floor_fails(self):
        holder, root = tree({'one/tests/test_x.py': PASSING})
        spec = manifest({'one': {'dir': 'one/tests', 'require': ['test_x'], 'min_tests': 3}})
        with holder:
            code, results, output = quiet_run(spec, root, 'any')
        self.assertEqual(code, 1)
        self.assertIn('at least 3 required', results['one']['detail'])

    def test_a_new_test_file_is_found_without_editing_the_manifest(self):
        holder, root = tree({'one/tests/test_x.py': PASSING, 'one/tests/test_new.py': PASSING})
        spec = manifest({'one': {'dir': 'one/tests', 'require': ['test_x'], 'min_tests': 4}})
        with holder:
            code, results, output = quiet_run(spec, root, 'any')
        self.assertEqual(code, 0, output)
        self.assertEqual(results['one']['ran'], 4)

    def test_a_required_module_that_vanished_fails_even_if_others_pass(self):
        holder, root = tree({'one/tests/test_other.py': PASSING})
        with holder:
            code, results, output = quiet_run(manifest(), root, 'any')
        self.assertEqual(code, 1)
        self.assertIn('test_x', results['one']['detail'])

    def test_a_directory_with_no_matching_module_fails(self):
        holder, root = tree({'one/tests/readme.txt': 'x'})
        spec = manifest({'one': {'dir': 'one/tests', 'require': [], 'min_tests': 1}})
        with holder:
            code, results, output = quiet_run(spec, root, 'any')
        self.assertEqual(code, 1)
        self.assertIn('no test module', results['one']['detail'])

    def test_helpers_are_imported_by_plain_name_and_extra_paths_apply(self):
        holder, root = tree({'one/tests/test_x.py': USES_HELPER, 'one/tests/helper_mod.py': SHARED_HELPER,
                             'one/tests/test_z.py': USES_EXTRA_PATH, 'extra/extra_pkg/__init__.py': "NAME = 'extra'\n"})
        spec = manifest({'one': {'dir': 'one/tests', 'pythonpath': ['extra'], 'require': ['test_x', 'test_z'], 'min_tests': 2}})
        with holder:
            code, results, output = quiet_run(spec, root, 'any')
        self.assertEqual(code, 0, output)
        self.assertEqual(results['one']['ran'], 2)

    def test_a_missing_suite_directory_fails(self):
        holder, root = tree({'two/tests/test_x.py': PASSING})
        spec = manifest({'one': {'dir': 'one/tests', 'require': ['test_x'], 'min_tests': 1},
                         'two': {'dir': 'two/tests', 'require': ['test_x'], 'min_tests': 1}})
        with holder:
            code, results, output = quiet_run(spec, root, 'branch-a')
        self.assertEqual(code, 1)
        self.assertEqual(results['one']['status'], suites.FAIL)
        self.assertEqual(results['two']['status'], suites.PASS)

    def test_the_manifest_can_excuse_a_missing_suite_for_its_own_branch_only(self):
        spec = manifest({'one': {'dir': 'one/tests', 'require': ['test_x'], 'min_tests': 1},
                         'two': {'dir': 'two/tests', 'require': ['test_x'], 'min_tests': 1}},
                        absent_ok={'branch-a': {'one': 'arrives with another branch'}})
        holder, root = tree({'two/tests/test_x.py': PASSING})
        with holder:
            code, results, output = quiet_run(spec, root, 'branch-a')
            other_code, other, other_output = quiet_run(spec, root, 'branch-b')
        self.assertEqual(code, 0, output)
        self.assertEqual(results['one']['status'], suites.ABSENT_ALLOWED)
        self.assertIn('arrives with another branch', results['one']['detail'])
        self.assertEqual(other_code, 1)
        self.assertEqual(other['one']['status'], suites.FAIL)

    def test_an_excuse_that_outlives_its_reason_fails(self):
        spec = manifest(absent_ok={'branch-a': {'one': 'was missing once'}})
        holder, root = tree({'one/tests/test_x.py': PASSING})
        with holder:
            code, results, output = quiet_run(spec, root, 'branch-a')
        self.assertEqual(code, 1)
        self.assertEqual(results['one']['status'], suites.PASS)
        self.assertEqual(results[suites.STALE_ROW]['status'], suites.FAIL)
        self.assertIn('stale absent_ok', results[suites.STALE_ROW]['detail'])
        self.assertIn('branch-a', results[suites.STALE_ROW]['detail'])

    def test_an_excuse_for_another_branch_fails_as_soon_as_the_directory_exists(self):
        # a branch that was merged into this one brings its excuse along: nothing may stay excused once the suite is here
        spec = manifest(absent_ok={'branch-a': {'one': 'was missing once'}})
        holder, root = tree({'one/tests/test_x.py': PASSING})
        with holder:
            code, results, output = quiet_run(spec, root, 'integration')
        self.assertEqual(code, 1)
        self.assertEqual(results['one']['status'], suites.PASS)
        self.assertIn('branch-a excuses one', results[suites.STALE_ROW]['detail'])

    def test_a_stale_excuse_is_looked_for_in_the_manifests_own_tree(self):
        # the job checks the scaffolding out at the commit it runs from and the tree under test at another ref
        spec = manifest(absent_ok={'branch-a': {'one': 'was missing once'}})
        scaffolding, scaffolding_root = tree({'unrelated.txt': 'x'})
        under_test, under_test_root = tree({'one/tests/test_x.py': PASSING})
        with scaffolding, under_test:
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink):
                code, results = suites.run(spec, under_test_root, 'branch-b', sys.executable, os.environ,
                                           manifest_root=scaffolding_root)
                other_code, other = suites.run(spec, under_test_root, 'branch-b', sys.executable, os.environ,
                                               manifest_root=under_test_root)
        self.assertEqual(code, 0, sink.getvalue())
        self.assertEqual([r['suite'] for r in results], ['one'])
        self.assertEqual(other_code, 1)
        self.assertEqual([r['suite'] for r in other], ['one', suites.STALE_ROW])

    def test_a_run_in_which_nothing_ran_fails_whatever_the_manifest_says(self):
        spec = manifest(absent_ok={'branch-a': {'one': 'not yet'}})
        holder, root = tree({'unrelated.txt': 'x'})
        with holder:
            code, results, output = quiet_run(spec, root, 'branch-a')
        self.assertEqual(results['one']['status'], suites.ABSENT_ALLOWED)
        self.assertEqual(code, 1)

    def test_no_branch_means_no_excuse(self):
        spec = manifest(absent_ok={'branch-a': {'one': 'not yet'}})
        holder, root = tree({'unrelated.txt': 'x'})
        with holder:
            code, results, output = quiet_run(spec, root, '')
        self.assertEqual(results['one']['status'], suites.FAIL)
        self.assertEqual(code, 1)


class DeadTestTests(unittest.TestCase):
    """A test file that the suite would never run is a failure, the dead-test case of the allowlist rule."""

    def test_a_test_file_in_a_subdirectory_fails_the_suite(self):
        holder, root = tree({'one/tests/test_x.py': PASSING, 'one/tests/sub/test_deep.py': PASSING})
        with holder:
            code, results, output = quiet_run(manifest(), root, 'any')
        self.assertEqual(code, 1)
        self.assertEqual(results['one']['status'], suites.FAIL)
        self.assertIn('sub/test_deep.py', results['one']['detail'])
        self.assertIn('never run', results['one']['detail'])

    def test_a_test_file_the_glob_misses_fails_the_suite(self):
        holder, root = tree({'one/tests/test_x.py': PASSING, 'one/tests/test_other.py': PASSING})
        spec = manifest({'one': {'dir': 'one/tests', 'glob': 'test_x*.py', 'require': ['test_x'], 'min_tests': 2}})
        with holder:
            code, results, output = quiet_run(spec, root, 'any')
        self.assertEqual(code, 1)
        self.assertIn('test_other.py', results['one']['detail'])

    def test_a_file_that_ends_in_test_is_a_test_file_too(self):
        holder, root = tree({'one/tests/test_x.py': PASSING, 'one/tests/other_test.py': PASSING})
        with holder:
            code, results, output = quiet_run(manifest(), root, 'any')
        self.assertEqual(code, 1)
        self.assertIn('other_test.py', results['one']['detail'])

    def test_caches_and_dot_directories_are_not_searched(self):
        holder, root = tree({'one/tests/test_x.py': PASSING, 'one/tests/__pycache__/test_old.py': PASSING,
                             'one/tests/.hidden/test_h.py': PASSING})
        with holder:
            code, results, output = quiet_run(manifest(), root, 'any')
        self.assertEqual(code, 0, output)

    def test_an_exclude_entry_with_a_reason_lets_one_file_stay_unrun(self):
        holder, root = tree({'one/tests/test_x.py': PASSING, 'one/tests/needs_card/test_card.py': PASSING,
                             'one/tests/test_late.py': BROKEN_IMPORT})
        spec = manifest({'one': {'dir': 'one/tests', 'require': ['test_x'], 'min_tests': 2, 'exclude': {
            'needs_card/test_card.py': 'needs a card', 'test_late.py': 'imports a module that only the image has'}}})
        with holder:
            code, results, output = quiet_run(spec, root, 'any')
        self.assertEqual(code, 0, output)
        self.assertEqual(results['one']['ran'], 2)
        self.assertEqual(results['one']['modules'], ['test_x'])

    def test_an_exclude_entry_for_a_file_that_is_gone_fails(self):
        holder, root = tree({'one/tests/test_x.py': PASSING})
        spec = manifest({'one': {'dir': 'one/tests', 'require': ['test_x'], 'min_tests': 2, 'exclude': {
            'test_gone.py': 'was here once'}}})
        with holder:
            code, results, output = quiet_run(spec, root, 'any')
        self.assertEqual(code, 1)
        self.assertIn('stale exclude', results['one']['detail'])
        self.assertIn('test_gone.py', results['one']['detail'])


class ChildInterpreterTests(unittest.TestCase):
    def test_a_suite_module_that_shares_a_name_with_a_scripts_ci_module_is_not_shadowed(self):
        self.assertTrue(os.path.isfile(os.path.join(HERE, 'vllm030_suites.py')))
        holder, root = tree({'one/tests/test_x.py': SHADOWS_A_CI_MODULE, 'one/tests/vllm030_suites.py': SHARED_HELPER})
        with holder:
            code, results, output = quiet_run(manifest({'one': {'dir': 'one/tests', 'require': ['test_x'], 'min_tests': 1}}), root,
                                              'any')
        self.assertEqual(code, 0, output)

    def test_scripts_ci_is_not_on_the_suites_import_path(self):
        probe = '''
        import sys
        import unittest
        class T(unittest.TestCase):
            def test_path(self):
                import os
                here = %r
                self.assertNotIn(here, [os.path.realpath(entry or os.getcwd()) for entry in sys.path])
        ''' % os.path.realpath(HERE)
        holder, root = tree({'one/tests/test_x.py': probe})
        with holder:
            code, results, output = quiet_run(manifest({'one': {'dir': 'one/tests', 'require': ['test_x'], 'min_tests': 1}}), root,
                                              'any')
        self.assertEqual(code, 0, output)

    def test_the_callers_pythonpath_is_not_passed_on(self):
        # the job's own-tests step runs with PYTHONPATH=scripts/ci: that must not reach the suites, at any position
        probe = '''
        import importlib.util
        import unittest
        class T(unittest.TestCase):
            def test_a_scripts_ci_module_cannot_be_imported(self):
                self.assertIsNone(importlib.util.find_spec('vllm030_suites'))
        '''
        holder, root = tree({'one/tests/test_x.py': probe})
        spec = manifest({'one': {'dir': 'one/tests', 'require': ['test_x'], 'min_tests': 1}})
        with holder:
            code, results, output = quiet_run(spec, root, 'any', dict(os.environ, PYTHONPATH=os.path.realpath(HERE)))
        self.assertEqual(code, 0, output)

    def test_a_module_loaded_from_outside_the_suite_directory_fails_the_suite(self):
        holder, root = tree({'one/tests/test_x.py': LOADED_FROM_ELSEWHERE})
        with holder:
            code, results, output = quiet_run(manifest({'one': {'dir': 'one/tests', 'require': ['test_x'], 'min_tests': 1}}), root,
                                              'any')
        self.assertEqual(code, 1)
        self.assertIn('not from the suite directory', results['one']['detail'])

    def test_a_hung_suite_times_out_with_its_stacks_in_the_log_and_the_others_still_run(self):
        holder, root = tree({'one/tests/test_x.py': HANGS, 'two/tests/test_y.py': PASSING})
        spec = manifest({'one': {'dir': 'one/tests', 'require': ['test_x'], 'min_tests': 1, 'timeout_s': 2},
                         'two': {'dir': 'two/tests', 'require': ['test_y'], 'min_tests': 2}})
        with holder:
            started = time.monotonic()
            code, results, output = quiet_run(spec, root, 'any')
            elapsed = time.monotonic() - started
        self.assertEqual(code, 1)
        self.assertEqual(results['one']['status'], suites.FAIL)
        self.assertIn('timeout', results['one']['detail'])
        self.assertIn('most recent call first', output, 'the hung child should dump its thread stacks')
        self.assertIn('test_waits_forever', output)
        self.assertEqual(results['two']['status'], suites.PASS)
        self.assertEqual(results['two']['ran'], 2)
        self.assertLess(elapsed, 60)

    def test_a_suite_that_exits_without_a_verdict_fails(self):
        holder, root = tree({'one/tests/test_x.py': 'import os\nos._exit(3)\n'})
        with holder:
            code, results, output = quiet_run(manifest({'one': {'dir': 'one/tests', 'require': ['test_x'], 'min_tests': 1}}), root,
                                              'any')
        self.assertEqual(code, 1)
        self.assertIn('exited 3 without a verdict', results['one']['detail'])


class CommandLineTests(unittest.TestCase):
    def write_manifest(self, directory, spec):
        path = os.path.join(directory, 'm.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(spec, handle)
        return path

    def run_main(self, argv):
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            code = suites.main(argv)
        return code, sink.getvalue()

    def test_dry_run_plans_without_running_and_the_json_and_summary_are_written(self):
        holder, root = tree({'one/tests/test_x.py': SKIPPING})
        with holder:
            path = self.write_manifest(root, manifest())
            out = os.path.join(root, 'out.json')
            summary = os.path.join(root, 'summary.md')
            code, text = self.run_main(['--manifest', path, '--root', root, '--branch', 'b', '--dry-run', '--json', out,
                                        '--summary', summary])
            self.assertEqual(code, 0, text)
            self.assertIn('dry run', text)
            with open(out, encoding='utf-8') as handle:
                data = json.load(handle)
            with open(summary, encoding='utf-8') as handle:
                table = handle.read()
        self.assertEqual(data['branch'], 'b')
        self.assertEqual(data['suites'][0]['modules'], ['test_x'])
        self.assertIn('| one |', table)

    def test_a_real_run_through_main_reports_a_skip_as_a_failure_with_an_annotation(self):
        holder, root = tree({'one/tests/test_x.py': SKIPPING})
        with holder:
            path = self.write_manifest(root, manifest())
            code, text = self.run_main(['--manifest', path, '--root', root, '--branch', 'b'])
        self.assertEqual(code, 1)
        self.assertIn('VLLM030 SUITES: FAIL', text)
        self.assertIn('::error title=one::', text)

    def test_branch_defaults_to_the_environment(self):
        holder, root = tree({'unrelated.txt': 'x'})
        with holder:
            path = self.write_manifest(root, manifest(absent_ok={'env-branch': {'one': 'not yet'}}))
            old = os.environ.get('CI_BRANCH')
            os.environ['CI_BRANCH'] = 'env-branch'
            try:
                code, text = self.run_main(['--manifest', path, '--root', root, '--dry-run'])
            finally:
                if old is None:
                    del os.environ['CI_BRANCH']
                else:
                    os.environ['CI_BRANCH'] = old
        self.assertIn('ABSENT-ALLOWED', text)
        self.assertEqual(code, 1)


class ManifestValidationTests(unittest.TestCase):
    def load(self, spec):
        holder = tempfile.TemporaryDirectory(prefix='vllm030_')
        with holder:
            path = os.path.join(holder.name, 'm.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(spec, handle)
            return suites.load_manifest(path)

    def test_a_bad_manifest_is_refused(self):
        good = manifest()
        self.assertEqual(self.load(good)['schema'], 1)
        bad = [
            dict(good, schema=2),
            dict(good, suites={}),
            dict(good, suites={'one': {'dir': '/abs', 'require': []}}),
            dict(good, suites={'one': {'dir': '../up', 'require': []}}),
            dict(good, suites={'one': {'dir': 'a', 'pythonpath': ['../x']}}),
            dict(good, suites={'one': {'dir': 'a', 'require': ['not a module']}}),
            dict(good, suites={'one': {'dir': 'a', 'min_tests': 0}}),
            dict(good, suites={'one': {'dir': 'a', 'glob': 'sub/test_*.py'}}),
            dict(good, suites={'one': {'dir': 'a', 'timeout_s': 0}}),
            dict(good, suites={'one': {'dir': 'a', 'timeout_s': '60'}}),
            dict(good, suites={'one': {'dir': 'a', 'exclude': ['test_x.py']}}),
            dict(good, suites={'one': {'dir': 'a', 'exclude': {'test_x.py': ' '}}}),
            dict(good, suites={'one': {'dir': 'a', 'exclude': {'../test_x.py': 'reason'}}}),
            dict(good, suites={'one': {'dir': 'a', 'exclude': {'test_x.txt': 'reason'}}}),
            dict(good, suites={'one': {'dir': 'a', 'exclude': {'sub\\test_x.py': 'reason'}}}),
            dict(good, absent_ok={'b': {'unknown': 'reason'}}),
            dict(good, absent_ok={'b': {'one': '  '}}),
            dict(good, absent_ok={'b': {}}),
        ]
        for spec in bad:
            with self.subTest(spec=spec):
                with self.assertRaises(suites.ManifestError):
                    self.load(spec)

    def test_the_checked_in_manifest_is_valid_and_only_a_suite_that_may_be_late_is_ever_excused(self):
        spec = suites.load_manifest(suites.DEFAULT_MANIFEST)
        self.assertEqual(sorted(spec['suites']), ['kv_region_read', 'kvp2p', 'scripts_p2p'])
        for branch, entries in spec.get('absent_ok', {}).items():
            with self.subTest(branch=branch):
                # the suites that exist on the base branch must always run; only the connector tests may arrive late
                self.assertEqual(sorted(entries), ['kvp2p'])
                # and only a work-package branch may be excused: never the integration branch, never main, never a tag or a sha
                self.assertTrue(fnmatch.fnmatchcase(branch, 'tp4/p2p-gemma-wp*'), branch)

    def test_every_suite_of_the_checked_in_manifest_has_a_time_limit_that_fits_the_job(self):
        spec = suites.load_manifest(suites.DEFAULT_MANIFEST)
        limits = [suite.get('timeout_s', suites.DEFAULT_TIMEOUT_S) for suite in spec['suites'].values()]
        # the job has 45 minutes and its install takes about 8: the three limits together must leave room for it
        self.assertLessEqual(sum(limits) + 10 * 60, 45 * 60, limits)
        # the room goes where the time goes: on a CI runner kv_region_read (the C++ emulation tests) took 452 s and kvp2p 210 s (integration review)
        by_suite = {name: suite.get('timeout_s', suites.DEFAULT_TIMEOUT_S) for name, suite in spec['suites'].items()}
        self.assertGreaterEqual(by_suite['kv_region_read'], 2 * 452, by_suite)
        self.assertGreaterEqual(by_suite['kvp2p'], 2 * 210, by_suite)


@unittest.skipUnless(os.path.isfile(NEW_WORKFLOW), 'no .github tree here (the serving image)')
class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        with open(NEW_WORKFLOW, encoding='utf-8') as handle:
            cls.text = handle.read()
        cls.doc = yaml.safe_load(cls.text)
        cls.spec = suites.load_manifest(suites.DEFAULT_MANIFEST)
        cls.job = cls.doc['jobs']['vllm030']

    def triggers(self):
        return self.doc.get('on', self.doc.get(True))

    def test_triggers(self):
        on = self.triggers()
        self.assertEqual(sorted(on), ['pull_request', 'push', 'workflow_dispatch'])
        self.assertEqual(on['push']['branches'], ['tp4/p2p-gemma', 'tp4/p2p-gemma-*'])
        self.assertNotIn('tags', on['push'])
        self.assertEqual(on['pull_request']['branches'], ['tp4/p2p-gemma', 'main'], 'a pull request to main runs the kvp2p suites too (review CI note)')
        ref = on['workflow_dispatch']['inputs']['ref']
        self.assertFalse(ref.get('required', False), 'a bare dispatch (gh workflow run --ref BRANCH) must work')

    def test_runner_label_and_permissions_match_the_other_cpu_workflows(self):
        self.assertEqual(self.job['runs-on'], 'ubuntu-latest')
        self.assertEqual(self.doc['permissions'], {'contents': 'read'})

    def test_actions_are_pinned_like_the_other_cpu_workflows(self):
        steps = [step['uses'] for step in self.job['steps'] if 'uses' in step]
        self.assertTrue(steps)
        for uses in steps:
            self.assertTrue(re.search(r'@[0-9a-f]{40}$', uses) or uses == 'actions/setup-python@v5', uses)

    def test_python_plugin_and_torch_pins_agree_with_the_manifest(self):
        pins = self.spec['pins']
        setup = [step for step in self.job['steps'] if str(step.get('uses', '')).startswith('actions/setup-python')]
        self.assertEqual([step['with']['python-version'] for step in setup], [pins['python']])
        checkouts = [step['with'] for step in self.job['steps'] if str(step.get('uses', '')).startswith('actions/checkout')
                     and step['with'].get('repository')]
        self.assertEqual([(c['repository'], c['ref']) for c in checkouts], [('tenstorrent/vllm-tt-plugin', pins['plugin'])])
        self.assertIn('torch==%s' % pins['torch'], self.text)
        self.assertIn('"%s"' % pins['vllm'], self.text)
        self.assertEqual(self.job['env']['VLLM_TARGET_DEVICE'], 'empty')

    def test_the_job_runs_the_suites_through_the_manifest_runner_and_checks_the_runner_itself_first(self):
        runs = [step.get('run', '') for step in self.job['steps']]
        own = [i for i, run in enumerate(runs) if 'required_tests.py' in run and 'test_vllm030_suites' in run]
        gate = [i for i, run in enumerate(runs) if 'scripts/ci/vllm030_suites.py' in run]
        self.assertEqual(len(own), 1)
        self.assertEqual(len(gate), 1)
        self.assertLess(own[0], gate[0])
        self.assertIn('CI_BRANCH', self.job['env'])

    def test_the_scaffolding_is_checked_out_apart_from_the_tree_under_test(self):
        # `-f ref=<other branch>` replaces the tree under test, so the runner, its manifest and its tests must not come from it
        steps = self.job['steps']
        checkouts = [step for step in steps if str(step.get('uses', '')).startswith('actions/checkout')
                     and not step['with'].get('repository')]
        self.assertEqual(len(checkouts), 2)
        tree_under_test, scaffolding = checkouts
        self.assertEqual(tree_under_test['with'].get('ref'), '${{ inputs.ref }}')
        self.assertNotIn('path', tree_under_test['with'])
        self.assertEqual(scaffolding['with'].get('ref'), '${{ github.sha }}')
        self.assertEqual(scaffolding['with'].get('path'), 'ci')
        self.assertLess(steps.index(tree_under_test), steps.index(scaffolding), 'a later checkout of the workspace would delete ci/')
        own = [step for step in steps if 'test_vllm030_suites' in step.get('run', '')]
        gate = [step for step in steps if 'vllm030_suites.py' in step.get('run', '')]
        self.assertEqual([step.get('working-directory') for step in own], ['ci'])
        self.assertEqual(len(gate), 1)
        command = ' '.join(gate[0]['run'].replace('\\\n', ' ').split())
        for wanted in ('python -B ci/scripts/ci/vllm030_suites.py', '--root .', '--manifest ci/scripts/ci/vllm030_suites.json',
                       '--manifest-root ci'):
            self.assertIn(wanted, command)
        self.assertNotIn('working-directory', gate[0])

    def test_this_file_is_on_the_integration_allowlist(self):
        with open(INTEGRATION_WORKFLOW, encoding='utf-8') as handle:
            integration = handle.read()
        self.assertRegex(integration, r'python -B -m unittest [^\n]*\btest_vllm030_suites\b')

    def test_the_allowlist_block_is_what_the_generator_writes_from_the_manifests(self):
        # the line sits in a block generated by make_fusion_profiles.py from scripts/ci/fusion-wp/*.json: a hand-typed line makes
        # test_fusion_wp fail in the integration job and is deleted by the next --write, so it is the manifest that must name the test
        import make_fusion_profiles
        sink = io.StringIO()
        with contextlib.redirect_stderr(sink):
            code = make_fusion_profiles.main(['--check'])
        self.assertEqual(code, 0, sink.getvalue())
        with open(os.path.join(HERE, 'fusion-wp', 'P2P.json'), encoding='utf-8') as handle:
            self.assertIn('test_vllm030_suites', json.load(handle)['tests'])


if __name__ == '__main__':
    unittest.main()
