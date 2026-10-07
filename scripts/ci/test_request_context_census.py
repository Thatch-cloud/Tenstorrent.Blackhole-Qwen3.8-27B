"""H1: the QWEN_DSPARK_REQUEST_CONTEXT census, held for the 262,144-token window (tp4/seats8-262k).

The drafter's request context (QWEN_DSPARK_REQUEST_CONTEXT) is a qualified constant, 131,072, in every serving profile: the frozen
evidence qualifies THAT geometry (frozen_context_geometry, pinned), and the attach qualifies it once per process
(serving_request_factory.attach_source_check -> target_t16_attention_gate.qualify). A 262,144-token WINDOW is a different
thing from the drafter's request context: the window is the page table (4,096 entries), the extent readers' capacity and the
contract's clamp; the drafter keeps its 131,072-token geometry, and a window's positions past it are read through the history
the drafter already has. So the 262k profiles set QWEN_FAST_MAX_POSITION 262,144 and leave the request context at 131,072.

This census lists every tracked non-test Python file that names the variable, by what it does with it (a reader, a writer, a source
patch or shell string for another file, a docs-only mention), and fails when a file joins or leaves the list: a new READER is a new
place that decides by the value, and must be read against the window before it is added here. It reads the tracked files (the
checkout), never the working tree's pycache. The non-Python mentions (workflows and harness scripts that each set their own
value for their own arm) are listed by count only: none is on the serving path."""

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
NAME = 'QWEN_DSPARK_REQUEST_CONTEXT'
PROFILES = HERE / 'qwen_c2_profiles.json'

# (file, kind). READ: the module decides by the value. WRITE: it sets the value for a canary. PATCHES: its text rewrites ANOTHER
# file's source or shell. DOCS: a docstring or message only.
READERS = ('dspark_context_selection.py', 'frozen_context_geometry.py', 'frozen_gdn_cache_scope.py', 'frozen_incremental_scope.py',
           'frozen_mlp_buffer_scope.py', 'frozen_wait_zone_scope.py', 'history_concat_scope.py',
           'shared_qk_norm_comparison_scope.py', 'probe_t16_gate_profile.py')
WRITERS = ('serving_canary_runner.py',)
PATCHES = ('frozen_ladder_stage.py', 'frozen_recipe_context.py', 'frozen_runtime_context.py')
DOCS = ('lever_n_m3native_gate.py',)
CENSUS = READERS + WRITERS + PATCHES + DOCS
# What each reader compares the value against (a pin the 262k profiles must not set, or the geometry it defaults to).
READS = {
    'dspark_context_selection.py': ("os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT', '4096')",),
    'frozen_context_geometry.py': ("os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT', '8192')",),
    'frozen_gdn_cache_scope.py': ("os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT') != '32768'",),
    'frozen_incremental_scope.py': ("os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT') != '32768'",),
    'frozen_mlp_buffer_scope.py': ("os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT') != '32768'",),
    'frozen_wait_zone_scope.py': ("os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT') != '32768'",),
    # the one that switches ON at a window value: 261,888 is its own rung (a 262,144-token prompt less the drafter's block), not 131,072
    'history_concat_scope.py': ("os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT') != '261888'",),
    'shared_qk_norm_comparison_scope.py': ("os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT') != '4096'",),
    'probe_t16_gate_profile.py': ("show('QWEN_DSPARK_REQUEST_CONTEXT', os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT'))",),
}
PINNED_VALUES = ('261888', '32768', '4096', '8192')
NON_PYTHON_FLOOR = 30          # the workflows and harness scripts that name it (set per arm); the count must not shrink silently


def tracked(pattern):
    try:
        out = subprocess.run(['git', 'ls-files', pattern], cwd=str(ROOT), capture_output=True, text=True, check=True,
                             timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return [line for line in out.splitlines() if line]


def python_files():
    names = tracked('scripts/ci/*.py')
    if names is None:                    # no git (an exported tree): the files beside this one, never the pycache
        names = ['scripts/ci/%s' % path.name for path in HERE.glob('*.py')]
    return sorted(names)


def mentions():
    found = {}
    for name in python_files():
        base = os.path.basename(name)
        if base.startswith('test_'):
            continue
        text = (ROOT / name).read_text(encoding='utf-8', errors='replace')
        if NAME in text:
            found[base] = text
    return found


def reads_variable(source):
    """Whether the module's CODE reads the variable: os.environ.get(NAME[, default]) or os.environ[NAME] (a mention inside a string
    that a patch rewrites into another file, or a docstring, is not a read)."""
    def is_environ(node):
        return (isinstance(node, ast.Attribute) and node.attr == 'environ' and isinstance(node.value, ast.Name)
                and node.value.id == 'os')

    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get'
                and is_environ(node.func.value) and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == NAME):
            return True
        if (isinstance(node, ast.Subscript) and is_environ(node.value) and isinstance(node.slice, ast.Constant)
                and node.slice.value == NAME):
            return True
    return False


def profile_values(fast_path=None):
    """{profile: value} for the profiles that set it; `fast_path` True keeps the C2 fast-path profiles (QWEN_FAST_ANY_REQUEST=1),
    False the others (the general profiles run the model's own path at their own contexts)."""
    profiles = json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']
    values = {}
    for name, profile in profiles.items():
        env = profile.get('env', {})
        if NAME in env and (fast_path is None or (env.get('QWEN_FAST_ANY_REQUEST') == '1') == fast_path):
            values[name] = env[NAME]
    return values


class CensusTests(unittest.TestCase):
    def test_the_non_test_python_files_naming_the_variable_are_exactly_the_census(self):
        found = mentions()
        self.assertEqual(sorted(found), sorted(CENSUS),
                         'a file that names %s joined or left the list: read it against the 262,144-token window, then '
                         'classify it here' % NAME)
        self.assertEqual(len(CENSUS), 14)

    def test_the_readers_are_nine_and_each_reads_it_as_recorded(self):
        found = mentions()
        self.assertEqual(len(READERS), 9)
        for name in READERS:
            with self.subTest(file=name):
                for expression in READS[name]:
                    self.assertIn(expression, found[name])

    def test_exactly_the_nine_readers_read_it_in_code_and_no_other_file_does(self):
        found = mentions()
        for name in READERS:
            with self.subTest(reader=name):
                self.assertTrue(reads_variable(found[name]), '%s no longer reads the variable: move it in the census' % name)
        for name in WRITERS + PATCHES + DOCS:
            with self.subTest(file=name):
                self.assertFalse(reads_variable(found[name]),
                                 '%s reads the variable in code: it is a reader, list it so' % name)

    def test_the_writer_is_the_canary_only(self):
        text = mentions()['serving_canary_runner.py']
        self.assertIn("QWEN_DSPARK_REQUEST_CONTEXT='4096'", text)
        self.assertEqual(text.count(NAME), 1)

    def test_history_concat_scope_is_the_only_reader_that_switches_on_at_a_window_value(self):
        found = mentions()
        for name in READERS:
            wide = [value for value in ('261888', '262144', '131072') if "'%s'" % value in found[name]]
            if name == 'history_concat_scope.py':
                self.assertEqual(wide, ['261888'])
            else:
                self.assertEqual(wide, [], name)
        self.assertFalse((HERE / 'history_concat_scope.py').read_text(encoding='utf-8').count("'131072'"))

    def test_the_non_python_mentions_are_the_arms_that_set_their_own_value(self):
        names = tracked('*')
        if names is None:
            self.skipTest('no git')
        count = 0
        for name in names:
            if name.endswith(('.pyc', '.png', '.jpg')) or name.startswith('docs/') or '/test_' in name or name.endswith('.py'):
                continue
            path = ROOT / name
            if path.is_file() and NAME in path.read_text(encoding='utf-8', errors='replace'):
                count += 1
        self.assertGreaterEqual(count, NON_PYTHON_FLOOR)
        dockerfile = (ROOT / 'docker' / 'qwen-c2-serving.Dockerfile').read_text(encoding='utf-8')
        self.assertEqual(len([line for line in dockerfile.splitlines() if NAME in line]), 1)
        self.assertTrue([line for line in dockerfile.splitlines() if NAME in line][0].lstrip().startswith('#'),
                        'the serving Dockerfile only mentions it, in a comment')
        environment = json.loads((ROOT / 'docker' / 'qwen-c2-v235-environment.json').read_text(encoding='utf-8'))
        self.assertEqual(environment['qwen_configuration'][NAME], '131072')


class ProfileTests(unittest.TestCase):
    def test_every_fast_path_profile_that_sets_it_sets_131072(self):
        values = profile_values(fast_path=True)
        self.assertGreater(len(values), 60)
        self.assertEqual(sorted(set(values.values())), ['131072'], {k: v for k, v in values.items() if v != '131072'})

    def test_the_general_profiles_that_set_it_keep_their_own_contexts(self):
        values = profile_values(fast_path=False)
        self.assertTrue(set(values.values()) <= {'65536', '131072'}, values)
        self.assertEqual(sorted(set(profile_values().values())), ['131072', '65536'])

    def test_every_262k_profile_sets_it_to_131072_and_the_window_elsewhere(self):
        profiles = json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']
        wide = sorted(name for name, profile in profiles.items() if profile['engine'].get('max-model-len') == 262144)
        self.assertEqual(wide, sorted(['c2-packed-tp4-262k-gate', 'c2-packed-tp4-8x262k', 'c2-packed-tp4-8x262k-diag-strace',
                                'c2-packed-tp4-8x262k-gate', 'c2-packed-tp4-8x262k-time-gate',
                                'c2-packed-tp4-8x262k-best', 'c2-packed-tp4-8x262k-best-audit', 'c2-packed-tp4-8x262k-best-levern-audit', 'c2-packed-tp4-8x262k-best-levern-control-audit', 'c2-packed-tp4-8x262k-best-levern-final-hold-time-gate', 'c2-packed-tp4-8x262k-best-levern-foreign-time-gate', 'c2-packed-tp4-8x262k-best-levern-hang-gate', 'c2-packed-tp4-8x262k-best-levern-r1-time-gate', 'c2-packed-tp4-8x262k-best-levern-time-gate',
                                'c2-packed-tp4-8x262k-best-time-gate', 'c2-packed-tp4-8x262k-ship', 'c2-packed-tp4-8x262k-prefix-gate', 'c2-packed-tp4-8x262k-prefix-time-gate', 'c2-packed-tp4-8x262k-ship-prefix', 'c2-packed-tp4-8x262k-ship-prefix-audit']
                                + ['c2-packed-tp4-8x262k-best-time-gate-' + lever for lever in ('nosamp', 's1', 'd2', 'dbf16', 'lookup', 'stack')]  # tp4/262k8-x
                                + ['c2-packed-tp4-8x262k-best-nosamp-audit', 'c2-packed-tp4-8x262k-best-stack-audit']
                                + ['c2-packed-tp4-8x262k-best-time-gate-u1', 'c2-packed-tp4-8x262k-best-u1-audit']
                                + ['c2-packed-tp4-8x262k-w1', 'c2-packed-tp4-8x262k-w1-audit', 'c2-packed-tp4-8x262k-w1-audit-nod1', 'c2-packed-tp4-8x262k-w1-lite', 'c2-packed-tp4-8x262k-w1-nod1']
                                + ['c2-packed-tp4-8x262k-hostgap-1', 'c2-packed-tp4-8x262k-hostgap-1-audit', 'c2-packed-tp4-8x262k-hostgap-2', 'c2-packed-tp4-8x262k-hostgap-2-audit']))
        for name in wide:
            env = profiles[name]['env']
            with self.subTest(profile=name):
                self.assertEqual(env[NAME], '131072')
                self.assertEqual(env['QWEN_FAST_MAX_POSITION'], '262144')
                for value in PINNED_VALUES:
                    self.assertNotEqual(env[NAME], value)

    def test_no_profile_sets_a_value_another_reader_pins(self):
        for name, value in profile_values().items():
            with self.subTest(profile=name):
                self.assertNotIn(value, PINNED_VALUES)

    def test_the_variable_is_a_context_not_a_window_the_two_are_different_settings(self):
        profiles = json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']
        for name, profile in profiles.items():
            if profile['engine'].get('max-model-len') == 262144:
                self.assertNotEqual(profile['env'][NAME], profile['env']['QWEN_FAST_MAX_POSITION'], name)
                self.assertNotEqual(profile['env'][NAME], str(profile['engine']['max-model-len']), name)


class AttachQualifiesTheServedGeometryTests(unittest.TestCase):
    """The attach's source qualification is the 131,072 geometry's (frozen_context_geometry), whatever the window."""

    def test_the_frozen_geometry_at_131072_is_the_one_the_profiles_name(self):
        import frozen_context_geometry as geometry

        text = (HERE / 'frozen_context_geometry.py').read_text(encoding='utf-8')
        self.assertIn('131072', text)
        os_environ = dict(os.environ)
        try:
            os.environ[NAME] = '131072'
            selected = geometry.selected_geometry()
        finally:
            os.environ.clear()
            os.environ.update(os_environ)
        self.assertEqual(selected['context'], 131072)

    def test_the_attach_source_check_asks_for_the_same_qualification_at_every_window(self):
        import serving_request_factory as factory

        factory._ATTACH_QUALIFICATION.clear()
        calls = []

        def qualify(directory):
            calls.append(str(directory))
            return dict(report_sha256='x')

        saved = os.environ.get('QWEN_FAST_TP')
        os.environ.pop('QWEN_FAST_TP', None)
        try:
            for directory in ('/a', '/b'):
                factory.attach_source_check(directory, qualify=qualify, log=lambda *a, **k: None)
        finally:
            factory._ATTACH_QUALIFICATION.clear()
            if saved is not None:
                os.environ['QWEN_FAST_TP'] = saved
        self.assertEqual(calls, [str(Path('/a')), str(Path('/b'))])


if __name__ == '__main__':
    unittest.main()
