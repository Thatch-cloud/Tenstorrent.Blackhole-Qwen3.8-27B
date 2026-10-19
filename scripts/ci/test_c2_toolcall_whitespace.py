"""c2_toolcall_whitespace: the API server keeps the edge whitespace of tool-call arguments.

vLLM 0.25.1's qwen3 / qwen3_xml tool parser strip()s every <parameter=NAME>VALUE</parameter> value (vllm/parser/qwen3.py,
_qwen3_arg_converter: `params[name] = value.strip()`, twice), so a file written with a trailing newline, an edit whose
old_string starts with indentation, or a value that is only whitespace reaches the client without it, in streaming and
non-streaming. The chat template renders '<parameter=NAME>\\n' + VALUE + '\\n</parameter>\\n'; the fix removes exactly that one
newline at each end (what vLLM 0.23's qwen3_coder parser did) and nothing else.

Groups:
- UnwrapTests, ConverterTests, PrefixTests, InstallTests, ArmTests, SmokeTests: the module's own logic against fakes and
  the template's rendering. vLLM-free and dependency-free, so the CPU suite runs them.
- BootTests: serving_c2_contract.boot arms the fix in the API server for every profile (default on), and the kill switches.
- EngineTests: vLLM 0.25.1's real parsers (the DelegatingParser the API server builds from --reasoning-parser qwen3 and
  --tool-call-parser qwen3_xml), non-streaming and streaming at several chunk schedules, with and without thinking:
  the stock parser loses the whitespace (the positive control, with the exact values a client gets), the fixed one
  returns the model's values, and render -> parse -> render is the identity. Where the parsers come from is
  test_c2_parser_rechunk's rule: the installed vLLM (the image's own test run, qwen-fast-vllm-cpu.yml), or
  QWEN_VLLM_SOURCE naming the pinned vLLM 0.25.1 sdist tree through its stubs; with neither they are skipped. They
  need `tokenizers` and `pydantic`. The tokenizer is a byte-level BPE built here (one token per byte, the real ids of
  the 33 added tokens), so any text can be streamed one token at a time.

Order independence. The image runs many test modules in ONE unittest process, so by the time these run, another module may
have imported vllm.parser.qwen3, armed this fix (its import hook fires, or it installs at once on an imported module) or
left an ImportHook in sys.meta_path. Every test therefore sets up the state it needs and restores what it found: the
Arm/Boot tests strip the leftover ImportHooks and choose the module's starting state (not imported, imported and stock,
imported and already patched) explicitly, and EngineTests reads the converter it finds, takes the stock one out of it
(fix.uninstall) and puts the found one back when the class is done.
"""

import ast
import functools
import io
import json
import os
import random
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c2_toolcall_whitespace as fix  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import toolcall_whitespace_smoke as smoke  # noqa: E402

PROFILES = os.path.join(HERE, 'qwen_c2_profiles.json')


def setUpModule():
    fix._LOGGED.add('live')         # the once-per-process live marker is checked where a test asks for it, not printed by the rest


def live_lines(run):
    """The live-marker lines run() makes stderr show, with the once-only latch reset around it."""
    err = io.StringIO()
    fix._LOGGED.discard('live')
    try:
        with mock.patch('sys.stderr', err):
            run()
    finally:
        fix._LOGGED.add('live')
    return [line for line in err.getvalue().splitlines() if 'toolcall whitespace: live' in line]


def own_hooks():
    """The ImportHooks of this fix that sys.meta_path holds right now."""
    return [hook for hook in sys.meta_path if isinstance(hook, fix.ImportHook)]


_ABSENT = object()


def isolate_import_state(imported=None):
    """Start from a known import state whatever ran before in this process, and return the function that puts the process
    back: no ImportHook of this fix in sys.meta_path (an earlier module's boot may have left one), and sys.modules[MODULE]
    absent (imported=None: the module is not imported yet, so arming registers the hook) or `imported` (a module that is
    already imported: arming installs at once). Only that one sys.modules entry and the meta_path list are touched."""
    meta_path, present = list(sys.meta_path), sys.modules.get(fix.MODULE, _ABSENT)

    def restore():
        sys.meta_path[:] = meta_path
        if present is _ABSENT:
            sys.modules.pop(fix.MODULE, None)
        else:
            sys.modules[fix.MODULE] = present

    sys.meta_path[:] = [hook for hook in meta_path if not isinstance(hook, fix.ImportHook)]
    if imported is None:
        sys.modules.pop(fix.MODULE, None)
    else:
        sys.modules[fix.MODULE] = imported
    return restore


# ------------------------------------------------------------------ the template and the stock converter

def render_tool_calls(calls, content=''):
    """The assistant tool_calls branch of the served chat template (Qwen3.8-27B chat_template.jinja), transliterated:

        {%- if loop.first %}{%- if content|trim %}{{- '\\n\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}
        {%- else %}{{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}{%- endif %}
        {%- else %}{{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}{%- endif %}
        {%- for args_name, args_value in tool_call.arguments|items %}
            {{- '<parameter=' + args_name + '>\\n' }}
            {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
            {{- args_value }}{{- '\\n</parameter>\\n' }}
        {%- endfor %}{{- '</function>\\n</tool_call>' }}
    calls: [(name, {argument: value})]; a string value is rendered as is, any other as JSON."""
    out = []
    for index, (name, arguments) in enumerate(calls):
        if index == 0:
            out.append(('\n\n' if content.strip() else '') + '<tool_call>\n<function=%s>\n' % name)
        else:
            out.append('\n<tool_call>\n<function=%s>\n' % name)
        for key, value in arguments.items():
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            out.append('<parameter=%s>\n%s\n</parameter>\n' % (key, text))
        out.append('</function>\n</tool_call>')
    return ''.join(out)


def arguments_body(params, name='f'):
    """The text between <function=NAME> and </function> that the template renders for these arguments."""
    return render_tool_calls([(name, params)]).split('<function=%s>\n' % name, 1)[1].rsplit('</function>', 1)[0]


def stock_converter(raw_args, partial):
    """vLLM 0.25.1's vllm/parser/qwen3.py _qwen3_arg_converter (lines 59-76), reproduced for comparison."""
    params = {}
    for match in fix.PARAM_RE.finditer(raw_args):
        name = match.group(1)
        value = match.group(2)
        params[name] = value.strip()
    if partial:
        remaining = fix.PARAM_RE.sub('', raw_args)
        m = fix.PARTIAL_PARAM_RE.search(remaining)
        if m:
            name = m.group(1)
            value = m.group(2)
            if name:
                params[name] = value.strip()
    return json.dumps(params, ensure_ascii=False)


# What the model writes for these values, and what the stock parser hands the client for each (the keys are the
# arguments of tool functions in the harness' tool list, write_file and edit_file among them).
WHITESPACE_CASES = [
    ('trailing newline', 'write_file', {'path': 'a.txt', 'content': 'line1\nline2\n'},
     {'path': 'a.txt', 'content': 'line1\nline2'}),
    ('leading indentation', 'edit_file',
     {'path': 'a.py', 'old_string': '    return x', 'new_string': '        return x + 1', 'replace_all': False},
     {'path': 'a.py', 'old_string': 'return x', 'new_string': 'return x + 1', 'replace_all': False}),
    ('spaces both ends', 'write_file', {'path': 'a', 'content': '  padded  '}, {'path': 'a', 'content': 'padded'}),
    ('only spaces', 'edit_file', {'path': 'a', 'old_string': 'x', 'new_string': '   '},
     {'path': 'a', 'old_string': 'x', 'new_string': ''}),
    ('only a newline', 'edit_file', {'path': 'a', 'old_string': 'x', 'new_string': '\n'},
     {'path': 'a', 'old_string': 'x', 'new_string': ''}),
    ('empty', 'edit_file', {'path': 'a', 'old_string': 'x', 'new_string': ''},
     {'path': 'a', 'old_string': 'x', 'new_string': ''}),
    ('blank lines both ends', 'write_file', {'path': 'a', 'content': '\n\ndef f():\n    pass\n\n'},
     {'path': 'a', 'content': 'def f():\n    pass'}),
    ('tabs and CRLF', 'write_file', {'path': 'a', 'content': '\tx\r\n\ty\r\n'}, {'path': 'a', 'content': 'x\r\n\ty'}),
    ('non-ASCII', 'write_file', {'path': 'a', 'content': ' caf\u00e9 \U0001f680\n'},
     {'path': 'a', 'content': 'caf\u00e9 \U0001f680'}),
    ('typed values around a padded string', 'run_command', {'command': 'ls  ', 'timeout': 30, 'background': True},
     {'command': 'ls', 'timeout': 30, 'background': True}),
]


# ------------------------------------------------------------------ the module's own logic (no vLLM)

ALPHABET = 'ab <>/\n\t'     # cannot spell '<parameter=' or '</parameter>': a value never ends its own parameter


def random_value(rng, longest=12):
    return ''.join(rng.choice(ALPHABET) for _ in range(rng.randint(0, longest)))


class UnwrapTests(unittest.TestCase):
    def test_one_newline_each_end_and_nothing_else(self):
        table = [('', ''), ('\n', ''), ('\n\n', ''), ('\nx\n', 'x'), ('x', 'x'), ('\n\nx\n\n', '\nx\n'),
                 ('\n  x  \n', '  x  '), ('  x  ', '  x  '), ('x\n', 'x'), ('\nx', 'x'), ('\n\n\n', '\n'),
                 ('\r\nx\r\n', '\r\nx\r'), ('\tx\t', '\tx\t')]
        for raw, expected in table:
            with self.subTest(raw=raw):
                self.assertEqual(fix.unwrap(raw), expected)

    def test_it_inverts_the_templates_framing_for_every_string(self):
        rng = random.Random(0)
        values = ['', '\n', '\n\n', ' ', 'x', 'x\n', '\nx', '\nx\n', '  x  ', '\n\n  x  \n\n'] + [random_value(rng) for _ in range(300)]
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(fix.unwrap('\n' + value + '\n'), value)

    def test_the_stock_strip_does_not(self):
        lost = [v for v in ('x\n', '\nx', '  x  ', '\n', ' ', 'x') if ('\n' + v + '\n').strip() != v]
        self.assertEqual(lost, ['x\n', '\nx', '  x  ', '\n', ' '])


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.convert = fix.make_converter()

    def args(self, params):
        return json.loads(self.convert(arguments_body(params), False))

    def test_every_whitespace_case_round_trips_closed(self):
        for label, name, params, _ in WHITESPACE_CASES:
            for key, value in params.items():
                if isinstance(value, str):
                    with self.subTest(case=label, key=key):
                        self.assertEqual(self.args({key: value}), {key: value})

    def test_render_parse_render_is_the_identity_on_random_values(self):
        rng = random.Random(1)
        for _ in range(400):
            params = {'k%d' % i: random_value(rng) for i in range(rng.randint(1, 3))}
            body = arguments_body(params)
            parsed = json.loads(self.convert(body, False))
            self.assertEqual(parsed, params)
            self.assertEqual(render_tool_calls([('f', parsed)]), render_tool_calls([('f', params)]))

    def test_it_is_the_stock_converter_where_no_value_has_an_edge_to_lose(self):
        rng = random.Random(2)
        different = 0
        for _ in range(600):
            params = {'k%d' % i: random_value(rng) for i in range(rng.randint(0, 3))}
            body = arguments_body(params)
            cuts = sorted(set([len(body)] + [rng.randint(0, len(body)) for _ in range(3)]))
            for cut in cuts:
                for partial in (False, True):
                    mine, stock = self.convert(body[:cut], partial), stock_converter(body[:cut], partial)
                    edge = any(isinstance(v, str) and (v != v.strip()) for v in json.loads(mine).values())
                    if not edge:
                        self.assertEqual(mine, stock, (body[:cut], partial))
                    else:
                        different += 1
        self.assertGreater(different, 20)      # the fuzz does reach values with an edge

    def test_unclosed_parameters_end_at_the_next_parameter_as_the_stock_does(self):
        raw = '<parameter=a>\n  x  \n<parameter=b>\ny\n</parameter>'
        self.assertEqual(json.loads(self.convert(raw, False)), {'a': '  x  ', 'b': 'y'})
        self.assertEqual(json.loads(stock_converter(raw, False)), {'a': 'x', 'b': 'y'})

    def test_a_parameter_the_model_wrote_without_framing_newlines_is_kept_as_written(self):
        # (a deviation from the template; the 0.23 reference parser did the same)
        self.assertEqual(json.loads(self.convert('<parameter=a>  x  </parameter>', False)), {'a': '  x  '})
        self.assertEqual(json.loads(self.convert('<parameter=a>x</parameter>', False)), {'a': 'x'})

    def test_the_live_marker_counts_the_first_kept_edge_once(self):
        seen = []
        convert = fix.make_converter(on_kept=seen.append)
        convert('<parameter=a>\nx\n</parameter>', False)
        self.assertEqual(seen, [])
        convert('<parameter=a>\n x\n</parameter><parameter=b>\ny \n</parameter>', False)
        self.assertEqual(seen, [' x', 'y '])
        convert('<parameter=a>\n x\n</parameter>', True)          # a partial conversion is not the final value
        self.assertEqual(seen, [' x', 'y '])


class PrefixTests(unittest.TestCase):
    """The streaming path drops a delta whose JSON does not extend what it sent (ParserEngine._compute_arg_delta): every
    partial value must be a prefix of every later one and of the final one."""

    def pieces(self, params):
        out = []
        for key, value in params.items():
            out += ['<parameter=%s>' % key, '\n'] + list(value) + ['\n', '</parameter>', '\n']
        return out

    def test_partial_values_only_grow_and_end_at_the_final_value(self):
        convert = fix.make_converter()
        rng = random.Random(3)
        for _ in range(250):
            params = {'k%d' % i: random_value(rng, 8) for i in range(rng.randint(1, 3))}
            pieces = self.pieces(params)
            seen = []
            for end in range(len(pieces) + 1):
                seen.append(json.loads(convert(''.join(pieces[:end]), True)))
            for earlier, later in zip(seen, seen[1:]):
                for key, value in earlier.items():
                    self.assertIn(key, later)
                    self.assertTrue(later[key].startswith(value), (params, earlier, later))
            self.assertEqual(seen[-1], params)
            self.assertEqual(json.loads(convert(''.join(pieces), False)), params)

    def test_the_open_value_holds_back_one_trailing_newline(self):
        convert = fix.make_converter()
        self.assertEqual(json.loads(convert('<parameter=a>\nfoo\n', True)), {'a': 'foo'})
        self.assertEqual(json.loads(convert('<parameter=a>\nfoo\n\n', True)), {'a': 'foo\n'})
        self.assertEqual(json.loads(convert('<parameter=a>\nfoo\n\n</parameter>', True)), {'a': 'foo\n'})
        self.assertEqual(json.loads(convert('<parameter=a>\n', True)), {'a': ''})
        self.assertEqual(json.loads(convert('<parameter=a>', True)), {'a': ''})


def fake_module(converter=stock_converter, pattern=None, config=True):
    """A stand-in for vllm.parser.qwen3: the converter, its two patterns and the cached config that reads the converter
    by name when it is called (as qwen3_config does)."""
    module = types.ModuleType('fake_vllm_parser_qwen3')
    module._PARAM_RE = fix.PARAM_RE if pattern is None else pattern
    module._PARTIAL_PARAM_RE = fix.PARTIAL_PARAM_RE
    module._qwen3_arg_converter = converter
    if config:
        @functools.cache
        def qwen3_config(thinking=True):
            return SimpleNamespace(arg_converter=module._qwen3_arg_converter, thinking=thinking)

        module.qwen3_config = qwen3_config
    return module


class InstallTests(unittest.TestCase):
    def test_install_swaps_the_converter_and_builds_new_configs_with_it(self):
        module = fake_module()
        stale = module.qwen3_config()
        self.assertIs(stale.arg_converter, stock_converter)
        lines = []
        replacement = fix.install(module, lines.append)
        self.assertIs(module._qwen3_arg_converter, replacement)
        self.assertIs(module.qwen3_config(thinking=False).arg_converter, replacement)
        self.assertIs(module.qwen3_config().arg_converter, replacement)       # the cache was cleared
        self.assertEqual(len(lines), 1)
        self.assertIn('toolcall whitespace: installed on fake_vllm_parser_qwen3._qwen3_arg_converter', lines[0])
        self.assertEqual(json.loads(replacement('<parameter=a>\n x\n</parameter>', False)), {'a': ' x'})

    def test_the_live_marker_is_logged_once_with_lengths_only(self):
        replacement = fix.install(fake_module(), lambda line: None)

        def run():
            replacement('<parameter=a>\n secret \n</parameter>', False)
            replacement('<parameter=a>\n again \n</parameter>', False)
            replacement('<parameter=a>\nplain\n</parameter>', False)

        lines = live_lines(run)
        self.assertEqual(len(lines), 1)
        self.assertIn('(8 characters, 6 after strip())', lines[0])
        self.assertNotIn('secret', lines[0])

    def test_install_is_idempotent_and_uninstall_restores(self):
        module = fake_module()
        first = fix.install(module, lambda line: None)
        lines = []
        self.assertIs(fix.install(module, lines.append), first)
        self.assertEqual(lines, [])
        self.assertIs(fix.uninstall(module), stock_converter)
        self.assertIs(module.qwen3_config().arg_converter, stock_converter)

    def test_install_refuses_what_is_not_vllm_0_25_1s_parser(self):
        import re

        with self.assertRaisesRegex(RuntimeError, 'has no _qwen3_arg_converter'):
            fix.install(types.ModuleType('empty'), lambda line: None)
        with self.assertRaisesRegex(RuntimeError, '_PARAM_RE is not the vLLM 0.25.1 pattern'):
            fix.install(fake_module(pattern=re.compile(r'<parameter=(.*?)>(.*?)</parameter>', re.DOTALL)), lambda line: None)
        with self.assertRaisesRegex(RuntimeError, 'does not behave as the vLLM 0.25.1 converter'):
            fix.install(fake_module(converter=lambda raw_args, partial: '{}'), lambda line: None)
        with self.assertRaisesRegex(RuntimeError, 'does not behave'):
            fix.install(fake_module(converter=lambda raw_args, partial: 1 / 0), lambda line: None)

    def test_install_leaves_a_converter_that_already_keeps_the_edges_alone(self):
        keeps = fix.make_converter()
        keeps.__dict__.pop(fix.INSTALLED)
        module = fake_module(converter=keeps)
        lines = []
        self.assertIs(fix.install(module, lines.append), keeps)
        self.assertIs(module._qwen3_arg_converter, keeps)
        self.assertEqual(len(lines), 1)
        self.assertIn(fix.NOT_NEEDED, lines[0])

    def test_install_refuses_a_config_that_does_not_use_the_converter(self):
        module = fake_module(config=False)

        @functools.cache
        def qwen3_config(thinking=True):
            return SimpleNamespace(arg_converter=stock_converter)      # bound at definition: the swap cannot reach it

        module.qwen3_config = qwen3_config
        with self.assertRaisesRegex(RuntimeError, 'builds a config that does not use'):
            fix.install(module, lambda line: None)

    def test_the_probe_names_each_converter(self):
        self.assertEqual(fix.probe(stock_converter), 'stock')
        self.assertEqual(fix.probe(fix.make_converter()), 'preserving')
        self.assertIsNone(fix.probe(lambda raw_args, partial: '{}'))


class ArmTests(unittest.TestCase):
    def setUp(self):
        self.saved = list(sys.meta_path)
        self.addCleanup(lambda: sys.meta_path.__setitem__(slice(None), self.saved))
        # a hook an earlier module (the image runs them in one process) left in sys.meta_path is not this test's
        sys.meta_path[:] = [hook for hook in self.saved if not isinstance(hook, fix.ImportHook)]

    def hooks(self):
        return own_hooks()

    def test_enabled_is_on_unless_the_environment_or_the_profile_says_otherwise(self):
        for profile, environ, expected in [({}, {}, True), ({}, {'QWEN_C2_TOOLCALL_WS': ''}, True),
                                           ({}, {'QWEN_C2_TOOLCALL_WS': '1'}, True), ({}, {'QWEN_C2_TOOLCALL_WS': '0'}, False),
                                           ({'toolcall_whitespace': False}, {}, False), ({'toolcall_whitespace': True}, {}, True),
                                           ({'toolcall_whitespace': False}, {'QWEN_C2_TOOLCALL_WS': '1'}, True),
                                           ({'toolcall_whitespace': True}, {'QWEN_C2_TOOLCALL_WS': '0'}, False)]:
            with self.subTest(profile=profile, environ=environ):
                self.assertIs(fix.enabled(profile, environ), expected)
        for profile, environ, message in [({}, {'QWEN_C2_TOOLCALL_WS': 'off'}, 'must be 0 or 1'),
                                          ({'toolcall_whitespace': 'no'}, {}, 'must be true or false'),
                                          ({'toolcall_whitespace': 0}, {}, None)]:
            if message is None:
                self.assertIs(fix.enabled(profile, environ), False)      # 0 == False: the profile's own JSON 0 is not a typo
                continue
            with self.assertRaisesRegex(ValueError, message):
                fix.enabled(profile, environ)

    def test_arm_hooks_the_qwen3_module_import(self):
        lines = []
        with mock.patch.dict(sys.modules):
            sys.modules.pop(fix.MODULE, None)
            self.assertTrue(fix.arm({'name': 'p'}, {}, lambda message, *values: lines.append(message % values if values else message)))
        self.assertEqual([hook.name for hook in self.hooks()], [fix.MODULE])
        self.assertIn('profile p: toolcall whitespace armed: vllm.parser.qwen3._qwen3_arg_converter', lines[-1])

    def test_arm_installs_at_once_when_the_module_is_already_imported(self):
        module = fake_module()
        lines = []
        with mock.patch.dict(sys.modules, {fix.MODULE: module}):
            self.assertTrue(fix.arm({'name': 'p'}, {}, lambda message, *values: lines.append(message % values if values else message)))
        self.assertEqual(self.hooks(), [])
        self.assertTrue(getattr(module._qwen3_arg_converter, fix.INSTALLED))
        self.assertIn('toolcall whitespace: installed on', lines[0])
        self.assertIn('toolcall whitespace armed', lines[1])

    def test_arm_off_installs_nothing_and_says_so(self):
        lines = []
        emit = lambda message, *values: lines.append(message % values if values else message)      # noqa: E731
        self.assertFalse(fix.arm({'name': 'p'}, {'QWEN_C2_TOOLCALL_WS': '0'}, emit))
        self.assertFalse(fix.arm({'name': 'q', 'toolcall_whitespace': False}, {}, emit))
        self.assertEqual(self.hooks(), [])
        self.assertEqual(len(lines), 2)
        self.assertIn('profile p: toolcall whitespace disabled', lines[0])
        self.assertIn('profile q: toolcall whitespace disabled', lines[1])

    def test_the_import_hook_runs_the_callback_after_the_module_executes_and_removes_itself(self):
        import importlib
        import tempfile

        calls = []
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, 'qwen_tw_hook_probe.py'), 'w', encoding='utf-8') as handle:
                handle.write('VALUE = 7\n')
            hook = fix.ImportHook('qwen_tw_hook_probe', lambda module: calls.append(module.VALUE))
            sys.meta_path.insert(0, hook)
            sys.path.insert(0, directory)
            try:
                importlib.invalidate_caches()
                module = importlib.import_module('qwen_tw_hook_probe')
            finally:
                sys.path.remove(directory)
                sys.modules.pop('qwen_tw_hook_probe', None)
        self.assertEqual(calls, [7])
        self.assertEqual(module.VALUE, 7)
        self.assertNotIn(hook, sys.meta_path)

    def test_the_module_is_stdlib_python_3_10(self):
        for name in ('c2_toolcall_whitespace.py', 'toolcall_whitespace_smoke.py'):
            with open(os.path.join(HERE, name), encoding='utf-8') as handle:
                tree = ast.parse(handle.read(), feature_version=(3, 10))
            imported = {alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
            imported |= {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
            self.assertLessEqual(imported, {'json', 're', 'sys', 'importlib', 'c2_toolcall_whitespace'}, name)


class SmokeTests(unittest.TestCase):
    BOOT = '[QWEN-C2] profile p: vLLM argv ["--port", "8000"]'
    ARMED = '[QWEN-C2] profile p: toolcall whitespace armed: vllm.parser.qwen3._qwen3_arg_converter keeps the edge whitespace'
    INSTALLED = '[QWEN-C2] toolcall whitespace: installed on vllm.parser.qwen3._qwen3_arg_converter: a tool argument keeps ...'
    DISABLED = '[QWEN-C2] profile p: toolcall whitespace disabled (QWEN_C2_TOOLCALL_WS=0 or "toolcall_whitespace": false in the profile)'

    def text(self, *lines):
        return '\n'.join(lines) + '\n'

    def test_a_healthy_boot_is_clean(self):
        self.assertEqual(smoke.problems({}, self.text(self.BOOT, self.ARMED, self.INSTALLED)), [])
        self.assertEqual(smoke.problems(None, self.text(self.BOOT, self.ARMED, '[QWEN-C2] ' + fix.NOT_NEEDED + ': ...')), [])

    def test_a_log_that_is_not_an_api_server_boot_is_not_judged(self):
        self.assertEqual(smoke.problems({}, 'status: ok\n'), [])

    def test_no_armed_line_is_a_problem(self):
        found = smoke.problems({}, self.text(self.BOOT))
        self.assertEqual(len(found), 1)
        self.assertIn('toolcall whitespace armed', found[0])

    def test_armed_without_the_install_line_is_a_problem(self):
        found = smoke.problems({}, self.text(self.BOOT, self.ARMED))
        self.assertEqual(len(found), 1)
        self.assertIn('installed on', found[0])

    def test_disabled_is_a_problem_unless_the_arm_asks_for_it(self):
        found = smoke.problems({}, self.text(self.BOOT, self.DISABLED))
        self.assertEqual(len(found), 1)
        self.assertIn('toolcall whitespace disabled', found[0])
        self.assertEqual(smoke.problems({'QWEN_C2_TOOLCALL_WS': '0'}, self.text(self.BOOT, self.DISABLED)), [])


class BootTests(unittest.TestCase):
    """serving_c2_contract.boot arms the fix in the vLLM API server, under every profile."""

    def boot(self, name, environ=None, orig_argv=None, profile_patch=None, imported=None):
        """Boot under profile `name` and return (the ImportHooks it registered, its stderr). The process starts with
        vllm.parser.qwen3 not imported (imported=None) or already imported as the module `imported`, and without any
        ImportHook of this fix, whatever ran before in the process; all of it is put back afterwards."""
        orig_argv = orig_argv or ['python3', '-m', contract.API_SERVER, '--port', '8000']
        restore_import_state = isolate_import_state(imported)
        saved = list(sys.meta_path), list(sys.path), list(sys.argv)
        err = io.StringIO()
        try:
            patches = [mock.patch.dict(os.environ, {'QWEN_C2_PROFILE': name}),
                       mock.patch.object(contract, 'resolve_snapshot', lambda profile: profile['snapshots'][-1]),
                       mock.patch.object(contract, 'install_teardown_skip', lambda: None), mock.patch('sys.stderr', err)]
            if profile_patch is not None:
                real = contract.load_profile
                patches.append(mock.patch.object(contract, 'load_profile', lambda *a, **k: dict(real(*a, **k), **profile_patch)))
            for patch in patches:
                patch.start()
            try:
                sys.argv[:] = ['-m', '--port', '8000']
                contract.boot(environ=dict({'QWEN_C2_SERVING': '1', 'QWEN_C2_PROFILES': PROFILES}, **(environ or {})), orig_argv=orig_argv)
                hooks = own_hooks()
            finally:
                for patch in reversed(patches):
                    patch.stop()
        finally:
            sys.meta_path[:], sys.path[:], sys.argv[:] = saved
            restore_import_state()
        return hooks, err.getvalue()

    def test_every_profile_family_arms_it_by_default(self):
        with open(PROFILES, encoding='utf-8') as handle:
            names = list(json.load(handle)['profiles'])
        for name in ('general', 'exact', 'coding', 'c2', 'c2-gate', 'c2-packed', 'c2-packed-gate'):
            self.assertIn(name, names)
            with self.subTest(profile=name):
                hooks, err = self.boot(name)
                self.assertEqual([hook.name for hook in hooks], [fix.MODULE])
                self.assertIn('[QWEN-C2] profile %s: toolcall whitespace armed: vllm.parser.qwen3._qwen3_arg_converter' % name, err)

    def test_every_profile_family_arms_it_whatever_state_the_parser_module_is_in(self):
        # (the image runs this after other modules: vllm.parser.qwen3 may be imported by then, even patched by the installer)
        for name in ('general', 'exact', 'coding', 'c2', 'c2-gate', 'c2-packed', 'c2-packed-gate'):
            armed = '[QWEN-C2] profile %s: toolcall whitespace armed: vllm.parser.qwen3._qwen3_arg_converter' % name
            installed = 'toolcall whitespace: installed on fake_vllm_parser_qwen3._qwen3_arg_converter'
            with self.subTest(profile=name, state='imported, stock'):
                module = fake_module()
                hooks, err = self.boot(name, imported=module)
                self.assertEqual(hooks, [])                                       # nothing left to hook: installed at once
                self.assertTrue(getattr(module._qwen3_arg_converter, fix.INSTALLED))
                self.assertIs(module.qwen3_config().arg_converter, module._qwen3_arg_converter)
                self.assertIn(armed, err)
                self.assertEqual(err.count(installed), 1)
            with self.subTest(profile=name, state='imported, already patched'):
                module = fake_module()
                replacement = fix.install(module, lambda line: None)
                hooks, err = self.boot(name, imported=module)
                self.assertEqual(hooks, [])
                self.assertIs(module._qwen3_arg_converter, replacement)           # idempotent: not wrapped twice
                self.assertIs(replacement.__wrapped__, stock_converter)
                self.assertIn(armed, err)
                self.assertNotIn(installed, err)
            with self.subTest(profile=name, state='not imported'):
                hooks, err = self.boot(name)
                self.assertEqual([hook.name for hook in hooks], [fix.MODULE])
                self.assertIn(armed, err)

    def test_a_hook_left_by_an_earlier_boot_does_not_change_what_this_boot_registers(self):
        stale = fix.ImportHook(fix.MODULE, lambda module: None)
        sys.meta_path.insert(0, stale)
        self.addCleanup(lambda: stale in sys.meta_path and sys.meta_path.remove(stale))
        hooks, err = self.boot('c2')
        self.assertEqual([hook.name for hook in hooks], [fix.MODULE])
        self.assertNotIn(stale, hooks)
        self.assertIn(stale, sys.meta_path)                                       # and the stale one is back afterwards

    def test_the_environment_switch_turns_it_off(self):
        hooks, err = self.boot('c2', environ={'QWEN_C2_TOOLCALL_WS': '0'})
        self.assertEqual(hooks, [])
        self.assertIn('[QWEN-C2] profile c2: toolcall whitespace disabled', err)
        self.assertNotIn('toolcall whitespace armed', err)

    def test_the_profile_key_turns_it_off(self):
        hooks, err = self.boot('general', profile_patch={'toolcall_whitespace': False})
        self.assertEqual(hooks, [])
        self.assertIn('toolcall whitespace disabled', err)

    def test_a_typo_in_the_switch_stops_the_boot(self):
        with self.assertRaisesRegex(ValueError, 'QWEN_C2_TOOLCALL_WS must be 0 or 1'):
            self.boot('c2', environ={'QWEN_C2_TOOLCALL_WS': 'no'})

    def test_it_is_not_armed_outside_the_api_server(self):
        hooks, err = self.boot('c2', orig_argv=['python3', '-m', 'serving.server'])
        self.assertEqual(hooks, [])
        self.assertNotIn('toolcall whitespace', err)

    def test_the_other_hooks_are_untouched(self):
        # the contract's own hooks (PostImportHook) are the profile-gated ones; this one is a different class
        self.assertFalse(issubclass(fix.ImportHook, contract.PostImportHook))
        self.assertFalse(issubclass(contract.PostImportHook, fix.ImportHook))


# ------------------------------------------------------------------ vLLM 0.25.1's real parsers

def byte_tokenizer():
    """A byte-level BPE (no merges): one token per UTF-8 byte of the text, and the 26 added tokens of the model's
    vocabulary that matter here (<|im_end|>, <tool_call>, </tool_call>, <think>, </think> ...) as special tokens. It has
    the TokenizerLike surface the parsers touch (get_vocab, decode, all_special_*, the fast backend _tokenizer)."""
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

    names = ['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>',
             '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>',
             '<|image_pad|>', '<|video_pad|>', '<tool_call>', '</tool_call>', '<|fim_prefix|>', '<|fim_middle|>',
             '<|fim_suffix|>', '<|fim_pad|>', '<|repo_name|>', '<|file_sep|>', '<tool_response>', '</tool_response>',
             '<think>', '</think>']
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(models.BPE(vocab={character: index for index, character in enumerate(alphabet)}, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([AddedToken(name, special=True, normalized=False) for name in names])
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    special = [(name, vocab[name]) for name in names]

    class ByteTokenizer(object):
        _tokenizer = tokenizer
        eos = vocab['<|im_end|>']
        all_special_tokens = [name for name, _ in special]
        all_special_ids = [token for _, token in special]

        def get_vocab(self):
            return vocab

        def decode(self, ids, skip_special_tokens=False):
            return tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    return ByteTokenizer()


PROMPTS = {'think': '<|im_start|>assistant\n<think>\n', 'nothink': '<|im_start|>assistant\n<think>\n\n</think>\n\n'}
PREAMBLE = "I'll make the change."


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_c2_parser_rechunk as base

        cls.base = base
        cls.environment = base.parser_environment()
        try:
            import tokenizers.decoders

            if not hasattr(tokenizers.decoders, 'DecodeStream'):
                raise unittest.SkipTest('tokenizers has no DecodeStream: the engine detokenizes another way')
        except ImportError:
            raise unittest.SkipTest('tokenizers is not installed')
        cls.tokenizer = byte_tokenizer()
        cls.module = sys.modules['vllm.parser.qwen3']
        # The converter this process holds now: the stock one, or ours when another module of the same unittest process
        # armed the fix (its import hook fired on the first import, or it installed on the imported module). The tests need
        # the stock one to compare against and to install onto, and the next module the one it found.
        found = cls.module._qwen3_arg_converter
        cls.addClassCleanup(cls.put_back, found)
        fix.uninstall(cls.module)
        cls.stock = staticmethod(cls.module._qwen3_arg_converter)
        if getattr(cls.stock, fix.INSTALLED, False):
            raise AssertionError('vllm.parser.qwen3 holds the fix and uninstall() could not take it out')
        cls.kind = fix.probe(cls.stock)
        parser_base = cls.environment.abstract_parser.DelegatingParser
        cls.Parser = type('_Parser', (parser_base,), dict(reasoning_parser_cls=cls.environment.reasoning,
                                                           tool_parser_cls=cls.environment.tool))
        sys.stderr.write('[%s] EngineTests on %s; the stock converter is %r\n' % (__name__, cls.environment.mode, cls.kind))

    @classmethod
    def put_back(cls, converter):
        """Leave vllm.parser.qwen3 holding `converter` (what the class found) and its cached config built from it."""
        setattr(cls.module, fix.FUNCTION, converter)
        config = getattr(cls.module, fix.CONFIG_FUNCTION, None)
        if hasattr(config, 'cache_clear'):
            config.cache_clear()

    def tearDown(self):
        fix.uninstall(self.module)
        self.assertIs(self.module._qwen3_arg_converter, self.stock)

    def use(self, patched):
        fix.uninstall(self.module)
        if patched:
            fix.install(self.module, lambda line: None)

    def model_text(self, calls, thinking):
        body = PREAMBLE + render_tool_calls(calls, PREAMBLE)
        return ('Plan the edit.\n</think>\n\n' if thinking else '') + body

    def parse(self, patched, text, thinking, sizes=None):
        """(content, [(name, arguments JSON)]) a client gets: non-streaming when sizes is None, else streamed in chunks of
        `sizes` tokens (a list, repeated), the way the API server feeds the parser."""
        self.use(patched)
        case = dict(name='t', prompt='think' if thinking else 'nothink', thinking=thinking, eos=True)
        request = self.base.Request(case, self.environment.tools_param)
        if sizes is None:
            parser = self.Parser(self.tokenizer, request.tools, chat_template_kwargs=request.chat_template_kwargs, model_config=None)
            reasoning, content, calls = parser.parse(text, request, enable_auto_tools=True)
            return content, [(call.name, call.arguments) for call in calls or []]
        ids = list(self.tokenizer._tokenizer.encode(text, add_special_tokens=False).ids) + [self.tokenizer.eos]
        prompt = list(self.tokenizer._tokenizer.encode(PROMPTS[case['prompt']], add_special_tokens=False).ids)
        corpus = SimpleNamespace(ids={'t': ids}, prompt=lambda _: prompt, tokenizer=self.tokenizer)
        stream = self.base.run_stream(self.Parser, corpus, self.environment, case, list(sizes) * (len(ids) + 1))
        _, content, calls = stream.result()
        return content, [(name, arguments) for name, arguments, _ in calls]

    def arguments(self, found):
        return [(name, json.loads(arguments)) for name, arguments in found]

    def test_the_reference_converter_is_the_stock_one(self):
        if self.kind != 'stock':
            self.skipTest('the installed vLLM converter is %r, not the 0.25.1 stripping one' % self.kind)
        rng = random.Random(4)
        for _ in range(200):
            params = {'k%d' % i: random_value(rng) for i in range(rng.randint(0, 3))}
            body = arguments_body(params)
            for cut in (len(body), rng.randint(0, len(body))):
                for partial in (False, True):
                    self.assertEqual(self.stock(body[:cut], partial), stock_converter(body[:cut], partial))

    def test_the_stock_parser_loses_the_whitespace_in_both_paths(self):
        """The positive control, with what a client receives today."""
        if self.kind != 'stock':
            self.skipTest('the installed vLLM converter is %r, not the 0.25.1 stripping one' % self.kind)
        for thinking in (False, True):
            for label, name, params, stripped in WHITESPACE_CASES:
                text = self.model_text([(name, params)], thinking)
                for sizes in (None, [1], [4]):
                    with self.subTest(case=label, thinking=thinking, sizes=sizes):
                        content, found = self.parse(False, text, thinking, sizes)
                        self.assertEqual(content.strip(), PREAMBLE)
                        self.assertEqual(self.arguments(found), [(name, stripped)])

    def test_the_fixed_parser_returns_the_values_the_model_wrote(self):
        for thinking in (False, True):
            for label, name, params, _ in WHITESPACE_CASES:
                text = self.model_text([(name, params)], thinking)
                for sizes in (None, [1], [2], [3], [7], [10 ** 6]):
                    with self.subTest(case=label, thinking=thinking, sizes=sizes):
                        content, found = self.parse(True, text, thinking, sizes)
                        self.assertEqual(content.strip(), PREAMBLE)
                        self.assertEqual(self.arguments(found), [(name, params)])

    def test_render_parse_render_is_the_identity(self):
        """What the client sends back in the next turn re-renders to the text the model generated; the stock parser's does not."""
        for label, name, params, _ in WHITESPACE_CASES:
            calls = [(name, params)]
            generated = render_tool_calls(calls, PREAMBLE)
            text = self.model_text(calls, False)
            for sizes in (None, [1], [5]):
                with self.subTest(case=label, sizes=sizes):
                    _, found = self.parse(True, text, False, sizes)
                    self.assertEqual(render_tool_calls([(n, json.loads(a)) for n, a in found], PREAMBLE), generated)
                    if self.kind == 'stock' and any(isinstance(v, str) and v != v.strip() for v in params.values()):
                        _, found = self.parse(False, text, False, sizes)
                        self.assertNotEqual(render_tool_calls([(n, json.loads(a)) for n, a in found], PREAMBLE), generated)

    def test_the_live_marker_shows_the_real_parser_ran_the_fix(self):
        name, params = WHITESPACE_CASES[0][1:3]
        text = self.model_text([(name, params)], False)
        self.assertEqual(len(live_lines(lambda: self.parse(True, text, False))), 1)
        self.assertEqual(len(live_lines(lambda: self.parse(True, text, False, [3]))), 1)
        plain = self.model_text([(name, {'path': 'a', 'content': 'no edges'})], False)
        self.assertEqual(live_lines(lambda: self.parse(True, plain, False)), [])

    def test_two_calls_and_no_whitespace_arguments_are_unchanged(self):
        calls = [('write_file', {'path': 'a.txt', 'content': 'one\n'}), ('edit_file', {'path': 'b', 'old_string': '  x', 'new_string': 'y'})]
        text = self.model_text(calls, False)
        for sizes in (None, [1], [6]):
            with self.subTest(sizes=sizes):
                content, found = self.parse(True, text, False, sizes)
                self.assertEqual(self.arguments(found), calls)
        # arguments with nothing to lose: the fixed parser and the stock one return the same bytes, content and reasoning too
        calls = [('write_file', {'path': 'a.txt', 'content': 'one\ntwo'}), ('run_command', {'command': 'ls', 'timeout': 5, 'background': False})]
        for thinking in (False, True):
            text = self.model_text(calls, thinking)
            for sizes in (None, [1], [3]):
                with self.subTest(thinking=thinking, sizes=sizes):
                    self.assertEqual(self.parse(False, text, thinking, sizes), self.parse(True, text, thinking, sizes))

    def test_every_chunk_schedule_streams_the_same_arguments_as_the_whole_output(self):
        label, name, params, _ = WHITESPACE_CASES[6]
        calls = [(name, params), ('edit_file', {'path': 'z', 'old_string': '\n    x\n', 'new_string': ' y '})]
        text = self.model_text(calls, True)
        whole = self.parse(True, text, True)
        self.assertEqual(self.arguments(whole[1]), calls)
        for seed in range(12):
            rng = random.Random(seed)
            sizes = [rng.randint(1, 16) for _ in range(40)]
            with self.subTest(seed=seed):
                content, found = self.parse(True, text, True, sizes)
                self.assertEqual(content.strip(), whole[0].strip())      # (the streaming path keeps the content's own newlines)
                self.assertEqual(found, whole[1])


if __name__ == '__main__':
    unittest.main()
