"""R21 fix M (c2_parser_rechunk): the C2 API server re-chunks multi-token deltas at the parsers' marker tokens.

Three groups:
- RechunkTests: M's own logic against fakes. Which deltas split and how, the merge (only
  fields that are not None), the guarded private DecodeStream and the logged switch-off, and
  install. vLLM-free and dependency-free, so the CPU suite (qwen-integration-cpu.yml) runs them.
- ProfileGatingTests: serving_c2_contract.boot arms M in the API server under c2 and c2-gate only;
  general and exact boot as before.
- CorpusTests: the R21 harness's validation (tmp/risks/parsers/harness.py; verified-r21-parsers.md
  section 4) on vLLM 0.25.1's real qwen3 and qwen3_xml parsers. The schedules are deltas of 1, 3, 5, 8
  and 16 tokens at every alignment, 40 random 1-16 schedules, and the harness's 32, 64 and
  whole-output ones. On every corpus case, including the K1 lookalikes, M's streams equal the
  one-token-per-step stream, and no chunk carries a null field. The stock parser does not
  (the positive control).
  - Where the parsers come from: the installed vLLM (qwen-fast-vllm-cpu.yml, and the C2 image's
    own build-time test run). Or QWEN_VLLM_SOURCE names the pinned sdist tree, sha256-checked,
    loaded through vLLM-free stubs (a port of the harness's vllm_stubs).
  - With neither they are skipped. The CPU suite has no pydantic or tokenizers. A vLLM that is
    installed but fails to import is an error, never a skip.

The corpus is the harness's 27 cases plus four:
- I6's two (tag ids in prose, and in tool arguments);
- a piece-spelled </tool_call> after a real call;
- a prompt longer than c2_parser_rechunk.PRIME_TOKENS whose tail starts mid-character.

c2_parser_corpus.json holds their token ids from Qwen3.8-27B's tokenizer.json (snapshot 1d4bf0f2),
and a reduced tokenizer. The reduced tokenizer is the ids the corpus uses plus the 33 added
tokens, as a WordLevel model with the original ByteLevel decoder. It decodes and streams those
ids exactly as the full tokenizer does (checked when the fixture is written), and
test_the_fixture_is_the_corpus holds every case to its text. Regenerate it with:
    python scripts/ci/test_c2_parser_rechunk.py --regenerate <path to Qwen3.8-27B tokenizer.json>
"""

import hashlib
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

import c2_parser_rechunk as rechunk  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

FIXTURE = os.path.join(HERE, 'c2_parser_corpus.json')
PROFILES = os.path.join(HERE, 'qwen_c2_profiles.json')
THINK_START, THINK_END, TOOL_START, TOOL_END, EOS = 248068, 248069, 248058, 248059, 248046
TAGS = {THINK_START: 'THINK_START', THINK_END: 'THINK_END', TOOL_START: 'TOOL_START', TOOL_END: 'TOOL_END'}

# The vLLM 0.25.1 sdist (vllm-0.25.1.tar.gz, sha256 ddbdec3f1c0f21af...) files whose code the corpus
# exercises, or which M mirrors (detokenizer.py) or wraps (abstract_parser.py).
PINNED = {
    'vllm/parser/abstract_parser.py': 'fd4eb7a64ea97f59cfaec2b08c359fb460daa06e4ac5b42906b6877b34360859',
    'vllm/parser/parser_manager.py': 'd3e8656eaa200d02a3778a8bfe197c0c876bfe6ca5524aac028767fc12b5bbca',
    'vllm/parser/engine/adapters.py': 'dc1c1317dbfb298e54b8d94ca0e66d2b0cb1e481c35cdcc60a815284bd8a6ef7',
    'vllm/parser/engine/events.py': '493543e5832b721c67640c09a6ad664823423a98afe0bb53ba78518608047f0a',
    'vllm/parser/engine/incremental_lexer.py': 'c58797ba16d60a6a4db365dc678b38b756df83eea566c6e34dac2785f55c4cce',
    'vllm/parser/engine/parser_engine.py': '886bf6293b6b4cd082882e883c4c3d1ff16597500ed741f871a4ae67826db178',
    'vllm/parser/engine/parser_engine_config.py': '0854bd50b239b3b5286f56e9254851f224822bf0e942c99f1fe3dabf3c2035a7',
    'vllm/parser/engine/registered_adapters.py': 'c6474bbc4ee4b0378dfb0dd74ac072e612a4037c2cbaff029a85f78e51e6b99d',
    'vllm/parser/engine/streaming_parser_engine.py': '4ac9135e12f286d32f5d8725630b49a54b68d902815ad598923f7130d0899e0b',
    'vllm/parser/engine/token_id_scanner.py': 'c9db6d6a29865d65ba1adc523015488aad4d7dbef46b7539041e9efbe81bd034',
    'vllm/parser/qwen3.py': '8c1e6ca9c6e2269a252cde2430bc49827a358cfd2707cdd4351f865838931858',
    'vllm/reasoning/qwen3_engine_reasoning_parser.py': '0a0038bf94ff7b77dffe1a5737c16208f37b06535e1876e2e085f9a63310c742',
    'vllm/tool_parsers/qwen3_engine_tool_parser.py': '3cf83a2a9408d72c79082825464b2c4dea1147ff390289dfb8936c5501114be9',
    'vllm/entrypoints/openai/engine/protocol.py': 'beab4f1a827b40963762a035bf2a955dc095a918c4293c3f1963f58b5e2562fa',
    'vllm/v1/engine/detokenizer.py': '52d43b97d5c0f84285df596b2da5f4ee32184b9f41b87c6ef99437225b5720b5',
}


# ------------------------------------------------------------------ the corpus (harness.py CASES)

def fn(name, props, required=()):
    params = {'type': 'object', 'properties': props}
    if required:
        params['required'] = list(required)
    return {'type': 'function', 'function': {'name': name, 'description': name, 'parameters': params}}


S = {'type': 'string'}
TOOLS = [
    fn('read_file', {'path': S, 'start_line': {'type': 'integer'}, 'end_line': {'type': 'integer'}}, ['path']),
    fn('write_file', {'path': S, 'content': S}, ['path', 'content']),
    fn('run_command', {'command': S, 'timeout': {'type': 'integer'}, 'background': {'type': 'boolean'}},
       ['command']),
    fn('edit_file', {'path': S, 'old_string': S, 'new_string': S, 'replace_all': {'type': 'boolean'}},
       ['path', 'old_string', 'new_string']),
    fn('search', {'query': S, 'paths': {'type': 'array', 'items': S}, 'options': {'type': 'object'}}, ['query']),
    {'type': 'function', 'function': {'name': 'note', 'description': 'free-form', 'parameters': {'type': 'object'}}},
]


def tc(name, **params):
    body = ''.join('<parameter=%s>\n%s\n</parameter>\n' % (k, v) for k, v in params.items())
    return '<tool_call>\n<function=%s>\n%s</function>\n</tool_call>' % (name, body)


SYSTEM = '<|im_start|>system\nYou are a coding assistant.<|im_end|>\n'
USER = '<|im_start|>user\nPlease help with the repository.<|im_end|>\n'
GEN_THINK = '<|im_start|>assistant\n<think>\n'              # the template opens the think block
GEN_NOTHINK = '<|im_start|>assistant\n<think>\n\n</think>\n\n'  # enable_thinking=False
GEN_BARE = '<|im_start|>assistant\n'                        # the model opens <think> itself
# 346 tokens; its last 64 start mid-character (the two continuation bytes of a split emoji).
LONG_USER = ('<|im_start|>user\n'
             + 'Review every file \U0001f4c4 and summarise the changes \U0001f680 to na\u00efve caf\u00e9 '
               '\U0001f9d1\u200d\U0001f4bb \u2014 \u68c0\u67e5\u4ed3\u5e93 \u2705. ' * 10
             + 'Keep it short, then call a tool.' + ' ok' * 5 + '<|im_end|>\n')
PROMPTS = {'think': SYSTEM + USER + GEN_THINK, 'nothink': SYSTEM + USER + GEN_NOTHINK,
           'bare': SYSTEM + USER + GEN_BARE, 'long': SYSTEM + LONG_USER + GEN_THINK}

HTML = ('<!DOCTYPE html>\n<html>\n  <body>\n    <div id="app"></div>\n    <object><param name="a" value="1">'
        '</object>\n    <script>\n      if (a < b && c > d) { console.log("<ok>"); }\n    </script>\n'
        '  </body>\n</html>')
PY_OLD = 'def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n\n    return s'
PY_NEW = 'def total(xs: list[int]) -> int:\n    """Sum xs."""\n    return sum(x for x in xs if x >= 0)'

CASES = [
    dict(name='plain_answer', text=(
        'The user wants a comparison helper. Generic Java uses List<String> and a < b checks; '
        'also x >= y edge cases and C++ #include <functional>.\n</think>\n\n'
        'Here is the function:\n\n```java\npublic static <T extends Comparable<T>> boolean less(T a, T b) {\n'
        '    return a.compareTo(b) < 0;\n}\n```\n\nIn HTML you would write `<div class="x"></div>` and '
        '`<param name="a">`; in C++ `#include <functional>` and `std::vector<int>`.\n')),
    dict(name='tool_with_preamble', text=(
        'I need to read the file first to understand the structure.\n</think>\n\nI\'ll read the file.\n\n'
        + tc('read_file', path='src/main.py', start_line='10', end_line='120'))),
    dict(name='tool_no_preamble', text=(
        'Let me run the tests.\n</think>\n\n'
        + tc('run_command', command='pytest -q tests/test_parser.py -k "stream and not slow"',
             timeout='300', background='false'))),
    dict(name='two_parallel_tools', text=(
        'Read both files.\n</think>\n\n' + tc('read_file', path='a.py') + '\n' + tc('read_file', path='b.py'))),
    dict(name='write_html', text='Create the page.\n</think>\n\n' + tc('write_file', path='web/index.html', content=HTML)),
    dict(name='edit_python', text=(
        'Refactor total().\n</think>\n\nUpdating the helper.\n\n'
        + tc('edit_file', path='lib/util.py', old_string=PY_OLD, new_string=PY_NEW, replace_all='true'))),
    dict(name='search_array_object', text=(
        'Search the code.\n</think>\n\n'
        + tc('search', query='parse_delta(', paths='["src", "tests"]', options='{"case": false, "max": 50}'))),
    dict(name='implicit_reasoning_end', text='I will call the tool now.\n\n' + tc('read_file', path='src/lib.rs')),
    dict(name='thinking_disabled_tool', prompt='nothink', thinking=False, text=(
        'Sure.\n\n' + tc('run_command', command='ls -la', timeout='30'))),
    dict(name='thinking_disabled_plain', prompt='nothink', thinking=False,
         text='Use `a < b` and `</div>` carefully.\n'),
    dict(name='model_opens_think', prompt='bare', text=(
        '<think>\nCheck the generic bounds <T> first.\n</think>\n\nThe bound is `<T extends Number>`.\n')),
    dict(name='unicode_cjk_emoji', text=(
        '\u7528\u6237\u60f3\u8981\u8bfb\u53d6\u6587\u4ef6 \U0001f4c4\uff0c\u7136\u540e\u5199\u5165\u8bf4\u660e\u3002'
        '\n</think>\n\n\u597d\u7684 \U0001f44d\uff0c\u6211\u6765\u5199\u5165\u3002\n\n'
        + tc('write_file', path='docs/\u8bf4\u660e.md',
             content='# \u6807\u9898 \U0001f680\n\n\u5185\u5bb9\uff1aa < b \u2705 \u2014 na\u00efve caf\u00e9'))),
    dict(name='truncated_mid_tool', eos=False, text=(
        'Write it.\n</think>\n\n<tool_call>\n<function=write_file>\n<parameter=path>\nx.py\n</parameter>\n'
        '<parameter=content>\ndef f():\n    return 1')),
    dict(name='truncated_mid_reasoning', eos=False, text='Thinking about a < b and <T> generics and </div'),
    dict(name='no_schema_tool', text=(
        'Take a note.\n</think>\n\n' + tc('note', title='Parser risk', body='Deltas carry 16 tokens; check "quotes" '
                                           'and \\backslashes\\ and <tags>.'))),
    dict(name='content_after_tool', text=(
        'Read then summarise.\n</think>\n\n' + tc('read_file', path='README.md') + '\n\nDone.')),
    dict(name='whitespace_values', text=(
        'Edit with blank lines.\n</think>\n\n'
        + tc('edit_file', path='a.txt', old_string='  indented  \n\n', new_string='\n\n  x  ', replace_all='false'))),
    dict(name='eos_inside_reasoning', text='The model stops without closing a < b think'),
    dict(name='think_end_then_eos', text='Nothing to add.\n</think>\n\n'),
    dict(name='short_answer', text='Easy.\n</think>\n\nYes.'),
    dict(name='unicode_boundaries', text=(
        '\u68c0\u67e5\u5b8c\u6210\U0001f680</think>\n\n\U0001f9d1\u200d\U0001f4bb done \u2705'
        + tc('write_file', path='\U0001f4c4.md', content='\U0001f680 x < y \U0001f680') + '\U0001f680')),
    dict(name='tool_choice_none', tool_choice='none', text=(
        'I need to read the file first.\n</think>\n\nI\'ll read the file.\n\n'
        + tc('read_file', path='src/main.py', start_line='10'))),
    dict(name='no_tools_in_request', tools=None, text=(
        'Answer directly.\n</think>\n\nUse `x < y` then ' + tc('read_file', path='a.py'))),
    dict(name='no_tools_plain', tools=None, text='Plain chat.\n</think>\n\nHello <b>world</b>.\n'),
    # K1: a prose mention of a tag generated as ordinary BPE pieces, not the tag id ('plain').
    dict(name='prose_tag_utf8', eos=False, segments=[
        ('text', 'Plan \U0001f680\n</think>\n\n\U0001f680 Close with '), ('plain', '</think>'), ('text', ' later.')]),
    dict(name='tool_arg_mentions_tag', segments=[
        ('text', 'Write the fixture.\n</think>\n\n<tool_call>\n<function=write_file>\n<parameter=path>\nt.txt\n'
                 '</parameter>\n<parameter=content>\n'),
        ('plain', '<tool_call>'), ('text', ' \U0001f680\n</parameter>\n</function>\n</tool_call>')]),
    dict(name='prose_mentions_tags', segments=[
        ('text', 'Explain the tags.\n</think>\n\nThe model closes reasoning with '),
        ('plain', '</think>'), ('text', ' and opens a call with '), ('plain', '<tool_call>'), ('text', '.\n')]),
    # Added here (verified-r21-parsers.md 3.2 I6 and the untested TOOL_BETWEEN lookalike).
    dict(name='prose_ids_answer', text=(
        'Explain the tags.\n</think>\n\nThe model closes reasoning with </think> and opens a call with '
        '<tool_call> then continues. Done.\n')),
    dict(name='prose_ids_fixture', text=(
        'Write the fixture.\n</think>\n\n' + tc('write_file', path='t.txt', content='a <tool_call> b </tool_call> c'))),
    dict(name='lookalike_tool_end_after_call', segments=[
        ('text', 'Read it.\n</think>\n\n' + tc('read_file', path='a.py') + '\n\nIt ended with '),
        ('plain', '</tool_call>'), ('text', ' there.\n')]),
    dict(name='long_prompt', prompt='long', text=(
        'Summarise.\n</think>\n\n' + tc('read_file', path='notes/\u7b14\u8bb0.md') + '\n\n\u5b8c\u6210 \u2705')),
]
CASE = {case['name']: case for case in CASES}


def case_text(case):
    return case['text'] if 'text' in case else ''.join(text for _, text in case['segments'])


def schedules(n):
    """harness.schedules, same order and seeds (so the report's K1 seeds 6, 10, 12, 27, 31 recur)."""
    for k in (1, 3, 5, 8, 16):
        for first in range(k):
            yield 'k=%d/off=%d' % (k, first), ([first] if first else []) + [k] * (n + 1)
    for k in (32, 64):  # the output collector merges rounds when the API server lags (output_processor.py:62-72)
        for first in (0, 1, 7, 13, 31):
            yield 'k=%d/off=%d' % (k, first), ([first] if first else []) + [k] * (n + 1)
    yield 'whole', [n]
    yield 'first1+rest', [1, n]
    for seed in range(40):
        rng = random.Random(seed)
        yield 'rand1-16/seed=%d' % seed, [1] + [rng.randint(1, 16) for _ in range(n + 1)]


def schedule(n, label):
    return next(sizes for name, sizes in schedules(n) if name == label)


# ------------------------------------------------------------------ the fixture

def split_keep(text, separators=('<', '>', '/')):
    out, current = [], ''
    for character in text:
        if character in separators:
            if current:
                out.append(current)
            out.append(character)
            current = ''
        else:
            current += character
    if current:
        out.append(current)
    return out


def build_fixture(tokenizer_path):
    """The corpus's ids under the full tokenizer, and a reduced tokenizer that decodes them identically."""
    from tokenizers import Tokenizer
    from tokenizers.decoders import DecodeStream

    with open(tokenizer_path, 'rb') as handle:
        raw = handle.read()
    full_spec = json.loads(raw.decode('utf-8'))
    full = Tokenizer.from_str(raw.decode('utf-8'))

    def encode(text):
        return full.encode(text, add_special_tokens=False).ids

    def encode_plain(text):  # harness Tok.encode_plain: added tokens cannot match a prose mention
        return [token for run in split_keep(text) for token in encode(run)]

    prompts = {name: encode(text) for name, text in PROMPTS.items()}
    cases = {}
    for case in CASES:
        if 'segments' in case:
            ids = [token for kind, text in case['segments'] for token in (encode(text) if kind == 'text'
                                                                          else encode_plain(text))]
        else:
            ids = encode(case['text'])
        cases[case['name']] = ids + ([EOS] if case.get('eos', True) else [])
    used = {token for ids in list(prompts.values()) + list(cases.values()) for token in ids}
    inverse = {token: piece for piece, token in full_spec['model']['vocab'].items()}
    vocab = {inverse[token]: token for token in sorted(used) if token in inverse}
    # Added tokens keep their ids only if the model's vocabulary names them (else they are renumbered).
    vocab.update({added['content']: added['id'] for added in full_spec['added_tokens']})
    spec = dict(version=full_spec['version'], truncation=None, padding=None, added_tokens=full_spec['added_tokens'],
                normalizer=None, pre_tokenizer=None, post_processor=None, decoder=full_spec['decoder'],
                model=dict(type='WordLevel', vocab=vocab, unk_token=inverse[min(inverse)]))
    reduced = Tokenizer.from_str(json.dumps(spec))
    for ids in list(prompts.values()) + list(cases.values()):
        for skip in (False, True):
            assert reduced.decode(ids, skip_special_tokens=skip) == full.decode(ids, skip_special_tokens=skip)
            one, two = DecodeStream(ids=[], skip_special_tokens=skip), DecodeStream(ids=[], skip_special_tokens=skip)
            assert [one.step(full, token) for token in ids] == [two.step(reduced, token) for token in ids]
    tail = full.decode(prompts['long'][-rechunk.PRIME_TOKENS:], skip_special_tokens=False)
    assert tail.startswith('\ufffd'), 'the long prompt\'s tail must start mid-character: %r' % tail[:8]
    return dict(about='test_c2_parser_rechunk corpus: token ids under Qwen3.8-27B tokenizer.json, and a reduced '
                      'tokenizer that decodes them identically; regenerate with --regenerate',
                tokenizer_sha256=hashlib.sha256(raw).hexdigest(), prompts=prompts, cases=cases, tokenizer=spec)


def dump_fixture(fixture):
    lines = ['{']
    keys = ('about', 'tokenizer_sha256', 'prompts', 'cases', 'tokenizer')
    for number, key in enumerate(keys):
        comma = ',' if number < len(keys) - 1 else ''
        value = fixture[key]
        if key in ('prompts', 'cases'):
            lines.append('%s: {' % json.dumps(key))
            names = sorted(value)
            for index, name in enumerate(names):
                lines.append('%s: %s%s' % (json.dumps(name), json.dumps(value[name], separators=(',', ':')),
                                           ',' if index < len(names) - 1 else ''))
            lines.append('}' + comma)
        else:
            lines.append('%s: %s%s' % (json.dumps(key), json.dumps(value, sort_keys=True, separators=(',', ':')),
                                       comma))
    lines.append('}')
    return '\n'.join(lines) + '\n'


def write_fixture(tokenizer_path):
    text = dump_fixture(build_fixture(tokenizer_path))
    with open(FIXTURE, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(text)
    print('wrote %s (%d bytes)' % (FIXTURE, len(text)))


# ------------------------------------------------------------------ vLLM-free fakes

class Fields(object):
    """A pydantic-like model: a field is set once the constructor or an assignment gives it a value,
    and dump() is model_dump(exclude_unset=True), so it shows what serving.py:743 sends."""

    DEFAULTS = {}

    def __init__(self, **values):
        object.__setattr__(self, 'fields_set', set())
        for name, default in self.DEFAULTS.items():
            object.__setattr__(self, name, [] if default is list else default)
        for name, value in values.items():
            setattr(self, name, value)

    def __setattr__(self, name, value):
        if name not in self.DEFAULTS:
            raise AttributeError(name)
        object.__setattr__(self, name, value)
        self.fields_set.add(name)

    def dump(self):
        out = {}
        for name in sorted(self.fields_set):
            value = getattr(self, name)
            if isinstance(value, Fields):
                value = value.dump()
            elif isinstance(value, list):
                value = [item.dump() if isinstance(item, Fields) else item for item in value]
            out[name] = value
        if out.get('tool_calls') == []:  # DeltaMessage._serialize drops an empty list
            out.pop('tool_calls')
        return out


class Message(Fields):
    DEFAULTS = dict(role=None, content=None, reasoning=None, tool_calls=list)


class Function(Fields):
    DEFAULTS = dict(name=None, arguments=None)


class Call(Fields):
    DEFAULTS = dict(id=None, type=None, index=None, function=None)


def nulls(value, path=''):
    """Paths of every null in a dumped message."""
    if value is None:
        return [path or '.']
    if isinstance(value, dict):
        return [found for key, item in value.items() for found in nulls(item, path + '.' + key)]
    if isinstance(value, list):
        return [found for index, item in enumerate(value) for found in nulls(item, '%s[%d]' % (path, index))]
    return []


class FakeRaw(object):
    """The fast tokenizer M steps its DecodeStream with; the fake stream reads the pieces here."""

    def __init__(self, pieces, added=None):
        self.pieces, self.added = dict(pieces), dict(added or {})

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(self.pieces[token] or '' for token in ids)

    def get_added_tokens_decoder(self):
        return {token: SimpleNamespace(content=content) for token, content in self.added.items()}


def fake_tokenizers(version='0.23.2', faults=None):
    """A tokenizers module whose DecodeStream renders FakeRaw.pieces, raising faults[token] in turn."""
    created = []

    class DecodeStream(object):
        def __init__(self, ids=None, skip_special_tokens=False):
            self.ids = None if ids is None else list(ids)
            created.append((self.ids, skip_special_tokens))

        def step(self, raw, token_id):
            queue = (faults or {}).get(token_id)
            if queue:
                raise queue.pop(0)
            return raw.pieces[token_id]

    module = types.ModuleType('tokenizers')
    module.__version__ = version
    module.decoders = types.ModuleType('tokenizers.decoders')
    module.decoders.DecodeStream = DecodeStream
    return module, created


MARKERS = dict(TAGS)
MARKERS[EOS] = '__DROP__'
PIECES = {1: 'a', 2: 'b', 3: ' ', 4: '\n\n', 5: 'Done', 6: '.', 7: None, 8: '\U0001f680',
          THINK_START: '<think>', THINK_END: '</think>', TOOL_START: '<tool_call>', TOOL_END: '</tool_call>',
          EOS: '<|im_end|>'}
REQUEST = SimpleNamespace(skip_special_tokens=False, spaces_between_special_tokens=True)


class FakeParser(object):
    """DelegatingParser as M reads it: engine-based, engines with a token-id map, a fast tokenizer;
    its stock parse_delta records each call and returns result(text, ids, finished)."""

    _engine_based = True

    def __init__(self, markers=MARKERS, pieces=PIECES, result=None, added=None):
        engine = SimpleNamespace(_resolved_token_ids=dict(markers))
        self._reasoning_parser = SimpleNamespace(_parser_engine=SimpleNamespace(_engine=engine))
        self._tool_parser = None
        self.model_tokenizer = SimpleNamespace(_tokenizer=FakeRaw(pieces, added))
        self.result = result or (lambda text, ids, finished: Message(content=text))
        self.calls = []

    def parse_delta(self, delta_text, delta_token_ids, request, prompt_token_ids=None, *, finished):
        self.calls.append((delta_text, list(delta_token_ids), finished))
        return self.result(delta_text, list(delta_token_ids), finished)


class MFake(FakeParser):
    parse_delta = rechunk.rechunked(FakeParser.parse_delta)


class FakeDelegatingParser(object):
    def parse_delta(self, delta_text, delta_token_ids, request, prompt_token_ids=None, *, finished):
        return Message(content=delta_text)


class RechunkCase(unittest.TestCase):
    """Fresh STATS and log dedupe, a fake tokenizers module, stderr captured."""

    faults = None
    version = '0.23.2'

    def setUp(self):
        self.saved_stats, self.saved_logged = dict(rechunk.STATS), set(rechunk._LOGGED)
        rechunk.STATS.update(dict.fromkeys(rechunk.STATS, 0))
        rechunk._LOGGED.clear()
        self.saved_modules = {name: sys.modules.get(name) for name in ('tokenizers', 'tokenizers.decoders')}
        self.tokenizers, self.created = fake_tokenizers(self.version, self.faults)
        sys.modules['tokenizers'], sys.modules['tokenizers.decoders'] = self.tokenizers, self.tokenizers.decoders
        self.err = io.StringIO()
        self.stderr = mock.patch('sys.stderr', self.err)
        self.stderr.start()

    def tearDown(self):
        self.stderr.stop()
        for name, module in self.saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        rechunk.STATS.clear()
        rechunk.STATS.update(self.saved_stats)
        rechunk._LOGGED.clear()
        rechunk._LOGGED.update(self.saved_logged)

    def use_faults(self, faults):
        self.tokenizers, self.created = fake_tokenizers(self.version, faults)
        sys.modules['tokenizers'], sys.modules['tokenizers.decoders'] = self.tokenizers, self.tokenizers.decoders

    def feed(self, parser, deltas, prompt=(11, 12, 13), request=REQUEST, finished_last=True):
        return [parser.parse_delta(text, list(ids), request, None if prompt is None else list(prompt),
                                   finished=finished_last and index == len(deltas) - 1)
                for index, (text, ids) in enumerate(deltas)]

    def lines(self):
        return [line for line in self.err.getvalue().splitlines() if line]

    def assert_inactive(self, parser, reason, deltas=(('a</think>', [1, THINK_END]),)):
        self.feed(parser, list(deltas))
        self.assertEqual(parser.calls, [(text, list(ids), index == len(deltas) - 1)
                                        for index, (text, ids) in enumerate(deltas)], 'the stock path, whole')
        self.assertEqual(self.lines(), ['[QWEN-C2] parser M: inactive for %s: %s' % (type(parser).__name__, reason)])
        self.assertEqual(rechunk.STATS['switched_off'], 0)


class RechunkTests(RechunkCase):
    def test_a_delta_with_a_marker_is_fed_one_token_per_sub_delta(self):
        parser = MFake()
        out = self.feed(parser, [('ab</think> a', [1, 2, THINK_END, 3, 1])])
        self.assertEqual(parser.calls, [('a', [1], False), ('b', [2], False), ('</think>', [THINK_END], False),
                                        (' ', [3], False), ('a', [1], True)])
        self.assertEqual(out[0].dump(), {'content': 'ab</think> a'})
        self.assertEqual(self.lines(), ['[QWEN-C2] parser M: live: first re-chunked delta in MFake (5 tokens)'])
        self.assertEqual(rechunk.STATS['rechunked'], 1)

    def test_a_delta_without_a_marker_or_of_one_token_stays_whole(self):
        parser = MFake()
        self.feed(parser, [('ab', [1, 2]), ('</think>', [THINK_END]), ('ba', [2, 1])])
        self.assertEqual(parser.calls, [('ab', [1, 2], False), ('</think>', [THINK_END], False), ('ba', [2, 1], True)])
        self.assertEqual(self.lines(), [])
        self.assertEqual(rechunk.STATS['rechunked'], 0)

    def test_every_special_id_the_engines_match_is_a_marker(self):
        """The EOS drop terminal (streaming_parser_engine.py:40-89) splits a delta like the tags."""
        parser = MFake()
        self.feed(parser, [('ab', [1, 2, EOS])])
        self.assertEqual(parser.calls, [('a', [1], False), ('b', [2], False), ('', [EOS], True)])

    def test_finished_reaches_only_the_last_sub_delta_and_a_stop_token_has_no_text(self):
        """detokenizer.py:107-113 leaves a stop token's text out of the final delta."""
        parser = MFake()
        self.feed(parser, [('a</think>b', [1, THINK_END, 2, EOS])])
        self.assertEqual(parser.calls, [('a', [1], False), ('</think>', [THINK_END], False), ('b', [2], False),
                                        ('', [EOS], True)])
        parser = MFake()  # a length finish: the last token's text is in the delta
        self.feed(parser, [('a</think>b', [1, THINK_END, 2])])
        self.assertEqual(parser.calls[-1], ('b', [2], True))

    def test_held_back_text_rides_with_the_token_that_completes_it(self):
        """A token whose bytes do not end a character renders nothing (DecodeStream returns None);
        the next one carries the character, exactly as at one token per step."""
        self.assertEqual((PIECES[7], PIECES[8]), (None, '\U0001f680'))
        parser = MFake()
        self.feed(parser, [('\U0001f680</think>', [7, 8, THINK_END])])
        self.assertEqual(parser.calls[:2], [('', [7], False), ('\U0001f680', [8], False)])

    def test_the_whitespace_window_after_a_tool_call_is_fed_one_token_per_sub_delta(self):
        """parser_engine.py:741-761 drops whitespace-only content chunks after a tool call until the
        first content that is not whitespace: '\\n\\n' + 'Done' in one delta would keep it (K2)."""
        class ToolParserModel(object):
            """Arguments inside a call, content outside it, whitespace-only content dropped."""

            in_tool = False

            def __call__(self, text, ids, finished):
                if TOOL_START in ids or TOOL_END in ids:
                    self.in_tool = TOOL_START in ids
                    return None
                if self.in_tool:
                    return Message(tool_calls=[Call(index=0, function=Function(arguments=text))])
                return Message(content=text) if text.strip() else None

        parser = MFake(result=ToolParserModel())
        out = self.feed(parser, [
            ('<tool_call>a', [TOOL_START, 1]),                      # a marker: split
            ('ab', [1, 2]),                                         # inside the call: whole
            ('b</tool_call>', [2, TOOL_END]),                       # a marker: split
            ('\n\nDone.', [4, 5, 6]),                               # the window: split
            ('\n\nab', [4, 1, 2]),                                  # after content: whole
        ])
        self.assertEqual([call[:2] for call in parser.calls], [
            ('<tool_call>', [TOOL_START]), ('a', [1]), ('ab', [1, 2]), ('b', [2]), ('</tool_call>', [TOOL_END]),
            ('\n\n', [4]), ('Done', [5]), ('.', [6]), ('\n\nab', [4, 1, 2])])
        self.assertEqual(out[3].dump(), {'content': 'Done.'})
        self.assertEqual(out[4].dump(), {'content': '\n\nab'})
        self.assertEqual((rechunk.STATS['rechunked'], rechunk.STATS['windowed']), (3, 1))

    def test_the_window_stays_shut_without_a_tool_call(self):
        parser = MFake(result=lambda text, ids, finished: None)
        self.feed(parser, [('</think>', [THINK_END]), ('\n\nDone.', [4, 5, 6])])
        self.assertEqual(parser.calls[-1][:2], ('\n\nDone.', [4, 5, 6]))

    def test_the_merge_sets_only_fields_that_are_not_none(self):
        """The prototype built DeltaMessage(role=..., content=...) and so sent "role":null and
        "content":null under exclude_unset (verified-r21-parsers.md Corrections 4)."""
        self.assertEqual(Message(role=None, content=None).dump(), {'role': None, 'content': None},
                         'the fake keeps pydantic\'s exclude_unset semantics')

        def result(text, ids, finished):
            return {'r': Message(reasoning='r'), '</think>': None, 'c': Message(content='c'),
                    'x': Message(tool_calls=[Call(index=0, id='call-1', type='function', function=Function(name='f'))]),
                    'y': Message(tool_calls=[Call(index=0, function=Function(arguments='{"a": 1}'))])}[text]

        pieces = {**PIECES, 1: 'r', 2: 'c', 3: 'x', 4: 'y'}
        parser = MFake(pieces=pieces, result=result)
        merged = self.feed(parser, [('r</think>cxy', [1, THINK_END, 2, 3, 4])])[0]
        self.assertEqual(merged.dump(), {'reasoning': 'r', 'content': 'c', 'tool_calls': [
            {'index': 0, 'id': 'call-1', 'type': 'function', 'function': {'name': 'f', 'arguments': '{"a": 1}'}}]})
        self.assertEqual(nulls(merged.dump()), [])
        self.assertNotIn('role', merged.fields_set)

    def test_merge_returns_a_lone_result_as_is_and_none_for_none(self):
        only = Message(content='x')
        self.assertIs(rechunk.merge([None, only, None]), only)
        self.assertIsNone(rechunk.merge([None, None]))
        empty = rechunk.merge([Message(), Message()])
        self.assertEqual(empty.dump(), {}, 'two empty messages merge to one empty one, not to None')
        self.assertEqual(rechunk.merge([Message(content=''), Message(reasoning='r')]).dump(),
                         {'content': '', 'reasoning': 'r'}, 'an empty string is a value, and is kept')

    def test_tool_call_deltas_coalesce_per_index_as_the_parser_engine_does(self):
        calls = [Call(index=0, id='a', type='function', function=Function(name='f')),
                 Call(index=0, function=Function(arguments='{')),
                 Call(index=1, id='b', type='function', function=Function(name='g', arguments='')),
                 Call(index=0, function=Function(arguments='}'))]
        merged = rechunk.coalesce(calls)
        self.assertEqual([call.dump() for call in merged], [
            {'index': 0, 'id': 'a', 'type': 'function', 'function': {'name': 'f', 'arguments': '{}'}},
            {'index': 1, 'id': 'b', 'type': 'function', 'function': {'name': 'g', 'arguments': ''}}])
        distinct = [Call(index=0, function=Function(arguments='x')), Call(index=1, function=Function(arguments='y'))]
        self.assertEqual(rechunk.coalesce(distinct), distinct)
        self.assertEqual(distinct[0].dump(), {'index': 0, 'function': {'arguments': 'x'}}, 'unset fields stay unset')

    def test_the_private_stream_is_primed_with_the_prompt_tail(self):
        """A stream primed with a 131,072-token prompt spends ~125 ms decoding it at its first step."""
        parser = MFake()
        prompt = list(range(100, 300))
        self.feed(parser, [('ab', [1, 2])], prompt=prompt)
        self.assertEqual(self.created, [(prompt[-rechunk.PRIME_TOKENS:], False)])
        parser = MFake()
        self.feed(parser, [('ab', [1, 2])], prompt=None)
        self.assertEqual(self.created[-1], ([], False))
        parser = MFake()
        self.feed(parser, [('ab', [1, 2])], request=SimpleNamespace(skip_special_tokens=True,
                                                                  spaces_between_special_tokens=True))
        self.assertEqual(self.created[-1][1], True, 'the engine\'s skip_special_tokens, from the request')

    def test_consecutive_added_tokens_render_as_the_engine_renders_them(self):
        """spaces_between_special_tokens=False: detokenizer.py:210-221 replaces the second of two
        consecutive added tokens with its raw content."""
        pieces = {**PIECES, TOOL_END: ' </tool_call>'}
        added = {TOOL_START: '<tool_call>', TOOL_END: '</tool_call>'}
        parser = MFake(pieces=pieces, added=added)
        request = SimpleNamespace(skip_special_tokens=False, spaces_between_special_tokens=False)
        self.feed(parser, [('<tool_call></tool_call>a', [TOOL_START, TOOL_END, 1])], request=request)
        self.assertEqual([call[0] for call in parser.calls], ['<tool_call>', '</tool_call>', 'a'])
        self.assertEqual(rechunk.STATS['switched_off'], 0)

    def test_the_switch_off_log_is_capped(self):
        with mock.patch.object(rechunk, 'MAX_SWITCH_OFF_LINES', 2):
            for _ in range(3):
                self.feed(MFake(), [('xx', [1])], finished_last=False)
        self.assertEqual(rechunk.STATS['switched_off'], 3)
        lines = self.lines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[1].endswith('(2 so far); further switch-offs are counted, not logged'))


class GuardTests(RechunkCase):
    """The private DecodeStream is guarded as FastIncrementalDetokenizer._protected_step
    (detokenizer.py:223-246), and whatever else M cannot handle switches M off for that request."""

    def test_invalid_prefix_resets_the_stream_as_the_engine_does(self):
        self.use_faults({1: [Exception('Invalid prefix encountered while decoding')]})
        parser = MFake()
        self.feed(parser, [('a</think>', [1, THINK_END])])
        self.assertEqual(self.created, [([11, 12, 13], False), (None, False)], 'a fresh, unprimed stream')
        self.assertEqual(parser.calls, [('a', [1], False), ('</think>', [THINK_END], True)])
        self.assertEqual(rechunk.STATS['switched_off'], 0)

    def test_overflow_and_type_errors_render_nothing_as_the_engine_s_do(self):
        self.use_faults({2: [OverflowError('id'), TypeError('id')]})
        parser = MFake()
        self.feed(parser, [('a</think>', [1, 2, THINK_END]), ('a</think>', [2, 1, THINK_END])])
        self.assertEqual([call[0] for call in parser.calls], ['a', '', '</think>', '', 'a', '</think>'])
        self.assertEqual(rechunk.STATS['switched_off'], 0)

    def assert_switched_off(self, parser, deltas, reason):
        out = self.feed(parser, deltas)
        self.assertEqual(parser.calls, [(text, list(ids), index == len(deltas) - 1)
                                        for index, (text, ids) in enumerate(deltas)], 'the stock path, whole')
        self.assertEqual(rechunk.STATS['switched_off'], 1)
        lines = self.lines()
        self.assertEqual(len(lines), 1, lines)
        self.assertIn('[QWEN-C2] parser M: switched off for a request, which continues on the stock path: ', lines[0])
        self.assertIn(reason, lines[0])
        return out

    def test_any_other_decode_error_switches_off_that_request_and_logs_once(self):
        self.use_faults({THINK_END: [RuntimeError('boom')]})
        self.assert_switched_off(MFake(), [('a</think>', [1, THINK_END]), ('ab', [1, 2]), ('a</think>', [1, THINK_END])],
                                 'the private decode stream failed: RuntimeError: boom')
        parser = MFake()
        self.feed(parser, [('a</think>', [1, THINK_END])])
        self.assertEqual(len(parser.calls), 2, 'the next request re-chunks again')

    def test_a_failed_reset_switches_off(self):
        self.use_faults({1: [Exception('Invalid prefix encountered'), Exception('Invalid prefix encountered')]})
        self.assert_switched_off(MFake(), [('a</think>', [1, THINK_END])],
                                 'the private decode stream failed: Exception: Invalid prefix encountered')

    def test_pieces_that_miss_the_engine_s_text_switch_off_and_log_no_text(self):
        """A stop string held back (detokenizer.py:82-89), or any other rendering the private
        stream does not share: M cannot tell which text belongs to which token."""
        self.assert_switched_off(MFake(), [('ab</thi', [1, 2, THINK_END]), ('nk>', [3])],
                                 'its 3-token delta is 7 characters, the private decode stream rendered 10')
        self.assertNotIn('</thi', self.lines()[0])

    def test_an_unknown_token_switches_off(self):
        self.assert_switched_off(MFake(), [('a?', [1, 999])], 'KeyError')

    def test_opening_the_stream_can_fail_without_raising(self):
        class Broken(MFake):
            @property
            def _reasoning_parser(self):
                raise RuntimeError('no engines')

            @_reasoning_parser.setter
            def _reasoning_parser(self, value):
                pass

        parser = Broken()
        out = self.feed(parser, [('a</think>', [1, THINK_END])])
        self.assertEqual(parser.calls, [('a</think>', [1, THINK_END], True)])
        self.assertEqual(out[0].dump(), {'content': 'a</think>'})
        self.assertIn('opening the private decode stream failed: RuntimeError: no engines', self.lines()[0])

    def test_a_merge_failure_keeps_the_parser_s_output_and_does_not_raise(self):
        class Brittle(Message):
            armed = False

            def __init__(self, **values):
                if Brittle.armed:
                    raise TypeError('cannot build')
                Message.__init__(self, **values)

        def result(text, ids, finished):
            message = Brittle(content=text)
            Brittle.armed = finished
            return message

        parser = MFake(result=result)
        out = self.feed(parser, [('a</think>', [1, THINK_END])])
        self.assertEqual(out[0].dump(), {'content': '</think>'}, 'the last sub-result: both merges failed')
        self.assertIn('merging sub-results failed: TypeError: cannot build', self.lines()[-1])

    def test_the_parser_s_own_errors_are_not_m_s_to_hide(self):
        def result(text, ids, finished):
            raise ValueError('parser bug')

        with self.assertRaisesRegex(ValueError, 'parser bug'):
            self.feed(MFake(result=result), [('a</think>', [1, THINK_END])])


class InactiveTests(RechunkCase):
    """Where M cannot apply it stays out of the way for every request, logged once per reason."""

    def test_a_parser_that_is_not_engine_based(self):
        class Legacy(MFake):
            _engine_based = False

        self.assert_inactive(Legacy(), 'not an engine-based parser')
        self.feed(Legacy(), [('a</think>', [1, THINK_END])])
        self.assertEqual(len(self.lines()), 1, 'logged once per reason')
        self.assertEqual(rechunk.STATS['inactive'], 2)

    def test_engines_without_token_ids(self):
        self.assert_inactive(MFake(markers={}), 'its parser engines match no token ids')

    def test_a_tokenizer_without_a_fast_backend(self):
        parser = MFake()
        parser.model_tokenizer = SimpleNamespace()
        self.assert_inactive(parser, 'its tokenizer has no fast (tokenizers) backend')

    def test_a_parser_without_instance_attributes(self):
        class Slotted(object):
            __slots__ = ('calls',)

            def __init__(self):
                self.calls = []

            def stock(self, delta_text, delta_token_ids, request, prompt_token_ids=None, *, finished):
                self.calls.append((delta_text, list(delta_token_ids), finished))
                return Message(content=delta_text)

            parse_delta = rechunk.rechunked(stock)

        parser = Slotted()
        self.feed(parser, [('a</think>', [1, THINK_END])])
        self.assertEqual(parser.calls, [('a</think>', [1, THINK_END], True)])


class OldTokenizersTests(RechunkCase):
    """tokenizers < 0.22: the engine detokenizes without a DecodeStream (detokenizer.py:22-24, 60-65)."""

    version = '0.21.4'

    def test_old_tokenizers(self):
        self.assert_inactive(MFake(), 'tokenizers < 0.22: the engine detokenizes without a DecodeStream')

    def test_no_tokenizers_at_all(self):
        sys.modules['tokenizers'] = None
        self.assert_inactive(MFake(), 'tokenizers < 0.22: the engine detokenizes without a DecodeStream')

    def test_version_parsing(self):
        for text, expected in (('0.22.0', (0, 22)), ('0.23.2rc1', (0, 23)), ('1', (1, 0)), ('abc', (0, 0)),
                               ('0.21.4.dev0', (0, 21))):
            self.assertEqual(rechunk.version_tuple(text), expected, text)


class InstallTests(RechunkCase):
    def module(self, parser_cls=None):
        module = types.ModuleType('fake_parser_module')
        module.DelegatingParser = parser_cls or type('DelegatingParser', (FakeDelegatingParser,), {
            'parse_delta': FakeDelegatingParser.parse_delta})
        return module

    def test_install_wraps_once_and_the_subclass_the_server_builds_inherits_it(self):
        module = self.module()
        original = module.DelegatingParser.__dict__['parse_delta']
        rechunk.install(module)
        rechunk.install(module)
        wrapped = module.DelegatingParser.__dict__['parse_delta']
        self.assertIs(wrapped.__wrapped__, original, 'wrapped once')
        served = type('_Parser', (module.DelegatingParser,), {})  # ParserManager.get_parser's class
        self.assertIs(served.parse_delta, wrapped)
        self.assertEqual(self.lines(), [
            '[QWEN-C2] parser M: installed on fake_parser_module.DelegatingParser.parse_delta: a multi-token delta is '
            'fed one token per sub-delta when it carries a marker token or falls in the whitespace window after a '
            'tool call, and the sub-results merge into one DeltaMessage per engine step'])

    def test_install_refuses_what_is_not_vllm_0_25_1_s_parser(self):
        with self.assertRaisesRegex(RuntimeError, 'has no DelegatingParser class'):
            rechunk.install(types.ModuleType('empty'))
        with self.assertRaisesRegex(RuntimeError, 'defines no parse_delta'):
            rechunk.install(self.module(type('DelegatingParser', (FakeDelegatingParser,), {})))

        def positional(self, delta_text, delta_token_ids, request, prompt_token_ids=None, finished=False):
            return None

        def renamed(self, text, delta_token_ids, request, prompt_token_ids=None, *, finished):
            return None

        for parse_delta in (positional, renamed):
            with self.subTest(parse_delta=parse_delta.__name__):
                with self.assertRaisesRegex(RuntimeError, 'is not the vLLM 0.25.1 signature'):
                    rechunk.install(self.module(type('DelegatingParser', (object,), {'parse_delta': parse_delta})))

    def test_the_wrapper_keeps_the_stock_signature(self):
        import inspect

        self.assertEqual(tuple(inspect.signature(MFake.parse_delta).parameters), rechunk.PARAMETERS)


class ProfileGatingTests(unittest.TestCase):
    """serving_c2_contract.boot arms M in the vLLM API server for c2 and c2-gate only."""

    def boot(self, name, orig_argv=None):
        orig_argv = orig_argv or ['python3', '-m', contract.API_SERVER, '--port', '8000']
        saved = list(sys.meta_path), list(sys.path), list(sys.argv)
        err = io.StringIO()
        try:
            with mock.patch.dict(os.environ, {'QWEN_C2_PROFILE': name}), \
                    mock.patch.object(contract, 'resolve_snapshot', lambda profile: profile['snapshots'][-1]), \
                    mock.patch.object(contract, 'install_teardown_skip', lambda: None), \
                    mock.patch('sys.stderr', err):
                sys.argv[:] = ['-m', '--port', '8000']
                contract.boot(environ={'QWEN_C2_SERVING': '1', 'QWEN_C2_PROFILES': PROFILES}, orig_argv=orig_argv)
                hooks = {hook.name: hook for hook in sys.meta_path if isinstance(hook, contract.PostImportHook)}
        finally:
            sys.meta_path[:], sys.path[:], sys.argv[:] = saved
        return hooks, err.getvalue()

    def profiles(self):
        with open(PROFILES, encoding='utf-8') as handle:
            return json.load(handle)['profiles']

    def test_only_c2_and_c2_gate_ask_for_m(self):
        # ...and their S2 twins, c2-packed and c2-packed-gate (design W8: c2's and c2-gate's limits and parser).
        asked = {name: profile.get('parser_rechunk') for name, profile in self.profiles().items()}
        self.assertEqual(asked, {'exact': None, 'c2': True, 'c2-gate': True, 'c2-packed': True, 'c2-packed-gate': True,
                                 'coding': None, 'general': None})
        for name, profile in self.profiles().items():
            self.assertEqual(contract.parser_rechunk(profile), name in ('c2', 'c2-gate', 'c2-packed', 'c2-packed-gate'),
                             name)
        with self.assertRaisesRegex(ValueError, 'parser_rechunk must be true or false'):
            contract.parser_rechunk({'parser_rechunk': 'yes'})

    def test_boot_arms_m_in_the_api_server_under_the_c2_profiles(self):
        for name in ('c2', 'c2-gate', 'c2-packed', 'c2-packed-gate'):
            with self.subTest(profile=name):
                hooks, err = self.boot(name)
                self.assertEqual(sorted(hooks), sorted([rechunk.MODULE, contract.INPUT_PROCESSOR]))
                self.assertIs(hooks[rechunk.MODULE].callback, rechunk.install)
                self.assertIn('[QWEN-C2] profile %s: parser M armed: vllm.parser.abstract_parser.DelegatingParser.'
                              'parse_delta re-chunks multi-token deltas at marker tokens (R21)' % name, err)

    def test_general_exact_and_coding_boot_as_before(self):
        """general decodes one token per step; exact is the gate's configuration, byte for byte."""
        expected = {'general': [], 'exact': [contract.INPUT_PROCESSOR], 'coding': [contract.INPUT_PROCESSOR]}
        for name, hook_names in expected.items():
            with self.subTest(profile=name):
                hooks, err = self.boot(name)
                self.assertEqual(sorted(hooks), hook_names)
                self.assertNotIn('parser M', err)

    def test_m_is_never_armed_outside_the_api_server(self):
        hooks, err = self.boot('c2', orig_argv=['python3', '-m', 'serving.server'])
        self.assertEqual(hooks, {})
        self.assertNotIn('parser M', err)

    def test_the_armed_hook_installs_m_when_vllm_imports_the_parser_module(self):
        hooks, _ = self.boot('c2')
        module = types.ModuleType(rechunk.MODULE)
        module.DelegatingParser = type('DelegatingParser', (FakeDelegatingParser,), {
            'parse_delta': FakeDelegatingParser.parse_delta})
        err = io.StringIO()
        with mock.patch('sys.stderr', err):
            hooks[rechunk.MODULE].callback(module)
        self.assertTrue(module.DelegatingParser.__dict__[rechunk.INSTALLED])
        self.assertIn('[QWEN-C2] parser M: installed on vllm.parser.abstract_parser.DelegatingParser.parse_delta', err.getvalue())


# ------------------------------------------------------------------ real vLLM 0.25.1 parsers

_ENVIRONMENT = []


def sha256_file(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def pinned_mismatches(root):
    """[(file, sha256 or None)] under root that are not the pinned sdist's."""
    found = []
    for relative, digest in sorted(PINNED.items()):
        path = os.path.join(root, *relative.split('/'))
        actual = sha256_file(path) if os.path.isfile(path) else None
        if actual != digest:
            found.append((relative, actual))
    return found


def install_stubs(package):
    """Load the pinned source's parser modules without vLLM's heavy packages: the harness's
    vllm_stubs.install. Real, unmodified: vllm/parser/engine/*, parser/qwen3.py,
    parser/abstract_parser.py, reasoning/abs_reasoning_parsers.py, tool_parsers/{abstract_tool_parser,
    utils,streaming,qwen3_engine_tool_parser}.py, reasoning/qwen3_engine_reasoning_parser.py,
    envs.py, utils/{collection_utils,import_utils}.py. The protocol classes (DeltaMessage,
    DeltaToolCall, ChatCompletionToolsParam, ...) are the real class bodies, extracted with ast.
    Stubbed, with no parser logic: the package __init__ files, logger, chat_utils' tool-id helpers,
    metrics, parser.utils' history count (0), the request classes, sampling_params, tokenizers,
    mcp, utils.mistral."""
    import ast
    import logging
    import uuid

    def package_module(name, path=None):
        module = types.ModuleType(name)
        module.__path__ = [path] if path else []
        sys.modules[name] = module
        parent, _, child = name.rpartition('.')
        if parent and parent in sys.modules:
            setattr(sys.modules[parent], child, module)
        return module

    def plain_module(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module
        parent, _, child = name.rpartition('.')
        if parent and parent in sys.modules:
            setattr(sys.modules[parent], child, module)
        return module

    def extract_classes(path, names, namespace):
        with open(path, encoding='utf-8') as handle:
            source = handle.read()
        lines, wanted = source.splitlines(), set(names)
        for node in ast.parse(source).body:
            if isinstance(node, ast.ClassDef) and node.name in wanted:
                start = min([decorator.lineno for decorator in node.decorator_list] + [node.lineno]) - 1
                exec(compile('\n'.join(lines[start:node.end_lineno]), '%s:%s' % (path, node.name), 'exec'), namespace)
                wanted.discard(node.name)
        if wanted:
            raise RuntimeError('classes not found in %s: %s' % (path, sorted(wanted)))

    package_module('vllm', package)
    sys.modules['vllm']._qwen_c2_stub = True
    for sub in ('parser', 'parser/engine', 'reasoning', 'tool_parsers', 'entrypoints', 'entrypoints/openai',
                'entrypoints/openai/engine', 'entrypoints/openai/chat_completion', 'entrypoints/openai/responses',
                'entrypoints/mcp', 'utils'):
        package_module('vllm.' + sub.replace('/', '.'), os.path.join(package, *sub.split('/')))
    plain_module('vllm.logger', init_logger=logging.getLogger)

    def make_tool_call_id(id_type='random', func_name=None, idx=None):
        if id_type == 'kimi_k2':
            return 'functions.%s:%s' % (func_name, idx)
        return 'chatcmpl-tool-%s' % uuid.uuid4().hex

    plain_module('vllm.entrypoints.chat_utils', make_tool_call_id=make_tool_call_id,
                 get_tool_call_id_type=lambda model_config: 'random', ChatCompletionMessageParam=dict)
    plain_module('vllm.parser.metrics', record_tool_parser_invocation=lambda **kw: None)
    plain_module('vllm.parser.utils', count_history_tool_calls=lambda request: 0)
    plain_module('vllm.entrypoints.mcp.tool_server', ToolServer=type('ToolServer', (), {}))
    plain_module('vllm.tokenizers', TokenizerLike=object)
    plain_module('vllm.utils.mistral', is_mistral_tokenizer=lambda tokenizer: False)

    class StructuredOutputsParams(object):
        def __init__(self, **values):
            self.__dict__.update(dict(json=None, structural_tag=None, regex=None, choice=None, grammar=None,
                                      json_object=None))
            self.__dict__.update(values)

    plain_module('vllm.sampling_params', StructuredOutputsParams=StructuredOutputsParams)
    from typing import Any, ClassVar, Literal
    from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

    namespace = dict(Any=Any, ClassVar=ClassVar, Literal=Literal, BaseModel=BaseModel, ConfigDict=ConfigDict,
                     Field=Field, model_serializer=model_serializer, model_validator=model_validator,
                     make_tool_call_id=make_tool_call_id, logger=logging.getLogger('protocol'))
    engine_protocol = ('OpenAIBaseModel', 'FunctionDefinition', 'FunctionCall', 'ToolCall', 'DeltaFunctionCall',
                       'DeltaToolCall', 'ExtractedToolCallInformation', 'DeltaMessage')
    extract_classes(os.path.join(package, 'entrypoints', 'openai', 'engine', 'protocol.py'), engine_protocol, namespace)
    plain_module('vllm.entrypoints.openai.engine.protocol', **{name: namespace[name] for name in engine_protocol})
    chat_protocol = ('ChatCompletionToolsParam', 'ChatCompletionNamedFunction', 'ChatCompletionNamedToolChoiceParam')
    extract_classes(os.path.join(package, 'entrypoints', 'openai', 'chat_completion', 'protocol.py'), chat_protocol,
                    namespace)
    plain_module('vllm.entrypoints.openai.chat_completion.protocol',
                 ChatCompletionRequest=type('ChatCompletionRequest', (), {}),
                 **{name: namespace[name] for name in chat_protocol})
    plain_module('vllm.entrypoints.openai.responses.protocol', ResponsesRequest=type('ResponsesRequest', (), {}))
    import vllm.parser.engine.registered_adapters  # noqa: F401
    import vllm.parser.abstract_parser  # noqa: F401
    import vllm.reasoning.qwen3_engine_reasoning_parser  # noqa: F401
    import vllm.tool_parsers.qwen3_engine_tool_parser  # noqa: F401


def parser_environment():
    """vLLM 0.25.1's parser classes, from the installed vLLM or the pinned source; SkipTest without either."""
    if _ENVIRONMENT:
        return _ENVIRONMENT[0]
    try:
        import vllm
    except ModuleNotFoundError as error:
        if error.name != 'vllm':
            raise  # installed but broken: an error, never a skip
        vllm = None
    if vllm is not None and not getattr(vllm, '_qwen_c2_stub', False):
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
        from vllm.parser import abstract_parser
        from vllm.parser.parser_manager import ParserManager
        from vllm.reasoning.qwen3_engine_reasoning_parser import Qwen3ParserReasoningAdapter
        from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser

        environment = SimpleNamespace(
            mode='installed vLLM %s' % getattr(vllm, '__version__', '?'), installed=True,
            root=os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__))), abstract_parser=abstract_parser,
            manager=ParserManager, reasoning=Qwen3ParserReasoningAdapter, tool=Qwen3EngineToolParser,
            tools_param=ChatCompletionToolsParam)
    else:
        source = os.environ.get('QWEN_VLLM_SOURCE')
        if not source:
            raise unittest.SkipTest('vLLM is not installed and QWEN_VLLM_SOURCE (the pinned vLLM 0.25.1 source tree) '
                                    'is not set')
        mismatched = pinned_mismatches(source)
        if mismatched:
            raise AssertionError('QWEN_VLLM_SOURCE=%s is not the pinned vLLM 0.25.1 sdist: %s' % (source, mismatched))
        install_stubs(os.path.join(source, 'vllm'))
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
        from vllm.parser import abstract_parser
        from vllm.reasoning.qwen3_engine_reasoning_parser import Qwen3ParserReasoningAdapter
        from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser

        environment = SimpleNamespace(
            mode='pinned vLLM 0.25.1 source %s (stubs)' % source, installed=False, root=source,
            abstract_parser=abstract_parser, manager=None, reasoning=Qwen3ParserReasoningAdapter,
            tool=Qwen3EngineToolParser, tools_param=ChatCompletionToolsParam)
    _ENVIRONMENT.append(environment)
    return environment


class FixtureTokenizer(object):
    """The TokenizerLike surface the parsers and M touch: get_vocab, decode, all_special_tokens/ids
    and the fast backend, _tokenizer. all_special_* are the added tokens flagged special (the
    harness's approximation of the HF tokenizer's special-tokens map)."""

    def __init__(self, spec):
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_str(json.dumps(spec))
        self._vocab = self._tokenizer.get_vocab(with_added_tokens=True)
        special = [(added['content'], added['id']) for added in spec['added_tokens'] if added['special']]
        self.all_special_tokens = [content for content, _ in special]
        self.all_special_ids = [token for _, token in special]

    def get_vocab(self):
        return self._vocab

    def decode(self, ids, skip_special_tokens=False):
        return self._tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)


class Corpus(object):
    _loaded = []

    @classmethod
    def load(cls):
        if not cls._loaded:
            with open(FIXTURE, encoding='utf-8') as handle:
                fixture = json.load(handle)
            corpus = cls()
            corpus.fixture = fixture
            corpus.tokenizer = FixtureTokenizer(fixture['tokenizer'])
            corpus.prompts, corpus.ids = fixture['prompts'], fixture['cases']
            cls._loaded.append(corpus)
        return cls._loaded[0]

    def prompt(self, case):
        return self.prompts[case.get('prompt', 'think')]


class Request(object):
    """The request fields the parsers and M read. skip_special_tokens is False, as
    ParserEngine.adjust_request (parser_engine.py:200-206) leaves the served request."""

    def __init__(self, case, tools_param):
        tools = case.get('tools', TOOLS)
        self.messages = []
        self.tools = [tools_param(**tool) for tool in tools] if tools else None
        self.tool_choice = case.get('tool_choice', 'auto' if tools else 'none')
        self.chat_template_kwargs = None if case.get('thinking', True) else {'enable_thinking': False}
        self.skip_special_tokens = False
        self.spaces_between_special_tokens = True
        self.include_reasoning = True
        self.structured_outputs = None
        self.response_format = None


def engine_deltas(tokenizer, prompt, ids, sizes, eos, skip_special_tokens=False, holdback=0):
    """(delta_text, delta_token_ids, finished) per engine step, as FastIncrementalDetokenizer renders
    them (detokenizer.py:95-246): a DecodeStream primed with the WHOLE prompt, the stop token's text
    left out. holdback moves that many characters of the first delta into the second (a held-back stop
    string's shape, detokenizer.py:147-164)."""
    import tokenizers.decoders

    raw = tokenizer._tokenizer
    stream = [tokenizers.decoders.DecodeStream(ids=list(prompt), skip_special_tokens=skip_special_tokens)]

    def step(token):
        try:
            piece = stream[0].step(raw, token)
        except (OverflowError, TypeError):
            piece = None
        except Exception as error:
            if not str(error).startswith(rechunk.INVALID_PREFIX):
                raise
            stream[0] = tokenizers.decoders.DecodeStream(skip_special_tokens=skip_special_tokens)
            piece = stream[0].step(raw, token)
        return piece or ''

    out, sizes, index = [], iter(sizes), 0
    while index < len(ids):
        chunk = ids[index:index + next(sizes)]
        index += len(chunk)
        finished = index >= len(ids)
        out.append([''.join(step(token) for token in (chunk[:-1] if finished and eos else chunk)), chunk, finished])
    if holdback and len(out) > 1:
        out[1][0] = out[0][0][-holdback:] + out[1][0]
        out[0][0] = out[0][0][:-holdback]
    return [tuple(delta) for delta in out]


class Reconstruction(object):
    """What a client rebuilds from the chunks: reasoning, content, and per tool index its name, its
    arguments and how many chunks carried an id."""

    def __init__(self):
        self.reasoning, self.content, self.calls, self.messages = '', '', {}, []

    def add(self, message):
        if message is None:
            return
        self.messages.append(message)
        self.reasoning += message.reasoning or ''
        self.content += message.content or ''
        for call in message.tool_calls or []:
            slot = self.calls.setdefault(call.index, ['', '', 0])
            slot[2] += call.id is not None
            if call.function is not None:
                slot[0] += call.function.name or ''
                slot[1] += call.function.arguments or ''

    def result(self):
        return self.reasoning, self.content, [tuple(self.calls[index]) for index in sorted(self.calls)]


def run_stream(parser_cls, corpus, environment, case, sizes, holdback=0):
    ids, prompt = corpus.ids[case['name']], corpus.prompt(case)
    request = Request(case, environment.tools_param)
    parser = parser_cls(corpus.tokenizer, request.tools, chat_template_kwargs=request.chat_template_kwargs,
                        model_config=None)
    stream, previous = Reconstruction(), 0
    for text, chunk, finished in engine_deltas(corpus.tokenizer, prompt, ids, sizes, case.get('eos', True),
                                               holdback=holdback):
        if not text and not chunk and not previous:  # serving.py:585-591
            continue
        stream.add(parser.parse_delta(text, list(chunk), request, list(prompt), finished=finished))
        previous += len(chunk)
    return stream


class CorpusTests(unittest.TestCase):
    """M on vLLM 0.25.1's real qwen3 / qwen3_xml parsers, against one token per step."""

    _run = None

    @classmethod
    def setUpClass(cls):
        cls.environment = parser_environment()
        import tokenizers

        if rechunk.version_tuple(tokenizers.__version__) < rechunk.FAST_DETOKENIZER:
            raise AssertionError('tokenizers %s: the engine detokenizes without a DecodeStream, and M is inactive'
                                 % tokenizers.__version__)
        cls.corpus = Corpus.load()
        base = cls.environment.abstract_parser.DelegatingParser
        cls.Stock = type('_Parser', (base,), dict(reasoning_parser_cls=cls.environment.reasoning,
                                                  tool_parser_cls=cls.environment.tool))
        cls.M = type('_Parser', (cls.Stock,), {'parse_delta': rechunk.rechunked(base.__dict__['parse_delta'])})
        cls.references = {}
        sys.stderr.write('[%s] CorpusTests on %s\n' % (__name__, cls.environment.mode))

    def stream(self, parser_cls, name, label=None, sizes=None, holdback=0):
        case, ids = CASE[name], self.corpus.ids[name]
        sizes = sizes if sizes is not None else schedule(len(ids), label)
        return run_stream(parser_cls, self.corpus, self.environment, case, sizes, holdback)

    def reference(self, name):
        if name not in self.references:
            self.references[name] = self.stream(self.Stock, name, 'k=1/off=0').result()
        return self.references[name]

    def corpus_run(self):
        """Every case under every schedule, through M: mismatches, nulls, and M's own counters."""
        if CorpusTests._run is None:
            before, err = dict(rechunk.STATS), io.StringIO()
            mismatches, null_chunks, streams, chunks = [], [], 0, 0
            # A fresh once-only log, so the run's own markers show whatever ran before it.
            with mock.patch('sys.stderr', err), mock.patch.object(rechunk, '_LOGGED', set()):
                for case in CASES:
                    reference = self.reference(case['name'])
                    for label, sizes in schedules(len(self.corpus.ids[case['name']])):
                        got = self.stream(self.M, case['name'], sizes=sizes)
                        streams += 1
                        if got.result() != reference:
                            mismatches.append((case['name'], label, got.result(), reference))
                        for message in got.messages:
                            chunks += 1
                            found = nulls(json.loads(message.model_dump_json(exclude_unset=True)))
                            if found:
                                null_chunks.append((case['name'], label, found))
            CorpusTests._run = dict(mismatches=mismatches, nulls=null_chunks, streams=streams, chunks=chunks,
                                    stats={key: rechunk.STATS[key] - before[key] for key in before}, log=err.getvalue())
        return CorpusTests._run

    def test_the_fixture_is_the_corpus(self):
        tokenizer = self.corpus.tokenizer
        for key, text in PROMPTS.items():
            self.assertEqual(tokenizer.decode(self.corpus.prompts[key]), text, key)
        self.assertEqual(sorted(self.corpus.ids), sorted(CASE))
        for case in CASES:
            with self.subTest(case=case['name']):
                ids = self.corpus.ids[case['name']]
                eos = case.get('eos', True)
                self.assertEqual(ids[-1] == EOS, eos)
                self.assertEqual(tokenizer.decode(ids[:-1] if eos else ids), case_text(case))
        for name in ('prose_tag_utf8', 'tool_arg_mentions_tag', 'prose_mentions_tags', 'lookalike_tool_end_after_call'):
            with self.subTest(lookalike=name):
                spelled = sum(case_text(CASE[name]).count(tag) for tag in ('<think>', '</think>', '<tool_call>',
                                                                            '</tool_call>'))
                self.assertLess(sum(token in TAGS for token in self.corpus.ids[name]), spelled,
                                'the lookalike is spelled as pieces, not as the tag id')
        self.assertIn(TOOL_START, self.corpus.ids['prose_ids_answer'], 'I6: the prose tag is the tag id')

    def test_the_markers_are_the_four_wrapper_tags_and_the_special_ids(self):
        request = Request(CASE['plain_answer'], self.environment.tools_param)
        parser = self.Stock(self.corpus.tokenizer, request.tools, model_config=None)
        markers = rechunk.marker_terminals(parser)
        self.assertEqual({token: markers.get(token) for token in TAGS}, TAGS)
        for token in self.corpus.tokenizer.all_special_ids:
            self.assertEqual(markers.get(token), '__DROP__', token)
        self.assertEqual(markers[EOS], '__DROP__')
        self.assertEqual(set(markers), set(TAGS) | set(self.corpus.tokenizer.all_special_ids))

    def test_m_streams_equal_one_token_per_step_on_every_case_and_schedule(self):
        run = self.corpus_run()
        self.assertEqual(run['streams'], sum(len(list(schedules(len(ids)))) for ids in self.corpus.ids.values()))
        self.assertEqual(run['mismatches'], [], '\n'.join('%s %s:\n  M   %r\n  ref %r' % item
                                                          for item in run['mismatches'][:10]))

    def test_m_re_chunked_and_never_switched_off_on_the_corpus(self):
        run = self.corpus_run()
        stats = run['stats']
        self.assertEqual((stats['switched_off'], stats['inactive']), (0, 0), run['log'])
        self.assertGreater(stats['rechunked'], 1000)
        self.assertGreater(stats['windowed'], 0, 'K2\'s residual schedules go through the window')
        self.assertIn('[QWEN-C2] parser M: live: first re-chunked delta in _Parser', run['log'])

    def test_no_chunk_carries_a_null_field(self):
        """serving.py:743 sends model_dump_json(exclude_unset=True)."""
        run = self.corpus_run()
        self.assertGreater(run['chunks'], 10000)
        self.assertEqual(run['nulls'], [])

    def test_the_stock_parser_diverges_where_m_does_not(self):
        """The positive control: the corpus exercises what M fixes (verified-r21-parsers.md 3.1)."""
        k1_reasoning = self.stream(self.Stock, 'prose_tag_utf8', 'rand1-16/seed=10').result()
        self.assertIn('</think>', k1_reasoning[0], 'K1: the real </think> bound to its lookalike')
        self.assertNotIn('</think>', self.reference('prose_tag_utf8')[0])
        k1_tool = self.stream(self.Stock, 'tool_arg_mentions_tag', 'k=64/off=0').result()
        self.assertIn('<tool_call>', k1_tool[1], 'K1: <tool_call> leaks into content')
        self.assertNotIn('<tool_call>', self.reference('tool_arg_mentions_tag')[1])
        k2 = self.stream(self.Stock, 'content_after_tool', 'k=3/off=0').result()
        self.assertEqual((k2[1], self.reference('content_after_tool')[1]), ('\n\nDone.', 'Done.'), 'K2')
        for name, label in (('prose_tag_utf8', 'rand1-16/seed=10'), ('prose_tag_utf8', 'k=16/off=1'),
                            ('prose_mentions_tags', 'whole'), ('tool_arg_mentions_tag', 'k=64/off=0'),
                            ('content_after_tool', 'k=3/off=0'), ('content_after_tool', 'rand1-16/seed=38')):
            with self.subTest(case=name, schedule=label):
                self.assertNotEqual(self.stream(self.Stock, name, label).result(), self.reference(name))
                self.assertEqual(self.stream(self.M, name, label).result(), self.reference(name))

    def test_a_tail_primed_stream_renders_what_the_fully_primed_engine_does(self):
        prompt, ids = self.corpus.prompts['long'], self.corpus.ids['long_prompt']
        self.assertGreater(len(prompt), rechunk.PRIME_TOKENS)
        tail = self.corpus.tokenizer.decode(prompt[-rechunk.PRIME_TOKENS:])
        self.assertTrue(tail.startswith('\ufffd'), 'the tail starts mid-character: %r' % tail[:4])
        engine = engine_deltas(self.corpus.tokenizer, prompt, ids, [1] * len(ids), eos=True)
        stream = rechunk.Stream(rechunk.decode_stream_class(), self.corpus.tokenizer._tokenizer, {}, prompt,
                                False, True)
        self.assertEqual([stream.step(token) for token in ids[:-1]], [text for text, _, _ in engine[:-1]])

    def test_a_decode_failure_switches_off_and_the_stream_still_completes(self):
        real = rechunk.decode_stream_class()

        def failing_after(steps):
            class Failing(object):
                def __init__(self, ids=None, skip_special_tokens=False):
                    self.inner, self.steps = real(ids=ids, skip_special_tokens=skip_special_tokens), 0

                def step(self, raw, token):
                    self.steps += 1
                    if self.steps > steps:
                        raise RuntimeError('injected decode failure')
                    return self.inner.step(raw, token)

            return Failing

        for steps in (0, 40):
            with self.subTest(fail_after=steps):
                before, err = rechunk.STATS['switched_off'], io.StringIO()
                with mock.patch.object(rechunk, 'decode_stream_class', lambda: failing_after(steps)), \
                        mock.patch('sys.stderr', err):
                    got = self.stream(self.M, 'tool_arg_mentions_tag', 'k=16/off=0').result()
                self.assertEqual(rechunk.STATS['switched_off'], before + 1)
                self.assertIn('the private decode stream failed: RuntimeError: injected decode failure', err.getvalue())
                self.assertEqual(got[2], self.reference('tool_arg_mentions_tag')[2], 'the tool call completes')
                if steps == 0:  # off before the first sub-delta: the stock stream, exactly
                    self.assertEqual(got, self.stream(self.Stock, 'tool_arg_mentions_tag', 'k=16/off=0').result())

    def test_text_the_private_stream_does_not_share_switches_off(self):
        before, err = rechunk.STATS['switched_off'], io.StringIO()
        with mock.patch('sys.stderr', err):
            got = self.stream(self.M, 'plain_answer', 'k=16/off=0', holdback=3).result()
        self.assertEqual(rechunk.STATS['switched_off'], before + 1)
        self.assertIn('its 16-token delta is', err.getvalue())
        self.assertEqual(got, self.stream(self.Stock, 'plain_answer', 'k=16/off=0', holdback=3).result())

    def test_install_wraps_the_parser_class_the_server_builds(self):
        module = self.environment.abstract_parser
        parser_cls = module.DelegatingParser
        original = parser_cls.__dict__['parse_delta']
        err = io.StringIO()
        try:
            with mock.patch('sys.stderr', err):
                rechunk.install(module)
                rechunk.install(module)
                self.assertIs(parser_cls.__dict__['parse_delta'].__wrapped__, original)
                if self.environment.manager is not None:
                    served = self.environment.manager.get_parser(tool_parser_name='qwen3_xml',
                                                                 reasoning_parser_name='qwen3', enable_auto_tools=True)
                    self.assertIs(served.reasoning_parser_cls, self.environment.reasoning)
                    self.assertIs(served.tool_parser_cls, self.environment.tool)
                else:
                    served = type('_Parser', (parser_cls,), dict(reasoning_parser_cls=self.environment.reasoning,
                                                                 tool_parser_cls=self.environment.tool))
                self.assertTrue(issubclass(served, parser_cls))
                self.assertNotIn('parse_delta', served.__dict__)
                got = self.stream(served, 'tool_arg_mentions_tag', 'k=64/off=0').result()
        finally:
            parser_cls.parse_delta = original
            if rechunk.INSTALLED in parser_cls.__dict__:
                delattr(parser_cls, rechunk.INSTALLED)
        self.assertEqual(got, self.reference('tool_arg_mentions_tag'))
        self.assertEqual(err.getvalue().count('[QWEN-C2] parser M: installed on vllm.parser.abstract_parser.'
                                              'DelegatingParser.parse_delta'), 1)

    def test_the_parser_sources_are_the_pinned_sdist(self):
        """Stubs refuse any other source; an installed vLLM is compared here. A difference is
        reported as a skip, not a failure: the corpus tests above are what hold M to the installed
        parsers, and this only says whether the R21 report's source is the one installed."""
        mismatched = pinned_mismatches(self.environment.root)
        if mismatched:
            self.skipTest('%s differs from the pinned sdist in %s' % (self.environment.mode, mismatched))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--regenerate':
        write_fixture(sys.argv[2])
    else:
        unittest.main()
