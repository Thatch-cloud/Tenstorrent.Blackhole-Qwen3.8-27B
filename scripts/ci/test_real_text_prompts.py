"""real_text_prompts: the real-text prompt set the M3native gate builds inside the container.

A fake tokenizer stands in for transformers (the CPU suite has none): it renders a Qwen-like
chat template, matches the template's special tokens atomically, and splits the rest into short
word pieces, so lengths, windows, special-token handling and determinism are all exercised."""

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import real_text_prompts as rtp  # noqa: E402

SPECIALS = {'<|im_start|>': 1, '<|im_end|>': 2, '<|endoftext|>': 3, '<think>': 4, '</think>': 5}
SPECIAL_SPLIT = re.compile('(%s)' % '|'.join(re.escape(s) for s in sorted(SPECIALS, key=len, reverse=True)))
PIECE = re.compile(r'\w{1,4}|\s+|[^\w\s]')


class FakeTokenizer:
    """A Qwen-shaped tokenizer: <|im_start|>role ... <|im_end|> turns, an empty think block when
    enable_thinking is False, specials matched atomically, everything else in <=4-char pieces."""

    def __init__(self, mapping=False, scale=1):
        self.mapping = mapping
        self.scale = scale
        self.calls = []

    @property
    def added_tokens_decoder(self):
        return {token_id: SimpleNamespace(content=text, special=text.startswith('<|'))
                for text, token_id in SPECIALS.items()}

    def encode(self, text):
        out = []
        for part in SPECIAL_SPLIT.split(text):
            if part in SPECIALS:
                out.append(SPECIALS[part])
            elif part:
                out.extend(100 + zlib.crc32(piece.encode()) % 50000 for piece in PIECE.findall(part))
        return out * self.scale if self.scale > 1 else out

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(dict(messages=messages, **kwargs))
        text = ''.join('<|im_start|>%s\n%s<|im_end|>\n' % (m['role'], m['content']) for m in messages)
        if kwargs.get('add_generation_prompt'):
            text += '<|im_start|>assistant\n'
            if kwargs.get('enable_thinking') is False:
                text += '<think>\n\n</think>\n\n'
        tokens = self.encode(text)
        return {'input_ids': tokens} if self.mapping else tokens


def code_corpus(files=40, lines=60, seed=0):
    """A deterministic pile of python-looking files, as {relative path: text}."""
    out = {}
    for index in range(files):
        body = ['"""Module %d of the fake package."""' % index, 'import os', '']
        for line in range(lines):
            body.append('def function_%d_%d(value, other=%d):' % (index, line, (line * 7 + seed) % 13))
            body.append('    return value * %d + other  # comment %d' % (line + 1, index))
        out['pkg/%s/module_%02d.py' % ('sub' if index % 3 else 'top', index)] = '\n'.join(body) + '\n'
    return out


def write_package(root, files):
    for relative, text in files.items():
        path = Path(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8', newline='\n')
    return Path(root, 'pkg')


class CorpusTests(unittest.TestCase):
    def test_corpus_is_every_py_file_sorted_by_relative_posix_path_with_a_file_line(self):
        with tempfile.TemporaryDirectory() as directory:
            files = {'pkg/b.py': 'B = 1\n', 'pkg/a.py': 'A = 1\n', 'pkg/sub/c.py': 'C = 1\n', 'pkg/notes.txt': 'no'}
            root = write_package(directory, files)
            corpus, info = rtp.build_corpus(root)
        self.assertEqual(corpus, 'File: pkg/a.py\nA = 1\n\n\nFile: pkg/b.py\nB = 1\n\n\nFile: pkg/sub/c.py\nC = 1\n')
        self.assertEqual(info['files'], 3)
        self.assertEqual(info['characters'], len(corpus))
        self.assertEqual(info['sha256'], hashlib.sha256(corpus.encode()).hexdigest())
        self.assertEqual((info['first_file'], info['last_file']), ('pkg/a.py', 'pkg/sub/c.py'))

    def test_the_package_is_found_without_being_imported(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'rtp_probe_pkg').mkdir()
            Path(directory, 'rtp_probe_pkg', '__init__.py').write_text('raise RuntimeError("imported")\n')
            sys.path.insert(0, directory)
            try:
                root = rtp.package_root('rtp_probe_pkg')
            finally:
                sys.path.remove(directory)
        self.assertEqual(root.name, 'rtp_probe_pkg')
        self.assertNotIn('rtp_probe_pkg', sys.modules)
        with self.assertRaises(RuntimeError):
            rtp.package_root('no_such_package_rtp_xyz')


class BuildTests(unittest.TestCase):
    TARGET = 3000

    def build(self, users=4, target=TARGET, tokenizer=None, corpus=None, **kwargs):
        corpus = corpus if corpus is not None else '\n\n'.join(
            'File: %s\n%s' % item for item in sorted(code_corpus().items()))
        return rtp.build_prompts(users, target, tokenizer=tokenizer or FakeTokenizer(), corpus=corpus, **kwargs)

    def test_every_prompt_is_exactly_the_target_and_recorded(self):
        """The image pins each request's position to the arm's request context (32768 / 131072):
        frozen_combined_runtime.validate_target_option refuses any other prompt length."""
        built = self.build()
        for entry in built['users']:
            with self.subTest(user=entry['user']):
                self.assertEqual(entry['prompt_tokens'], self.TARGET)
                self.assertEqual(len(entry['tokens']), self.TARGET)
                self.assertEqual(entry['chunks_of_2048'], -(-self.TARGET // 2048))
                self.assertLessEqual(entry['padding_tokens'], rtp.MAX_PADDING)
                self.assertEqual(entry['padding_at'] is None, entry['padding_tokens'] == 0)
        self.assertEqual(rtp.summary(built)['prompt_lengths'], [self.TARGET] * 4)

    def test_windows_are_disjoint_contiguous_and_start_at_i_times_len_over_n(self):
        built = self.build()
        length = built['corpus']['cleaned_characters']
        for index, entry in enumerate(built['users']):
            with self.subTest(user=index):
                self.assertEqual(entry['window_start'], index * length // 4)
                self.assertEqual(entry['window_end'], (index + 1) * length // 4)
                self.assertEqual(entry['excerpt_start'], entry['window_start'])
                self.assertLessEqual(entry['excerpt_end'], entry['window_end'])
                self.assertEqual(entry['excerpt_end'] - entry['excerpt_start'], entry['excerpt_characters'])
        ranges = sorted((e['excerpt_start'], e['excerpt_end']) for e in built['users'])
        for (_, end), (start, _) in zip(ranges, ranges[1:]):
            self.assertLessEqual(end, start)

    def test_provenance_hashes_recompute(self):
        corpus = '\n\n'.join('File: %s\n%s' % item for item in sorted(code_corpus().items()))
        built = self.build(corpus=corpus)
        self.assertEqual(built['corpus']['sha256'], hashlib.sha256(corpus.encode()).hexdigest())
        self.assertEqual(built['corpus']['cleaned_sha256'], built['corpus']['sha256'], 'nothing to neutralise here')
        for entry in built['users']:
            excerpt = corpus[entry['excerpt_start']:entry['excerpt_end']]
            self.assertEqual(entry['excerpt_sha256'], hashlib.sha256(excerpt.encode()).hexdigest())
            self.assertEqual(entry['prompt_sha256'], hashlib.sha256(
                json.dumps(entry['tokens'], separators=(',', ':')).encode()).hexdigest())
            self.assertEqual(entry['task_index'], entry['user'] % 4)
            self.assertEqual(entry['task'], rtp.TASK_NAMES[entry['user'] % 4])

    def test_the_build_is_deterministic(self):
        first, second = self.build(), self.build()
        self.assertEqual([e['tokens'] for e in first['users']], [e['tokens'] for e in second['users']])
        self.assertEqual([e['prompt_sha256'] for e in first['users']], [e['prompt_sha256'] for e in second['users']])
        # Everything but the wall-clock cost is identical from one build to the next.
        untimed = lambda built: [{k: v for k, v in e.items() if k != 'tokenizer_seconds'}
                                 for e in rtp.summary(built)['users']]
        self.assertEqual(untimed(first), untimed(second))
        self.assertEqual(first['corpus'], second['corpus'])

    def test_each_user_gets_its_task_inside_make_context_prompts_framing(self):
        tokenizer = FakeTokenizer()
        self.build(tokenizer=tokenizer)
        seen = {}
        for call in tokenizer.calls:
            system, user = call['messages']
            self.assertEqual(system, {'role': 'system', 'content': rtp.SYSTEM})
            self.assertTrue(user['content'].startswith(rtp.HEADER))
            task = next(t for t in rtp.TASKS if user['content'].endswith(rtp.FOOTER + t))
            seen[task] = True
            self.assertEqual({k: call[k] for k in ('tokenize', 'add_generation_prompt', 'return_dict', 'enable_thinking')},
                             dict(tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=False))
        self.assertEqual(len(seen), 4)

    def test_the_framing_is_coding_context_requests_own(self):
        """make_context_prompt is not mounted in the container, so the framing is copied; this
        is what keeps the copy honest."""
        from coding_context_request import make_context_prompt
        from coding_request import TASK
        tokenizer = FakeTokenizer()
        make_context_prompt(tokenizer, context_tokens=4096)
        call = tokenizer.calls[-1]
        system, user = call['messages']
        self.assertEqual(system['content'], rtp.SYSTEM)
        self.assertTrue(user['content'].startswith(rtp.HEADER))
        self.assertTrue(user['content'].endswith(rtp.FOOTER + TASK))
        excerpt = user['content'][len(rtp.HEADER):-len(rtp.FOOTER + TASK)]
        mine = FakeTokenizer()
        rtp.encode_prompt(mine, excerpt, TASK)
        self.assertEqual(mine.calls[-1], call)

    def test_a_mapping_return_is_read_as_input_ids(self):
        flat = self.build(users=2)
        mapped = self.build(users=2, tokenizer=FakeTokenizer(mapping=True))
        self.assertEqual([e['tokens'] for e in flat['users']], [e['tokens'] for e in mapped['users']])

    def test_the_search_is_cheap(self):
        built = self.build(users=4, target=12000, corpus='\n\n'.join(
            'File: %s\n%s' % item for item in sorted(code_corpus(files=120).items())))
        for entry in built['users']:
            self.assertLessEqual(entry['tokenizer_calls'], 12, {k: v for k, v in entry.items() if k != 'tokens'})
        self.assertEqual(built['tokenizer_calls'], sum(e['tokenizer_calls'] for e in built['users']))
        self.assertIn('total', built['seconds'])

    def test_a_corpus_too_small_for_n_windows_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'too small'):
            self.build(users=4, corpus='x = 1\n' * 200)
        corpus = '\n\n'.join('File: %s\n%s' % item for item in sorted(code_corpus(files=6).items()))
        with self.assertRaisesRegex(ValueError, 'too small'):
            self.build(users=4, target=4000, corpus=corpus)

    def test_bad_arguments_are_refused(self):
        for users, target in ((0, 3000), (True, 3000), (4, 0), (4, '3000')):
            with self.subTest(users=users, target=target), self.assertRaises(ValueError):
                self.build(users=users, target=target)
        with self.assertRaises(ValueError):
            rtp.build_prompts(1, 3000, corpus='x')

    def test_write_prompts_keeps_the_tokens_and_summary_drops_them(self):
        built = self.build(users=2)
        with tempfile.TemporaryDirectory() as directory:
            path = rtp.write_prompts(Path(directory, 'real-text-prompts.json'), built)
            loaded = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(loaded['users'][1]['tokens'], built['users'][1]['tokens'])
        self.assertNotIn('tokens', rtp.summary(built)['users'][0])
        self.assertIsNone(rtp.write_prompts(Path(directory, 'gone', 'x.json'), built))


class SpecialTokenTests(unittest.TestCase):
    MARKERS = 'TEMPLATE = "<|im_start|>user\\n{x}<|im_end|>"\nTHINK = ("<think>", "</think>")\nEOT = "<|endoftext|>"\n'

    def corpus(self):
        """Every file opens with the template's own markers, so every user's excerpt holds some."""
        files = {path: self.MARKERS + text for path, text in code_corpus().items()}
        return '\n\n'.join('File: %s\n%s' % item for item in sorted(files.items()))

    def test_template_markers_in_the_source_are_neutralised_and_the_prompt_carries_only_the_templates(self):
        corpus = self.corpus()
        tokenizer = FakeTokenizer()
        built = rtp.build_prompts(4, 3000, tokenizer=tokenizer, corpus=corpus)
        self.assertEqual(built['corpus']['neutralised'], {'</think>': 40, '<think>': 40, '<|endoftext|>': 40,
                                                          '<|im_end|>': 40, '<|im_start|>': 40})
        for entry in built['users']:
            excerpt = rtp.neutralise(corpus, list(SPECIALS))[0][entry['excerpt_start']:entry['excerpt_end']]
            self.assertIn('< |im_start|>', excerpt, 'the excerpt did carry the markers, broken')
        self.assertNotEqual(built['corpus']['cleaned_sha256'], built['corpus']['sha256'])
        template = tokenizer.apply_chat_template([
            {'role': 'system', 'content': rtp.SYSTEM}, {'role': 'user', 'content': rtp.HEADER + rtp.FOOTER}],
            tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=False)
        expected = sorted(t for t in template if t < 100)
        for entry in built['users']:
            self.assertEqual(sorted(t for t in entry['tokens'] if t < 100), expected)

    def test_neutralise_breaks_every_occurrence_and_counts_them(self):
        text, counts = rtp.neutralise('a <|im_start|> b <think></think> <|im_start|>', ['<|im_start|>', '<think>', '</think>'])
        self.assertEqual(text, 'a < |im_start|> b < think>< /think> < |im_start|>')
        self.assertEqual(counts, {'</think>': 1, '<think>': 1, '<|im_start|>': 2})

    def test_a_special_id_outside_the_template_refuses_the_build(self):
        with mock.patch.object(rtp, 'neutralise', lambda corpus, texts: (corpus, {})):
            with self.assertRaisesRegex(ValueError, 'survived neutralisation'):
                rtp.build_prompts(4, 3000, tokenizer=FakeTokenizer(), corpus=self.corpus())

    def test_guarded_tokens_come_from_the_added_vocabulary(self):
        self.assertEqual(rtp.guarded_tokens(FakeTokenizer()), SPECIALS)
        legacy = SimpleNamespace(get_added_vocab=lambda: {'<|im_end|>': 2, '<think>': 4, 'plain': 9},
                                 all_special_tokens=['<|im_end|>'])
        self.assertEqual(rtp.guarded_tokens(legacy), {'<|im_end|>': 2, '<think>': 4})


class SteppingTokenizer(FakeTokenizer):
    """Word and punctuation pieces are two ids, whitespace one: a count can be stepped over, as
    byte-level BPE steps over one when a character re-merges the last tokens."""

    def encode(self, text):
        out = []
        for part in SPECIAL_SPLIT.split(text):
            if part in SPECIALS:
                out.append(SPECIALS[part])
            elif part:
                for piece in PIECE.findall(part):
                    token = 100 + zlib.crc32(piece.encode()) % 50000
                    out.extend([token] if piece.isspace() else [token, token + 1])
        return out


class FitPrefixTests(unittest.TestCase):
    def test_a_token_count_that_jumps_past_the_target_is_refused_not_padded_far(self):
        def encode(excerpt):
            return [7] * (10 + 200 * (len(excerpt) // 50))
        characters, tokens, template = rtp.fit_prefix(encode, 'x' * 100000, 3000)
        self.assertLess(len(tokens), 3000)
        self.assertGreater(len(encode('x' * (characters + 1))), 3000, 'the longest prefix below the target')
        with self.assertRaisesRegex(ValueError, 'no prefix of the window comes within 16 tokens of 3000'):
            rtp.pad_to(tokens, template, 3000, 9)

    def test_it_converges_on_a_nonlinear_tokenizer(self):
        def encode(excerpt):
            return [7] * (20 + len(excerpt) // 3 + len(excerpt) // 997)
        characters, tokens, template = rtp.fit_prefix(encode, 'y' * 400000, 50000)
        self.assertEqual(len(tokens), 50000)
        self.assertEqual(len(template), 20)
        self.assertEqual(len(tokens), len(encode('y' * characters)))

    def test_a_target_bpe_steps_over_is_padded_just_before_the_footer(self):
        tokenizer = SteppingTokenizer()
        template = rtp.encode_prompt(tokenizer, '', rtp.TASKS[0])
        # Every 'xxxx' piece is two ids and the excerpt splits the template's one blank-line piece
        # into two newline pieces, so a prompt is the template + 1 + an even count: an even
        # distance from the template is unreachable.
        target = len(template) + 2 * 700
        built = rtp.build_prompts(1, target, tokenizer=tokenizer, corpus='x' * 20000)
        entry = built['users'][0]
        self.assertEqual(len(entry['tokens']), target)
        self.assertEqual(entry['prompt_tokens'], target)
        self.assertEqual(entry['padding_tokens'], 1)
        filler = rtp.filler_token(tokenizer)
        self.assertEqual(entry['padding_token_id'], filler)
        at = entry['padding_at']
        self.assertEqual(entry['tokens'][at], filler)
        tail = entry['tokens'][at + 1:]
        self.assertGreater(len(tail), 20, 'the footer, the task and the generation prompt follow the filler')
        self.assertEqual(tail, template[len(template) - len(tail):])
        unpadded = rtp.encode_prompt(tokenizer, 'x' * entry['excerpt_characters'], rtp.TASKS[0])
        self.assertEqual(entry['tokens'][:at] + tail, unpadded)
        self.assertEqual(entry['prompt_sha256'], hashlib.sha256(
            json.dumps(entry['tokens'], separators=(',', ':')).encode()).hexdigest())

    def test_the_filler_must_be_one_token(self):
        self.assertEqual(rtp.filler_token(FakeTokenizer()), FakeTokenizer().encode(rtp.FILLER_TEXT)[0])
        with self.assertRaisesRegex(ValueError, 'exactly one token'):
            rtp.filler_token(FakeTokenizer(scale=2))

    def test_the_splice_never_reaches_into_the_shared_head(self):
        self.assertEqual(rtp.splice_index([1, 2, 3, 9, 9, 4, 5], [1, 2, 3, 4, 5]), 5)
        self.assertEqual(rtp.splice_index([1, 2, 9, 2], [1, 2]), 4, 'a tail equal to the head is not the footer')
        self.assertEqual(rtp.pad_to([1, 2, 9, 4], [1, 2, 4], 6, 0), ([1, 2, 9, 0, 0, 4], 2, 3))
        self.assertEqual(rtp.pad_to([1, 2], [1], 2, 0), ([1, 2], 0, None))


class PerUserTargetTests(unittest.TestCase):
    """targets=[...]: the C2 serving gate's real-text matrix, one prompt length per user."""

    def corpus(self, files=120):
        return '\n\n'.join('File: %s\n%s' % item for item in sorted(code_corpus(files=files).items()))

    def compact_overhead(self, task_index=0):
        return len(rtp.encode_compact(FakeTokenizer(), '', rtp.COMPACT_TASKS[task_index]))

    def repository_overhead(self, task_index=0):
        return len(rtp.encode_prompt(FakeTokenizer(), '', rtp.TASKS[task_index]))

    def test_each_user_gets_exactly_its_own_length(self):
        short = self.compact_overhead() + 5
        targets = [short, 2047, 2048, 2049, 4096]
        built = rtp.build_prompts(5, None, tokenizer=FakeTokenizer(), corpus=self.corpus(), targets=targets)
        self.assertEqual([len(e['tokens']) for e in built['users']], targets)
        self.assertEqual(rtp.summary(built)['prompt_lengths'], targets)
        self.assertEqual(built['targets'], targets)
        self.assertIsNone(built['target'])
        self.assertEqual([e['target'] for e in built['users']], targets)
        self.assertEqual(built['users'][0]['framing'], 'compact')
        self.assertEqual({e['framing'] for e in built['users'][1:]}, {'repository'})
        self.assertEqual(built['users'][0]['task'], rtp.COMPACT_TASK_NAMES[0])
        self.assertEqual(built['users'][1]['task'], rtp.TASK_NAMES[1])

    def test_the_uniform_build_is_unchanged(self):
        """No targets: no framing key, the top-level target, and the same tokens as a per-user
        build whose lengths are all that target (every one fits the repository framing)."""
        uniform = rtp.build_prompts(4, 3000, tokenizer=FakeTokenizer(), corpus=self.corpus())
        self.assertNotIn('framing', uniform['users'][0])
        self.assertNotIn('targets', uniform)
        self.assertEqual(uniform['target'], 3000)
        per_user = rtp.build_prompts(4, None, tokenizer=FakeTokenizer(), corpus=self.corpus(), targets=[3000] * 4)
        self.assertEqual([e['tokens'] for e in uniform['users']], [e['tokens'] for e in per_user['users']])

    def test_a_target_below_the_repository_framing_takes_the_compact_one(self):
        tokenizer = FakeTokenizer()
        boundary = self.repository_overhead(0) + rtp.MIN_EXCERPT_TOKENS
        self.assertEqual(rtp.choose_framing(tokenizer, boundary, 0), 'repository')
        self.assertEqual(rtp.choose_framing(tokenizer, boundary - 1, 0), 'compact')
        built = rtp.build_prompts(1, None, tokenizer=tokenizer, corpus=self.corpus(), targets=[boundary - 1])
        entry = built['users'][0]
        self.assertEqual((entry['framing'], len(entry['tokens'])), ('compact', boundary - 1))
        call = tokenizer.calls[-1]
        self.assertEqual(len(call['messages']), 1, 'one user turn, no system turn')
        content = call['messages'][0]['content']
        self.assertTrue(content.startswith(rtp.COMPACT_TASKS[0] + rtp.COMPACT_HEADER))
        self.assertEqual({k: call[k] for k in ('tokenize', 'add_generation_prompt', 'return_dict', 'enable_thinking')},
                         dict(tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=False))

    def test_the_solo_arm_builds_the_concurrent_arms_prompts(self):
        targets = [self.compact_overhead() + 9, 2048, 5000]
        first = rtp.build_prompts(3, None, tokenizer=FakeTokenizer(), corpus=self.corpus(), targets=targets)
        second = rtp.build_prompts(3, None, tokenizer=FakeTokenizer(), corpus=self.corpus(), targets=targets)
        self.assertEqual([e['prompt_sha256'] for e in first['users']], [e['prompt_sha256'] for e in second['users']])

    def test_windows_stay_disjoint_with_mixed_lengths(self):
        built = rtp.build_prompts(3, None, tokenizer=FakeTokenizer(), corpus=self.corpus(),
                                  targets=[self.compact_overhead() + 3, 6000, 2500])
        ranges = sorted((e['excerpt_start'], e['excerpt_end']) for e in built['users'])
        for (_, end), (start, _) in zip(ranges, ranges[1:]):
            self.assertLessEqual(end, start)

    def test_bad_target_lists_are_refused(self):
        with self.assertRaisesRegex(ValueError, '2 prompt lengths for 3 users'):
            rtp.build_prompts(3, None, tokenizer=FakeTokenizer(), corpus=self.corpus(), targets=[3000, 3000])
        for text in ('', '60,,255', '60,-1', '60,x', '0'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                rtp.parse_targets(text)
        self.assertEqual(rtp.parse_targets('60, 255,2047'), [60, 255, 2047])
        self.assertEqual(rtp.parse_targets([60, 255]), [60, 255])
        with self.assertRaisesRegex(ValueError, 'no room'):
            rtp.build_prompts(1, None, tokenizer=FakeTokenizer(), corpus=self.corpus(),
                              targets=[self.compact_overhead() - 1])


class TokenizerLoadTests(unittest.TestCase):
    def test_the_snapshot_is_loaded_local_only_without_remote_code(self):
        calls = []
        fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: calls.append((args, kwargs)) or 'tokenizer'))
        with mock.patch.dict(sys.modules, {'transformers': fake}):
            self.assertEqual(rtp.load_tokenizer('/models/snapshot'), 'tokenizer')
        self.assertEqual(calls, [(('/models/snapshot',), dict(local_files_only=True, trust_remote_code=False))])



SNAPSHOT_SUFFIX = os.path.join('models--Qwen--Qwen3.8-27B', 'snapshots', '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
SNAPSHOT_ROOTS = (os.environ.get('QWEN_SNAPSHOT_ROOT', ''), '/models', '/models/hub', '/home/thatch/hf-cache/hub',
                  os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub'))


def served_snapshot():
    for root in SNAPSHOT_ROOTS:
        path = os.path.join(root, SNAPSHOT_SUFFIX) if root else ''
        if path and os.path.isfile(os.path.join(path, 'tokenizer.json')):
            return path
    return None


class QwenTemplate(object):
    """The served snapshot's tokenizer.json under the Qwen3 chat template (a system turn if given, the
    user turn, the generation prompt, and the empty think block that enable_thinking=False writes) -
    for a snapshot carrying no tokenizer_config.json with the template (a dev PC's copy). The repository
    framing's 115-133 tokens (real_text_prompts' docstring, measured on the rig) validate it."""

    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(os.path.join(path, 'tokenizer.json'))

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_dict=False,
                            enable_thinking=True):
        text = ''.join('<|im_start|>%s\n%s<|im_end|>\n' % (m['role'], m['content']) for m in messages)
        text += '<|im_start|>assistant\n' + ('<think>\n\n</think>\n\n' if not enable_thinking else '')
        return self.tokenizer.encode(text, add_special_tokens=False).ids


class TemplateTokenTests(unittest.TestCase):
    """Review finding 12: the compact framing was never measured on the real tokenizer; a compact
    template past a target raises "no room" inside the container and kills the arm before a request."""

    def test_the_pinned_lengths_leave_room_for_the_ladder(self):
        self.assertEqual(len(rtp.COMPACT_TEMPLATE_TOKENS), len(rtp.COMPACT_TASKS))
        self.assertEqual(len(rtp.REPOSITORY_TEMPLATE_TOKENS), len(rtp.TASKS))
        self.assertEqual(rtp.COMPACT_MIN_TARGET, max(rtp.COMPACT_TEMPLATE_TOKENS) + 1)
        self.assertLess(rtp.COMPACT_MIN_TARGET, 60, 'the G4 ladder\'s shortest rung carries code')
        self.assertEqual((min(rtp.REPOSITORY_TEMPLATE_TOKENS), max(rtp.REPOSITORY_TEMPLATE_TOKENS)), (115, 133))

    def test_the_pinned_lengths_are_the_served_tokenizers(self):
        path = served_snapshot()
        if path is None:
            self.skipTest('the served snapshot\'s tokenizer is not on this machine')
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            candidate = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            if getattr(candidate, 'chat_template', None):
                tokenizer = candidate
        except Exception:
            tokenizer = None
        if tokenizer is None:
            try:
                tokenizer = QwenTemplate(path)
            except ImportError:
                self.skipTest('neither transformers with the chat template nor tokenizers is installed')
        self.assertEqual(tuple(len(rtp.encode_prompt(tokenizer, '', task)) for task in rtp.TASKS),
                         rtp.REPOSITORY_TEMPLATE_TOKENS)
        self.assertEqual(tuple(len(rtp.encode_compact(tokenizer, '', task)) for task in rtp.COMPACT_TASKS),
                         rtp.COMPACT_TEMPLATE_TOKENS)

if __name__ == '__main__':
    unittest.main()
