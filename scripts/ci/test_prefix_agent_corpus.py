"""prefix_agent_corpus: the real-text coding-agent corpus of the prefix-reuse gates, held on CPU.

No tokenizer here: sizes are the module's character estimate, and fit_tokens runs against counters
that misbehave the way a BPE tokenizer does (a filler that merges with trailing whitespace)."""

import ast
import json
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prefix_agent_corpus as pc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

PYTHON = '''"""A module."""
import os


def load_items(path, limit=None):
    """Read items."""
    with open(path) as handle:
        lines = handle.read().splitlines()
    return lines[:limit] if limit else lines


class Store(object):
    def __init__(self, root):
        self.root = root

    def put(self, key, value):
        with open(os.path.join(self.root, key), 'w') as handle:
            handle.write(value)
'''


def make_tree(root, copies=6):
    os.makedirs(os.path.join(root, 'scripts', 'ci'))
    os.makedirs(os.path.join(root, 'docs'))
    for index in range(copies):
        with open(os.path.join(root, 'scripts', 'ci', 'mod_%d.py' % index), 'w', encoding='utf-8', newline='') as handle:
            handle.write((PYTHON * (index + 2)).replace('\n', '\r\n' if index == 0 else '\n'))
    with open(os.path.join(root, 'docs', 'README.md'), 'w', encoding='utf-8') as handle:
        handle.write('# Project\n\nRun the tests with `python -m unittest`.\n' * 40)
    with open(os.path.join(root, 'scripts', 'ci', 'tiny.py'), 'w') as handle:
        handle.write('x = 1\n')
    with open(os.path.join(root, 'scripts', 'ci', 'data.bin'), 'wb') as handle:
        handle.write(b'\xff\xfe' * 400)
    with open(os.path.join(root, 'scripts', 'ci', 'latin.py'), 'wb') as handle:
        handle.write(b'# \xe9\xe9\n' * 300)


class CorpusCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        make_tree(cls.tmp)
        cls.sources = pc.load_sources(cls.tmp)
        cls.corpus = pc.Corpus(cls.sources)


class SourceTests(CorpusCase):
    def test_sources_are_the_readable_text_files_sorted_and_normalised(self):
        paths = [source.path for source in self.sources]
        self.assertEqual(paths, sorted(paths))
        self.assertIn('docs/README.md', paths)
        self.assertIn('scripts/ci/mod_0.py', paths)
        self.assertNotIn('scripts/ci/tiny.py', paths, 'below MIN_FILE_CHARS')
        self.assertNotIn('scripts/ci/data.bin', paths, 'not a text extension')
        self.assertNotIn('scripts/ci/latin.py', paths, 'not UTF-8')
        first = [source for source in self.sources if source.path == 'scripts/ci/mod_0.py'][0]
        self.assertNotIn('\r', first.text, 'CRLF is normalised')

    def test_corpus_info_is_deterministic_and_content_sensitive(self):
        again = pc.corpus_info(pc.load_sources(self.tmp))
        self.assertEqual(again, self.corpus.info)
        changed = list(self.sources)
        changed[0] = pc.Source(changed[0].path, changed[0].text + '#')
        self.assertNotEqual(pc.corpus_info(changed)['sha256'], self.corpus.info['sha256'])

    def test_an_empty_corpus_is_refused(self):
        with self.assertRaises(ValueError):
            pc.Corpus([])

    def test_numbered_is_the_read_tools_shape(self):
        text = pc.numbered(['a', 'b', 'c'], 1, 1000)
        self.assertEqual(text, '     2\tb\n     3\tc')
        self.assertEqual(pc.numbered(['x' * 50], 0, 10), '     1\t' + 'x' * 50, 'one line always, even over budget')


class SystemTests(CorpusCase):
    def test_the_full_block_carries_the_readme_and_nine_tools(self):
        full, compact = self.corpus.system_text('full'), self.corpus.system_text('compact')
        self.assertIn('Run the tests with', full)
        self.assertIn(pc.ENVIRONMENT, full)
        self.assertLess(len(compact), len(full) / 3)
        self.assertEqual([tool['function']['name'] for tool in self.corpus.tools('full')],
                         ['bash', 'read', 'edit', 'write', 'grep', 'glob', 'todowrite', 'task', 'webfetch'])
        self.assertEqual([tool['function']['name'] for tool in self.corpus.tools('compact')], ['read', 'grep', 'bash'])
        with self.assertRaises(ValueError):
            self.corpus.system_text('huge')

    def test_the_environment_block_names_no_moving_date(self):
        """A line that changes per request breaks the prefix there (P0b: a date line at 4,223)."""
        self.assertIn("Today's date: 2026-09-26", pc.ENVIRONMENT)

    def test_tools_are_copies(self):
        tools = self.corpus.tools('compact')
        tools[0]['function']['name'] = 'changed'
        self.assertEqual(self.corpus.tools('compact')[0]['function']['name'], 'read')


class ConversationTests(CorpusCase):
    def answer(self, calls=(), content='Done.', reasoning='Thinking.'):
        return dict(content=content, reasoning=reasoning, tool_calls=[
            dict(id='chatcmpl-tool-%d' % index, type='function', function=dict(name=name, arguments=json.dumps(args)))
            for index, (name, args) in enumerate(calls)])

    def test_the_same_seed_and_name_build_the_same_messages(self):
        a = pc.Conversation(self.corpus, 'c', 7, first_tokens=500)
        b = pc.Conversation(self.corpus, 'c', 7, first_tokens=500)
        c = pc.Conversation(self.corpus, 'd', 7, first_tokens=500)
        self.assertEqual(a.body(), b.body())
        self.assertNotEqual(a.messages[1], c.messages[1])
        for conv in (a, b):
            conv.add_answer(self.answer([('read', {'file_path': '/x/scripts/ci/mod_2.py'})]))
            conv.extend(800)
        self.assertEqual(a.body(), b.body(), 'a cold and a hit arm are sent byte-identical messages')

    def test_the_first_message_is_a_task_on_a_real_file_with_an_attachment(self):
        conv = pc.Conversation(self.corpus, 'c', 1, system='compact', first_tokens=600)
        user = conv.messages[1]['content']
        self.assertTrue(any(source.path in user for source in self.sources))
        self.assertIn('```', user)
        self.assertGreater(len(user), 600 * pc.CHARS_PER_TOKEN * 0.7)
        bare = pc.Conversation(self.corpus, 'c', 1, system='compact')
        self.assertNotIn('```', bare.messages[1]['content'])

    def test_tool_calls_get_one_result_each_in_order_with_their_ids(self):
        conv = pc.Conversation(self.corpus, 'c', 1)
        conv.add_answer(self.answer([('read', {'file_path': '/w/scripts/ci/mod_3.py', 'offset': 5}),
                                     ('grep', {'pattern': 'def put'}), ('todowrite', {'todos': []}),
                                     ('bash', {'command': 'pytest -q', 'description': 'tests'})]))
        conv.extend(2000)
        tools = conv.messages[-4:]
        self.assertEqual([m['role'] for m in tools], ['tool'] * 4)
        self.assertEqual([m['tool_call_id'] for m in tools], ['chatcmpl-tool-%d' % i for i in range(4)])
        self.assertTrue(tools[0]['content'].startswith('     5\t'), 'read honours the offset of the named file')
        self.assertIn('scripts/ci/', tools[1]['content'])
        self.assertIn('def put', tools[1]['content'])
        self.assertLess(len(tools[2]['content']), 200, 'todowrite gets a small share')
        self.assertTrue(tools[3]['content'].startswith('$ pytest -q'))
        total = sum(len(m['content']) for m in tools)
        self.assertGreater(total, 2000 * pc.CHARS_PER_TOKEN * 0.6)
        self.assertLess(total, 2000 * pc.CHARS_PER_TOKEN * 1.6)

    def test_no_tool_call_gets_a_user_follow_up_with_real_text(self):
        conv = pc.Conversation(self.corpus, 'c', 1)
        conv.add_answer(self.answer())
        conv.extend(500)
        self.assertEqual(conv.messages[-1]['role'], 'user')
        self.assertTrue(any(conv.messages[-1]['content'].startswith(text.split(' ')[0]) for text in pc.FOLLOWUPS))
        self.assertIn('\t', conv.messages[-1]['content'])

    def test_the_answer_goes_back_as_served(self):
        message = pc.assistant_message(dict(content=None, reasoning='r', tool_calls=[
            dict(id='t1', function=dict(name='read', arguments='{"file_path": "/a"}'))]))
        self.assertEqual(message, {'role': 'assistant', 'content': '', 'reasoning': 'r', 'tool_calls': [
            {'id': 't1', 'type': 'function', 'function': {'name': 'read', 'arguments': '{"file_path": "/a"}'}}]})
        self.assertNotIn('reasoning', pc.assistant_message(dict(content='x')))
        self.assertNotIn('tool_calls', pc.assistant_message(dict(content='x', tool_calls=[])))

    def test_unparseable_arguments_still_get_a_result(self):
        self.assertEqual(pc.parse_arguments('{not json'), {})
        self.assertEqual(pc.parse_arguments('[1]'), {})
        conv = pc.Conversation(self.corpus, 'c', 1)
        conv.messages.append({'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'x', 'type': 'function', 'function': {'name': 'read', 'arguments': '{oops'}}]})
        conv.extend(300)
        self.assertEqual(conv.messages[-1]['tool_call_id'], 'x')

    def test_a_fork_is_independent_and_differs_only_in_what_follows(self):
        conv = pc.Conversation(self.corpus, 'c', 1)
        conv.salt = 'salt'
        conv.add_answer(self.answer())
        fork = conv.fork('suffix')
        conv.extend(400)
        fork.extend(400)
        self.assertEqual(fork.salt, 'salt')
        self.assertEqual(conv.messages[:-1], fork.messages[:-1])
        self.assertNotEqual(conv.messages[-1], fork.messages[-1])
        fork.messages[0]['content'] = 'changed'
        self.assertNotEqual(conv.messages[0]['content'], 'changed')

    def test_diverge_at_replaces_one_early_message(self):
        conv = pc.Conversation(self.corpus, 'c', 1)
        conv.add_answer(self.answer())
        conv.extend(600)
        conv.add_answer(self.answer(content='Second.'))
        early = conv.diverge_at(3)
        self.assertEqual(early.messages[:3], conv.messages[:3])
        self.assertNotEqual(early.messages[3]['content'], conv.messages[3]['content'])
        self.assertEqual(early.messages[4:], conv.messages[4:])
        keep = min(len(conv.messages[3]['content']) // 4, 200)
        self.assertEqual(early.messages[3]['content'][:keep], conv.messages[3]['content'][:keep])
        with self.assertRaises(ValueError):
            conv.diverge_at(2)

    def test_compaction_and_followups_are_real_text(self):
        rng = random.Random(1)
        text = self.corpus.compaction(rng, 1000)
        self.assertTrue(text.startswith(pc.COMPACTION))
        self.assertIn('Continue with the last task', text)


class SizingTests(unittest.TestCase):
    def test_growth_input_reaches_the_target_and_never_goes_below_the_minimum(self):
        self.assertEqual(pc.growth_input(9000, 4000, 1000), 4000)
        self.assertEqual(pc.growth_input(9000, 8900, 1000), int(pc.MIN_INPUT_CHARS / pc.CHARS_PER_TOKEN))

    def test_first_attachment_subtracts_the_measured_system_block(self):
        self.assertEqual(pc.first_attachment('compact', 2600), 2600 - pc.SYSTEM_TOKENS['compact'])
        self.assertEqual(pc.first_attachment('full', 100), 0)

    def test_the_metering_shape_is_bounded_and_centred_near_its_mean(self):
        rng = random.Random(5)
        sizes = [pc.metering_input(rng) for _ in range(4000)]
        self.assertGreaterEqual(min(sizes), pc.METERING['input_tokens_min'])
        self.assertLessEqual(max(sizes), pc.METERING['input_tokens_max'])
        self.assertLess(abs(sum(sizes) / float(len(sizes)) - pc.METERING['input_tokens_mean']), 250)
        gaps = [pc.think_gap(rng, 15.0) for _ in range(4000)]
        self.assertGreaterEqual(min(gaps), 0.0)
        self.assertLess(abs(sum(gaps) / len(gaps) - 15.0), 1.5)
        self.assertEqual(pc.think_gap(rng, 0), 0.0)

    def test_chain_targets_sorts_the_hits_after_the_first(self):
        self.assertEqual(pc.chain_targets(2600, (9000, 4200)), [2600, 4200, 9000])


class FitTests(unittest.TestCase):
    PAD = ('word ' * 30 + '          ') * 40

    def build(self, pad, filler):
        return dict(text='task ' + self.PAD[:pad].rstrip() + filler)

    @staticmethod
    def count(body):
        """A BPE-like counter: a word per token, a whitespace run of 4+ an extra token, and ' ok'
        after whitespace merging into two."""
        text = body['text']
        return len(text.split()) + text.count('    ') // 4 + text.count('  ok')

    def test_an_exact_count_is_reached(self):
        for target in (50, 51, 97, 160):
            body, tokens = pc.fit_tokens(self.build, self.count, target, len(self.PAD))
            self.assertEqual(tokens, target)
            self.assertEqual(self.count(body), target)

    def test_a_pad_that_starts_past_the_target_is_refused(self):
        with self.assertRaises(pc.FitError):
            pc.fit_tokens(self.build, self.count, 0, len(self.PAD))

    def test_a_counter_that_jumps_by_two_is_refused_rather_than_faked(self):
        with self.assertRaises(pc.FitError):
            pc.fit_tokens(lambda pad, filler: dict(n=2 * (pad + len(filler))), lambda body: body['n'], 7, 100)

    def test_the_filler_is_tried_in_order(self):
        seen = []

        def count(body):
            seen.append(body['text'][-3:])
            return len(body['text'].split())

        pc.fit_tokens(self.build, count, 40, len(self.PAD))
        self.assertTrue(all(isinstance(item, str) for item in seen))


class HostSyntaxTests(unittest.TestCase):
    def test_the_host_modules_parse_as_python_37(self):
        """The prefix gate runs on the rig host (python3.7 syntax, stdlib only)."""
        for name in ('prefix_agent_corpus.py', 'prefix_markers.py', 'prefix_judge.py', 'prefix_report.py',
                     'prefix_replay.py', 'c2_prefix_gate.py', 'c2_serving_job.py'):
            with open(os.path.join(HERE, name), encoding='utf-8') as handle:
                source = handle.read()
            # On 3.8+ the parser is held to 3.7's grammar; on 3.7 itself a plain parse is the check.
            ast.parse(source, name, **(dict(feature_version=(3, 7)) if sys.version_info >= (3, 8) else {}))
            self.assertNotIn('\r', source, '%s must be LF' % name)
            tree = ast.parse(source)
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split('.')[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split('.')[0])
            third_party = imported & {'numpy', 'torch', 'yaml', 'requests', 'transformers', 'vllm'}
            self.assertEqual(third_party, set(), '%s imports %s' % (name, third_party))


if __name__ == '__main__':
    unittest.main()
