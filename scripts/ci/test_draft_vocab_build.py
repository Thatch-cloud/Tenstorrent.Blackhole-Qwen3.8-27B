"""draft_vocab_build.py (the coding draft-vocabulary generator) and the committed list it wrote.

The pure parts are held here (selection, the required tokens, the mixture, coverage, the splits, the rendering of public text into the model's output shape, the packed
format). The command line needs the tokenizer, a parquet reader and the corpora, none of which the CPU suite has: the committed list is held instead (its sha256, its
order, its required tokens, its stated coverage) and the public-repository hygiene of its sidecar."""

import json
from pathlib import Path
import re
import unittest

import draft_vocab_build as build

HERE = Path(__file__).resolve().parent
SIDECAR = HERE / 'draft_vocab_coding_40960.json'
LIST = HERE / 'draft_vocab_coding_40960.ids'


class RequiredTokenTests(unittest.TestCase):
    def test_the_byte_alphabet_is_256_distinct_printable_characters(self):
        mapping = build.bytes_to_unicode()
        self.assertEqual(len(mapping), 256)
        self.assertEqual(len(set(mapping.values())), 256)
        self.assertTrue(all(len(character) == 1 for character in mapping.values()))
        # the printable ASCII range maps to itself; space and newline do not
        self.assertEqual(mapping[ord('a')], 'a')
        self.assertNotEqual(mapping[ord(' ')], ' ')
        self.assertEqual(mapping[ord(' ')], 'Ġ')
        self.assertEqual(mapping[ord('\n')], 'Ċ')

    def test_byte_token_ids_find_every_single_byte_token_or_refuse(self):
        vocabulary = {character: index for index, character in enumerate(build.bytes_to_unicode().values())}
        self.assertEqual(build.byte_token_ids(vocabulary), tuple(range(256)))
        del vocabulary['a']
        with self.assertRaisesRegex(ValueError, 'no single-byte token'):
            build.byte_token_ids(vocabulary)

    def test_required_from_tokenizer_takes_every_added_token_special_or_not(self):
        vocabulary = {character: index for index, character in enumerate(build.bytes_to_unicode().values())}
        vocabulary['ab'] = 300
        tokenizer = dict(model=dict(vocab=vocabulary), added_tokens=[dict(id=301, special=True), dict(id=302, special=False), dict(id=303, special=False)])
        added, single, real = build.required_from_tokenizer(tokenizer)
        self.assertEqual(added, (301, 302, 303))
        self.assertEqual(single, tuple(range(256)))
        self.assertEqual(real, 304)


class SelectionTests(unittest.TestCase):
    def test_required_tokens_are_kept_whatever_their_frequency(self):
        mixed = {token: float(1000 - token) for token in range(1000)}
        chosen = build.select(mixed, required=(900, 999), rows=128, real_tokens=1000)
        self.assertEqual(len(chosen), 128)
        self.assertIn(900, chosen)
        self.assertIn(999, chosen)
        self.assertEqual(chosen, tuple(sorted(chosen)))
        # the rest is the 126 most frequent
        self.assertEqual(set(chosen) - {900, 999}, set(range(126)))

    def test_ties_go_to_the_lower_id(self):
        chosen = build.select({token: 1.0 for token in range(500)}, required=(), rows=128, real_tokens=500)
        self.assertEqual(chosen, tuple(range(128)))

    def test_padding_ids_are_never_chosen_and_uncounted_tokens_fill_from_the_bottom(self):
        chosen = build.select({10: 5.0, 400: 9.0, 900: 99.0}, required=(), rows=128, real_tokens=500)
        self.assertNotIn(900, chosen)
        self.assertEqual(chosen[:2], (0, 1))
        self.assertEqual(len(chosen), 128)
        self.assertIn(400, chosen)

    def test_bad_sizes_and_required_sets_are_refused(self):
        for rows in (0, 100, 129, -128):
            with self.assertRaises(ValueError):
                build.select({1: 1.0}, (), rows, 1000)
        with self.assertRaises(ValueError):
            build.select({1: 1.0}, (), 2048, 1000)
        with self.assertRaises(ValueError):
            build.select({1: 1.0}, tuple(range(129)), 128, 1000)
        with self.assertRaises(ValueError):
            build.select({1: 1.0}, (1000,), 128, 1000)

    def test_the_mixture_weights_categories_not_sizes(self):
        counts = {'edit': {1: 1000, 2: 1000}, 'prose': {3: 1}}
        mixed, used = build.mixture(counts, (('edit', 0.5), ('prose', 0.5), ('shell', 0.5)))
        self.assertEqual(set(used), {'edit', 'prose'})
        self.assertAlmostEqual(sum(used.values()), 1.0)
        # one prose token outweighs each of the edit tokens: 0.5 x 1 against 0.5 x 0.5
        self.assertGreater(mixed[3], mixed[1])
        self.assertAlmostEqual(mixed[1], mixed[2])
        with self.assertRaises(ValueError):
            build.mixture({})

    def test_coverage_counts_occurrences(self):
        self.assertAlmostEqual(build.coverage({1: 90, 2: 10}, {1}), 0.9)
        self.assertEqual(build.coverage({}, {1}), None)
        held = {'edit': {1: 90, 2: 10}, 'prose': {1: 50, 3: 50}, 'oss_py': {1: 100}}
        weighted, worst, per = build.code_coverage(held, {1}, {'edit': 0.5, 'prose': 0.5, 'oss_py': 0.5})
        self.assertAlmostEqual(per['edit'], 0.9)
        self.assertNotIn('prose', per)         # prose is not code
        self.assertAlmostEqual(worst, 0.9)
        self.assertAlmostEqual(weighted, 0.95)

    def test_the_weights_sum_to_one_and_name_the_code_categories(self):
        self.assertAlmostEqual(sum(weight for unused, weight in build.WEIGHTS), 1.0)
        names = {name for name, unused in build.WEIGHTS}
        self.assertTrue(set(build.CODE_CATEGORIES) <= names)


class SplitAndFormatTests(unittest.TestCase):
    def test_the_split_is_stable_and_about_one_in_ten(self):
        keys = ['repo%d/name' % index for index in range(5000)]
        first = [build.split_of(key) for key in keys]
        self.assertEqual(first, [build.split_of(key) for key in keys])
        held = first.count('heldout')
        self.assertTrue(400 < held < 600, held)

    def test_ids_round_trip_as_uint32_little_endian(self):
        ids = (0, 1, 255, 65536, 248076)
        data = build.pack_ids(ids)
        self.assertEqual(len(data), 20)
        self.assertEqual(data[:8], b'\x00\x00\x00\x00\x01\x00\x00\x00')
        self.assertEqual(build.unpack_ids(data), ids)
        with self.assertRaises(ValueError):
            build.unpack_ids(b'\x00\x00\x00')

    PATCH = ('diff --git a/pkg/mod.py b/pkg/mod.py\nindex 1..2 100644\n--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1,3 +1,4 @@ def old():\n'
             ' keep = 1\n-removed = 2\n+added = 3\n+more = 4\n tail = 5\n'
             'diff --git a/pkg/new.py b/pkg/new.py\nnew file mode 100644\nindex 0..3\n--- /dev/null\n+++ b/pkg/new.py\n@@ -0,0 +1,2 @@\n+def fresh(x):\n+    return x\n')

    def test_hunks_split_a_unified_diff_into_old_and_new_sides(self):
        found = build.hunks(self.PATCH)
        self.assertEqual(found[0], ('pkg/mod.py', False, 'keep = 1\nremoved = 2\ntail = 5', 'keep = 1\nadded = 3\nmore = 4\ntail = 5'))
        self.assertEqual(found[1], ('pkg/new.py', True, '', 'def fresh(x):\n    return x'))
        self.assertEqual(len(found), 2)

    def test_edits_render_in_the_served_chat_templates_tool_call_markup(self):
        documents = build.render_edits(self.PATCH)
        self.assertEqual(len(documents), 2)
        self.assertTrue(documents[0].startswith('<tool_call>\n<function=str_replace_editor>\n<parameter=command>\nstr_replace\n</parameter>\n'))
        self.assertIn('<parameter=old_str>\nkeep = 1\nremoved = 2\ntail = 5\n</parameter>\n', documents[0])
        self.assertIn('<parameter=new_str>\nkeep = 1\nadded = 3\nmore = 4\ntail = 5\n</parameter>\n', documents[0])
        self.assertTrue(documents[0].rstrip().endswith('</function>\n</tool_call>'))
        self.assertIn('<parameter=command>\ncreate\n</parameter>', documents[1])
        self.assertIn('<parameter=file_text>\ndef fresh(x):\n    return x\n</parameter>', documents[1])

    def test_prose_keeps_long_text_as_thinking_and_drops_the_rest(self):
        long_text = 'The parser fails on empty input. ' * 20
        documents = build.render_prose(long_text, None)
        self.assertEqual(len(documents), 1)
        self.assertTrue(documents[0].startswith('<think>\n') and documents[0].endswith('\n</think>\n\n'))
        self.assertEqual(build.render_prose('short', ''), [])

    def test_shell_calls_name_the_failing_tests_the_touched_names_and_files(self):
        documents = build.render_shell(self.PATCH, ['tests/test_mod.py::test_old', 'tests/test_mod.py::test_new', 'tests/test_other.py::test_x'])
        joined = '\n'.join(documents)
        self.assertIn('python -m pytest tests/test_mod.py::test_old -x -q', joined)
        self.assertNotIn('test_other', joined)
        self.assertIn('grep -rn "fresh" pkg/', joined)
        self.assertIn("sed -n '1,120p' pkg/mod.py", joined)
        self.assertTrue(all(document.startswith('<tool_call>\n<function=bash>\n<parameter=command>\n') for document in documents))

    def test_a_patch_with_nothing_renders_nothing(self):
        self.assertEqual(build.render_edits(''), [])
        self.assertEqual(build.render_shell('', None), [])
        self.assertEqual(build.identifiers('+x = 1\n'), [])


class CommittedListTests(unittest.TestCase):
    def meta(self):
        return json.loads(SIDECAR.read_text(encoding='utf-8'))

    def test_the_committed_list_verifies(self):
        self.assertEqual(build.verify(str(SIDECAR)), [])

    def test_the_list_is_the_packed_ascending_ids_and_nothing_else(self):
        data = LIST.read_bytes()
        ids = build.unpack_ids(data)
        meta = self.meta()
        self.assertEqual((len(ids), len(data)), (40960, 163840))
        self.assertEqual(build.sha256_of(data), meta['ids_sha256'])
        self.assertEqual(build.pack_ids(ids), data)
        self.assertTrue(all(a < b for a, b in zip(ids, ids[1:])))
        self.assertLess(ids[-1], meta['real_tokens'])

    def test_the_sidecar_states_the_selection_and_the_coverage(self):
        meta = self.meta()
        self.assertEqual(meta['format'], build.FORMAT)
        self.assertEqual(meta['vocab_size'], 248320)
        self.assertEqual(meta['rows'] % build.ROW_MULTIPLE, 0)
        self.assertEqual(meta['selection']['required_count'] + meta['selection']['by_frequency'], meta['rows'])
        self.assertAlmostEqual(sum(meta['selection']['weights'].values()), 1.0, places=3)
        coverage = meta['coverage']
        self.assertEqual(coverage['bar'], build.COVERAGE_BAR)
        self.assertGreaterEqual(coverage['heldout_code_weighted'], coverage['bar'])
        self.assertGreaterEqual(coverage['heldout_code_worst_category'], coverage['bar'])
        for name, value in coverage['heldout_by_category'].items():
            self.assertGreaterEqual(value, coverage['bar'], name)
        # the curve is monotone in rows, and the committed size is on it
        curve = coverage['curve_by_rows']
        rows = sorted(int(key) for key in curve)
        values = [curve[str(key)]['code_weighted'] for key in rows]
        self.assertEqual(values, sorted(values))
        self.assertEqual(curve['40960']['code_weighted'], coverage['heldout_code_weighted'])

    def test_a_held_out_split_was_used(self):
        meta = self.meta()
        documents = meta['corpus']['documents']
        for category in ('edit', 'test', 'prose', 'shell'):
            held, train = documents[category + '/heldout']['documents'], documents[category + '/train']['documents']
            self.assertTrue(0.05 < held / float(held + train) < 0.2, category)
        self.assertGreater(meta['corpus']['swe_rebench']['instances']['heldout'], 1000)

    def test_the_sidecar_names_no_path_host_address_or_person(self):
        text = SIDECAR.read_text(encoding='utf-8')
        for pattern in (r'/home', r'/Users', r'[A-Za-z]:\\', r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', r'@', r'\.local\b', r'secret|password|api[_-]?key|bearer', r'\bssh\b'):
            self.assertIsNone(re.search(pattern, text, re.I), pattern)
        # only the public dataset is named as a source
        self.assertIn('nebius/SWE-rebench', text)

    def test_the_generator_names_no_host_or_home_path(self):
        text = (HERE / 'draft_vocab_build.py').read_text(encoding='utf-8')
        for pattern in (r'/home/', r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'):
            self.assertIsNone(re.search(pattern, text), pattern)


if __name__ == '__main__':
    unittest.main()
