"""CPU tests of the ESTIMATE of the K-split (ksplit_estimate.py) and of its agreement with docs/tp4-prefill3-ksplit.md: the tables in the document are the script's output, the figures its prose
quotes are the script's, and the script's arithmetic has the properties the document relies on (only the 512- and 1,024-row buckets, low <= central <= high, the planner's slice counts).

Run from this directory:  python3 -B -m unittest test_ksplit_estimate
"""

from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ksplit_estimate as estimate  # noqa: E402
import ksplit_planner as planner  # noqa: E402

DOC = HERE.parents[2] / 'docs' / 'tp4-prefill3-ksplit.md'


class ArithmeticTests(unittest.TestCase):
    def test_the_documents_own_assertions_hold(self):
        self.assertEqual(estimate.check(), 0)

    def test_only_the_512_and_1024_row_buckets_engage_and_their_slice_counts_are_the_planners(self):
        self.assertEqual(estimate.ENGAGED_BUCKETS, planner.QUALIFIED_ROWS)
        for bucket in (128, 256, 2048):
            for level in ('low', 'central', 'high'):
                self.assertEqual(estimate.saving_ms(bucket, 120832, level), 0.0)
        self.assertEqual([planner.split_k_slices(bucket) for bucket in estimate.ENGAGED_BUCKETS], [4, 2])

    def test_the_levels_are_ordered_at_every_context_and_the_saving_grows_with_the_context(self):
        for bucket in estimate.ENGAGED_BUCKETS:
            previous = 0.0
            for context in estimate.CONTEXTS:
                low, central, high = (estimate.saving_ms(bucket, context, level) for level in ('low', 'central', 'high'))
                self.assertLessEqual(low, central + 1e-9)
                self.assertLessEqual(central, high + 1e-9)
                self.assertGreaterEqual(low, 0.0)
                self.assertGreater(central, previous, (bucket, context))
                previous = central

    def test_a_saving_never_exceeds_the_attention_it_acts_on(self):
        for bucket in estimate.ENGAGED_BUCKETS:
            for context in estimate.CONTEXTS:
                self.assertLess(estimate.saving_ms(bucket, context, 'high'), estimate.tail_attention_ms(bucket, context))

    def test_the_clock_penalties(self):
        self.assertAlmostEqual(estimate.clock_penalty(512, 'central'), 0.63 / 0.505)
        self.assertAlmostEqual(estimate.clock_penalty(512, 'high'), 1.0)
        self.assertAlmostEqual(estimate.clock_penalty(512, 'low'), 2.108 / 1.274, places=3)
        self.assertAlmostEqual(estimate.clock_penalty(1024, 'low'), 2.108 / 1.251, places=3)

    def test_the_population_is_the_documented_one(self):
        self.assertEqual(sum(estimate.POPULATION.values()), 1022)
        self.assertEqual(estimate.prompts_taken(), 369)
        passes = estimate.population_passes()
        self.assertEqual(len(passes), 1022)
        self.assertAlmostEqual(estimate.population_saving_s('central'), estimate.population_saving_s('central', passes))

    def test_a_per_pass_list_replaces_the_mean(self):
        deep = [(1024, 160000)] * 10
        shallow = [(1024, 16000)] * 10
        self.assertGreater(estimate.population_saving_s('central', deep), estimate.population_saving_s('central', shallow))
        self.assertEqual(estimate.population_saving_s('central', [(2048, 160000)] * 5), 0.0)

    def test_the_older_row_is_the_cost_models(self):
        import prefill_cost_model as model
        for tokens, bucket in ((900, 1024), (400, 512)):
            self.assertEqual(model.bucket_for(tokens), bucket)
            self.assertAlmostEqual(estimate.cost_model_saving_ms(bucket, 120832), model.split_k_saving_ms(120832, tokens), places=6)
            self.assertGreater(estimate.cost_model_saving_ms(bucket, 120832), estimate.saving_ms(bucket, 120832, 'high'), 'the older row is the larger one')
        self.assertEqual(model.split_k_saving_ms(120832, 3112), 0.0, 'a 2,048-row tail has one slice')


class DocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = DOC.read_text(encoding='utf-8')

    def test_the_tables_in_the_document_are_the_scripts_output(self):
        self.assertEqual(estimate.embedded(self.text), estimate.format_tables())

    def test_the_prose_quotes_the_scripts_figures(self):
        central = estimate.population_saving_s('central')
        low, high = estimate.population_saving_s('low'), estimate.population_saving_s('high')
        route = estimate.route_s(False)
        self.assertIn('%.1f seconds = %.2f percent of today\'s route time' % (central, 100 * central / route), self.text)
        self.assertIn('%.2f percent' % (100 * central / route), self.text)
        self.assertIn('%.2f-%.2f percent over the low-high band' % (100 * low / route, 100 * high / route), self.text)
        self.assertIn('%.1f percent after exact resume' % (100 * central / estimate.route_s(True)), self.text)
        self.assertIn('%.1f to %.1f' % (low, high), self.text)
        for bucket, context, key in ((1024, 120832, 'central'), (512, 120832, 'central')):
            self.assertIn('%.0f ms' % estimate.saving_ms(bucket, context, key), self.text)

    def test_the_documents_markers_are_in_order_and_once(self):
        self.assertEqual(self.text.count(estimate.BEGIN), 1)
        self.assertEqual(self.text.count(estimate.END), 1)
        self.assertLess(self.text.index(estimate.BEGIN), self.text.index(estimate.END))

    def test_with_tables_replaces_only_the_block(self):
        changed = self.text.replace(estimate.format_tables(), 'x')
        self.assertNotEqual(changed, self.text)
        self.assertEqual(estimate.with_tables(changed), self.text)
        with self.assertRaises(ValueError):
            estimate.with_tables('no markers here')

    def test_the_document_names_what_it_must(self):
        for needle in ('ESTIMATE', 'not run', 'numerics waiver', 'Not waived', 'Rollback', 'Abort criteria', 'G1', 'G7', 'not compiled', 'relay'):
            self.assertIn(needle.lower(), self.text.lower(), needle)
        for forbidden in ('merge core is', ):
            self.assertNotIn(forbidden, self.text)

    def test_ascii_lf_and_nothing_private(self):
        raw = DOC.read_bytes()
        self.assertNotIn(b'\r', raw)
        self.assertEqual({char for char in self.text if ord(char) > 127}, set())
        private = re.compile(r'\d{1,3}(\.\d{1,3}){3}|th[a]tch|blackhole-(?!0{8})[0-9A-F]{8}|/hom[e]/|\bss[h]\b|gh[p]_|sha256:[0-9a-f]{16}|\.loca[l]\b', re.I)
        for number, line in enumerate(self.text.splitlines(), 1):
            self.assertIsNone(private.search(line), '%d: %s' % (number, line[:140]))


if __name__ == '__main__':
    unittest.main()
