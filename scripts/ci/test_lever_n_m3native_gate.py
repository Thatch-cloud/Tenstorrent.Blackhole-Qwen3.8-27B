"""lever_n_m3native_gate.retired_binder_leaks: only retired binders count.

Also lever_n_m3native_gate.compare_prefix: a profiling run capped at a small
--max-tokens produces a stream shorter than its single-user reference, which must
compare as a partial prefix match rather than a divergence."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lever_n_m3native_gate import RETIRED_LABELS, compare_prefix, retired_binder_leaks  # noqa: E402

V4_ROUND = {'decode norm': 129, 'full-attention forward': 16, 'MLP forward': 0, 'GDN output projection': 0}


class RetiredBinderLeakTests(unittest.TestCase):
    def test_the_v4_payload_is_not_a_leak(self):
        self.assertEqual(retired_binder_leaks([V4_ROUND, V4_ROUND]), [])

    def test_a_retired_mlp_call_is_a_leak(self):
        leaked = dict(V4_ROUND, **{'MLP forward': 3})
        self.assertEqual(retired_binder_leaks([V4_ROUND, leaked]), [{'MLP forward': 3}])

    def test_native_attn_guards_are_retired_labels(self):
        self.assertIn('sliced attn_decode_prep', RETIRED_LABELS)
        self.assertIn('two-tile head concat', RETIRED_LABELS)
        leaked = dict(V4_ROUND, **{'two-tile head concat': 16})
        self.assertEqual(retired_binder_leaks([leaked]), [{'two-tile head concat': 16}])

    def test_unknown_labels_are_ignored(self):
        self.assertEqual(retired_binder_leaks([{'something else': 5}]), [])


class ComparePrefixTests(unittest.TestCase):
    def test_exact_full_length_match_is_unchanged(self):
        self.assertEqual(compare_prefix('hello world', 'hello world'), (True, False))

    def test_full_length_mismatch_is_unchanged(self):
        self.assertEqual(compare_prefix('hello WORLD', 'hello world'), (False, False))

    def test_longer_actual_is_still_checked_as_a_prefix_match(self):
        self.assertEqual(compare_prefix('hello world and more tokens', 'hello world'), (True, False))

    def test_longer_actual_that_diverges_is_not_identical(self):
        self.assertEqual(compare_prefix('hello WORLD and more tokens', 'hello world'), (False, False))

    def test_shorter_actual_that_is_a_true_prefix_is_partial_and_identical(self):
        self.assertEqual(compare_prefix('hello wor', 'hello world'), (True, True))

    def test_shorter_actual_that_diverges_is_partial_and_not_identical(self):
        self.assertEqual(compare_prefix('hello xyz', 'hello world'), (False, True))

    def test_empty_actual_is_a_partial_prefix_of_any_reference(self):
        self.assertEqual(compare_prefix('', 'hello world'), (True, True))


if __name__ == '__main__':
    unittest.main()
