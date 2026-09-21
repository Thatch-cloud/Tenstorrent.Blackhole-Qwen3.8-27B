"""lever_n_m3native_gate.retired_binder_leaks: only retired binders count.

Also lever_n_m3native_gate.compare_prefix: a profiling run capped at a small
--max-tokens produces a stream shorter than its single-user reference, which must
compare as a partial prefix match rather than a divergence."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lever_n_m3native_gate import select_diagnostic  # noqa: E402
from lever_n_m3native_gate import (  # noqa: E402
    RETIRED_LABELS, compare_prefix, evaluate_gate, retired_binder_leaks)

V4_ROUND = {'decode norm': 129, 'full-attention forward': 16, 'MLP forward': 0, 'GDN output projection': 0}

# A complete, passing four-user round: every ingredient evaluate_gate checks for.
COMPLETE_KWARGS = dict(
    ready=True, users=4,
    checked=[{'user': index, 'identical_prefix': True} for index in range(4)],
    allow_missing_references=False, native_m3_marker_present=True,
    packed_phase={'rounds': 3, 'trace_ms_min': 1.0, 'trace_ms_mean': 1.0, 'trace_ms_max': 1.0},
    binder_rounds=[V4_ROUND], retired_binder_calls_nonzero=[])


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


class EvaluateGateTests(unittest.TestCase):
    """evaluate_gate: the --allow-missing-references flag only widens what counts as
    complete reference coverage; it never relaxes a mismatch, readiness, the native_m3
    marker, packed_phase, or the retired-binder checks."""

    def test_a_complete_passing_round_passes_by_default(self):
        self.assertTrue(evaluate_gate(**COMPLETE_KWARGS))

    def test_a_missing_reference_fails_by_default_unchanged_behaviour(self):
        # Exactly today's pre-flag behaviour: fewer checked users than the run served.
        kwargs = dict(COMPLETE_KWARGS, checked=COMPLETE_KWARGS['checked'][:3])
        self.assertFalse(evaluate_gate(**kwargs))

    def test_no_references_at_all_fails_by_default(self):
        # The 131k arm with no reference file yet: checked is empty.
        kwargs = dict(COMPLETE_KWARGS, checked=[])
        self.assertFalse(evaluate_gate(**kwargs))

    def test_no_references_at_all_passes_with_the_flag_when_everything_else_holds(self):
        kwargs = dict(COMPLETE_KWARGS, checked=[], allow_missing_references=True)
        self.assertTrue(evaluate_gate(**kwargs))

    def test_partial_references_pass_with_the_flag_if_all_present_ones_match(self):
        kwargs = dict(COMPLETE_KWARGS, checked=COMPLETE_KWARGS['checked'][:2],
                      allow_missing_references=True)
        self.assertTrue(evaluate_gate(**kwargs))

    def test_partial_references_still_fail_without_the_flag(self):
        kwargs = dict(COMPLETE_KWARGS, checked=COMPLETE_KWARGS['checked'][:2],
                      allow_missing_references=False)
        self.assertFalse(evaluate_gate(**kwargs))

    def test_the_flag_never_forgives_an_actual_mismatch(self):
        checked = [dict(COMPLETE_KWARGS['checked'][0], identical_prefix=False)]
        kwargs = dict(COMPLETE_KWARGS, checked=checked, allow_missing_references=True)
        self.assertFalse(evaluate_gate(**kwargs))

    def test_the_flag_never_forgives_an_unready_server(self):
        kwargs = dict(COMPLETE_KWARGS, checked=[], allow_missing_references=True, ready=False)
        self.assertFalse(evaluate_gate(**kwargs))

    def test_the_flag_never_forgives_a_missing_native_m3_marker(self):
        # This is the actual 131k-attach-arm outcome today: the T16 gate refuses the
        # request inside engine construction before any packed round runs (see
        # docs/lever-n-131k-attach-arm.md), so no decode round ever prints the marker.
        kwargs = dict(COMPLETE_KWARGS, checked=[], allow_missing_references=True,
                      native_m3_marker_present=False, packed_phase=None, binder_rounds=[])
        self.assertFalse(evaluate_gate(**kwargs))

    def test_the_flag_never_forgives_a_retired_binder_leak(self):
        kwargs = dict(COMPLETE_KWARGS, checked=[], allow_missing_references=True,
                      retired_binder_calls_nonzero=[{'MLP forward': 1}])
        self.assertFalse(evaluate_gate(**kwargs))


class DiagnosticFilterTests(unittest.TestCase):
    def test_every_packed_audit_family_and_the_phase_lines_pass_the_filter(self):
        lines = ['x [PACKED] request=a segment=0', 'x [PACKED-PHASE] round=1', 'x [PACKED-COMMIT] round=1',
                 'x [PACKED-COMMIT-HOST] round=1', 'x [PACKED-PROPOSE] round=1', 'x [PHASE] step a begin',
                 'x [PINDIAG] dram after', 'plain server chatter', 'x ERROR boom', 'Traceback (most recent call last):',
                 'TT_FATAL @ llrt.cpp:594: Timed out', 'Segmentation fault (core dumped)', 'terminate called after throwing',
                 'Engine core proc EngineCore_0 died unexpectedly']
        kept = select_diagnostic(lines)
        self.assertEqual(kept, [line for line in lines if line != 'plain server chatter'])

    def test_the_cap_keeps_both_ends_and_says_how_much_it_dropped(self):
        lines = ['[PHASE] %d' % index for index in range(10)]
        self.assertEqual(select_diagnostic(lines, cap=4),
                         ['[PHASE] 0', '[PHASE] 1', '... 6 diagnostic lines omitted', '[PHASE] 8', '[PHASE] 9'])
        self.assertEqual(select_diagnostic(lines, cap=10), lines)


if __name__ == '__main__':
    unittest.main()
