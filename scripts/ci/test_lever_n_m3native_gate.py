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
    RETIRED_LABELS, compare_prefix, evaluate_gate, load_references, retired_binder_leaks,
    write_candidate)

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


class StreamCompletionTests(unittest.TestCase):
    def test_an_errored_stream_fails_the_gate_even_with_a_matching_prefix(self):
        kwargs = dict(COMPLETE_KWARGS)
        kwargs['checked'] = [dict(c, error='EngineCore encountered an issue') for c in COMPLETE_KWARGS['checked']]
        self.assertFalse(evaluate_gate(**kwargs))

    def test_a_stream_cut_short_of_its_reference_fails_unless_short_output_was_requested(self):
        kwargs = dict(COMPLETE_KWARGS)
        kwargs['checked'] = [dict(c, actual_len=12, reference_len=240) for c in COMPLETE_KWARGS['checked']]
        self.assertFalse(evaluate_gate(**kwargs))
        self.assertTrue(evaluate_gate(**dict(kwargs, full_output_required=False)))
        kwargs['checked'] = [dict(c, actual_len=960, reference_len=240) for c in COMPLETE_KWARGS['checked']]
        self.assertTrue(evaluate_gate(**kwargs))


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



class ReferenceSelectionTests(unittest.TestCase):
    """I0 of the 4 x 131k plan: references become full-length and prompt-length keyed."""

    def _write(self, directory, name, text):
        import json
        from pathlib import Path
        Path(directory, name).write_text(json.dumps({'streams': [{'text': text, 'text_sha256': None}]}),
                                         encoding='utf-8')

    def test_a_longer_generic_reference_that_extends_the_old_one_wins_without_conflict(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            self._write(directory, 'single-user-1001-35493236124.json', 'abc')
            self._write(directory, 'single-user-1001-35900000000.json', 'abcdef')
            reference = load_references(directory, 32768)[1001]
        self.assertEqual(reference['text'], 'abcdef')
        # Unsegmented references ARE 32,768-token ones, so at 32k they are an exact match.
        self.assertEqual(reference['context'], 'exact')
        self.assertEqual(reference['conflicts'], [])

    def test_two_disagreeing_references_are_a_conflict_and_fail_the_gate(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            self._write(directory, 'single-user-1001-35493236124.json', 'abX')
            self._write(directory, 'single-user-1001-35900000000.json', 'abcdef')
            reference = load_references(directory, 32768)[1001]
        self.assertEqual(len(reference['conflicts']), 1)
        checked = [dict(c, reference_conflicts=reference['conflicts']) if c['user'] == 1 else c
                   for c in COMPLETE_KWARGS['checked']]
        self.assertFalse(evaluate_gate(**dict(COMPLETE_KWARGS, checked=checked)))

    def test_a_prompt_length_reference_is_used_only_at_its_own_length(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            self._write(directory, 'single-user-35492921706.json', 'thirty-two')
            self._write(directory, 'single-user-1000-p131072-35900000001.json', 'one-three-one')
            at_131k = load_references(directory, 131072)[1000]
            at_32k = load_references(directory, 32768)[1000]
        self.assertEqual((at_131k['text'], at_131k['context']), ('one-three-one', 'exact'))
        self.assertEqual((at_32k['text'], at_32k['context']), ('thirty-two', 'exact'))

    def test_a_131k_run_without_its_own_reference_falls_back_and_says_so(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            self._write(directory, 'single-user-1002-35498370154.json', 'generic')
            reference = load_references(directory, 131072)[1002]
        self.assertEqual(reference['context'], 'generic-32768')

    def test_a_segmented_reference_never_stands_in_for_another_length(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            self._write(directory, 'single-user-1003-p131072-35900000002.json', 'long only')
            self.assertEqual(load_references(directory, 32768), {})

    def test_the_tracked_references_still_load_by_base(self):
        from pathlib import Path
        tracked = Path(__file__).parent / 'references' / 'packed-gate'
        loaded = load_references(tracked, 32768)
        self.assertEqual(sorted(loaded), [1000, 1001, 1002, 1003])
        for base, reference in loaded.items():
            with self.subTest(base=base):
                self.assertEqual(reference['conflicts'], [])

    def test_a_candidate_is_written_in_the_tracked_format_and_loads_back(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = write_candidate(directory, 1001, 131072, {'text': 'hello', 'text_sha256': 'x'})
            self.assertEqual(path.name, 'reference-candidate-1001-p131072.json')
            promoted = Path(directory, 'single-user-1001-p131072-35900000003.json')
            path.rename(promoted)
            self.assertEqual(load_references(directory, 131072)[1001]['text'], 'hello')
            self.assertIsNone(write_candidate(directory, 1002, 131072, {'text': 'x', 'error': 'boom'}))
            self.assertIsNone(write_candidate(directory, 1002, 131072, None))


class SequentialReferenceRunTests(unittest.TestCase):
    """--sequential-users N: N single-stream requests, one at a time, on one server."""

    def _main(self, argv):
        import io
        import tempfile
        from contextlib import redirect_stdout
        from unittest import mock
        import lever_n_m3native_gate as gate

        calls = []

        def stream(port, prompt, max_tokens, results, index, timeout):
            calls.append((index, prompt[0], len(prompt)))
            results[index] = {'text': 'stream-%d' % index, 'error': None}

        with tempfile.TemporaryDirectory() as directory:
            results = str(directory)
            with mock.patch.object(gate, 'start_server', return_value=(None, None, gate.Path(results, 'server.log'), ['x'])), \
                    mock.patch.object(gate, 'stop_server'), \
                    mock.patch.object(gate, 'stream_once', side_effect=stream), \
                    mock.patch.object(gate.threading, 'Thread', side_effect=AssertionError('no threads')), \
                    mock.patch.object(sys, 'argv', ['gate'] + argv + ['--results', results,
                                                                     '--references', results]), \
                    redirect_stdout(io.StringIO()) as out:
                code = gate.main()
            written = sorted(p.name for p in gate.Path(results).glob('reference-candidate-*.json'))
        return code, calls, written, out.getvalue()

    def test_each_base_runs_alone_in_order_and_becomes_a_candidate(self):
        code, calls, written, _ = self._main(['--users', '1', '--sequential-users', '4',
                                              '--prompt-tokens', '128'])
        self.assertEqual(calls, [(0, 1000, 128), (1, 1001, 128), (2, 1002, 128), (3, 1003, 128)])
        self.assertEqual(written, ['reference-candidate-%d-p128.json' % b for b in (1000, 1001, 1002, 1003)])

    def test_sequential_mode_refuses_more_than_one_concurrent_user(self):
        with self.assertRaises(SystemExit):
            self._main(['--users', '4', '--sequential-users', '4'])



class ReferenceRunVerdictTests(unittest.TestCase):
    """v97: four clean lone streams went red on packed-round terms they can never meet."""

    LONE = dict(COMPLETE_KWARGS, native_m3_marker_present=False, packed_phase=None, binder_rounds=[])

    def test_a_clean_reference_run_passes_without_packed_round_evidence(self):
        self.assertTrue(evaluate_gate(**dict(self.LONE, reference_run=True)))

    def test_the_same_evidence_still_fails_a_packed_run(self):
        self.assertFalse(evaluate_gate(**self.LONE))

    def test_a_reference_run_still_needs_every_reference_term(self):
        mismatch = [dict(c, identical_prefix=c['user'] != 2) for c in COMPLETE_KWARGS['checked']]
        short = [dict(c, actual_len=10, reference_len=20) for c in COMPLETE_KWARGS['checked']]
        conflict = [dict(c, reference_conflicts=['x']) for c in COMPLETE_KWARGS['checked']]
        for label, changes in (('mismatch', dict(checked=mismatch)), ('short', dict(checked=short)),
                               ('conflict', dict(checked=conflict)), ('uncovered', dict(checked=[])),
                               ('unready', dict(ready=False))):
            with self.subTest(case=label):
                self.assertFalse(evaluate_gate(**dict(self.LONE, reference_run=True, **changes)))



class FlagMarkerTests(unittest.TestCase):
    """A capacity flag that reached the server without its marker is not a pass."""

    def _report(self, environ, log, users=4):
        from lever_n_m3native_gate import flag_marker_report
        return flag_marker_report(environ, users, log)

    def test_no_flags_require_nothing(self):
        self.assertEqual(self._report({}, '')['missing'], [])

    def test_a_skip_flag_without_its_marker_is_missing(self):
        report = self._report({'QWEN_FAST_SKIP_BLOCK_STREAM': '1'}, '[PINDIAG] block stream NOT skipped: ...')
        self.assertEqual(report['missing'], ['QWEN_FAST_SKIP_BLOCK_STREAM: [PINDIAG] block stream skipped for the 64-row block'])
        self.assertFalse(evaluate_gate(**dict(COMPLETE_KWARGS, missing_markers=report['missing'])))

    def test_single_gateup_needs_all_four_markers_and_supersedes_the_skip_marker(self):
        from lever_n_m3native_gate import SINGLE_GATEUP_MARKERS
        environ = {'QWEN_FAST_SINGLE_GATEUP': '1', 'QWEN_FAST_SKIP_BLOCK_STREAM': '1'}
        log = chr(10).join(m + ' 64 layers' for m in SINGLE_GATEUP_MARKERS)
        self.assertEqual(self._report(environ, log)['missing'], [])
        self.assertEqual(len(self._report(environ, chr(10).join(log.split(chr(10))[:3]))['missing']), 1)

    def test_a_model_that_still_holds_the_copy_does_not_count(self):
        environ = {'QWEN_FAST_SINGLE_GATEUP': '1'}
        log = '[PINDIAG] block stream skipped for the single gate/up copy: w_gate_up present on 64 of 64 layers'
        self.assertIn('QWEN_FAST_SINGLE_GATEUP: [PINDIAG] block stream skipped for the single gate/up copy: '
                      'w_gate_up present on 0 of', self._report(environ, log)['missing'])

    def test_bf8_and_the_ledger_markers(self):
        environ = {'QWEN_FAST_DRAFT_BF8': '1', 'QWEN_FAST_MEMORY_LEDGER': '1'}
        log = ('[PINDIAG] draft weights lent to x (borrowers=1 tensors=40) projections dtype=bf8 x36' + chr(10)
               + '[MEMLEDGER] phase=P7 point=after_attach check=residual status=passed limit=1.500GB')
        report = self._report(environ, log)
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['ledger_residual'], 'passed')
        self.assertEqual(len(self._report(environ, '')['missing']), 3)

    def test_gdn_user_batch_needs_one_forward_batching_every_layer_at_four_users(self):
        environ = {'QWEN_FAST_GDN_USER_BATCH': '1'}
        partial = '[PINDIAG] gdn user_batched calls this captured forward: 0 of 48 GDN layers'
        full = '[PINDIAG] gdn user_batched calls this captured forward: 48 of 48 GDN layers'
        self.assertEqual(len(self._report(environ, partial)['missing']), 1)
        self.assertEqual(self._report(environ, partial + chr(10) + full)['missing'], [])
        self.assertEqual(self._report(environ, partial, users=1)['missing'], [])


if __name__ == '__main__':
    unittest.main()
