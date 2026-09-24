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
        from lever_n_m3native_gate import C1C_MARKER, SINGLE_GATEUP_MARKERS
        environ = {'QWEN_FAST_SINGLE_GATEUP': '1', 'QWEN_FAST_SKIP_BLOCK_STREAM': '1'}
        log = chr(10).join([m + ' 64 layers' for m in SINGLE_GATEUP_MARKERS] + [C1C_MARKER + ', product-only concat'])
        self.assertEqual(self._report(environ, log)['missing'], [])
        self.assertEqual(len(self._report(environ, chr(10).join(log.split(chr(10))[:3] + [C1C_MARKER]))['missing']), 1)

    def test_single_gateup_needs_the_marker_of_the_prefill_mlp_branch_that_ran(self):
        """C1c is the default: a rerun of a SINGLE_GATEUP tag that fell through to C1 does not
        pass; under QWEN_FAST_C1_LEGACY=1 C1's own 2D-branch marker is the proof instead."""
        from lever_n_m3native_gate import C1C_MARKER, C1_LEGACY_MARKER, SINGLE_GATEUP_MARKERS
        base = [m + ' 64 layers' for m in SINGLE_GATEUP_MARKERS]
        c1c_log = chr(10).join(base + [C1C_MARKER + ', product-only concat: rows=2048 slices of 1024 rows'])
        c1_log = chr(10).join(base + [C1_LEGACY_MARKER + ': rows=2048 fused=False x=None slices of 1024 rows'])
        single = {'QWEN_FAST_SINGLE_GATEUP': '1'}
        legacy = {'QWEN_FAST_SINGLE_GATEUP': '1', 'QWEN_FAST_C1_LEGACY': '1'}
        self.assertEqual(self._report(single, c1c_log)['missing'], [])
        self.assertEqual(self._report(single, c1_log)['missing'], ['QWEN_FAST_SINGLE_GATEUP: ' + C1C_MARKER])
        self.assertEqual(self._report(legacy, c1_log)['missing'], [])
        self.assertEqual(self._report(legacy, c1c_log)['missing'], ['QWEN_FAST_SINGLE_GATEUP: ' + C1_LEGACY_MARKER])
        self.assertFalse(evaluate_gate(**dict(COMPLETE_KWARGS, missing_markers=self._report(single, c1_log)['missing'])))
        # Neither flag alone (without SINGLE_GATEUP) promises anything.
        for environ in ({'QWEN_FAST_C1_LEGACY': '1'}, {'QWEN_FAST_C1_AGMM': '1'}):
            self.assertEqual(self._report(environ, '')['missing'], [])

    def test_the_prefill_profile_flush_flag_needs_its_first_flush_marker(self):
        from lever_n_m3native_gate import PREFILL_FLUSH_MARKER
        environ = {'QWEN_PREFILL_PROFILE_FLUSH': '1'}
        self.assertEqual(self._report(environ, '')['missing'], ['QWEN_PREFILL_PROFILE_FLUSH: ' + PREFILL_FLUSH_MARKER])
        log = PREFILL_FLUSH_MARKER + ' at prompt 1 chunk 0 layer 15 (every 16 layers)'
        self.assertEqual(self._report(environ, log, users=1)['missing'], [])

    def test_a_model_that_still_holds_the_copy_does_not_count(self):
        environ = {'QWEN_FAST_SINGLE_GATEUP': '1'}
        log = '[PINDIAG] block stream skipped for the single gate/up copy: w_gate_up present on 64 of 64 layers'
        self.assertIn('QWEN_FAST_SINGLE_GATEUP: [PINDIAG] block stream skipped for the single gate/up copy: '
                      'w_gate_up present on 0 of', self._report(environ, log)['missing'])

    def test_round_b1_needs_its_engaged_marker(self):
        """QWEN_FAST_ROUND_B1 (M3NATIVE_ROUND_B1): the image module logs the marker once, at
        the first build-1 path that runs; the gate's copy of it must be that same text."""
        from dflash_packed_proposal import ROUND_B1_MARKER as logged
        from lever_n_m3native_gate import ROUND_B1_MARKER, required_flag_markers
        self.assertEqual(ROUND_B1_MARKER, logged)
        environ = {'QWEN_FAST_ROUND_B1': '1'}
        self.assertEqual(required_flag_markers(environ, 4), {'QWEN_FAST_ROUND_B1': [ROUND_B1_MARKER]})
        self.assertEqual(required_flag_markers({'QWEN_FAST_ROUND_B1': '0'}, 4), {})
        self.assertEqual(self._report(environ, '[PINDIAG] draft weights lent')['missing'],
                         ['QWEN_FAST_ROUND_B1: ' + ROUND_B1_MARKER])
        log = '2026-09-23 | INFO | ' + ROUND_B1_MARKER + ' site=publication cuts=C1,C2,C7,C8,M0a'
        self.assertEqual(self._report(environ, log)['missing'], [])
        self.assertEqual(self._report(environ, log, users=1)['missing'], [])

    def test_round_b1_audit_needs_its_first_line_and_fails_on_a_mismatch(self):
        """QWEN_FAST_ROUND_B1_AUDIT (M3NATIVE_ROUND_B1_AUDIT): at four users the first audited
        round's line is required; any mismatch line fails, whatever the user count (the pair
        fallback can swallow the raise); and the last line's counts must all be non-zero. The
        gate's copies of the texts are the image module's."""
        import dflash_packed_proposal as image
        from lever_n_m3native_gate import (ROUND_B1_AUDIT_COUNTS, ROUND_B1_AUDIT_MARKER, ROUND_B1_AUDIT_MISMATCH,
                                           ROUND_B1_MARKER, required_flag_markers)
        self.assertEqual((ROUND_B1_AUDIT_MARKER, ROUND_B1_AUDIT_MISMATCH, ROUND_B1_AUDIT_COUNTS),
                         (image.ROUND_B1_AUDIT_MARKER, image.ROUND_B1_AUDIT_MISMATCH, image.ROUND_B1_AUDIT_COUNTS))
        environ = {'QWEN_FAST_ROUND_B1': '1', 'QWEN_FAST_ROUND_B1_AUDIT': '1'}
        self.assertEqual(required_flag_markers(environ, 4),
                         {'QWEN_FAST_ROUND_B1': [ROUND_B1_MARKER],
                          'QWEN_FAST_ROUND_B1_AUDIT': [ROUND_B1_AUDIT_MARKER + ' 1 exact=True']})
        self.assertEqual(required_flag_markers(environ, 1), {'QWEN_FAST_ROUND_B1': [ROUND_B1_MARKER]})
        self.assertEqual(required_flag_markers({'QWEN_FAST_ROUND_B1_AUDIT': '1'}, 4), {})
        engaged = ROUND_B1_MARKER + ' site=publication cuts=C1,C2,C7,C8,M0a'
        good = chr(10).join([engaged,
                             '[PINDIAG] round b1 audit 1 exact=True select=4 rope=2 retain=610 borrowed=40 release=8',
                             '[PINDIAG] round b1 audit 2 exact=True select=8 rope=4 retain=1220 borrowed=80 release=16'])
        self.assertEqual(self._report(environ, good)['missing'], [])
        self.assertEqual(self._report(environ, engaged)['missing'],
                         ['QWEN_FAST_ROUND_B1_AUDIT: ' + ROUND_B1_AUDIT_MARKER + ' 1 exact=True'])
        self.assertEqual(self._report(environ, engaged, users=1)['missing'], [])
        mismatch = good + chr(10) + ROUND_B1_AUDIT_MISMATCH + ' cut=C8 live key RoPE differs'
        for users in (4, 1):
            with self.subTest(users=users):
                missing = self._report(environ, mismatch, users=users)['missing']
                self.assertEqual(len(missing), 1)
                self.assertTrue(missing[0].startswith('QWEN_FAST_ROUND_B1_AUDIT: no mismatch (' + ROUND_B1_AUDIT_MISMATCH))
        idle = good + chr(10) + '[PINDIAG] round b1 audit 3 exact=True select=12 rope=0 retain=1830 borrowed=120 release=24'
        self.assertEqual(self._report(environ, idle)['missing'], ['QWEN_FAST_ROUND_B1_AUDIT: every cut compared (none for rope)'])
        self.assertEqual(self._report({'QWEN_FAST_ROUND_B1': '1'}, engaged + chr(10) + mismatch)['missing'], [],
                         'without the audit flag its lines are not judged')

    def test_the_image_audit_line_is_the_one_the_gate_reads(self):
        from contextlib import ExitStack
        from unittest.mock import patch

        import dflash_packed_proposal as image
        from lever_n_m3native_gate import ROUND_B1_AUDIT_LINE

        lines = []
        counts = dict(rounds=0, select=4, rope=2, retain=9, borrowed=3, release=1)
        with ExitStack() as stack:
            stack.enter_context(patch.object(image, '_ROUND_B1_AUDIT', counts))
            stack.enter_context(patch.dict('sys.modules', {'loguru': None}))
            stack.enter_context(patch('builtins.print', side_effect=lambda message, **kwargs: lines.append(message)))
            image.note_round_b1_audit()
        self.assertEqual(len(lines), 1)
        match = ROUND_B1_AUDIT_LINE.search(lines[0])
        self.assertIsNotNone(match, lines[0])
        self.assertEqual(match.groups(), ('1', '4', '2', '9', '3', '1'))

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

    def test_the_tail_sdpa_mode_needs_its_in_path_and_factory_markers(self):
        from lever_n_m3native_gate import SDPA_MODES_MARKERS, required_flag_markers
        log = chr(10).join((
            '2026-09-23 | INFO | [PINDIAG] sdpa qwen-modes binary /opt/tt-metal/build_Release/lib/_ttnncpp.so carries the [QWEN-SDPA] branch',
            "2026-09-23 | INFO | [PINDIAG] sdpa qwen-modes modes=tail rows=16 capacity=33024 bundles=[3, 1] flags=['0x1', '0x1'] mask=wide",
            '                 Op | INFO     | [QWEN-SDPA] flags=0x1 B=3 PNHt=2 St=1032 mask_width_t=1032 kv_share=false scratch_slots=4 cb_bytes=790592'))
        for value in ('tail', ' tail ', 'tail,tail'):
            with self.subTest(value=value):
                environ = {'QWEN_FAST_SDPA_MODES': value}
                self.assertEqual(required_flag_markers(environ, 4), {'QWEN_FAST_SDPA_MODES': list(SDPA_MODES_MARKERS)})
                self.assertEqual(self._report(environ, log)['missing'], [])
        environ = {'QWEN_FAST_SDPA_MODES': 'tail'}
        for index, marker in enumerate(SDPA_MODES_MARKERS):
            with self.subTest(dropped=marker):
                partial = chr(10).join(line for number, line in enumerate(log.split(chr(10))) if number != index)
                report = self._report(environ, partial)
                self.assertEqual(report['missing'], ['QWEN_FAST_SDPA_MODES: ' + marker])
                self.assertFalse(evaluate_gate(**dict(COMPLETE_KWARGS, missing_markers=report['missing'])))
        # A legacy-only program (flags=0x0 never happens; a legacy call logs nothing) is not the tail branch.
        self.assertIn('QWEN_FAST_SDPA_MODES: [QWEN-SDPA] flags=0x1 ',
                      self._report(environ, log.replace('flags=0x1 B=3', 'flags=0x3 B=3'))['missing'])
        for environ in ({}, {'QWEN_FAST_SDPA_MODES': ''}, {'QWEN_FAST_SDPA_MODES': 'narrow'}):
            with self.subTest(environ=environ):
                self.assertEqual(required_flag_markers(environ, 4), {})
                self.assertEqual(self._report(environ, '')['missing'], [])

    def test_the_share_sdpa_mode_needs_the_factory_line_with_the_share_flag(self):
        """tail,share (stage 3): the modes line names both, and the factory must have built a program
        with flag 0x3 - a bundle of twins, kv_share=true - not merely a tail program (0x1)."""
        from lever_n_m3native_gate import SDPA_MODES_MARKERS, required_flag_markers
        log = chr(10).join((
            '2026-09-24 | INFO | [PINDIAG] sdpa qwen-modes binary /opt/tt-metal/build_Release/lib/_ttnncpp.so '
            'carries the [QWEN-SDPA] branch with KV share',
            "2026-09-24 | INFO | [PINDIAG] sdpa qwen-modes modes=share,tail rows=16 capacity=33024 bundles=[2] "
            "flags=['0x3'] mask=wide",
            '                 Op | INFO     | [QWEN-SDPA] flags=0x3 B=2 PNHt=3 St=1032 mask_width_t=1032 kv_share=true '
            'scratch_slots=4 cb_bytes=1042496'))
        expected = [SDPA_MODES_MARKERS[0], '[PINDIAG] sdpa qwen-modes modes=share,tail ', '[QWEN-SDPA] flags=0x3 ']
        for value in ('tail,share', 'share,tail', ' share , tail '):
            with self.subTest(value=value):
                environ = {'QWEN_FAST_SDPA_MODES': value}
                self.assertEqual(required_flag_markers(environ, 4), {'QWEN_FAST_SDPA_MODES': expected})
                self.assertEqual(self._report(environ, log)['missing'], [])
        environ = {'QWEN_FAST_SDPA_MODES': 'tail,share'}
        for index, marker in enumerate(expected):
            with self.subTest(dropped=marker):
                partial = chr(10).join(line for number, line in enumerate(log.split(chr(10))) if number != index)
                report = self._report(environ, partial)
                self.assertEqual(report['missing'], ['QWEN_FAST_SDPA_MODES: ' + marker])
                self.assertFalse(evaluate_gate(**dict(COMPLETE_KWARGS, missing_markers=report['missing'])))
        # Only tail programs were built (a stage-1 .so, or every bundle single-entry): not share.
        tail_only = log.replace('flags=0x3 B=2', 'flags=0x1 B=2').replace('kv_share=true', 'kv_share=false')
        self.assertEqual(self._report(environ, tail_only)['missing'], ['QWEN_FAST_SDPA_MODES: [QWEN-SDPA] flags=0x3 '])
        # The stage-1 modes line (tail only) does not satisfy a share run either.
        self.assertIn('QWEN_FAST_SDPA_MODES: [PINDIAG] sdpa qwen-modes modes=share,tail ',
                      self._report(environ, log.replace('modes=share,tail', 'modes=tail'))['missing'])
        # share alone: flag 0x2; narrow is refused by the reader, so its modes line never appears.
        self.assertEqual(required_flag_markers({'QWEN_FAST_SDPA_MODES': 'share'}, 4)['QWEN_FAST_SDPA_MODES'][1:],
                         ['[PINDIAG] sdpa qwen-modes modes=share ', '[QWEN-SDPA] flags=0x2 '])
        self.assertEqual(required_flag_markers({'QWEN_FAST_SDPA_MODES': 'narrow,tail'}, 4)['QWEN_FAST_SDPA_MODES'][1:],
                         ['[PINDIAG] sdpa qwen-modes modes=narrow,tail ', '[QWEN-SDPA] flags=0x1 '])



class VerifyT2GateTests(unittest.TestCase):
    """QWEN_FAST_VERIFY_T2 (M3NATIVE_VERIFY_T2): the engaged marker is promised; at four users every
    captured verify trace engaged each cut not skipped, in every layer, with the knob's launch width
    and the warm forward's single chain; any fell-back, kv-shared or audit-mismatch line fails the
    arm at any user count; the audit needs its first line; one problem never hides another. The
    gate's copies of the texts are the image module's (verify_trace_t2)."""

    BATCH = {'QWEN_FAST_VERIFY_T2': '1', 'QWEN_FAST_GDN_USER_BATCH': '1'}

    BATCHED = '[PINDIAG] gdn user_batched calls this captured forward: 48 of 48 GDN layers'

    def _report(self, environ, log, users=4):
        from lever_n_m3native_gate import flag_marker_report
        return flag_marker_report(environ, users, chr(10).join([self.BATCHED, log]))

    @staticmethod
    def packed(**changes):
        import verify_trace_t2
        fields = dict(windows=48, windows_fallback=0, kv_chains=32, kv_fallback=0, kv_rows=64,
                      warm_chain='single', audit=0)
        fields.update(changes)
        return '2026-09-24 | INFO | ' + verify_trace_t2.engaged_line('packed_verify', **fields)

    def test_the_gate_texts_are_the_image_modules(self):
        import lever_n_m3native_gate as gate
        import verify_trace_t2 as image
        self.assertEqual((gate.VERIFY_T2_FLAG, gate.VERIFY_T2_AUDIT_FLAG, gate.VERIFY_T2_SKIP_FLAG,
                          gate.VERIFY_T2_KV_ROWS_FLAG, gate.VERIFY_T2_MARKER, gate.VERIFY_T2_AUDIT_MARKER,
                          gate.VERIFY_T2_AUDIT_MISMATCH, gate.VERIFY_T2_FALLBACK, gate.VERIFY_T2_KV_SHARED,
                          gate.VERIFY_T2_CUTS),
                         (image.FLAG, image.AUDIT_FLAG, image.SKIP_FLAG, image.KV_ROWS_FLAG, image.MARKER,
                          image.AUDIT_MARKER, image.AUDIT_MISMATCH, image.FALLBACK, image.KV_SHARED, image.CUTS))

    def test_the_flag_promises_the_marker_and_the_audit_its_first_line(self):
        from lever_n_m3native_gate import VERIFY_T2_AUDIT_MARKER, VERIFY_T2_MARKER, required_flag_markers
        self.assertEqual(required_flag_markers({'QWEN_FAST_VERIFY_T2': '1'}, 4),
                         {'QWEN_FAST_VERIFY_T2': [VERIFY_T2_MARKER]})
        audit = {'QWEN_FAST_VERIFY_T2': '1', 'QWEN_FAST_VERIFY_T2_AUDIT': '1'}
        self.assertEqual(required_flag_markers(audit, 4),
                         {'QWEN_FAST_VERIFY_T2': [VERIFY_T2_MARKER],
                          'QWEN_FAST_VERIFY_T2_AUDIT': [VERIFY_T2_AUDIT_MARKER + ' 1 exact=True']})
        self.assertEqual(required_flag_markers(audit, 1), {'QWEN_FAST_VERIFY_T2': [VERIFY_T2_MARKER]})
        self.assertEqual(required_flag_markers(dict(audit, QWEN_FAST_VERIFY_T2_SKIP='windows'), 4),
                         {'QWEN_FAST_VERIFY_T2': [VERIFY_T2_MARKER]}, 'no windows, nothing to audit')
        self.assertEqual(required_flag_markers({'QWEN_FAST_VERIFY_T2_AUDIT': '1'}, 4), {})
        self.assertEqual(required_flag_markers({'QWEN_FAST_VERIFY_T2': '0'}, 4), {})

    def test_a_four_user_arm_passes_when_both_cuts_engaged_everywhere(self):
        report = self._report(self.BATCH, self.packed())
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['verify_t2_packed'], [dict(windows=48, windows_fallback=0, kv_chains=32, kv_fallback=0,
                                                           kv_rows=64, warm_chain='single', audit=0)])
        self.assertIsNone(self._report({}, self.packed())['verify_t2_packed'])
        self.assertEqual(self._report(self.BATCH, '', users=1)['missing'],
                         ['QWEN_FAST_VERIFY_T2: [PINDIAG] verify t2 engaged'])

    def test_a_cut_that_did_not_engage_everywhere_fails(self):
        for changes, field in ((dict(windows=47), 'windows=47'), (dict(windows_fallback=1), 'windows_fallback=1'),
                               (dict(kv_chains=0), 'kv_chains=0'), (dict(kv_fallback=16), 'kv_fallback=16'),
                               (dict(kv_rows=32), 'kv_rows=32'), (dict(warm_chain='none'), 'warm_chain=none')):
            with self.subTest(changes=changes):
                missing = self._report(self.BATCH, self.packed(**changes))['missing']
                self.assertEqual(len(missing), 1)
                self.assertIn('capture 1 engaged ' + field + ' (expected', missing[0])
        missing = self._report(self.BATCH, chr(10).join([self.packed(), self.packed(kv_chains=16)]))['missing']
        self.assertEqual(len(missing), 1)
        self.assertIn('capture 2 engaged kv_chains=16', missing[0])
        self.assertIn('site=packed_verify', self._report(self.BATCH, '[PINDIAG] verify t2 engaged site=other')['missing'][0])

    def test_skips_and_the_knob_move_the_expectation(self):
        skip = dict(self.BATCH, QWEN_FAST_VERIFY_T2_SKIP='kv_chains')
        self.assertEqual(self._report(skip, self.packed(kv_chains=0, kv_rows=0, warm_chain='none'))['missing'], [])
        self.assertEqual(len(self._report(skip, self.packed())['missing']), 1)
        windows = {'QWEN_FAST_VERIFY_T2': '1', 'QWEN_FAST_VERIFY_T2_SKIP': 'windows'}
        self.assertEqual(self._report(windows, self.packed(windows=0))['missing'], [], 'no user batch needed')
        knob = dict(self.BATCH, QWEN_FAST_VERIFY_T2_KV_ROWS='32')
        self.assertEqual(self._report(knob, self.packed(kv_rows=32))['missing'], [])
        bad = dict(self.BATCH, QWEN_FAST_VERIFY_T2_SKIP='windows,chains', QWEN_FAST_VERIFY_T2_KV_ROWS='16')
        missing = self._report(bad, self.packed(windows=0))['missing']
        self.assertTrue(any(line.startswith('QWEN_FAST_VERIFY_T2_SKIP: names only cuts (chains') for line in missing))
        self.assertTrue(any(line.startswith("QWEN_FAST_VERIFY_T2_KV_ROWS: 64 or 32, not '16'") for line in missing))
        # a knob that is not a number is reported, not a crash of the gate
        words = dict(self.BATCH, QWEN_FAST_VERIFY_T2_KV_ROWS='sixty-four')
        self.assertEqual(self._report(words, self.packed())['missing'],
                         ["QWEN_FAST_VERIFY_T2_KV_ROWS: 64 or 32, not 'sixty-four'"])

    def test_the_windows_cut_needs_the_user_batch(self):
        missing = self._report({'QWEN_FAST_VERIFY_T2': '1'}, self.packed(windows=0))['missing']
        self.assertEqual(missing, ['QWEN_FAST_VERIFY_T2: windows engages only under QWEN_FAST_GDN_USER_BATCH=1 '
                                   '(or skip it)'])

    def test_the_user_batch_problem_does_not_hide_the_kv_fields(self):
        missing = self._report({'QWEN_FAST_VERIFY_T2': '1'}, self.packed(windows=0, kv_chains=16,
                                                                         warm_chain='none'))['missing']
        self.assertEqual(len(missing), 2, missing)
        self.assertTrue(missing[0].startswith('QWEN_FAST_VERIFY_T2: windows engages only under'))
        self.assertIn('capture 1 engaged kv_chains=16 (expected 32), warm_chain=none (expected single)', missing[1])

    def test_fell_back_kv_shared_and_mismatch_lines_fail_at_any_user_count(self):
        lines = ('[PINDIAG] verify t2 fell back site=kv_chains reason=grid',
                 '[PINDIAG] verify t2 kv shared site=ineligible verify t2 kv tile rows shared: users 0,2 page 7 tile row 0',
                 '[PINDIAG] verify t2 kv shared site=proposal_rows verify t2 kv tile rows shared: users 0,2 page 7 '
                 'tile row 0',
                 '[PINDIAG] verify t2 audit mismatch round=3 layers=2,3 layer 2 user 0 slot 1 chip 0: 4')
        for line in lines:
            for users in (4, 1):
                with self.subTest(line=line[:40], users=users):
                    missing = self._report(self.BATCH, chr(10).join([self.packed(), line]), users=users)['missing']
                    self.assertTrue(any(line[:30] in entry for entry in missing), missing)

    def test_the_audit_needs_its_first_line_and_the_kv_audit_line_is_not_it(self):
        audit = dict(self.BATCH, QWEN_FAST_VERIFY_T2_AUDIT='1')
        kv = '[PINDIAG] verify t2 audit kv_rows_per_user=1,1,2,1'
        engaged = chr(10).join([self.packed(audit=1), kv])
        self.assertEqual(self._report(audit, engaged)['missing'],
                         ['QWEN_FAST_VERIFY_T2_AUDIT: [PINDIAG] verify t2 audit 1 exact=True'])
        good = chr(10).join([engaged, '[PINDIAG] verify t2 audit 1 exact=True layers=0-47 windows=768'])
        self.assertEqual(self._report(audit, good)['missing'], [])


if __name__ == '__main__':
    unittest.main()
