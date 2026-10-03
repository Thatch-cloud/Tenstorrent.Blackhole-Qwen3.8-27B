"""tp4/warm4: the request-width warm before the ONE 64-row block's capture (QWEN_FAST_M3_REQUEST_WARM), the profiles that carry it and
the smoke rule that reads it.

The warm is the eight-seat fix (request_width_warm) generalised to one block: attach_combined_runtime runs it right after the
prefill warm and before PackedVerifierEngine, so the rows 1/2/4 programs and state exist before the process's first trace capture.
Production (c2-packed-tp4) must not move: it carries no flag, so nothing in its attach differs.
"""

import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_smoke_check as check  # noqa: E402
import request_width_warm  # noqa: E402
import serving_runtime  # noqa: E402
import test_serving_runtime as base  # noqa: E402

FLAG = 'QWEN_FAST_M3_REQUEST_WARM'
PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text())['profiles']
M3 = (True, 'users=4 FOUR_AS_TWO=0 PACKED_STEP=1')
POLICY = {'scheduler_requests': 4}
M3_ENV = {'QWEN_FAST_FOUR_AS_TWO': '0', 'QWEN_FAST_PACKED_STEP': '1', 'QWEN_FAST_TP': '4'}


class FlagTests(unittest.TestCase):
    def test_unset_and_zero_are_off(self):
        self.assertEqual(serving_runtime.M3_REQUEST_WARM_FLAG, FLAG)
        self.assertIsNone(serving_runtime.m3_request_warm(1, POLICY, {}))
        self.assertIsNone(serving_runtime.m3_request_warm(1, POLICY, {FLAG: '0'}))
        # off is off at every shape: nothing is refused when the flag is not asked for
        self.assertIsNone(serving_runtime.m3_request_warm(1, {'scheduler_requests': 2}, {FLAG: '0'}))

    def test_one_and_even_name_their_widths_at_the_four_card_m3_shape(self):
        self.assertEqual(serving_runtime.m3_request_warm(1, POLICY, {**M3_ENV, FLAG: '1'}), (1, 2, 4))
        self.assertEqual(serving_runtime.m3_request_warm(1, POLICY, {**M3_ENV, FLAG: 'even'}), (1, 2, 4, 1))
        self.assertEqual(request_width_warm.WIDTHS, (1, 2, 4))
        self.assertEqual(request_width_warm.EVEN_WIDTHS, (1, 2, 4, 1))

    def test_a_malformed_value_is_refused_naming_the_flag(self):
        for value in ('', '2', 'true', 'EVEN', ' 1', 'on'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, FLAG):
                serving_runtime.m3_request_warm(1, POLICY, {**M3_ENV, FLAG: value})

    def test_it_is_refused_off_the_m3_shape_and_at_tp2(self):
        for value in ('1', 'even'):
            for name, policy, environ in (
                    ('four_as_two', POLICY, {**M3_ENV, 'QWEN_FAST_FOUR_AS_TWO': '1'}),
                    ('four_as_two unset', POLICY, {k: v for k, v in M3_ENV.items() if k != 'QWEN_FAST_FOUR_AS_TWO'}),
                    ('no packed step', POLICY, {**M3_ENV, 'QWEN_FAST_PACKED_STEP': '0'}),
                    ('two users', {'scheduler_requests': 2}, M3_ENV),
                    ('eight users', {'scheduler_requests': 8}, M3_ENV)):
                with self.subTest(value=value, shape=name), self.assertRaisesRegex(ValueError, FLAG):
                    serving_runtime.m3_request_warm(1, policy, {**environ, FLAG: value})
            for tp in ({'QWEN_FAST_TP': '2'}, {}):
                environ = {k: v for k, v in M3_ENV.items() if k != 'QWEN_FAST_TP'}
                with self.subTest(value=value, tp=tp), self.assertRaisesRegex(ValueError, 'QWEN_FAST_TP'):
                    serving_runtime.m3_request_warm(1, POLICY, {**environ, **tp, FLAG: value})


class AttachHookTests(unittest.TestCase):
    """The hook, through the real attach (test_serving_runtime's harness): the warm sits after the prefill warm and before the
    block's construction, which is the process's first trace capture."""

    EXTRA = {'QWEN_FAST_TP': '4', 'QWEN_FAST_ANY_REQUEST': '1'}

    def attach(self, extra_env, **kwargs):
        order = []
        import packed_verifier

        def prefill(*args, **options):
            order.append('prefill_warm')

        def warm(operations, model, helpers, sampler, page_width, widths=None, **options):
            # the block constructor's mock is the harness's: it has not been called when the warm runs
            order.append(('request_warm', packed_verifier.PackedVerifierEngine.call_count, len(helpers), page_width, widths))

        harness = base.RuntimeAttachmentTests('test_combined_recipe_lives_until_request_traces_are_closed')
        with patch.object(serving_runtime, 'prefill_warm_before_traces', prefill), \
                patch.object(serving_runtime, 'attach_source_check', lambda *args, **options: {}), \
                patch('request_width_warm.warm_request_widths', warm):
            harness.exercise(packed=True, users=4, four_as_two=False, extra_env=extra_env, **kwargs)
        return order

    def test_the_warm_runs_after_the_prefill_warm_and_before_the_block(self):
        self.assertEqual(self.attach({**self.EXTRA, FLAG: '1'}),
                         ['prefill_warm', ('request_warm', 0, 48, 68, (1, 2, 4))])

    def test_even_widths_reach_the_warm(self):
        self.assertEqual(self.attach({**self.EXTRA, FLAG: 'even'})[1], ('request_warm', 0, 48, 68, (1, 2, 4, 1)))

    def test_flag_unset_or_zero_runs_no_warm_and_the_call_sequence_is_unchanged(self):
        for extra in (self.EXTRA, {**self.EXTRA, FLAG: '0'}):
            with self.subTest(extra=extra):
                self.assertEqual(self.attach(extra), ['prefill_warm'])

    def test_the_flag_is_refused_before_anything_is_built_at_tp2_and_off_the_m3_shape(self):
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_TP'):
            self.attach({'QWEN_FAST_ANY_REQUEST': '1', FLAG: '1'}, refused=True)
        with self.assertRaisesRegex(ValueError, FLAG):
            self.attach({**self.EXTRA, FLAG: 'yes'}, refused=True)
        harness = base.RuntimeAttachmentTests('test_combined_recipe_lives_until_request_traces_are_closed')
        with self.assertRaisesRegex(ValueError, FLAG):
            harness.exercise(packed=True, users=4, extra_env={**self.EXTRA, FLAG: '1'}, refused=True)  # four_as_two default
        with self.assertRaisesRegex(ValueError, FLAG):
            harness.exercise(packed=False, users=4, four_as_two=False, extra_env={**self.EXTRA, FLAG: '1'}, refused=True)

    def test_the_hook_is_one_contiguous_block_in_the_source(self):
        source = (HERE / 'serving_runtime.py').read_text()
        self.assertEqual(source.count('warm_request_widths('), 2, 'one call before the single block, one between the two blocks phases')
        hook = source.index('        if request_widths and not m3_blocks_two:\n')
        self.assertLess(source.index('        prefill_warm_before_traces(runner'), hook)
        self.assertLess(hook, source.index('        if packed_shapes:\n            from packed_verifier import PackedVerifierEngine'))
        self.assertIn("with stall_watch.scope('build', 'request widths warm'):", source)


SEATS8 = {'c2-packed-tp4-8', 'c2-packed-tp4-8-gate', 'c2-packed-tp4-8-time-gate', 'c2-packed-tp4-8-diag-strace',
          'c2-packed-tp4-8-diag-strace-nowarm', 'c2-packed-tp4-8-diag-strace-rshard', 'c2-packed-tp4-8-best', 'c2-packed-tp4-8-best-quad', 'c2-packed-tp4-8-best-quad-gate'} | {name for name in PROFILES if name.startswith('c2-packed-tp4-8x262k')}
WARM4 = {'c2-packed-tp4-warm4-diag', 'c2-packed-tp4-warm4-control', 'c2-packed-tp4-warm4-even-diag', 'c2-packed-tp4-warm4-gate',
         'c2-packed-tp4-speed-warm4', 'c2-packed-tp4-warm4-diag-oldtail'}


def delta(profile, base_profile):
    left, right = PROFILES[profile]['env'], PROFILES[base_profile]['env']
    added = {k: v for k, v in left.items() if right.get(k) != v}
    removed = sorted(k for k in right if k not in left)
    return added, removed


class ProfileTests(unittest.TestCase):
    def test_every_twin_is_its_base_plus_exact_deltas(self):
        diag_added = {'QWEN_FAST_BUDGET_CAP': '1', 'QWEN_FAST_STALL_BUILD_S': '150', FLAG: '1'}
        expected = {
            'c2-packed-tp4-warm4-diag': ('c2-packed-tp4-diag', diag_added, []),
            'c2-packed-tp4-warm4-control': ('c2-packed-tp4-warm4-diag', {FLAG: '0'}, []),
            'c2-packed-tp4-warm4-even-diag': ('c2-packed-tp4-warm4-diag', {FLAG: 'even'}, []),
            'c2-packed-tp4-warm4-gate': ('c2-packed-tp4-gate', {FLAG: '1'}, []),
            'c2-packed-tp4-speed-warm4': ('c2-packed-tp4-speed-strace', {FLAG: '1'}, ['QWEN_FAST_PACKED_SAMPLER_IN_TRACE']),
            'c2-packed-tp4-warm4-diag-oldtail': ('c2-packed-tp4-diag', {'QWEN_FAST_STALL_BUILD_S': '150', FLAG: '1'}, []),
        }
        self.assertEqual(set(expected), WARM4)
        for name, (base_name, added, removed) in expected.items():
            with self.subTest(profile=name):
                self.assertEqual(delta(name, base_name), (added, removed))
                for key in PROFILES[name]:
                    if key not in ('env', 'description'):
                        self.assertEqual(PROFILES[name][key], PROFILES[base_name][key], key)

    def test_warm4_diag_is_production_less_the_in_trace_sampler_plus_the_warm_and_the_instruments(self):
        added, removed = delta('c2-packed-tp4-warm4-diag', 'c2-packed-tp4')
        self.assertEqual(removed, ['QWEN_FAST_PACKED_SAMPLER_IN_TRACE'])
        self.assertEqual(added, {'QWEN_C2_GATE_PROFILE': '1', FLAG: '1', 'QWEN_FAST_SEQ_STAGE_LOG': '1', 'QWEN_FAST_TRACE_CENSUS': '1',
                                 'QWEN_FAST_TRACE_CENSUS_GRAPH': '0', 'QWEN_FAST_CCL_HANDLE_LOG': '1',
                                 'QWEN_FAST_CCL_HANDLE_GUARD': 'log', 'QWEN_FAST_MEMORY_LEDGER_L1': '1',
                                 'QWEN_FAST_STALL_DEADLINE_S': '120', 'QWEN_FAST_STALL_BUILD_S': '150'})
        env = PROFILES['c2-packed-tp4-warm4-diag']['env']
        self.assertEqual((env['QWEN_FAST_BUDGET_CAP'], env['QWEN_FAST_SEQ_DEADLINE_S']), ('1', '120'))

    def test_the_timed_twin_is_speed_plus_the_warm_and_nothing_else(self):
        self.assertEqual(delta('c2-packed-tp4-speed-warm4', 'c2-packed-tp4-speed'), ({FLAG: '1'}, []))

    def test_every_twin_is_gate_only_and_carries_the_waiver_marker(self):
        for name in WARM4:
            with self.subTest(profile=name):
                self.assertIs(PROFILES[name]['gate_only'], True)
                self.assertEqual(PROFILES[name]['env']['QWEN_C2_GATE_PROFILE'], '1')

    def test_production_is_unchanged_and_no_other_profile_carries_the_flag(self):
        production = PROFILES['c2-packed-tp4']
        self.assertNotIn(FLAG, production['env'])
        self.assertEqual(production['env']['QWEN_FAST_PACKED_SAMPLER_IN_TRACE'], '1')
        self.assertFalse(production.get('gate_only'))
        carriers = {name for name, profile in PROFILES.items() if FLAG in profile.get('env', {})}
        self.assertEqual(carriers - SEATS8 - {'c2-packed-tp4-best-ship-warm4'}, WARM4)
        self.assertEqual(carriers & SEATS8, SEATS8, 'the eight-seat profiles carry the flag with QWEN_FAST_M3_BLOCKS=2')
        for name in SEATS8:
            self.assertEqual(PROFILES[name]['env']['QWEN_FAST_M3_BLOCKS'], '2', name)

    def test_every_carrier_meets_the_attach_rule_laid_over_the_m3_env(self):
        for name in WARM4:
            env = dict(PROFILES[name]['env'])
            value = env[FLAG]
            with self.subTest(profile=name):
                self.assertEqual(env['QWEN_FAST_TP'], '4')
                result = serving_runtime.m3_request_warm(1, POLICY, {**M3_ENV, **env})
                self.assertEqual(result, {'0': None, '1': (1, 2, 4), 'even': (1, 2, 4, 1)}[value])


CAPTURE = '[PINDIAG] verify t1 engaged site=packed_verify audit=0 coalesce_fallback=0 in_trace=0'
ENGINE = '[PINDIAG] four-card engine programs=%d->%d window=1'


def warm_line(rows='(1, 2, 4)', before=700, after=807, ms=4100):
    return '[PINDIAG] request widths warmed before the packed traces: rows=%s programs=%d->%d ms=%d' % (rows, before, after, ms)


class SmokeRuleTests(unittest.TestCase):
    def problems(self, lines, flag='1'):
        return check.request_warm_problems('\n'.join(lines), flag)

    def test_a_warm_before_the_capture_anchor_that_compiled_programs_is_clean_and_recorded(self):
        for rows, flag in (('(1, 2, 4)', '1'), ('(1, 2, 4, 1)', 'even')):
            with self.subTest(flag=flag):
                problems, facts = self.problems([warm_line(rows), CAPTURE, ENGINE % (807, 939)], flag)
                self.assertEqual(problems, [])
                self.assertEqual((facts['request_warm_programs'], facts['request_warm_ms'], facts['first_engine_programs']),
                                 (107, 4100, 132))
                self.assertEqual((facts['request_warm_line'], facts['capture_anchor_line']), (1, 2))

    def test_block_zero_capture_line_is_the_anchor_when_present(self):
        text = [CAPTURE, warm_line(), '[PINDIAG] packed blocks capture block=0 programs=1->2']
        self.assertEqual(self.problems(text)[0], [])

    def test_a_missing_warm_line_fails_naming_the_flag(self):
        problems, facts = self.problems([CAPTURE])
        self.assertEqual(len(problems), 1)
        self.assertIn('QWEN_FAST_M3_REQUEST_WARM=1', problems[0])
        self.assertIsNone(facts['request_warm_programs'])

    def test_a_warm_after_the_capture_fails(self):
        problems, _ = self.problems([CAPTURE, warm_line()])
        self.assertEqual(len(problems), 1)
        self.assertIn('came after', problems[0])

    def test_no_capture_anchor_fails(self):
        problems, _ = self.problems([warm_line()])
        self.assertEqual(len(problems), 1)
        self.assertIn('to order the request warm against', problems[0])

    def test_a_warm_that_compiled_nothing_fails(self):
        problems, _ = self.problems([warm_line(before=700, after=700), CAPTURE])
        self.assertEqual(len(problems), 1)
        self.assertIn('compiled 0 programs', problems[0])

    def test_unreadable_program_counts_are_recorded_not_failed(self):
        problems, facts = self.problems([warm_line().replace('programs=700->807', 'programs=None->None'), CAPTURE])
        self.assertEqual(problems, [])
        self.assertIsNone(facts['request_warm_programs'])

    def test_a_three_width_line_under_the_wrong_rows_is_not_a_warm_line(self):
        problems, _ = self.problems([warm_line('(1, 2)'), CAPTURE])
        self.assertIn('never ran', problems[0])

    def test_check_applies_the_rule_only_under_the_flag(self):
        env = dict(PROFILES['c2-packed-tp4-speed-warm4']['env'])
        text = '\n'.join([CAPTURE])
        facts = check.check('', text, False, env=env)[1]
        self.assertIn('request_warm_programs', facts)
        facts = check.check('', text, False, env={**env, FLAG: '0'})[1]
        self.assertNotIn('request_warm_programs', facts)
        facts = check.check('', text, False, env={k: v for k, v in env.items() if k != FLAG})[1]
        self.assertNotIn('request_warm_programs', facts)


if __name__ == '__main__':
    unittest.main()
