"""Lever N at TP4: the smoke rules (c2_smoke_check.levern_problems) and the arm-against-arm comparison (levern_compare), on synthetic container logs.

The rules read what the lever LOGS, so every rule has a log that satisfies it and one negative control per way it can fail: the markers that prove the
lever engaged (not merely installed), the route ledger of a split prompt, the alternation between prefill steps, the progress of every seat inside the
arrival's window, the digest lines of the audit. levern_compare is held to the same standard: identical arms pass, every field that can differ is
caught, and a comparison that compared nothing is not a pass."""

import hashlib
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO

import c2_smoke_check as check
import levern_compare as compare
import levern_policy as policy

ENV = {'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_PREFILL_SHARE': '0.5', 'QWEN_FAST_TP': '4'}
AUDIT_ENV = dict(ENV, QWEN_FAST_LEVERN_AUDIT='1')
CONTROL_ENV = {'QWEN_FAST_LEVERN_AUDIT': '1', 'QWEN_FAST_TP': '4'}


def route(req, start, end, prompt, programs='900->900', ms=812.4, window=0):
    final = int(end == prompt)
    return policy.ROUTE_LINE.format(req, start, end, prompt, final, final, ms, *programs.split('->'), window)


def route_lines(req, prompt, decoding=True, **kwargs):
    return [route(req, start, end, prompt, **kwargs) for start, end in policy.plan(prompt, decoding=decoding)]


def step(n, kind, seats, req='-', start='-', tokens=0, end='-', prompt='-', final='-', reason='paid', prev='prefill:700.0', owed='350', rounds=0):
    return policy.STEP_LINE.format(n, kind, seats, req, start, tokens, end, prompt, final, reason, prev.split(':')[0], prev.split(':')[1], owed, rounds)


def interleaved_steps(req, prompt, seats=7, first=1):
    """The scheduler's lines for a prefill split by the plan with one decode step between each pair of prefill steps."""
    lines, n = [], first
    plan = policy.plan(prompt)
    for index, (start, end) in enumerate(plan):
        lines.append(step(n, 'prefill', seats, req, start, end - start, end, prompt, int(end == prompt), 'paid'))
        n += 1
        if index < len(plan) - 1:
            lines.append(step(n, 'decode', seats, reason='owed'))
            n += 1
    return lines


def solo_steps(req, prompt):
    """The scheduler's lines for a prompt split with no decoder running (back to back steps, nobody to yield to)."""
    return [step(n, 'prefill', 0, req, start, end - start, end, prompt, int(end == prompt), 'idle')
            for n, (start, end) in enumerate(policy.plan(prompt, decoding=False), 1)]


def engaged_log(extra=()):
    return '\n'.join([policy.INSTALLED_LINE.format('vllm_tt_plugin.scheduler.TTScheduler', 2048, 16384, 0.5, '-', 8),
                      policy.PLATFORM_LINE.format(262144, 0),
                      policy.ROUTE_INSTALLED_LINE.format('route=1 audit=0'),
                      *route_lines('__levern_warm__', 6208),
                      policy.ROUTE_WARM_LINE.format(3, 880, 900, 4100.0),
                      *extra])


class EngagedTests(unittest.TestCase):
    def problems(self, log, env=ENV, smoke=None):
        return check.levern_problems(env, log, smoke)[0]

    def test_a_clean_log_has_no_problem(self):
        log = engaged_log(route_lines('cmpl-a', 10000) + interleaved_steps('cmpl-a', 10000))
        self.assertEqual(self.problems(log), [])

    def test_the_flags_off_is_nothing_at_all(self):
        self.assertEqual(check.levern_problems({'QWEN_FAST_TP': '4'}, '', None), ([], {}))
        self.assertEqual(check.levern_problems({}, 'garbage', None), ([], {}))

    def test_each_engaged_marker_is_required(self):
        full = engaged_log().splitlines()
        for needle, what in (('lever N installed on', 'scheduler install line'), ('chunked prefill kept', 'platform wrap line'),
                             ('route installed', 'route install line'), ('route warmed before the packed traces', 'route warm line')):
            with self.subTest(needle=needle):
                log = '\n'.join(line for line in full if needle not in line)
                found = self.problems(log)
                self.assertTrue(any('did not engage' in problem and what in problem for problem in found), (what, found))

    def test_the_warm_must_have_three_steps(self):
        lines = [line for line in engaged_log().splitlines() if '__levern_warm__' not in line or 'start=2048' not in line]
        self.assertTrue(any('route warm logged 2 step line' in problem for problem in self.problems('\n'.join(lines))))

    def test_the_warm_must_come_before_the_first_allocation_with_a_trace_live(self):
        log = engaged_log().splitlines()
        log.insert(1, check.UNSAFE_ALLOCATION)
        self.assertTrue(any('came after the first allocation made with a trace live' in problem for problem in self.problems('\n'.join(log))))
        log = engaged_log().splitlines() + [check.UNSAFE_ALLOCATION]
        self.assertEqual(self.problems('\n'.join(log)), [])

    def test_a_refused_line_is_a_problem(self):
        log = engaged_log([policy.REFUSED_LINE.format('no cap for cmpl-a: prompt=None computed=0')])
        self.assertTrue(any('REFUSED' in problem for problem in self.problems(log)))

    def test_facts_count_the_split_prompts_and_their_steps(self):
        log = engaged_log(route_lines('cmpl-a', 10000) + interleaved_steps('cmpl-a', 10000) + route_lines('cmpl-b', 5000))
        problems, facts = check.levern_problems(ENV, log, None)
        self.assertEqual((facts['split_prompts'], facts['route_steps']), (2, 4 + 2))
        self.assertEqual(problems, [])


class RouteLedgerTests(unittest.TestCase):
    def problems(self, lines):
        return check.levern_route_problems(check.levern_facts('\n'.join(lines))['routes'])

    def test_every_plan_is_clean(self):
        for prompt in (2049, 4096, 4097, 6145, 12288, 131077, 253920):
            with self.subTest(prompt=prompt):
                self.assertEqual(self.problems(route_lines('r', prompt)), [])

    def test_a_skipped_chunk(self):
        lines = route_lines('r', 10000)
        del lines[1]
        self.assertTrue(any('starts at 4096, the previous ended at 2048' in problem for problem in self.problems(lines)))

    def test_a_replayed_chunk(self):
        lines = route_lines('r', 10000)
        lines.insert(2, lines[1])
        self.assertTrue(any('starts at 2048' in problem for problem in self.problems(lines)))

    def test_a_non_final_end_off_the_boundary(self):
        lines = [route('r', 0, 3000, 10000, ), route('r', 3000, 10000, 10000)]
        found = self.problems(lines)
        self.assertTrue(any('off the 2048-token chunk boundary' in problem for problem in found), found)

    def test_a_slot_write_before_the_final_step(self):
        lines = route_lines('r', 10000)
        lines[0] = lines[0].replace('wrote_slot=0', 'wrote_slot=1')
        self.assertTrue(any('wrote_slot=1' in problem for problem in self.problems(lines)))

    def test_a_final_step_that_wrote_no_slot(self):
        lines = route_lines('r', 10000)
        lines[-1] = lines[-1].replace('wrote_slot=1', 'wrote_slot=0')
        self.assertTrue(any('wrote_slot=0' in problem for problem in self.problems(lines)))

    def test_a_step_after_the_final_step(self):
        lines = route_lines('r', 10000) + [route('r', 10000, 12288, 20000)]
        self.assertTrue(self.problems(lines))

    def test_program_growth_in_a_step(self):
        lines = route_lines('r', 10000, programs='900->903')
        self.assertTrue(any('compiled 3 program(s)' in problem for problem in self.problems(lines)))

    def test_growth_the_window_snapshot_explains_is_not_a_hang(self):
        # the four-card tripwire's rule: B - A - W > 0 fails, B - A == W does not (the snapshot compiles per prompt geometry)
        final = route('r', 4096, 6145, 6145, programs='600->602', window=2)
        self.assertEqual(self.problems([route('r', 0, 2048, 6145), route('r', 2048, 4096, 6145), final]), [])
        short = route('r', 4096, 6145, 6145, programs='600->602', window=1)
        found = self.problems([route('r', 0, 2048, 6145), route('r', 2048, 4096, 6145), short])
        self.assertTrue(any('compiled 1 program(s) beyond its 1 window program(s)' in problem for problem in found), found)

    def test_the_warm_requests_programs_are_not_judged(self):
        self.assertEqual(self.problems(route_lines('__levern_warm__', 6208, programs='500->540')), [])

    def test_an_unknown_program_count_is_not_growth(self):
        self.assertEqual(self.problems(route_lines('r', 10000, programs='None->None')), [])

    def test_an_aborted_prompt_is_legal(self):
        self.assertEqual(self.problems(route_lines('r', 20000)[:3]), [])

    def test_two_prompts_are_judged_apart(self):
        interleaved = []
        a, b = route_lines('a', 10000), route_lines('b', 8000)
        for left, right in zip(a, b):
            interleaved += [left, right]
        interleaved += a[len(b):]
        self.assertEqual(self.problems(interleaved), [])


LONG = 32785


def busy_smoke(*lengths):
    return {'levern_equal_busy': dict(prompts=dict((str(length), dict()) for length in lengths), lengths=list(lengths))}


class ExercisedRuleTests(unittest.TestCase):
    """NOT_EXERCISED is not a pass: a row of 4,096 tokens or more must have been split as the plan says."""

    def problems(self, extra, smoke, env=ENV):
        return check.levern_problems(env, engaged_log(extra), smoke)[0]

    def test_rows_that_were_never_split_are_a_problem(self):
        smoke = {'levern_equal': dict(prompts={'2049': dict(), str(LONG): dict()}, lengths=[2049, LONG])}
        found = self.problems([], smoke)
        self.assertEqual(len(found), 1, found)
        self.assertIn('levern_equal: the prompt of %d tokens was not split as the plan says' % LONG, found[0])
        self.assertIn('NOT_EXERCISED', found[0])

    def test_the_planned_split_is_clean_and_a_short_row_judges_nothing(self):
        smoke = {'levern_equal': dict(prompts={'2049': dict(), str(LONG): dict()}, lengths=[2049, LONG])}
        self.assertEqual(self.problems(route_lines('solo', LONG, decoding=False) + solo_steps('solo', LONG), smoke), [])

    def test_the_single_user_rows_are_planned_without_decoders_and_the_busy_ones_with(self):
        smoke = {'levern_equal': dict(prompts={str(LONG): dict()}, lengths=[LONG])}
        found = self.problems(route_lines('solo', LONG, decoding=True), smoke)
        self.assertTrue(any('decoding=False' in problem for problem in found), found)

    def test_a_split_at_other_ends_than_the_plan_is_a_problem(self):
        smoke = {'levern_equal': dict(prompts={'10000': dict()}, lengths=[10000])}
        lines = [route('r', 0, 4096, 10000), route('r', 4096, 8192, 10000), route('r', 8192, 10000, 10000)]
        self.assertTrue(self.problems(lines, smoke))

    def test_the_busy_rows_need_a_decode_step_between_two_prefill_steps(self):
        smoke = busy_smoke(10000)
        lines = route_lines('b', 10000)
        found = self.problems(lines + [s for s in interleaved_steps('b', 10000) if 'kind=decode' not in s], smoke)
        self.assertTrue(any('nothing was interleaved' in problem for problem in found), found)
        self.assertEqual(self.problems(lines + interleaved_steps('b', 10000), smoke), [])

    def test_a_decode_step_serving_nobody_does_not_count_as_interleaving(self):
        lines = route_lines('b', 10000)
        steps = [line.replace('kind=decode seats=7', 'kind=decode seats=0') for line in interleaved_steps('b', 10000)]
        self.assertTrue(any('nothing was interleaved' in problem for problem in self.problems(lines + steps, busy_smoke(10000))))

    def test_a_row_that_errored_is_not_judged_here(self):
        smoke = {'levern_equal': dict(prompts={str(LONG): dict(error='x')}, lengths=[LONG], error='x')}
        self.assertEqual(self.problems([], smoke), [])

    def test_the_control_arm_is_not_judged(self):
        smoke = {'levern_equal': dict(prompts={str(LONG): dict()}, lengths=[LONG])}
        found = check.levern_problems(CONTROL_ENV, digest('x', LONG), smoke)[0]
        self.assertEqual(found, [])


class AlternationRuleTests(unittest.TestCase):
    def problems(self, lines, env=ENV):
        return check.levern_alternation_problems(check.levern_facts('\n'.join(lines))['steps'], env)

    def test_a_decode_step_between_every_pair_is_clean(self):
        self.assertEqual(self.problems(interleaved_steps('a', 20000)), [])

    def test_two_prefill_steps_back_to_back_with_decoders_running_is_a_problem(self):
        lines = interleaved_steps('a', 20000)
        del lines[1]
        found = self.problems(lines)
        self.assertEqual(len(found), 1)
        self.assertIn('did not yield', found[0])

    def test_a_decode_step_that_served_nobody_is_not_a_yield(self):
        lines = interleaved_steps('a', 20000)
        lines[1] = step(2, 'decode', 0, reason='owed')
        self.assertTrue(self.problems(lines))

    def test_with_no_decoder_running_back_to_back_is_fine(self):
        lines = []
        for n, (start, end) in enumerate(policy.plan(20000, decoding=False), 1):
            lines.append(step(n, 'prefill', 0, 'a', start, end - start, end, 20000, int(end == 20000), 'idle'))
        self.assertEqual(self.problems(lines), [])

    def test_a_full_share_runs_chunks_back_to_back_by_design(self):
        lines = interleaved_steps('a', 20000)
        del lines[1]
        self.assertEqual(self.problems(lines, dict(ENV, QWEN_FAST_LEVERN_PREFILL_SHARE='1.0')), [])
        self.assertTrue(self.problems(lines, dict(ENV, QWEN_FAST_LEVERN_PREFILL_SHARE='1.0', QWEN_FAST_LEVERN_ROUNDS='1')))

    def test_a_decode_step_from_the_final_hold_counts(self):
        lines = interleaved_steps('a', 5000)
        lines.insert(2, step(9, 'decode', 7, reason='final-dram'))
        self.assertEqual(self.problems(lines), [])

    def test_a_split_without_any_step_line_means_the_alternation_never_ran(self):
        log = engaged_log(route_lines('cmpl-a', 10000))
        found = check.levern_problems(ENV, log, None)[0]
        self.assertTrue(any('alternation never ran' in problem for problem in found), found)

    def test_decoders_beside_prefills_and_no_decode_step_at_all(self):
        steps = [step(n, 'prefill', 7, 'a', 2048 * (n - 1), 2048, 2048 * n, 20000, 0, 'paid') for n in range(1, 4)]
        found = check.levern_problems(dict(ENV, QWEN_FAST_LEVERN_PREFILL_SHARE='1.0'), engaged_log(route_lines('a', 20000) + steps), None)[0]
        self.assertTrue(any('no decode step was ever yielded' in problem for problem in found), found)


def stall_entry(progress, window_s=30.0, seats=7):
    windows = [dict(seat=index, window_s=window_s, chunks=chunks, chunk_rate=round(chunks / window_s, 3), est_tok_s=1.0) for index, chunks in
               enumerate(progress)]
    return dict(users=[dict(tokens=100, finish='length', text='hello world code ' * 5) for _ in range(seats + 1)], arrival_ttft_s=40.0,
                seat_gaps=[dict(seat=index) for index in range(seats)], longest_gap_s=4.0, seat_windows=windows,
                window=dict(seats=seats, seats_progressing=len([value for value in progress if value]), min_chunk_rate=0.0,
                            total_est_tok_s=7.0))


class WindowRuleTests(unittest.TestCase):
    def test_every_seat_progressing_is_clean(self):
        smoke = {'stall8_cold262k': stall_entry([40] * 7)}
        log = engaged_log()
        self.assertEqual(check.levern_problems(ENV, log, smoke)[0], [])

    def test_a_frozen_seat_in_a_long_window_is_a_problem(self):
        smoke = {'stall8_cold128k': stall_entry([40, 40, 0, 40, 40, 40, 40])}
        found = check.levern_problems(ENV, engaged_log(), smoke)[0]
        self.assertTrue(any('6 of 7 decoding seats progressed' in problem for problem in found), found)

    def test_a_short_window_judges_nothing(self):
        smoke = {'stall8_cold262k': stall_entry([0] * 7, window_s=2.0)}
        self.assertEqual(check.levern_problems(ENV, engaged_log(), smoke)[0], [])

    def test_the_control_arm_is_not_judged_by_this_rule(self):
        smoke = {'stall8_cold262k': stall_entry([0] * 7)}
        self.assertEqual(check.levern_problems(CONTROL_ENV, '', smoke)[0], [])


def token_sha(prompt, salt=''):
    return hashlib.sha256(('%s%s' % (prompt, salt)).encode()).hexdigest()[:32]


def digest(req, prompt, tag='a', salt=''):
    return policy.DIGEST_LINE.format(req, prompt, token_sha(prompt, salt), tag * 32, 'b' * 32, 'c' * 32)


class DigestRuleTests(unittest.TestCase):
    SMOKE = {'levern_equal': dict(prompts={'4097': dict(), '6145': dict()}, lengths=[4097, 6145])}

    def test_a_digest_line_per_prompt_is_clean_on_both_audit_arms(self):
        log = '\n'.join([digest('x', 4097), digest('y', 6145)])
        self.assertEqual(check.levern_problems(CONTROL_ENV, log, self.SMOKE)[0], [])
        split = (route_lines('x', 4097, decoding=False) + route_lines('y', 6145, decoding=False) + solo_steps('x', 4097) + solo_steps('y', 6145))
        self.assertEqual(check.levern_problems(AUDIT_ENV, engaged_log([digest('x', 4097), digest('y', 6145)] + split), self.SMOKE)[0], [])

    def test_a_prompt_with_no_digest_is_a_problem(self):
        found = check.levern_problems(CONTROL_ENV, digest('x', 4097), self.SMOKE)[0]
        self.assertEqual(len(found), 1)
        self.assertIn('no digest line for the prompt of 6145 tokens', found[0])

    def test_digests_are_keyed_on_the_prompt_not_on_log_order(self):
        # two prompts of one length admitted in a different order across the arms: each is compared with its own twin
        control = '\n'.join([digest('p', 4097, 'a', 'one'), digest('q', 4097, 'b', 'two')])
        swapped = '\n'.join([digest('q', 4097, 'b', 'two'), digest('p', 4097, 'a', 'one')])
        self.assertEqual(compare.compare_digests(control, swapped), ([], 2))
        different = '\n'.join([digest('q', 4097, 'b', 'two'), digest('p', 4097, 'e', 'one')])
        mismatches, compared = compare.compare_digests(control, different)
        self.assertEqual((len(mismatches), compared), (1, 2))
        self.assertIn('the GDN slot differs', mismatches[0])

    def test_no_audit_flag_no_digest_rule(self):
        split = (route_lines('x', 4097, decoding=False) + route_lines('y', 6145, decoding=False) + solo_steps('x', 4097) + solo_steps('y', 6145))
        self.assertEqual(check.levern_problems(ENV, engaged_log(split), self.SMOKE)[0], [])

    def test_the_check_runs_the_rules_through_check(self):
        smoke_line = 'SMOKE_JSON ' + json.dumps({'warmup': {'value': 200}})
        problems, facts = check.check(smoke_line, engaged_log(route_lines('a', 10000) + interleaved_steps('a', 10000)), False, env=dict(ENV, QWEN_FAST_EXTENT_REPLAY='1'))
        self.assertEqual(facts['levern']['split_prompts'], 1)
        self.assertFalse([problem for problem in problems if 'lever' in problem.lower()], problems)


def smoke_text(**tests):
    return 'prefix\nSMOKE_JSON ' + json.dumps(tests) + '\n'


def users(text='a', tokens=100, finish='length', count=2):
    return [dict(content_sha256=text * 8, reasoning_sha256='r' * 8, tokens=tokens, finish=finish, text='code ' * 30) for _ in range(count)]


def rows(tag='a', tokens=256):
    return {'4097': dict(content_sha256=tag * 8, tokens=tokens, finish='length', prompt_tokens=4097, prompt_tokens_sent=4097),
            '6145': dict(content_sha256=tag * 8, tokens=tokens, finish='length', prompt_tokens=6145, prompt_tokens_sent=6145)}


class CompareTests(unittest.TestCase):
    def run_compare(self, control, levern, control_container=None, levern_container=None):
        return compare.compare(control, levern, control_container, levern_container)

    def test_identical_arms_pass_and_count_what_they_compared(self):
        control = smoke_text(levern_equal=dict(prompts=rows()), stall8_cold128k=dict(users=users(count=8), arrival_ttft_s=2.0, longest_gap_s=60.0,
                                                                                      window=dict(seats=7, seats_progressing=0)))
        levern = smoke_text(levern_equal=dict(prompts=rows()), stall8_cold128k=dict(users=users(count=8), arrival_ttft_s=3.0, longest_gap_s=1.0,
                                                                                     window=dict(seats=7, seats_progressing=7)))
        mismatches, compared, report = self.run_compare(control, levern)
        self.assertEqual(mismatches, [])
        self.assertEqual(compared, 2 + 8)
        self.assertTrue(any('arrival TTFT interleaved / control = 1.50' in line for line in report))
        self.assertTrue(any('seats_progressing=7/7' in line for line in report))

    def test_each_field_that_can_differ_is_caught(self):
        base = users()
        for key in ('content_sha256', 'reasoning_sha256', 'tokens', 'finish'):
            with self.subTest(key=key):
                other = users()
                other[1][key] = 'zzzzzzzz' if isinstance(other[1][key], str) else 99
                mismatches, compared, _ = self.run_compare(smoke_text(t=dict(users=base)), smoke_text(t=dict(users=other)))
                self.assertEqual(compared, 2)
                self.assertEqual(len(mismatches), 1, mismatches)
                self.assertIn('t user 1', mismatches[0])

    def test_each_row_field_is_caught(self):
        for key in ('content_sha256', 'tokens', 'finish', 'prompt_tokens'):
            with self.subTest(key=key):
                other = rows()
                other['6145'][key] = 'zzzz' if isinstance(other['6145'][key], str) else 1
                mismatches, _, _ = self.run_compare(smoke_text(levern_equal=dict(prompts=rows())), smoke_text(levern_equal=dict(prompts=other)))
                self.assertEqual(len(mismatches), 1, mismatches)
                self.assertIn('levern_equal prompt 6145', mismatches[0])

    def test_an_arm_that_errored_or_ran_other_lengths_is_a_mismatch(self):
        mismatches, _, _ = self.run_compare(smoke_text(t=dict(users=users())), smoke_text(t=dict(error='boom')))
        self.assertTrue(any('an arm errored' in text for text in mismatches))
        mismatches, _, _ = self.run_compare(smoke_text(levern_equal=dict(prompts=rows())), smoke_text(levern_equal=dict(prompts={'4097': rows()['4097']})))
        self.assertTrue(any('different prompt lengths' in text for text in mismatches))
        mismatches, _, _ = self.run_compare(smoke_text(t=dict(users=users(count=2))), smoke_text(t=dict(users=users(count=3))))
        self.assertTrue(any('2 users in the control, 3 in the interleaved' in text for text in mismatches))
        errored = users()
        errored[0] = dict(error='timeout')
        mismatches, _, _ = self.run_compare(smoke_text(t=dict(users=users())), smoke_text(t=dict(users=errored)))
        self.assertTrue(any('has no answer to compare' in text for text in mismatches))

    def test_the_digests_are_compared_per_prompt_in_log_order(self):
        control = '\n'.join([digest('x', 4097), digest('y', 6145), digest('z', 4097, 'd')])
        same = '\n'.join([digest('p', 4097), digest('q', 6145), digest('r', 4097, 'd')])
        mismatches, compared = compare.compare_digests(control, same)
        self.assertEqual((mismatches, compared), ([], 3))
        different = '\n'.join([digest('p', 4097), digest('q', 6145, 'e'), digest('r', 4097, 'd')])
        mismatches, compared = compare.compare_digests(control, different)
        self.assertEqual(compared, 3)
        self.assertEqual(len(mismatches), 1)
        self.assertIn('6145-token prompt (run 1): the GDN slot differs', mismatches[0])
        missing = '\n'.join([digest('p', 4097), digest('r', 4097, 'd')])
        self.assertTrue(any('6145-token prompt (token sha' in text and '1 in the control, 0 in the interleaved' in text
                            for text in compare.compare_digests(control, missing)[0]))

    def test_each_part_of_a_digest_is_named(self):
        for part, name in ((3, 'GDN slot'), (4, 'logits'), (5, 'KV pages')):
            with self.subTest(part=name):
                left = policy.DIGEST_LINE.format('x', 4097, token_sha(4097), 'a' * 32, 'b' * 32, 'c' * 32)
                fields = ['a' * 32, 'b' * 32, 'c' * 32]
                fields[part - 3] = 'f' * 32
                right = policy.DIGEST_LINE.format('y', 4097, token_sha(4097), *fields)
                mismatches, _ = compare.compare_digests(left, right)
                self.assertEqual(len(mismatches), 1)
                self.assertIn('the %s differs' % name, mismatches[0])

    def test_main_exit_codes(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a.log').write_text(smoke_text(levern_equal=dict(prompts=rows())), encoding='utf-8')
            (root / 'b.log').write_text(smoke_text(levern_equal=dict(prompts=rows())), encoding='utf-8')
            (root / 'c.log').write_text(smoke_text(levern_equal=dict(prompts=rows('z'))), encoding='utf-8')
            (root / 'none.log').write_text(smoke_text(other=dict(users=users())), encoding='utf-8')
            (root / 'empty.log').write_text('no smoke here', encoding='utf-8')
            args = lambda a, b: ['--control', str(root / a), '--levern', str(root / b)]  # noqa: E731
            buffer = StringIO()
            with redirect_stdout(buffer):
                self.assertEqual(compare.main(args('a.log', 'b.log')), 0)
                self.assertEqual(compare.main(args('a.log', 'c.log')), 1)
                self.assertEqual(compare.main(args('a.log', 'none.log')), 2)
                self.assertEqual(compare.main(args('a.log', 'empty.log')), 1)
            self.assertIn('"identical": true', buffer.getvalue())
            self.assertIn('LEVERN_COMPARE MISMATCH: levern_equal prompt 4097: content hash differs', buffer.getvalue())
            self.assertEqual(compare.main(['--control', str(root / 'a.log'), '--levern', str(root / 'missing.log')]), 2)
            self.assertEqual(compare.main(['--control', str(root / 'a.log'), '--levern', str(root / 'b.log'),
                                           '--control-container', str(root / 'a.log')]), 2)


# ---------------------------------------------------------------------------------------------------------------------------------------
# The merged route (Lever N beside prefix reuse, docs/lever-n-prefix-merged-route.md): the same rules with a hit's first step, a four-step warm, parking
# ---------------------------------------------------------------------------------------------------------------------------------------

MERGED_ENV = dict(ENV, QWEN_PREFIX_REUSE='1', QWEN_FAST_STICKY_SESSIONS='1', QWEN_FAST_LEVERN_PARK='host')


def merged_route(req, start, end, prompt, source, captured='-', programs='900->900'):
    final = int(end == prompt)
    return policy.ROUTE_LINE_MERGED.format(req, start, end, prompt, final, final, 812.4, *programs.split('->'), 0, source, captured)


def merged_route_lines(req, prompt, hit=0, captured=None):
    steps = policy.plan_from(prompt, hit)
    return [merged_route(req, start, end, prompt, ('CHECKPOINT' if hit else 'COLD') if index == 0 else 'SCRATCH',
                         (captured or {}).get(end, '-')) for index, (start, end) in enumerate(steps)]


def merged_log(extra=()):
    return '\n'.join(['install sticky=1 lookahead=16 drop_last=True ceiling=floor2048(P-2048)', 'install chunked=levern: chunked prefill beside the Lever N cap',
                      policy.INSTALLED_LINE.format('vllm_tt_plugin.scheduler.TTScheduler', 2048, 16384, 0.5, '-', 8),
                      policy.PLATFORM_LINE.format(262144, 0),
                      policy.ROUTE_INSTALLED_LINE.format('route=1 audit=0 merged=COLD,CHECKPOINT,SCRATCH,PARKED'),
                      policy.ROUTE_WARM_LINE.format(4, 880, 900, 4100.0),
                      *extra])


class MergedRouteTests(unittest.TestCase):
    def problems(self, log, env=MERGED_ENV, smoke=None):
        return check.levern_problems(env, log, smoke)[0]

    def test_a_clean_merged_log_has_no_problem(self):
        log = merged_log(merged_route_lines('first', 9000) + merged_route_lines('second', 13000, hit=6144) + interleaved_steps('first', 9000))
        self.assertEqual(self.problems(log), [])

    def test_the_merged_warm_has_four_steps_and_the_stage_one_warm_three(self):
        self.assertTrue(any('route warm line with 4 steps' in problem for problem in self.problems(merged_log().replace('steps=4', 'steps=3'))))
        self.assertEqual(self.problems(merged_log()), [])
        self.assertEqual(check.levern_problems(ENV, engaged_log(), None)[0], [])

    def test_the_graft_must_say_it_runs_chunked_beside_the_cap(self):
        log = '\n'.join(line for line in merged_log().splitlines() if 'chunked=levern' not in line)
        self.assertTrue(any('chunked=levern' in problem and 'did not engage' in problem for problem in self.problems(log)))

    def test_a_hits_first_step_starts_at_its_q_and_the_ledger_continues_from_there(self):
        log = merged_log(merged_route_lines('second', 13000, hit=6144) + interleaved_steps('second', 13000))
        self.assertEqual(self.problems(log), [])
        lines = merged_route_lines('second', 13000, hit=6144)
        del lines[1]
        self.assertTrue(any('starts at 10240, the previous ended at 8192' in problem for problem in self.problems(merged_log(lines + interleaved_steps('second', 13000)))))

    def test_a_continuation_must_come_from_the_scratch_or_a_park_and_a_first_step_from_neither(self):
        wrong = chr(10).join([merged_route('r', 0, 2048, 8000, 'COLD'), merged_route('r', 2048, 4096, 8000, 'COLD')])
        self.assertTrue(any('step 2 came from source COLD' in problem for problem in check.levern_route_problems(check.levern_facts(wrong)['routes'])))
        first = merged_route('r', 0, 2048, 8000, 'SCRATCH')
        self.assertTrue(any('step 1 came from source SCRATCH' in problem for problem in check.levern_route_problems(check.levern_facts(first)['routes'])))
        resumed = '\n'.join([merged_route('r', 0, 2048, 8000, 'COLD'), merged_route('r', 2048, 4096, 8000, 'PARKED'), merged_route('r', 4096, 8000, 8000, 'SCRATCH')])
        self.assertEqual(check.levern_route_problems(check.levern_facts(resumed)['routes']), [])

    def test_a_park_restore_with_no_park_is_a_problem_and_a_pair_is_not(self):
        out = '[PINDIAG] lever N park out req=a at=6144 ms=90.0 bytes=154000000 parked_now=1'
        back = '[PINDIAG] lever N park in req=a at=6144 ms=95.0 bytes=154000000 parked_now=0'
        self.assertEqual(self.problems(merged_log([out, back])), [])
        self.assertTrue(any('park at 6144 that was never taken' in problem for problem in self.problems(merged_log([back]))))
        problems, facts = check.levern_problems(MERGED_ENV, merged_log([out, back]), None)
        self.assertEqual((facts['parks'], facts['unparks']), (1, 1))

    def test_the_warm_requests_park_is_not_counted(self):
        warm_out = '[PINDIAG] lever N park out req=__levern_warm__ at=4096 ms=90.0 bytes=154000000 parked_now=1'
        self.assertEqual(check.levern_problems(MERGED_ENV, merged_log([warm_out]), None)[1]['parks'], 0)

    def test_the_route_line_carries_the_source_and_the_captures_and_stage_one_lines_still_parse(self):
        facts = check.levern_facts(merged_route('a', 6144, 8192, 13000, 'CHECKPOINT', '8192:stored:80ms') + '\n' + route('b', 0, 2048, 6000))
        self.assertEqual([(row['source'], row['captured']) for row in facts['routes']], [('CHECKPOINT', '8192:stored:80ms'), (None, None)])
        self.assertEqual(check.levern_problems(ENV, engaged_log(route_lines('a', 6000) + interleaved_steps('a', 6000)), None)[0], [])

    def test_a_step_line_with_the_governed_share_still_parses(self):
        line = policy.STEP_LINE_MERGED.format(1, 'prefill', 7, 'a', 0, 2048, 2048, 10000, 0, 'paid', 'prefill', '700.0', '350', 0, 0.54)
        facts = check.levern_facts(line)
        self.assertEqual((facts['steps'][0]['req'], facts['steps'][0]['tokens']), ('a', 2048))


if __name__ == '__main__':
    unittest.main()
