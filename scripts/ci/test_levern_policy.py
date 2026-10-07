"""Lever N at TP4: the chunking arithmetic, the alternation and the flags (levern_policy). Stdlib only, no device."""

import unittest

import levern_policy as policy
from levern_policy import CHUNK

BOUNDARY_PROMPTS = (1, 2047, 2048, 2049, 4095, 4096, 4097, 6143, 6144, 6145, 8191, 8192, 32785, 131077, 253920, 262111)


class FinalStartTests(unittest.TestCase):
    def test_the_final_step_start_at_the_boundaries(self):
        expected = {1: 0, 2047: 0, 2048: 0, 2049: 0, 4095: 0, 4096: 2048, 4097: 2048, 6143: 2048, 6144: 4096,
                    6145: 4096, 8191: 4096, 8192: 6144, 253920: 122 * CHUNK}
        for prompt, start in expected.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(policy.final_start(prompt), start)

    def test_253920_is_a_123_chunk_prompt_with_a_two_chunk_tail_free_split(self):
        # 253,920 = 123 x 2048 + 2,016: the last full chunk starts at 122 x 2048 and the tail follows it
        self.assertEqual(253920 // CHUNK, 123)
        self.assertEqual(policy.final_start(253920), 122 * CHUNK)

    def test_a_prompt_below_two_chunks_is_never_split(self):
        for prompt in range(1, 2 * CHUNK):
            self.assertEqual(policy.final_start(prompt), 0, prompt)

    def test_the_final_step_is_at_most_4095_tokens_and_holds_the_draft_window(self):
        for prompt in list(range(1, 3 * CHUNK + 5)) + [10 * CHUNK + k for k in (0, 1, 2047)] + list(BOUNDARY_PROMPTS):
            start = policy.final_start(prompt)
            self.assertLessEqual(prompt - start, 4095, prompt)
            if prompt >= 2 * CHUNK and prompt % CHUNK == 0:
                self.assertEqual(prompt - start, CHUNK, prompt)       # an exact multiple: just the last full chunk
            # the drafter's window [P - 2048, P) lies inside the final step
            self.assertLessEqual(start, max(0, prompt - CHUNK), prompt)
            self.assertEqual(start % CHUNK, 0, prompt)

    def test_bad_prompts_are_refused(self):
        for bad in (0, -1, 1.0, None, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                policy.final_start(bad)


class PlanTests(unittest.TestCase):
    def check(self, prompt, steps):
        self.assertEqual(steps[0][0], 0)
        self.assertEqual(steps[-1][1], prompt)
        for (start, end), (next_start, _) in zip(steps, steps[1:]):
            self.assertEqual(end, next_start, 'no gap and no replay')
            self.assertEqual(end % CHUNK, 0, 'every non-final end is on the model chunk boundary: %r' % ((start, end),))
        for start, end in steps:
            self.assertEqual(start % CHUNK, 0)
            self.assertGreater(end, start)
        final_start, final_end = steps[-1]
        self.assertLessEqual(final_end - final_start, 4095)
        # the draft window is inside the final step
        self.assertLessEqual(final_start, max(0, prompt - CHUNK))

    def test_every_boundary_prompt_with_decoders(self):
        for prompt in BOUNDARY_PROMPTS:
            with self.subTest(prompt=prompt):
                self.check(prompt, policy.plan(prompt, decoding=True))

    def test_every_boundary_prompt_with_no_decoder(self):
        for prompt in BOUNDARY_PROMPTS:
            with self.subTest(prompt=prompt):
                self.check(prompt, policy.plan(prompt, decoding=False))

    def test_a_dense_sweep_of_prompt_lengths(self):
        for prompt in range(1, 12 * CHUNK + 3, 7):
            self.check(prompt, policy.plan(prompt, decoding=True))
        for prompt in range(1, 40 * CHUNK + 3, 211):
            self.check(prompt, policy.plan(prompt, decoding=False))

    def test_short_prompts_run_whole(self):
        for prompt in (1, 100, 2047, 2048, 2049, 4095):
            self.assertEqual(policy.plan(prompt), [(0, prompt)], prompt)

    def test_4096_is_two_chunks(self):
        self.assertEqual(policy.plan(4096), [(0, 2048), (2048, 4096)])
        self.assertEqual(policy.plan(4097), [(0, 2048), (2048, 4097)])
        self.assertEqual(policy.plan(6145), [(0, 2048), (2048, 4096), (4096, 6145)])

    def test_the_cold_254k_prompt(self):
        steps = policy.plan(253920)
        self.assertEqual(len(steps), 123)                      # 122 chunk steps and the final step
        self.assertEqual(steps[-1], (122 * CHUNK, 253920))     # the last full chunk and the 2,016-token tail
        self.assertEqual(steps[-1][1] - steps[-1][0], 2048 + 2016)
        self.assertTrue(all(end - start == CHUNK for start, end in steps[:-1]))

    def test_solo_steps_are_longer_and_bounded(self):
        steps = policy.plan(253920, decoding=False)
        self.assertEqual(steps[0], (0, policy.DEFAULT_SOLO))
        self.assertTrue(all(end - start <= policy.DEFAULT_SOLO for start, end in steps[:-1]))
        self.assertEqual(steps[-1][0], 122 * CHUNK)

    def test_step_tokens_scale_the_steps(self):
        steps = policy.plan(65536, step=4 * CHUNK)
        self.assertEqual(steps[0], (0, 4 * CHUNK))
        self.assertTrue(all((end - start) % CHUNK == 0 for start, end in steps[:-1]))

    def test_another_waiting_prompt_keeps_the_short_step(self):
        self.assertEqual(policy.step_budget(65536, 0, decoding=False, others_waiting=1), CHUNK)
        self.assertEqual(policy.step_budget(65536, 0, decoding=False, others_waiting=0), policy.DEFAULT_SOLO)
        self.assertEqual(policy.step_budget(65536, 0, decoding=True, others_waiting=0), CHUNK)

    def test_an_unaligned_start_is_refused(self):
        for computed in (1, 1024, 2049, 4000):
            with self.subTest(computed=computed), self.assertRaises(ValueError):
                policy.step_budget(65536, computed, decoding=True)

    def test_an_unaligned_start_inside_the_final_step_is_taken_whole(self):
        # the final step takes what is left, wherever it starts (the model's own entry refuses a misaligned start)
        self.assertEqual(policy.step_budget(6145, 4096, decoding=True), 2049)

    def test_a_computed_outside_the_prompt_is_refused(self):
        for computed in (-1, 65536, 70000, 1.0, None):
            with self.subTest(computed=computed), self.assertRaises(ValueError):
                policy.step_budget(65536, computed, decoding=True)

    def test_bad_steps_are_refused(self):
        for step in (0, 1024, 3000, 2048.0, None):
            with self.subTest(step=step), self.assertRaises(ValueError):
                policy.step_budget(65536, 0, decoding=True, step=step)


class FlagTests(unittest.TestCase):
    def test_the_default_is_off_and_unconfigured(self):
        self.assertFalse(policy.enabled({}))
        self.assertFalse(policy.audit_enabled({}))
        self.assertEqual(policy.config({}), policy.Config(2048, 16384, 0.5, None, 8))
        self.assertEqual(policy.config_problems({}), [])
        self.assertIsNone(policy.fault({}))

    def test_the_master_switch_is_strict(self):
        self.assertTrue(policy.enabled({'QWEN_FAST_LEVER_N': '1'}))
        self.assertFalse(policy.enabled({'QWEN_FAST_LEVER_N': '0'}))
        for bad in ('', 'true', '2', ' 1', 'on'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                policy.enabled({'QWEN_FAST_LEVER_N': bad})

    def test_the_audit_switch_is_strict_and_stands_alone(self):
        self.assertTrue(policy.audit_enabled({'QWEN_FAST_LEVERN_AUDIT': '1'}))
        self.assertEqual(policy.config_problems({'QWEN_FAST_LEVERN_AUDIT': '1'}), [])
        with self.assertRaises(ValueError):
            policy.audit_enabled({'QWEN_FAST_LEVERN_AUDIT': 'yes'})

    def test_step_tokens_must_be_a_multiple_of_the_model_chunk(self):
        for value in ('1024', '3000', '2049', '0', '2048.0', '-2048', ' 2048', '', '0x800', '4096\n'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                policy.config({'QWEN_FAST_LEVERN_STEP_TOKENS': value})
        self.assertEqual(policy.config({'QWEN_FAST_LEVERN_STEP_TOKENS': '4096'}).step, 4096)
        self.assertEqual(policy.config({'QWEN_FAST_LEVERN_SOLO_STEP_TOKENS': '32768'}).solo, 32768)

    def test_solo_is_not_below_step(self):
        with self.assertRaises(ValueError):
            policy.config({'QWEN_FAST_LEVERN_STEP_TOKENS': '8192', 'QWEN_FAST_LEVERN_SOLO_STEP_TOKENS': '4096'})

    def test_share_is_in_zero_one(self):
        for value in ('0', '0.0', '1.5', '2', 'nan', 'inf', '-0.5', '1e-1', ' 0.5', '', 'half'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                policy.config({'QWEN_FAST_LEVERN_PREFILL_SHARE': value})
        for value, want in (('0.5', 0.5), ('1', 1.0), ('1.0', 1.0), ('.33', 0.33), ('0.25', 0.25)):
            self.assertEqual(policy.config({'QWEN_FAST_LEVERN_PREFILL_SHARE': value}).share, want)

    def test_rounds_and_max_rounds(self):
        self.assertEqual(policy.config({'QWEN_FAST_LEVERN_ROUNDS': '1'}).rounds, 1)
        self.assertEqual(policy.config({'QWEN_FAST_LEVERN_MAX_ROUNDS': '16'}).max_rounds, 16)
        for name in ('QWEN_FAST_LEVERN_ROUNDS', 'QWEN_FAST_LEVERN_MAX_ROUNDS'):
            for value in ('0', '65', '-1', '1.5', 'x', ''):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    policy.config({name: value})

    def test_fault_names_are_closed(self):
        self.assertEqual(policy.fault({'QWEN_FAST_LEVERN_FAULT': 'foreign'}), 'foreign')
        with self.assertRaises(ValueError):
            policy.config({'QWEN_FAST_LEVERN_FAULT': 'other'})

    def test_a_sibling_without_the_master_switch_is_a_problem(self):
        for name in policy.SIBLING_FLAGS:
            value = {'QWEN_FAST_LEVERN_FAULT': 'foreign', 'QWEN_FAST_LEVERN_PARK': 'host',
                     'QWEN_FAST_LEVERN_EPOCH_SCOPE': 'route'}.get(name, '2048' if 'TOKENS' in name else '1')
            with self.subTest(name=name):
                problems = policy.config_problems({name: value})
                self.assertTrue(problems, name)
                self.assertIn('QWEN_FAST_LEVER_N', problems[0])
                self.assertEqual(policy.config_problems({name: value, 'QWEN_FAST_LEVER_N': '1'}), [])

    def test_a_malformed_value_is_a_problem_not_a_default(self):
        problems = policy.config_problems({'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_STEP_TOKENS': '3000'})
        self.assertTrue(problems and 'multiple of 2048' in problems[0])


class Clock(object):
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance_ms(self, ms):
        self.now += ms / 1000.0


def run_steps(alternator, clock, chunks, chunk_ms, round_ms, decodes=7, pending_after_last=False):
    """Drive the alternator through `chunks` prefill steps of `chunk_ms`, with decode steps of `round_ms` whenever the
    alternator yields. Returns the list of (kind, ms) in order."""
    done, order = 0, []
    guard = 0
    while done < chunks:
        guard += 1
        assert guard < 10000
        decision = alternator.begin(decodes, True)
        kind = decision.kind
        order.append(kind)
        alternator.end(kind)
        clock.advance_ms(round_ms if kind == 'decode' else chunk_ms)
        if kind == 'prefill':
            done += 1
    return order


class AlternatorTests(unittest.TestCase):
    def make(self, **flags):
        clock = Clock()
        environ = {key: str(value) for key, value in flags.items()}
        return policy.Alternator(policy.config(environ), clock=clock), clock

    def test_the_first_step_is_never_delayed(self):
        alternator, clock = self.make()
        decision = alternator.begin(7, True)
        self.assertEqual(decision.kind, 'prefill')

    def test_half_share_gives_about_one_to_one_wall_time(self):
        alternator, clock = self.make(QWEN_FAST_LEVERN_PREFILL_SHARE='0.5', QWEN_FAST_LEVERN_MAX_ROUNDS=64)
        order = run_steps(alternator, clock, chunks=50, chunk_ms=1000.0, round_ms=250.0)
        decode_ms, prefill_ms = 250.0 * order.count('decode'), 1000.0 * order.count('prefill')
        # a decode step pays its own time: at f = 0.5 the decoders get about as much wall time as the prefill
        self.assertAlmostEqual(decode_ms / prefill_ms, 1.0, delta=0.25)

    def test_a_third_share_gives_the_decoders_twice_the_prefill_time(self):
        alternator, clock = self.make(QWEN_FAST_LEVERN_PREFILL_SHARE='0.33', QWEN_FAST_LEVERN_MAX_ROUNDS=64)
        order = run_steps(alternator, clock, chunks=50, chunk_ms=1000.0, round_ms=250.0)
        decode_ms, prefill_ms = 250.0 * order.count('decode'), 1000.0 * order.count('prefill')
        self.assertAlmostEqual(decode_ms / prefill_ms, 0.67 / 0.33, delta=0.4)

    def test_full_share_runs_chunks_back_to_back(self):
        alternator, clock = self.make(QWEN_FAST_LEVERN_PREFILL_SHARE='1')
        order = run_steps(alternator, clock, chunks=20, chunk_ms=1000.0, round_ms=250.0)
        self.assertEqual(order, ['prefill'] * 20)

    def test_static_rounds_run_exactly_r_decode_steps_between_prefill_steps(self):
        for rounds in (1, 2, 3):
            alternator, clock = self.make(QWEN_FAST_LEVERN_ROUNDS=rounds)
            order = run_steps(alternator, clock, chunks=6, chunk_ms=700.0, round_ms=245.0)
            self.assertEqual(order, ['prefill'] + (['decode'] * rounds + ['prefill']) * 5, rounds)

    def test_static_rounds_override_the_share(self):
        alternator, clock = self.make(QWEN_FAST_LEVERN_ROUNDS=1, QWEN_FAST_LEVERN_PREFILL_SHARE='1')
        order = run_steps(alternator, clock, chunks=4, chunk_ms=700.0, round_ms=245.0)
        self.assertEqual(order, ['prefill', 'decode', 'prefill', 'decode', 'prefill', 'decode', 'prefill'])

    def test_at_least_one_decode_step_per_prefill_step_when_the_share_is_below_one(self):
        alternator, clock = self.make(QWEN_FAST_LEVERN_PREFILL_SHARE='0.99')
        order = run_steps(alternator, clock, chunks=10, chunk_ms=1.0, round_ms=250.0)
        for first, second in zip(order, order[1:]):
            self.assertFalse(first == second == 'prefill', 'two prefill steps back to back at f < 1')

    def test_with_no_decoder_nothing_is_owed_and_nothing_yields(self):
        alternator, clock = self.make(QWEN_FAST_LEVERN_ROUNDS=3)
        order = run_steps(alternator, clock, chunks=5, chunk_ms=900.0, round_ms=245.0, decodes=0)
        self.assertEqual(order, ['prefill'] * 5)

    def test_the_last_decoder_leaving_drops_what_is_owed(self):
        alternator, clock = self.make(QWEN_FAST_LEVERN_ROUNDS=3)
        self.assertEqual(alternator.begin(7, True).kind, 'prefill')
        alternator.end('prefill')
        clock.advance_ms(900)
        self.assertEqual(alternator.begin(7, True).kind, 'decode')
        alternator.end('decode')
        clock.advance_ms(245)
        decision = alternator.begin(0, True)        # the decoders all finished meanwhile
        self.assertEqual(decision.kind, 'prefill')
        self.assertEqual(alternator.owed_rounds, 0)

    def test_what_is_owed_is_capped_at_kmax_rounds(self):
        alternator, clock = self.make(QWEN_FAST_LEVERN_PREFILL_SHARE='0.5', QWEN_FAST_LEVERN_MAX_ROUNDS=8)
        alternator.begin(7, True)
        alternator.end('prefill')
        clock.advance_ms(4000)                      # a 4 s final step with its engine build
        decision = alternator.begin(7, True)
        self.assertEqual(decision.kind, 'decode')
        self.assertLessEqual(decision.owed_ms, 8 * policy.INITIAL_ROUND_MS + 1e-6)
        kinds = [decision.kind]
        alternator.end('decode')
        for _ in range(40):
            clock.advance_ms(250)
            decision = alternator.begin(7, True)
            kinds.append(decision.kind)
            alternator.end(decision.kind)
            if decision.kind == 'prefill':
                break
        self.assertLessEqual(kinds.count('decode'), 9, 'a 4 s step does not owe 16 rounds')

    def test_nothing_pending_resets_the_account(self):
        alternator, clock = self.make(QWEN_FAST_LEVERN_ROUNDS=2)
        alternator.begin(7, True)
        alternator.end('prefill')
        clock.advance_ms(800)
        decision = alternator.begin(7, False)       # the prefill finished: plain decoding
        self.assertEqual(decision.kind, 'prefill')
        self.assertEqual((alternator.owed_ms, alternator.owed_rounds), (0.0, 0))
        alternator.end('decode')
        clock.advance_ms(250)
        # the next arrival starts a fresh account and is admitted at once
        self.assertEqual(alternator.begin(7, True).kind, 'prefill')

    def test_a_prefill_step_with_no_decoder_owes_nothing_later(self):
        alternator, clock = self.make(QWEN_FAST_LEVERN_ROUNDS=2)
        alternator.begin(0, True)
        alternator.end('prefill')
        clock.advance_ms(900)
        # a decoder arrives (an earlier prompt's seat joined): the step that ran with none owed it nothing
        self.assertEqual(alternator.begin(1, True).kind, 'prefill')

    def test_a_yield_is_a_decode_decision_and_an_unknown_kind_is_refused(self):
        alternator, clock = self.make(QWEN_FAST_LEVERN_ROUNDS=1)
        with self.assertRaises(ValueError):
            alternator.end('both')

    def test_the_decision_carries_the_previous_step_wall_time(self):
        alternator, clock = self.make(QWEN_FAST_LEVERN_ROUNDS=1)
        alternator.begin(7, True)
        alternator.end('prefill')
        clock.advance_ms(812.5)
        decision = alternator.begin(7, True)
        self.assertEqual(decision.prev_kind, 'prefill')
        self.assertAlmostEqual(decision.prev_ms, 812.5, places=3)
        self.assertEqual(decision.owed_rounds, 1)


if __name__ == '__main__':
    unittest.main()
