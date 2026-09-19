import unittest

from lever_n_prefill_range import (DEFAULT_CHUNK_SIZE, plan_batch, plan_step,
                                   steps_for_prompt, validate_chunking)

CHUNK = DEFAULT_CHUNK_SIZE


class WholePromptTests(unittest.TestCase):
    def test_a_short_prompt_is_one_step_that_does_everything(self):
        plan = plan_step(0, 433, 433)
        self.assertTrue(plan.first and plan.last)
        self.assertEqual(plan.full_chunk_indices, ())   # no whole chunk in 433 tokens
        self.assertTrue(plan.resets_state and plan.emits_logits and plan.writes_slot)
        self.assertTrue(plan.runs_tail)
        self.assertEqual(plan.tail_tokens, 433)

    def test_an_exact_chunk_has_no_tail(self):
        plan = plan_step(0, CHUNK, CHUNK)
        self.assertTrue(plan.last)
        self.assertEqual(plan.full_chunk_indices, (0,))
        self.assertEqual(plan.tail_tokens, 0)
        self.assertFalse(plan.runs_tail)
        self.assertTrue(plan.writes_slot)


class ResumeTests(unittest.TestCase):
    def test_a_30k_prompt_splits_into_chunk_windows(self):
        windows = steps_for_prompt(30000)
        self.assertEqual(windows[0], (0, CHUNK))
        self.assertEqual(len(windows), 15)            # 14 whole chunks plus a tail
        self.assertEqual(windows[-1], (14 * CHUNK, 30000))

    def test_only_the_first_step_resets_and_only_the_last_emits(self):
        prompt_len = 30000
        plans = [plan_step(s, e, prompt_len) for s, e in steps_for_prompt(prompt_len)]
        self.assertEqual([p.resets_state for p in plans].count(True), 1)
        self.assertTrue(plans[0].resets_state)
        self.assertEqual([p.emits_logits for p in plans].count(True), 1)
        self.assertTrue(plans[-1].emits_logits)
        self.assertEqual([p.writes_slot for p in plans].count(True), 1)

    def test_every_whole_chunk_is_replayed_exactly_once(self):
        prompt_len = 163840
        seen = []
        for start, end in steps_for_prompt(prompt_len):
            seen.extend(plan_step(start, end, prompt_len).full_chunk_indices)
        self.assertEqual(seen, sorted(seen))
        self.assertEqual(len(seen), len(set(seen)), 'a chunk was replayed twice')
        self.assertEqual(seen, list(range(prompt_len // CHUNK)))

    def test_the_tail_runs_only_on_the_last_step(self):
        prompt_len = 30000
        plans = [plan_step(s, e, prompt_len) for s, e in steps_for_prompt(prompt_len)]
        self.assertEqual([p.runs_tail for p in plans].count(True), 1)
        self.assertTrue(plans[-1].runs_tail)
        self.assertEqual(plans[-1].tail_tokens, 30000 - 14 * CHUNK)

    def test_preemption_resumes_as_a_fresh_first_chunk(self):
        """The plugin replays prompt plus generated tokens, so start==0 is correct."""
        plan = plan_step(0, CHUNK, 30000)
        self.assertTrue(plan.resets_state)
        self.assertFalse(plan.emits_logits)


class InvariantTests(unittest.TestCase):
    def test_continuations_must_start_on_a_chunk_boundary(self):
        for start in (1, 100, CHUNK - 1, CHUNK + 1):
            with self.assertRaises(ValueError):
                plan_step(start, 30000, 30000)

    def test_a_non_final_step_must_end_on_a_chunk_boundary(self):
        with self.assertRaises(ValueError):
            plan_step(0, 1000, 30000)
        plan_step(0, CHUNK, 30000)        # boundary end is fine
        plan_step(14 * CHUNK, 30000, 30000)  # a final step may end mid-chunk

    def test_degenerate_windows_rejected(self):
        for start, end, prompt_len in ((0, 0, 100), (5, 5, 100), (0, 101, 100),
                                       (-1, 10, 100), (0, 10, 0)):
            with self.assertRaises(ValueError):
                plan_step(start, end, prompt_len)

    def test_scheduler_budget_must_equal_the_chunk_size(self):
        self.assertTrue(validate_chunking(CHUNK))
        for budget in (1024, 4096, 2047, None, '2048'):
            with self.assertRaises(ValueError):
                validate_chunking(budget)

    def test_batch_planning_requires_matching_lengths(self):
        plans = plan_batch([0, 0], [CHUNK, 433], [30000, 433])
        self.assertFalse(plans[0].last)
        self.assertTrue(plans[1].last)
        with self.assertRaises(ValueError):
            plan_batch([0], [CHUNK, 433], [30000, 433])


if __name__ == '__main__':
    unittest.main()
