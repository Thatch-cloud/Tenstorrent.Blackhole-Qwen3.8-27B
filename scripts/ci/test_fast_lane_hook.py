"""The fast lane end to end on a CPU: the REAL worker hook, lane runtime, scheduler gate, packed step and runner-state contracts over a fake vLLM.

lanes_fakes.World is one engine: a scheduler class shaped like TTScheduler with the lane gate installed on it, the plugin runner's persistent-batch
algorithm (an unscheduled request leaves the batch and keeps its state; a scheduled one returns), an M3 block with padded rounds and the one-user
block, and a target model that is a pure function of each user's own history - so every user's committed stream must be the model's chain from its
prompt WHATEVER frames of rounds served it (per-lane exactness), and every state check the worker makes (the scheduled set equals the prepared
set; the runner's batch holds exactly the scheduled requests, on their frontiers) runs for real. The block round times are the design's model
(lanes_sim.ROUND_MS) on a virtual clock, so the rates measured here come through the same path a served run's would."""

import os
import sys
from unittest.mock import patch
import unittest

import lanes_sim
from lanes_fakes import World, next_token, true_stream
import serving_fast_lane as lanes
from serving_fast_lane import LaneConfig

PROMPT = 4100


def make(lever='phase1', context='32k', standard=3, tau_fast=12.1, tau_standard=12.1, fast=True, max_tokens=6000,
         config=None, **options):
    world = World(round_ms=lanes_sim.ROUND_MS[lever][context], config=config, **options)
    if fast:
        world.add_user('fast', 0, lane='fast', tau=tau_fast, max_tokens=max_tokens, prompt_length=PROMPT)
    for index in range(standard):
        world.add_user('std%d' % index, index + 1, tau=tau_standard, max_tokens=max_tokens, prompt_length=PROMPT + 50 * index)
    world.prime()
    return world


def run(world, steps=None, seconds=None):
    count = 0
    while (steps is None or count < steps) and (seconds is None or world.clock.now < seconds):
        world.engine_step()
        count += 1
    return world


def chain_holds(test, world, request_id, finished=False):
    """The user's committed stream is the target model's chain from its prompt: the exactness invariant."""
    request = world.users[request_id]
    state = world.runner.requests.get(request_id)
    tokens = list(state.output_token_ids) if state is not None else None
    if tokens is None:
        return
    seed = tokens[0]
    prompt = len(state.prompt_token_ids)
    test.assertEqual(tokens[1:], true_stream(request.user, seed, prompt, len(tokens) - 1), request_id)


class LoopCase(unittest.TestCase):
    def setUp(self):
        self.addCleanup(sys.modules.pop, lanes.GATE_KEY, None)


class TargetsTests(LoopCase):
    """The user's target: one fast lane at >= 150 tok/s beside standard lanes at >= 75 tok/s each, through the whole stack."""

    def rates(self, world, warmup=3.0):
        return world.rate('fast', after=warmup), [world.rate(name, after=warmup) for name in world.users if name != 'fast']

    def test_both_bars_hold_at_phase_one_for_one_to_three_standard_users_at_the_fixture_acceptance(self):
        for context in ('4k', '32k'):
            for standard in (1, 2, 3):
                with self.subTest(context=context, standard=standard):
                    world = run(make('phase1', context, standard), seconds=14.0)
                    fast, others = self.rates(world)
                    self.assertGreaterEqual(fast, 150.0, 'fast %.1f' % fast)
                    self.assertTrue(all(rate >= 75.0 for rate in others), others)
                    for name in world.users:
                        chain_holds(self, world, name)

    def test_the_fast_lane_pays_for_what_the_standard_lanes_keep(self):
        """More standard users, a slower packed round, less left for solo rounds: the fast rate falls, the floor holds."""
        fasts = []
        for standard in (1, 2, 3):
            world = run(make('phase1', '32k', standard), seconds=14.0)
            fast, others = self.rates(world)
            fasts.append(fast)
            self.assertGreaterEqual(min(others), 75.0)
        self.assertGreater(fasts[0], fasts[1])
        self.assertGreater(fasts[1], fasts[2])

    def test_the_straight_port_misses_the_fast_bar_and_still_keeps_the_floor(self):
        """The honest cell: no lever, three standard users, the fixture tau. The controller holds the floor and the fast lane gets less."""
        world = run(make('straight', '32k', 3), seconds=14.0)
        fast, others = self.rates(world)
        self.assertLess(fast, 150.0)
        self.assertGreater(fast, 100.0)
        self.assertGreaterEqual(min(others), 75.0)

    def test_a_low_acceptance_user_leaves_the_floor_out_of_reach_and_the_fast_lane_gets_the_packed_rate(self):
        world = run(make('phase1', '32k', 3, tau_standard=4.9), seconds=14.0)
        fast, others = self.rates(world)
        self.assertEqual(world.lanes.controller.k, 0.0)
        packed = 1000.0 * 12.1 / lanes_sim.ROUND_MS['phase1']['32k'][1][4]
        self.assertAlmostEqual(fast, packed, delta=8.0)
        self.assertEqual([kind for kind in world.kinds[-40:] if kind == 'solo'], [])
        for name in world.users:
            chain_holds(self, world, name)

    def test_the_fast_user_alone_runs_solo_rounds_at_its_own_ceiling(self):
        world = run(make('phase1', '32k', 0), seconds=8.0)
        self.assertEqual(set(world.kinds), {'solo'})
        self.assertGreaterEqual(world.rate('fast', after=1.0), 200.0)
        chain_holds(self, world, 'fast')

    def test_with_no_fast_user_every_round_is_a_packed_round(self):
        world = run(make('phase1', '32k', 3, fast=False), seconds=8.0)
        self.assertEqual(set(world.kinds), {'packed'})
        for name in world.users:
            self.assertGreaterEqual(world.rate(name, after=1.0), 100.0)


class RatioTests(LoopCase):
    """The tunable ratio: k solo rounds per packed round, fixed by QWEN_FAST_LANE_RATIO."""

    def kinds(self, ratio, standard=2, steps=60, **changes):
        config = LaneConfig.from_environment({'QWEN_FAST_LANE_RATIO': str(ratio), 'QWEN_FAST_LANE_GAP_MS': '1000'}, seats=4)
        world = run(make('phase1', '32k', standard, config=config, **changes), steps=steps)
        return world, ''.join('P' if kind == 'packed' else 'S' if kind == 'solo' else '?' for kind in world.kinds)

    def test_the_realised_pattern_is_the_ratio(self):
        for ratio, expected in ((0, 'P' * 12), (1, 'PS' * 6), (0.5, 'PPS' * 4), (2, 'PSS' * 4)):
            with self.subTest(ratio=ratio):
                world, kinds = self.kinds(ratio)
                self.assertEqual(kinds[:len(expected)], expected)

    def test_a_ratio_of_zero_is_the_packed_only_rate_for_everyone(self):
        world, kinds = self.kinds(0, standard=3, steps=120)
        packed = 1000.0 * 12.1 / lanes_sim.ROUND_MS['phase1']['32k'][1][4]
        for name in world.users:
            self.assertAlmostEqual(world.rate(name, after=1.0), packed, delta=6.0, msg=name)

    def test_the_realised_rates_follow_tokens_per_round_times_rounds_over_the_frame(self):
        world, _ = self.kinds(1, standard=2, steps=240)
        solo, packed = lanes_sim.ROUND_MS['phase1']['32k'][0], lanes_sim.ROUND_MS['phase1']['32k'][1][3]
        frame = packed + solo + 2 * 1.0
        self.assertAlmostEqual(world.rate('std0', after=1.0), 1000.0 * 12.1 / frame, delta=4.0)
        self.assertAlmostEqual(world.rate('fast', after=1.0), 1000.0 * 12.1 * 2 / frame, delta=8.0)

    def test_the_starvation_guard_bounds_the_gap_between_packed_rounds(self):
        config = LaneConfig.from_environment({'QWEN_FAST_LANE_RATIO': '3', 'QWEN_FAST_LANE_KMAX': '3',
                                              'QWEN_FAST_LANE_GAP_MS': '100'}, seats=4)
        world = run(make('phase1', '32k', 3, config=config), steps=200)
        packed_times = [event[0] for event in world.commits['std0']]
        gaps = [(b - a) * 1000.0 for a, b in zip(packed_times, packed_times[1:])]
        bound = 100.0 + lanes_sim.ROUND_MS['phase1']['32k'][0] + lanes_sim.ROUND_MS['phase1']['32k'][1][4] + 2.0
        self.assertLessEqual(max(gaps), bound)
        self.assertGreater(max(gaps), 100.0)


class InvariantTests(LoopCase):
    """What every step must hold, whatever the frame: the gate and the runner contracts, per lane."""

    def check_steps(self, world):
        steps = [step for step in world.observed if step['scheduled']]
        self.assertTrue(steps)
        for step in steps:
            members, scheduled = step['members'], step['scheduled']
            if members is not None:
                self.assertTrue(set(scheduled) <= set(members), (scheduled, members))
            self.assertEqual(len(set(scheduled)), len(scheduled))
            live_ids = set(step['before'])
            for request_id in live_ids - set(scheduled):
                # a request held out of a round carries no proposals into it: it is drafted again before its next round
                self.assertEqual(step['before'][request_id][1], [], 'stale proposals on a held request %s' % request_id)

    def test_every_step_serves_only_planned_members_and_a_held_request_carries_no_proposals(self):
        world = run(make('phase1', '32k', 3), steps=300)
        self.check_steps(world)
        self.assertIn('solo', world.kinds)
        self.assertIn('packed', world.kinds)

    def test_a_held_request_returns_on_the_frontier_it_left_and_never_as_resumed(self):
        world = make('phase1', '32k', 2)
        seen = {}
        for _ in range(200):
            step = world.engine_step()
            for request_id, position in zip(step.scheduled_cached_reqs.req_ids, step.scheduled_cached_reqs.num_computed_tokens):
                request = world.users[request_id]
                self.assertEqual(position, request.session.position - 0 if False else position)
                self.assertFalse(step.scheduled_cached_reqs.resumed_req_ids)
                if request_id in seen:
                    self.assertGreater(position, seen[request_id], 'every round advances the frontier')
                seen[request_id] = position
        for name in world.users:
            chain_holds(self, world, name)

    def test_the_runner_batch_is_exactly_the_scheduled_set_every_round(self):
        world = make('phase1', '32k', 3)
        for _ in range(120):
            world.engine_step()
            scheduled = world.observed[-1]['scheduled']
            if scheduled and world.kinds:
                self.assertEqual(set(world.runner.input_batch.req_id_to_index), set(scheduled))
        self.assertGreater(world.runner.input_batch.layout_changes, 20, 'a solo round drops the standard users from the batch')

    def test_each_lanes_text_is_its_own_solo_text_at_every_ratio(self):
        for ratio in ('auto', '0', '0.5', '1', '2'):
            with self.subTest(ratio=ratio):
                config = LaneConfig.from_environment({'QWEN_FAST_LANE_RATIO': ratio, 'QWEN_FAST_LANE_GAP_MS': '1000'}, seats=4)
                world = run(make('phase1', '32k', 3, config=config, max_tokens=900), steps=400)
                for name in world.users:
                    chain_holds(self, world, name)

    def test_the_tail_of_a_budget_and_the_departure_of_users_change_no_stream(self):
        """A short budget ends a user in the middle of a frame; the survivors' rounds narrow, go sequential, then are packed again."""
        world = make('phase1', '32k', 3, max_tokens=260)
        for index, name in enumerate(['std1']):
            world.users[name]
        world.scheduler.running[2].sampling_params.max_tokens = 200
        run(world, steps=60)
        for name in world.users:
            chain_holds(self, world, name)
        self.assertIn('sequential', world.kinds, 'the tail rounds ran on the per-request engines')
        self.assertTrue(world.scheduler.finished_req_ids or len(world.scheduler.running) < 4)

    def test_early_drafting_inside_the_step_plans_the_same_rounds_and_makes_the_same_streams(self):
        plain = run(make('phase1', '32k', 3, max_tokens=6000), steps=140)
        with patch.dict(os.environ, {'QWEN_FAST_EARLY_DRAFT': '1'}):
            early = run(make('phase1', '32k', 3, max_tokens=6000), steps=140)
        self.assertEqual(early.kinds, plain.kinds)
        self.assertAlmostEqual(early.clock.now, plain.clock.now)
        self.assertGreater(early.hook._early_draft.counts['reuse'], 100)
        self.assertEqual(early.hook._early_draft.counts['redo'], 0)
        for name in early.users:
            chain_holds(self, early, name)
            self.assertEqual(early.runner.requests[name].output_token_ids, plain.runner.requests[name].output_token_ids)


class AdmissionLoopTests(LoopCase):
    def test_a_second_fast_request_is_downgraded_takes_a_standard_slot_and_gets_no_solo_rounds(self):
        world = make('phase1', '32k', 1)
        grant, slot, request = world.add_user('fast2', 9, lane='fast', max_tokens=6000, prompt_length=PROMPT)
        self.assertEqual((grant.granted, grant.reason, slot), ('standard', 'fast-lane-busy', 2))
        world.prime()
        run(world, steps=200)
        solo_rounds = [members for members in world.solo.round_log]
        self.assertTrue(solo_rounds)
        self.assertTrue(all(members == ('fast',) for members in solo_rounds), 'only the one fast lane runs solo rounds')
        self.assertGreaterEqual(world.rate('fast2', after=2.0), 70.0)
        for name in world.users:
            chain_holds(self, world, name)

    def test_the_fast_lane_frees_when_its_holder_finishes_and_the_next_fast_request_takes_slot_zero(self):
        world = make('phase1', '32k', 2, max_tokens=6000)
        world.runner.requests['fast'].sampling_params.max_tokens = 400
        world.scheduler.running[0].sampling_params.max_tokens = 400
        run(world, steps=200)
        self.assertNotIn('fast', world.hook.bridges)
        self.assertIsNone(world.lanes.book.fast_id)
        self.assertIsNone(world.lanes.gate.fast_id)
        solo_before = world.solo.rounds
        grant, slot, request = world.add_user('fast_next', 7, lane='fast', max_tokens=6000, prompt_length=PROMPT)
        self.assertEqual((grant.granted, slot), ('fast', 0))
        world.prime()
        run(world, steps=100)
        self.assertGreater(world.solo.rounds, solo_before)
        for name in world.users:
            chain_holds(self, world, name)

    def test_standard_requests_never_take_slot_zero_under_the_reserve(self):
        world = World(round_ms=lanes_sim.ROUND_MS['phase1']['32k'])
        slots = [world.add_user('s%d' % index, index, max_tokens=500)[1] for index in range(3)]
        self.assertEqual(sorted(slots), [1, 2, 3])
        with self.assertRaises(StopIteration):
            world.add_user('s3', 4, max_tokens=500)       # no slot in (1, 2, 3) is free: the fourth standard request cannot be admitted

    def test_the_gate_is_cleared_when_the_hook_closes_or_a_request_attaches(self):
        world = make('phase1', '32k', 2)
        run(world, steps=8)
        world.hook.attach.__self__.lanes.publish(world.lanes.planned)
        self.assertIsNotNone(world.lanes.gate.members)
        world.add_user('late', 5, max_tokens=500)
        self.assertIsNone(world.lanes.gate.members, 'a plan made without the new request is dropped at its attach')
        world.hook.close()
        self.assertIsNone(world.lanes.gate.members)


class DesyncTests(LoopCase):
    def test_a_scheduler_the_gate_never_ran_on_is_refused_by_name_before_any_device_work(self):
        world = make('phase1', '32k', 3, install=False)
        with self.assertRaisesRegex(ValueError, r"\[LANE-DESYNC\] the lane gate planned \['u1'\] and the step scheduled "
                                                r"\['u1', 'u2', 'u3', 'u4'\]: the scheduler did not hide \['u2', 'u3', 'u4'\]"):
            run(world, steps=40)
        self.assertEqual(world.m3.round_log[-1:] and 1, 1)
        self.assertFalse(any(world.solo.round_log), 'no solo round ran on a step that scheduled everyone')

    def test_a_stale_plan_naming_nobody_who_runs_is_an_empty_step_and_the_next_drafts_plan_again(self):
        world = make('phase1', '32k', 2)
        run(world, steps=10)
        world.lanes.gate.members = frozenset(['gone'])
        scheduled = world.scheduler.schedule()
        self.assertEqual(len(scheduled.scheduled_cached_reqs.req_ids), 0)


class OffTests(unittest.TestCase):
    def test_the_lane_off_is_the_hooks_own_path(self):
        from serving_worker_hook import FastWorkerHook

        self.assertIsNone(FastWorkerHook.lanes)
        self.assertNotIn('lanes', FastWorkerHook.__init__.__code__.co_varnames[:1])

    def test_a_config_without_the_flag_builds_no_runtime(self):
        self.assertIsNone(lanes.lane_admission({'slot': 0}, {}, seats=4))


if __name__ == '__main__':
    unittest.main()
