"""Eight seats on two M3 blocks (QWEN_FAST_M3_BLOCKS=2), the whole host stack on fakes (A3's whole-stack test).

The REAL worker hook (the per-block drafting: widths, stale-ticket discards, budget narrowing), the REAL packed step over two M3
blocks with per-block widths, the real round partition (packed_device_rounds, the sequential fallback for a block that cannot run
packed, S2 D1 narrowing), execute_packed_decode, the runner-state contracts (the persistent batch must hold exactly the scheduled
requests on their frontiers) and the pool's REAL placement (ServingBufferPool.placement_slot) - over lanes_fakes' vLLM-shaped
scheduler, runner and deterministic target model: the target's token after (position, previous token) is a pure function of the
user's own history, so WHATEVER frame of rounds served a user - packed in a full block, padded, narrowed to the sequential step, held
back, with arrivals and departures between any two rounds - its committed stream is the model's chain from its prompt. A stale
proposal, a wrong frontier, a segment mix-up or a request served twice in a round breaks the chain or a state check.

Eight users with random arrivals, departures and budgets; every committed stream equals the target chain from its prompt, and every
round serves each scheduled request exactly once (through block A, block B or the sequential step, never two of them).
"""

import collections
import random
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from lanes_fakes import (Clock, FakeTTScheduler, LaneRequest, LaneRunner, LaneWorker, PaddedModelBlock, SchedRequest,
                         next_token, true_stream)
import serving_buffer_pool
from serving_packed_step import PackedStep
from serving_runner_bridge import FastRunnerBridge
from serving_worker_hook import FastWorkerHook

SEATS = 8
BLOCK_SLOTS = ((0, 1, 2, 3), (4, 5, 6, 7))


class Seats8World:
    """Eight pool slots over two M3 blocks, one hook, one scheduler. Arrivals take the slot the pool's placement names."""

    def __init__(self, *, tau=6.0):
        self.clock = Clock()
        self.tau = tau
        self.stepped = []
        self.block_a = PaddedModelBlock(self.clock, lambda live: 60.0 + 14.5 * live)
        self.block_b = PaddedModelBlock(self.clock, lambda live: 60.0 + 14.5 * live)
        self.blocks = (self.block_a, self.block_b)
        self.step = PackedStep(self.blocks, per_block_widths=True)
        self.pool = SimpleNamespace(slots=[SimpleNamespace(index=index, lent=False) for index in range(SEATS)])
        serving_buffer_pool.ServingBufferPool.place_blocks(self.pool, BLOCK_SLOTS)
        self.runner = LaneRunner()
        self.worker = LaneWorker(self.runner)
        self.scheduler = FakeTTScheduler()
        self.scheduler.max_num_running_reqs = SEATS
        self.hook = None
        self.users = collections.OrderedDict()          # request id -> LaneRequest
        self.slot_of = {}
        self.streams = collections.defaultdict(list)    # request id -> every token the runner committed for it
        self.outputs = ModuleType('vllm.v1.outputs')
        self.outputs.ModelRunnerOutput = SimpleNamespace
        self.outputs.DraftTokenIds = SimpleNamespace
        self.rounds = []                                # per executed step: (block A's users, block B's users, sequential users)
        self.mixed = 0

    @property
    def live(self):
        return len(self.users)

    def add_user(self, request_id, user, *, prompt_length, max_tokens):
        slot = serving_buffer_pool.ServingBufferPool.placement_slot(self.pool)
        if slot is None:
            raise AssertionError('no free seat')
        slot.lent = True
        block, segment = self.blocks[slot.index // 4], slot.index % 4
        seed = next_token(user, prompt_length - 1, 1)
        params = SimpleNamespace(extra_args=None, max_tokens=max_tokens)
        state = SimpleNamespace(req_id=request_id, prompt_token_ids=list(range(prompt_length)), output_token_ids=[seed],
                                block_ids=([1, 2],), sampling_params=params, num_computed_tokens=prompt_length)
        self.runner.requests[request_id] = state
        request = LaneRequest(request_id, user, prompt_length, seed, self.tau, self.stepped, max_new_tokens=256)
        request.engine.widths = (1, 2, 4)           # beside the 64-row block the per-request engines capture only these
        block.bind(request.engine, segment, user)
        binding = SimpleNamespace(engine=request.engine, refresh=lambda *args, **options: None)
        bridge = FastRunnerBridge(self.runner, request, binding)
        self.users[request_id] = request
        self.slot_of[request_id] = slot.index
        self.scheduler.running.append(SchedRequest(request_id, params, prompt_length, state))
        if self.hook is None:
            self.hook = FastWorkerHook(self.worker, bridge, cancelled=lambda: False, packed_step=self.step)
        else:
            self.hook.attach(bridge)
        return slot.index

    def draft(self):
        with patch.dict(sys.modules, {'vllm.v1.outputs': self.outputs}):
            drafts = self.worker.take_draft_token_ids()
        if drafts is not None:
            self.scheduler.update_draft_token_ids(drafts)

    def engine_step(self):
        """One EngineCore step: schedule, detach what finished, execute, update, draft. Returns the scheduled request ids."""
        scheduler = self.scheduler
        scheduled = scheduler.schedule()
        for request_id in sorted(scheduled.finished_req_ids):
            if request_id in self.hook.bridges:
                request = self.users.pop(request_id)
                self.hook.detach(request_id)
                slot = self.slot_of.pop(request_id)
                self.blocks[slot // 4].bound.pop(id(request.engine), None)
                self.pool.slots[slot].lent = False
        if not scheduled.total_num_scheduled_tokens:
            return []
        ids = list(scheduled.scheduled_cached_reqs.req_ids)
        before = [len(block.round_log) for block in self.blocks], len(self.stepped)
        with patch.dict(sys.modules, {'vllm.v1.outputs': self.outputs}):
            output = self.runner.execute_model(scheduled)
            scheduler.update_from_output(scheduled, output)
            for request_id, tokens in zip(output.req_ids, output.sampled_token_ids):
                self.streams[request_id].extend(tokens)
            drafts = self.worker.take_draft_token_ids()
        if drafts is not None:
            scheduler.update_draft_token_ids(drafts)
        served_a = [name for entry in self.block_a.round_log[before[0][0]:] for name in entry]
        served_b = [name for entry in self.block_b.round_log[before[0][1]:] for name in entry]
        sequential = [name for name, flag in self.stepped[before[1]:]]
        self.rounds.append((served_a, served_b, sequential))
        if (served_a or served_b) and sequential:
            self.mixed += 1
        return ids


class WholeStackTests(unittest.TestCase):
    def run_world(self, seed, steps=260, arrivals=0.35):
        rng = random.Random(seed)
        world = Seats8World(tau=rng.choice((4.0, 6.0, 9.0)))
        prompts = {}
        next_user = 0
        for _ in range(steps):
            while world.live < SEATS and (world.live == 0 or rng.random() < arrivals) and next_user < 40:
                request_id = 'u%d' % next_user
                prompts[request_id] = (next_user, 4100 + 37 * next_user)
                world.add_user(request_id, next_user, prompt_length=prompts[request_id][1],
                               max_tokens=rng.choice((20, 45, 80, 150, 400)))
                next_user += 1
                world.draft()
                if rng.random() < 0.5:
                    break
            ids = world.engine_step()
            if ids:
                served_a, served_b, sequential = world.rounds[-1]
                every = served_a + served_b + sequential
                self.assertEqual(sorted(every), sorted(ids), 'each scheduled request exactly once: %r' % (world.rounds[-1],))
                self.assertEqual(len(every), len(set(every)))
        return world, prompts

    def assert_chains(self, world, prompts):
        checked = 0
        for request_id, tokens in world.streams.items():
            user, prompt = prompts[request_id]
            seed = next_token(user, prompt - 1, 1)
            self.assertEqual(tokens, true_stream(user, seed, prompt, len(tokens)), request_id)
            checked += len(tokens)
        self.assertGreater(checked, 1000)

    def test_eight_users_with_random_arrivals_departures_and_budgets_commit_the_target_chain_each_request_once_a_round(self):
        shapes = collections.Counter()
        for seed in range(6):
            with self.subTest(seed=seed):
                world, prompts = self.run_world(seed)
                self.assert_chains(world, prompts)
                self.assertGreaterEqual(len(world.streams), 8, 'arrivals and departures happened')
                for served_a, served_b, sequential in world.rounds:
                    shapes[(len(served_a), len(served_b), len(sequential))] += 1
        # not vacuous: both blocks ran packed, rounds were padded, and a lone user ran sequentially beside a packed block
        self.assertTrue(any(a == 4 and b == 4 for a, b, s in shapes), shapes)
        self.assertTrue(any(a in (2, 3) or b in (2, 3) for a, b, s in shapes), shapes)
        self.assertTrue(any(s and (a or b) for a, b, s in shapes), 'a mixed packed + sequential round: %r' % shapes)

    def test_a_four_and_one_split_runs_block_a_packed_and_the_lone_user_alone(self):
        world = Seats8World()
        for index in range(5):
            world.add_user('u%d' % index, index, prompt_length=4100 + 50 * index, max_tokens=400)
        # placement: four in block A, the fifth alone in block B
        self.assertEqual([world.slot_of['u%d' % index] for index in range(5)], [0, 1, 2, 3, 4])
        world.draft()
        for _ in range(12):
            world.engine_step()
        self.assertEqual(world.mixed, 12, 'every round: block A packed and the lone user of block B sequential')
        self.assertTrue(all(len(a) == 4 and not b and s == ['u4'] for a, b, s in world.rounds), world.rounds[:2])
        for request_id in world.users:
            user = int(request_id[1:])
            tokens = world.streams[request_id]
            prompt = 4100 + 50 * user
            self.assertEqual(tokens, true_stream(user, next_token(user, prompt - 1, 1), prompt, len(tokens)))
        # a sixth arrival joins the block that holds exactly one user: the split goes away
        self.assertEqual(world.add_user('u5', 5, prompt_length=4350, max_tokens=400), 5)
        world.draft()
        for _ in range(6):
            world.engine_step()
        self.assertTrue(all(len(a) == 4 and len(b) == 2 and not s for a, b, s in world.rounds[-6:]), world.rounds[-6:])

    def test_a_block_that_empties_leaves_the_other_running_alone(self):
        world = Seats8World()
        for index in range(8):
            world.add_user('u%d' % index, index, prompt_length=4100, max_tokens=400 if index >= 4 else 30)
        world.draft()
        for _ in range(40):
            world.engine_step()
        self.assertTrue(any(a == [] or len(a) == 0 for a, b, s in world.rounds if b), 'block B alone after block A emptied')
        self.assertEqual(sorted(world.users), ['u4', 'u5', 'u6', 'u7'])
        for request_id, tokens in world.streams.items():
            user = int(request_id[1:])
            self.assertEqual(tokens, true_stream(user, next_token(user, 4099, 1), 4100, len(tokens)))


class WholeStackUnderTheProductionBudgetCapTests(WholeStackTests):
    """The same worlds with QWEN_FAST_BUDGET_CAP=1, the production flag: the per-block width and the commit limit both read the
    request's remaining budget (the fake session honours commit(max_rows=), as GreedySession does)."""

    def setUp(self):
        patcher = patch.dict('os.environ', {'QWEN_FAST_BUDGET_CAP': '1'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_cap_is_on(self):
        import serving_packed_step
        self.assertTrue(serving_packed_step.budget_cap_enabled())


if __name__ == '__main__':
    unittest.main()
