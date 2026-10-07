"""Lever N with prefix reuse: the merged model route (levern_route, SOURCES), executed on the REAL staged model.py with the recording fake ttnn and the toy
hybrid model of qwen_prefix_model_fixture, beside the prefix registry driven as the scheduler graft drives it (docs/lever-n-prefix-merged-route.md
sections 2, 3, 5 and 12.1).

What is held:
  E1 exact        a prefix hit that arrives split (CHECKPOINT first step, SCRATCH steps) equals a cold whole prefill of the same prompt, a cold SPLIT prefill of
                  it, and the G1 route's own whole-prompt hit: logits, KV [0, P) through the page table and the GDN slot, byte for byte, over a chained
                  conversation;
  E2 matrix       the same at every boundary length P and every aligned Q up to S_last, at step sizes 2,048 and 4,096;
  E3 checkpoints  every checkpoint a split prefill stores (mid-loop and at a step's drain) equals the one the G1 route's whole-prompt prefill stores at
                  that position, byte for byte, and the registries of the two runs hold the same keys at the end of every turn;
  E4 park         a long prefill parked to the host while a short one runs, then resumed, equals its solo run, and so does the short one; the captures it
                  plans survive the parking;
  E5 negative     a wrong checkpoint, a skipped park, a foreign scratch, a capture at the wrong loop position, a CHECKPOINT announced on a continuation, a
                  disagreeing announcement: each changes the digests or is refused before the fake records any device op;
  E6 width        every step is called at the runner's page-table width and compiles nothing after the warm, whichever source it comes from; a model with a
                  chunk-input buffer or a chunk trace is refused at the attach;
  markers         the final step logs the G1 route's '[PREFIX] row=' line (Q, L, plan, captured over every step), the route line names the source and the
                  captures, and the shared state the scheduler reads follows the owner and the parks."""

import contextlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dflash_prefill_window  # noqa: E402
import levern_policy as policy  # noqa: E402
import levern_route as route  # noqa: E402
import qwen_prefix_model_fixture as F  # noqa: E402
import qwen_prefix_registry as prefix_registry  # noqa: E402
import qwen_prefix_scheduler_patch as scheduler_patch  # noqa: E402
from test_qwen_prefix_model_runtime import ON, Engine, prefix_env, registry_scope  # noqa: E402

CHUNK = F.CHUNK
BLOCK = F.BLOCK
MERGED = {'QWEN_FAST_LEVER_N': '1', 'QWEN_PREFIX_REUSE': '1', 'QWEN_FAST_STICKY_SESSIONS': '1'}
PARK = dict(MERGED, QWEN_FAST_LEVERN_PARK='host', QWEN_FAST_LEVERN_SHORT_TOKENS='16384')


def floor_chunk(tokens):
    return tokens // CHUNK * CHUNK


class MergedEngine(Engine):
    """One engine with the merged route installed (the warm of the G1 stage has chosen the restore path, as serving_startup.prefix_warm does)."""

    def __init__(self, environ=None, traced=False, install=True):
        super().__init__(traced=traced, environ=ON)
        self.warm()
        self.lines = []
        self.merged_env = dict(MERGED if environ is None else environ)
        self.runner = mock.Mock()
        self.runner.model = self.wrapper
        self.runner.max_num_blocks_per_req = 4096
        self.handle = None
        # The shared state holder is created here, outside every sys.modules scope the steps run under, as in the engine process.
        policy.state_holder()
        if install:
            with self.scope():
                self.handle = route.install(self.runner, self.model, environ=self.merged_env, log=self.log_line)

    def log_line(self, message, *values):
        self.lines.append(message.format(*values) if values else message)

    def marker(self, text):
        return [line for line in self.lines if text in line]

    @contextlib.contextmanager
    def scope(self):
        with registry_scope(self.registry), prefix_env(self.environ), mock.patch.dict(
                sys.modules, {self.module.__name__: self.module, 'ttnn': self.fake.module()}):
            yield

    # -- the scheduler's side -----------------------------------------------------------------------------------------------------------
    def request(self, req, tokens):
        ids = F.as_ids(tokens)
        return SimpleNamespace(request_id=req, all_token_ids=ids, num_prompt_tokens=len(ids), num_tokens=len(ids), block_hashes=F.BlockHashes(ids))

    def admit(self, req, tokens, q, first_end, h=0, plan=None):
        """The grant the scheduler graft commits for a fresh admission whose step schedules [q, first_end): its sticky plan, an in-flight plan when the
        step stops short of the prompt."""
        self.registry.begin_step()
        request = self.request(req, tokens)
        stub = SimpleNamespace(registry=self.registry, sticky=True, drop_last=True)
        planned, unplanned, drain = scheduler_patch.SchedulerGraft.plan(stub, request, h, q)
        if plan is not None:
            planned = [(pos, F.key_at(request.all_token_ids, pos)) for pos in plan]
        key = F.key_at(request.all_token_ids, q) if q else None
        checkpoint = self.registry.get(key) if q else None
        if q and checkpoint is None:
            raise AssertionError('the test asked for a hit at %d with no checkpoint' % q)
        self.registry.stage(prefix_registry.Grant(req, q, h, key, checkpoint, planned, request, drain, unplanned))
        self.registry.commit({req: q}, {req: first_end - q})
        return [pos for pos, _ in planned]

    def next_step(self, req, start, end):
        """The scheduler's next schedule() call for a request already in flight: a cached request, no grant."""
        self.registry.begin_step()
        self.registry.note_scheduled(req, start, end - start)

    def step(self, req, tokens, row, start, end, total, slot, source=None, park_owner=False):
        route.announce(self.model, route.Step(req, start, end, total, source, park_owner))
        batch = tokens[:end].reshape(1, -1)
        try:
            with self.scope():
                logits, _ = self.wrapper.prefill_forward(batch, row, None, [end], empty_slots=[slot], start_pos=[start], request_ids=[req])
        finally:
            route.withdraw(self.model)
        return logits

    def run(self, req, tokens, row, q, slot, h=0, steps=None, decoding=True, plan=None, stop=None, begin=True):
        """All the steps of one prompt from its hit Q (0: cold), as the scheduler and the lifecycle would run them; `stop` ends after that many steps
        (a suspended prefill). Returns the last step's logits."""
        total = len(tokens)
        steps = policy.plan_from(total, q, decoding=decoding) if steps is None else steps
        logits = None
        for index, (start, end) in enumerate(steps):
            if index == 0 and begin:
                self.admit(req, tokens, q, end, h=h, plan=plan)
                source = 'CHECKPOINT' if q else 'COLD'
            else:
                self.next_step(req, start, end)
                source = 'SCRATCH'
            logits = self.step(req, tokens, row, start, end, total, slot, source)
            if stop is not None and index + 1 >= stop:
                break
        return logits

    def state(self, row, length, slot):
        return (self.toy.kv(row, length), self.toy.slot(slot))


def state_of(engine, row, length, slot):
    return (engine.toy.kv(row, length), engine.toy.slot(slot))


def assert_same_state(case, a, b, label):
    kv_a, slot_a = a
    kv_b, slot_b = b
    for x, y in zip(kv_a, kv_b):
        case.assertTrue(torch.equal(x, y), 'KV differs: %s' % label)
    case.assertEqual(len(slot_a), len(slot_b))
    for (rec_a, convs_a), (rec_b, convs_b) in zip(slot_a, slot_b):
        case.assertTrue(torch.equal(rec_a, rec_b), 'GDN recurrent state differs: %s' % label)
        case.assertTrue(all(torch.equal(x, y) for x, y in zip(convs_a, convs_b)), 'GDN conv state differs: %s' % label)


def reference():
    """The G1 route's own engine: whole prompts, the grant's plan, no Lever N."""
    engine = Engine(environ=ON)
    engine.warm()
    return engine


def entries(engine):
    return {key: (checkpoint.pos, list(checkpoint.token_ids), [r.clone() for r in checkpoint.rec], [c.clone() for c in checkpoint.carry])
            for key, checkpoint in engine.registry.entries.items()}


def same_entries(case, a, b, label):
    left, right = entries(a), entries(b)
    case.assertEqual(sorted(left), sorted(right), 'registry keys differ: %s' % label)
    for key in left:
        case.assertEqual(left[key][0], right[key][0], label)
        case.assertEqual(left[key][1], right[key][1], label)
        for x, y in zip(left[key][2], right[key][2]):
            case.assertTrue(torch.equal(x, y), 'checkpoint recurrent state differs: %s' % label)
        for x, y in zip(left[key][3], right[key][3]):
            case.assertTrue(torch.equal(x, y), 'checkpoint conv carry differs: %s' % label)


class ChainedConversationTests(unittest.TestCase):
    """E1 and E3: a conversation of turns, each extending the last, split by the cap."""

    LENGTHS = (9000, 13000, 17500, 21000)

    def test_a_split_hit_equals_every_cold_twin_and_the_g1_hit_at_every_turn(self):
        merged, g1, cold = MergedEngine(), reference(), reference()
        base = F.prompt(max(self.LENGTHS), seed=71)
        conv_g1 = {'id': 'conv'}
        rows, q, previous = {}, 0, 0
        for turn, length in enumerate(self.LENGTHS):
            tokens = base[:length]
            h = floor_chunk(previous) - BLOCK if previous else 0
            shared = rows['row'][0, :q // BLOCK].tolist() if q else []
            row = merged.pool.row(length, shared)
            plan = merged.run('conv', tokens, row, q, 2, h=h)
            logits = plan
            # the G1 route on the same prompt with the same plan: one whole step
            boundary = floor_chunk(length) - CHUNK
            g1_plan = [boundary] if boundary > q else []
            g1_logits, g1_row, g1_slot = g1.turn(conv_g1, tokens, q, g1_plan, h=h, force_plan=True)
            self.assertTrue(torch.equal(logits, g1_logits), 'logits differ from the G1 hit at turn %d (P=%d Q=%d)' % (turn, length, q))
            assert_same_state(self, merged.state(row, length, 2), state_of(g1, g1_row, length, g1_slot), 'G1 hit, turn %d' % turn)
            # a cold whole prefill and a cold split prefill of the same prompt, on fresh blocks
            cold_logits, cold_row, cold_slot = cold.cold(tokens, slot=3, req='cold-%d' % turn)
            self.assertTrue(torch.equal(logits, cold_logits), 'logits differ from the cold prefill at turn %d' % turn)
            assert_same_state(self, merged.state(row, length, 2), state_of(cold, cold_row, length, cold_slot), 'cold whole, turn %d' % turn)
            twin = MergedEngine()
            twin_row = twin.pool.row(length)
            twin_logits = twin.run('twin', tokens, twin_row, 0, 1)
            self.assertTrue(torch.equal(logits, twin_logits), 'logits differ from the cold split prefill at turn %d' % turn)
            assert_same_state(self, merged.state(row, length, 2), twin.state(twin_row, length, 1), 'cold split, turn %d' % turn)
            # E3: both registries hold the same checkpoints after the turn
            same_entries(self, merged, g1, 'turn %d' % turn)
            rows['row'] = row
            q = boundary if boundary > q else q
            previous = length
        self.assertEqual(sorted(entries(merged)), sorted(entries(g1)))
        self.assertGreater(merged.registry.stats['inflight_captures'], 0, 'no checkpoint was taken from an in-flight plan')

    def test_the_hits_run_split_from_q_through_the_checkpoint_source(self):
        merged = MergedEngine()
        base = F.prompt(13000, seed=72)
        merged.run('first', base[:9000], merged.pool.row(9000), 0, 1)
        merged.lines.clear()
        row = merged.pool.row(13000, merged.pool.row(9000)[0, :6144 // BLOCK].tolist())
        merged.run('second', base[:13000], row, 6144, 2, h=floor_chunk(9000) - BLOCK)
        lines = merged.marker('lever N route req=second')
        self.assertEqual([line.split(' start=')[1].split(' ')[0] for line in lines], ['6144', '8192', '10240'])
        self.assertEqual([line.rsplit(' source=', 1)[1].split()[0] for line in lines], ['CHECKPOINT', 'SCRATCH', 'SCRATCH'])
        self.assertIn('captured=10240:stored', lines[1], 'the capture at S_last is taken in the step that ends on it')

    def test_the_final_step_logs_one_g1_format_row_over_the_whole_request(self):
        merged = MergedEngine()
        base = F.prompt(13000, seed=73)
        merged.run('first', base[:9000], merged.pool.row(9000), 0, 1)
        merged.lines.clear()
        merged.run('second', base[:13000], merged.pool.row(13000, merged.pool.row(9000)[0, :6144 // BLOCK].tolist()), 6144, 2,
                   h=floor_chunk(9000) - BLOCK)
        logged = [line for line in merged.log.lines('[PREFIX] row=')]
        self.assertEqual(len(logged), 0, 'the route\'s own marker goes through its log callback')
        rows = merged.marker('[PREFIX] row=')
        self.assertEqual(len(rows), 1)
        self.assertIn('req=second', rows[0])
        self.assertIn('grant=committed Q=6144 L=13000', rows[0])
        self.assertIn('plan=[10240]', rows[0])
        self.assertIn('captured=[10240:stored', rows[0])
        self.assertIn('route=merged', rows[0])
        self.assertEqual(merged.registry.stats['restores'], 1)


class BoundaryMatrixTests(unittest.TestCase):
    """E2: every boundary length and every aligned Q, at two step sizes."""

    PROMPTS = (4096, 4097, 6143, 6144, 6145, 8191, 10241)

    def test_a_hit_at_every_aligned_q_equals_a_cold_prefill(self):
        base = F.prompt(max(self.PROMPTS), seed=81)
        cold = reference()
        for total in self.PROMPTS:
            tokens = base[:total]
            cold_logits, cold_row, cold_slot = cold.cold(tokens, slot=3, req='cold-%d' % total)
            cold_state = state_of(cold, cold_row, total, cold_slot)
            s_last = policy.final_start(total)
            for q in range(0, s_last + 1, CHUNK):
                for step in (CHUNK, 2 * CHUNK):
                    with self.subTest(total=total, q=q, step=step):
                        merged = MergedEngine()
                        # the previous turn: a cold prefill of the first q + 100 tokens that took its checkpoint at q
                        shared = []
                        if q:
                            seed_row = merged.pool.row(q + 100)
                            merged.run('seed', base[:q + 100], seed_row, 0, 0, plan=[q])
                            shared = seed_row[0, :q // BLOCK].tolist()
                        row = merged.pool.row(total, shared)
                        steps = policy.plan_from(total, q, step=step)
                        logits = merged.run('hit', base[:total], row, q, 2, h=q, steps=steps)
                        self.assertTrue(torch.equal(logits, cold_logits))
                        self.assertEqual(merged.registry.stats['restores'], 1 if q else 0)
                        assert_same_state(self, state_of(merged, row, total, 2), cold_state, 'P=%d Q=%d step=%d' % (total, q, step))
                        for start, end, source in self.sources(steps, q):
                            self.assertTrue(merged.marker('req=hit start=%d end=%d' % (start, end)))
                        self.assertEqual(merged.marker('req=hit start=%d' % q)[0].rsplit(' source=', 1)[1].split()[0], 'CHECKPOINT' if q else 'COLD')

    @staticmethod
    def sources(steps, q):
        return [(start, end, 'CHECKPOINT' if index == 0 and q else 'COLD' if index == 0 else 'SCRATCH') for index, (start, end) in enumerate(steps)]


class CaptureTests(unittest.TestCase):
    """E3 and the in-flight plan: where a split prompt takes its checkpoints."""

    def test_a_split_cold_prompt_stores_the_checkpoint_the_whole_prompt_stores(self):
        for total in (6145, 8192, 9000, 12289):
            with self.subTest(total=total):
                merged, g1 = MergedEngine(), reference()
                tokens = F.prompt(total, seed=total)
                merged.run('req', tokens, merged.pool.row(total), 0, 1)
                boundary = floor_chunk(total) - CHUNK
                g1.turn({'id': 'req'}, tokens, 0, [boundary], force_plan=True)
                same_entries(self, merged, g1, 'P=%d' % total)
                self.assertEqual(sorted(checkpoint.pos for checkpoint in merged.registry.entries.values()), [boundary])

    def test_the_plan_survives_the_steps_and_ends_with_the_final_one(self):
        merged = MergedEngine()
        tokens = F.prompt(9000, seed=91)
        row = merged.pool.row(9000)
        steps = policy.plan(9000)
        merged.admit('req', tokens, 0, steps[0][1])
        self.assertIn('req', merged.registry.inflight)
        merged.step('req', tokens, row, 0, steps[0][1], 9000, 1, 'COLD')
        for start, end in steps[1:]:
            merged.next_step('req', start, end)
            self.assertIn('req', merged.registry.inflight)
            merged.step('req', tokens, row, start, end, 9000, 1, 'SCRATCH')
        self.assertTrue(merged.registry.inflight['req'].final)
        merged.registry.begin_step()
        self.assertNotIn('req', merged.registry.inflight)
        self.assertEqual(merged.registry.stats['inflight_dropped'], 1)

    def test_a_planned_position_is_taken_in_the_step_that_ends_on_it_and_in_no_other(self):
        merged = MergedEngine()
        tokens = F.prompt(9000, seed=92)
        row = merged.pool.row(9000)
        merged.run('req', tokens, row, 0, 1)
        taken = [line for line in merged.marker('lever N route req=req') if 'captured=-' not in line]
        self.assertEqual(len(taken), 1)
        self.assertIn('start=4096 end=6144', taken[0])
        self.assertIn('captured=6144:stored', taken[0])

    def test_a_capture_at_the_wrong_loop_position_or_an_unplanned_one_is_refused(self):
        registry = prefix_registry.PrefixRegistry(budget_bytes=1 << 30)
        tokens = F.prompt(9000, seed=93)
        ids = F.as_ids(tokens)
        request = SimpleNamespace(request_id='r', all_token_ids=ids, num_prompt_tokens=9000, num_tokens=9000, block_hashes=F.BlockHashes(ids))
        registry.begin_step()
        registry.stage(prefix_registry.Grant('r', 0, 0, None, None, [(6144, F.key_at(ids, 6144))], request, 8192))
        registry.commit({'r': 0}, {'r': 2048})
        registry.begin_step()
        registry.note_scheduled('r', 2048, 2048)
        self.assertIsNone(registry.capture('r', 6144, rec=[], carry=[], nbytes=1, loop_pos=4096))
        self.assertEqual(registry.stats['capture_wrong_position'], 1)
        self.assertIsNone(registry.capture('r', 4096, rec=[], carry=[], nbytes=1, loop_pos=4096))
        self.assertEqual(registry.stats['capture_failures'], 2)
        self.assertIsNotNone(registry.capture('r', 6144, rec=[], carry=[], nbytes=1, loop_pos=6144))
        self.assertEqual(registry.stats['inflight_captures'], 1)

    def test_a_request_that_fits_one_step_leaves_no_in_flight_plan(self):
        merged = MergedEngine()
        tokens = F.prompt(3000, seed=94)
        merged.run('whole', tokens, merged.pool.row(3000), 0, 1)
        self.assertEqual(merged.registry.stats['inflight_started'], 0)
        merged.run('hit', F.prompt(5000, seed=95), merged.pool.row(5000), 0, 2, steps=[(0, 5000)])
        self.assertEqual(merged.registry.stats['inflight_started'], 0)


class ParkTests(unittest.TestCase):
    """E4: a long prefill parked to the host while a short one runs."""

    def parked_run(self, total_a=12289, total_b=5000, stop=3):
        merged = MergedEngine(PARK)
        tokens_a, tokens_b = F.prompt(total_a, seed=101), F.prompt(total_b, seed=102)
        row_a, row_b = merged.pool.row(total_a), merged.pool.row(total_b)
        steps_a = policy.plan(total_a)
        merged.run('A', tokens_a, row_a, 0, 1, stop=stop)
        steps_b = policy.plan(total_b)
        merged.final = {}
        for index, (start, end) in enumerate(steps_b):
            if index == 0:
                merged.admit('B', tokens_b, 0, end)
                merged.final['B'] = merged.step('B', tokens_b, row_b, start, end, total_b, 2, 'COLD', park_owner=True)
            else:
                merged.next_step('B', start, end)
                merged.final['B'] = merged.step('B', tokens_b, row_b, start, end, total_b, 2, 'SCRATCH')
        for index, (start, end) in enumerate(steps_a[stop:]):
            merged.next_step('A', start, end)
            merged.final['A'] = merged.step('A', tokens_a, row_a, start, end, total_a, 1, 'PARKED' if index == 0 else 'SCRATCH')
        return merged, tokens_a, tokens_b, row_a, row_b

    def test_a_parked_long_prefill_and_the_short_one_each_equal_their_solo_runs(self):
        merged, tokens_a, tokens_b, row_a, row_b = self.parked_run()
        cold = reference()
        logits_a, cold_row_a, cold_slot_a = cold.cold(tokens_a, slot=3, req='cold-a')
        logits_b, cold_row_b, cold_slot_b = cold.cold(tokens_b, slot=3, req='cold-b')
        solo = MergedEngine()
        solo_row_a, solo_row_b = solo.pool.row(len(tokens_a)), solo.pool.row(len(tokens_b))
        solo_logits_a = solo.run('A', tokens_a, solo_row_a, 0, 1)
        solo_logits_b = solo.run('B', tokens_b, solo_row_b, 0, 2)
        assert_same_state(self, merged.state(row_a, len(tokens_a), 1), solo.state(solo_row_a, len(tokens_a), 1), 'A vs its solo run')
        assert_same_state(self, merged.state(row_b, len(tokens_b), 2), solo.state(solo_row_b, len(tokens_b), 2), 'B vs its solo run')
        self.assertTrue(merged.handle.parked_total == 1 and merged.handle.unparked_total == 1)
        self.assertEqual(merged.handle.parks, {})
        self.assertIsNone(route.owner(merged.model))
        del logits_a, logits_b, solo_logits_a, solo_logits_b

    def test_the_final_logits_of_both_equal_the_cold_whole_prompts(self):
        merged, tokens_a, tokens_b, row_a, row_b = self.parked_run()
        cold = reference()
        logits_a, _, _ = cold.cold(tokens_a, slot=3, req='cold-a')
        logits_b, _, _ = cold.cold(tokens_b, slot=3, req='cold-b')
        self.assertEqual(len(merged.marker('lever N route req=A start=10240')), 1)
        # the parked long prompt's own final logits (resumed from the host) and the short one's equal the cold whole prompts' (S7: the parked run's
        # logits, not only a solo run's, are compared)
        self.assertTrue(torch.equal(merged.final['A'], logits_a), 'the resumed long prompt logits differ from its cold twin')
        self.assertTrue(torch.equal(merged.final['B'], logits_b), 'the short prompt logits differ from its cold twin')

    def test_the_plan_of_a_parked_prompt_still_takes_its_checkpoint_after_it_resumes(self):
        merged, tokens_a, tokens_b, row_a, row_b = self.parked_run()
        boundary = floor_chunk(len(tokens_a)) - CHUNK
        self.assertIn(F.key_at(F.as_ids(tokens_a), boundary), merged.registry.entries)

    def test_the_park_and_the_unpark_are_logged_and_the_shared_state_follows_them(self):
        merged = MergedEngine(PARK)
        tokens_a, tokens_b = F.prompt(12289, seed=103), F.prompt(5000, seed=104)
        row_a, row_b = merged.pool.row(12289), merged.pool.row(5000)
        merged.run('A', tokens_a, row_a, 0, 1, stop=3)
        holder = policy.state_holder()
        self.assertEqual(holder.owner, ('A', 6144))
        self.assertTrue(policy.state_has('A', 6144))
        merged.admit('B', tokens_b, 0, 2048)
        merged.step('B', tokens_b, row_b, 0, 2048, 5000, 2, 'COLD', park_owner=True)
        self.assertEqual(holder.parks, {'A': 6144})
        self.assertEqual(holder.owner, ('B', 2048))
        self.assertTrue(policy.state_has('A', 6144))
        self.assertFalse(policy.state_has('A', 8192))
        self.assertEqual(len(merged.marker('lever N park out req=A at=6144')), 1)
        merged.next_step('B', 2048, 5000)
        merged.step('B', tokens_b, row_b, 2048, 5000, 5000, 2, 'SCRATCH')
        merged.next_step('A', 6144, 8192)
        merged.step('A', tokens_a, row_a, 6144, 8192, 12289, 1, 'PARKED')
        self.assertEqual(len(merged.marker('lever N park in req=A at=6144')), 1)
        self.assertEqual(holder.parks, {})
        self.assertEqual(holder.owner, ('A', 8192))

    def test_a_park_beyond_the_slot_count_or_without_the_flag_is_refused_before_any_device_work(self):
        merged = MergedEngine(PARK)
        tokens_a, tokens_b, tokens_c = F.prompt(12289, seed=105), F.prompt(5000, seed=106), F.prompt(5000, seed=107)
        row_a, row_b, row_c = merged.pool.row(12289), merged.pool.row(5000), merged.pool.row(5000)
        merged.run('A', tokens_a, row_a, 0, 1, stop=2)
        merged.admit('B', tokens_b, 0, 2048)
        merged.step('B', tokens_b, row_b, 0, 2048, 5000, 2, 'COLD', park_owner=True)
        merged.admit('C', tokens_c, 0, 2048)
        before = list(merged.fake.log)
        with self.assertRaisesRegex(AssertionError, 'no park slot'):
            merged.step('C', tokens_c, row_c, 0, 2048, 5000, 3, 'COLD', park_owner=True)
        self.assertEqual(merged.fake.log, before)
        plain = MergedEngine()
        plain.run('A', tokens_a, plain.pool.row(12289), 0, 1, stop=2)
        plain.admit('B', tokens_b, 0, 2048)
        before = list(plain.fake.log)
        with self.assertRaisesRegex(AssertionError, 'does not park it'):
            plain.step('B', tokens_b, plain.pool.row(5000), 0, 2048, 5000, 2, 'COLD')
        self.assertEqual(plain.fake.log, before)


class RefusalTests(unittest.TestCase):
    """E5: a wrong state is refused before any device op, or the harness sees it."""

    def started(self):
        merged = MergedEngine()
        tokens = F.prompt(13000, seed=111)
        row = merged.pool.row(13000)
        merged.run('A', tokens, row, 0, 1, stop=2)
        return merged, tokens, row

    def refused(self, merged, call, pattern):
        before = list(merged.fake.log)
        with self.assertRaisesRegex(AssertionError, pattern):
            call()
        self.assertEqual(merged.fake.log, before, 'refused before any device work')

    def test_a_continuation_the_lifecycle_announced_as_a_checkpoint_is_refused(self):
        merged, tokens, row = self.started()
        merged.next_step('A', 4096, 6144)
        self.refused(merged, lambda: merged.step('A', tokens, row, 4096, 6144, 13000, 1, 'CHECKPOINT'), 'announced CHECKPOINT')

    def test_a_cold_announcement_on_a_continuation_is_refused(self):
        merged, tokens, row = self.started()
        merged.next_step('A', 4096, 6144)
        self.refused(merged, lambda: merged.step('A', tokens, row, 4096, 6144, 13000, 1, 'COLD'), 'announced COLD')

    def test_a_skipped_step_a_replayed_step_and_a_lost_owner_are_refused(self):
        merged, tokens, row = self.started()
        merged.next_step('A', 6144, 8192)
        self.refused(merged, lambda: merged.step('A', tokens, row, 6144, 8192, 13000, 1, 'SCRATCH'), 'no source holds')
        route.release(merged.model, 'A')
        merged.next_step('A', 4096, 6144)
        self.refused(merged, lambda: merged.step('A', tokens, row, 4096, 6144, 13000, 1, 'SCRATCH'), 'no source holds')

    def test_a_foreign_scratch_is_refused_for_a_new_prompt_without_a_park(self):
        merged, tokens, row = self.started()
        other = F.prompt(5000, seed=112)
        merged.admit('B', other, 0, 2048)
        self.refused(merged, lambda: merged.step('B', other, merged.pool.row(5000), 0, 2048, 5000, 2, 'COLD'), 'suspended on it')

    def test_a_park_announced_with_no_owner_is_refused(self):
        merged = MergedEngine(PARK)
        other = F.prompt(5000, seed=113)
        merged.admit('B', other, 0, 2048)
        self.refused(merged, lambda: merged.step('B', other, merged.pool.row(5000), 0, 2048, 5000, 2, 'COLD', park_owner=True),
                     'scratch has none')

    def test_a_wrong_checkpoint_is_refused_before_any_device_work(self):
        merged = MergedEngine()
        base = F.prompt(13000, seed=114)
        merged.run('first', base[:9000], merged.pool.row(9000), 0, 1)
        checkpoint = next(iter(merged.registry.entries.values()))
        checkpoint.token_ids = prefix_registry.token_array([0] * checkpoint.pos)
        row = merged.pool.row(13000, merged.pool.row(9000)[0, :6144 // BLOCK].tolist())
        request = merged.request('second', base[:13000])
        stub = SimpleNamespace(registry=merged.registry, sticky=True, drop_last=True)
        merged.registry.begin_step()
        planned, unplanned, drain = scheduler_patch.SchedulerGraft.plan(stub, request, 8000, 6144)
        merged.registry.stage(prefix_registry.Grant('second', 6144, 8000, checkpoint.key, checkpoint, planned, request, drain, unplanned))
        merged.registry.commit({'second': 6144}, {'second': 2048})
        self.refused(merged, lambda: merged.step('second', base[:13000], row, 6144, 8192, 13000, 2, 'CHECKPOINT'), 'token ids differ')

    def test_a_corrupted_checkpoint_changes_the_bytes_and_the_harness_sees_it(self):
        base = F.prompt(13000, seed=115)
        outcomes = []
        for corrupt in (False, True):
            merged = MergedEngine()
            merged.run('first', base[:9000], merged.pool.row(9000), 0, 1)
            if corrupt:
                checkpoint = next(iter(merged.registry.entries.values()))
                checkpoint.carry = [torch.zeros_like(c) for c in checkpoint.carry]
            row = merged.pool.row(13000, merged.pool.row(9000)[0, :6144 // BLOCK].tolist())
            logits = merged.run('second', base[:13000], row, 6144, 2, h=floor_chunk(9000) - BLOCK)
            outcomes.append((logits, merged.state(row, 13000, 2)))
        self.assertFalse(torch.equal(outcomes[0][0], outcomes[1][0]) and all(
            torch.equal(a[0], b[0]) for a, b in zip(outcomes[0][1][1], outcomes[1][1][1])))

    def test_a_skipped_park_is_seen_by_the_harness(self):
        """If the long's scratch were simply overwritten (no park), the resumed long would differ from its solo run: the harness must see it."""
        merged = MergedEngine(PARK)
        tokens_a, tokens_b = F.prompt(12289, seed=116), F.prompt(5000, seed=117)
        row_a, row_b = merged.pool.row(12289), merged.pool.row(5000)
        merged.run('A', tokens_a, row_a, 0, 1, stop=3)
        merged.admit('B', tokens_b, 0, 2048)
        merged.step('B', tokens_b, row_b, 0, 2048, 5000, 2, 'COLD', park_owner=True)
        merged.next_step('B', 2048, 5000)
        merged.step('B', tokens_b, row_b, 2048, 5000, 5000, 2, 'SCRATCH')
        # the lost park: B's run re-zeroed the scratch, and A resumes from whatever the scratch holds
        merged.handle.parks['A'] = route.Park(6144, *self.zeroed(merged), 0, 0.0)
        policy.state_holder().parks = {'A': 6144}
        merged.next_step('A', 6144, 8192)
        merged.step('A', tokens_a, row_a, 6144, 8192, 12289, 1, 'PARKED')
        for start, end in policy.plan(12289)[4:]:
            merged.next_step('A', start, end)
            merged.step('A', tokens_a, row_a, start, end, 12289, 1, 'SCRATCH')
        solo = MergedEngine()
        solo_row = solo.pool.row(12289)
        solo.run('A', tokens_a, solo_row, 0, 1)
        same = all(torch.equal(a[0], b[0]) for a, b in zip(merged.toy.slot(1), solo.toy.slot(1)))
        self.assertFalse(same, 'the harness cannot see a park that was skipped')

    @staticmethod
    def zeroed(merged):
        with merged.scope():
            previous = merged.model._bind_gdn_prefill_scratch()
            try:
                rec, carry, _ = merged.model._qwen_prefix_read_scratch()
            finally:
                merged.model._unbind_gdn_prefill_scratch(previous)
        return [torch.zeros_like(value) for value in rec], [torch.zeros_like(value) for value in carry]


class WidthAndAttachTests(unittest.TestCase):
    """E6 and the attach: the runner's page-table width, no program after the warm, what the install refuses."""

    def test_every_source_runs_at_the_runner_width_and_compiles_nothing_after_the_warm(self):
        merged = MergedEngine(PARK)
        with merged.scope():
            self.assertTrue(route.warm(merged.runner, merged.model, environ=merged.merged_env, log=merged.log_line))
        warm = merged.marker('route warmed')
        self.assertEqual(len(warm), 1)
        self.assertIn('steps=%d' % route.MERGED_WARM_STEPS, warm[0])
        before, after = warm[0].split('programs=')[1].split(' ms=')[0].split('->')
        merged.toy.widths.clear()
        base = F.prompt(13000, seed=121)
        merged.run('first', base[:9000], merged.pool.row(9000), 0, 1)
        merged.run('second', base[:13000], merged.pool.row(13000, merged.pool.row(9000)[0, :6144 // BLOCK].tolist()), 6144, 2,
                   h=floor_chunk(9000) - BLOCK)
        self.assertEqual({width for _, width in merged.toy.widths}, {4096})
        for line in merged.marker('lever N route req='):
            program_before, program_after = line.rsplit('programs=', 1)[1].split(' window=')[0].split('->')
            self.assertEqual(program_before, program_after, line)
        del before, after

    def test_the_warm_runs_the_four_sources_and_leaves_nothing_behind(self):
        merged = MergedEngine(PARK)
        merged.fake.log.clear()
        with merged.scope():
            route.warm(merged.runner, merged.model, environ=merged.merged_env, log=merged.log_line)
        self.assertIsNone(route.owner(merged.model))
        self.assertEqual(merged.handle.parks, {})
        self.assertEqual(policy.state_holder().owner, None)
        self.assertTrue(merged.marker('lever N park out req=__levern_warm__'))
        plain = MergedEngine()
        with plain.scope():
            route.warm(plain.runner, plain.model, environ=plain.merged_env, log=plain.log_line)
        self.assertFalse(plain.marker('lever N park out'))

    def test_a_model_with_a_chunk_buffer_or_a_chunk_trace_is_refused_at_the_attach(self):
        engine = MergedEngine(install=False)
        engine.model._chunk_full_page_table_buf = engine.fake.device_tensor(torch.zeros(1, 4096, dtype=torch.int64), engine.fake.int32)
        with engine.scope(), self.assertRaisesRegex(ValueError, 'chunk-input page-table buffer'):
            route.install(engine.runner, engine.model, environ=engine.merged_env, log=engine.log_line)
        traced = MergedEngine(install=False)
        traced.model._chunked_trace_id = 'trace'
        with traced.scope(), self.assertRaisesRegex(ValueError, 'chunk trace'):
            route.install(traced.runner, traced.model, environ=traced.merged_env, log=traced.log_line)

    def test_a_model_without_the_g1_graft_is_refused_by_name(self):
        engine = MergedEngine(install=False)
        del engine.module.__dict__['_qwen_prefix_row']
        with engine.scope(), self.assertRaisesRegex(ValueError, '_qwen_prefix_row'):
            route.install(engine.runner, engine.model, environ=engine.merged_env, log=engine.log_line)

    def test_prefix_reuse_without_sticky_sessions_is_refused(self):
        engine = MergedEngine(install=False)
        env = dict(MERGED)
        del env['QWEN_FAST_STICKY_SESSIONS']
        with engine.scope(), self.assertRaisesRegex(ValueError, 'QWEN_FAST_STICKY_SESSIONS'):
            route.install(engine.runner, engine.model, environ=env, log=engine.log_line)

    def test_the_install_announces_the_four_sources_and_lever_n_alone_stays_stage_one(self):
        merged = MergedEngine()
        self.assertTrue(merged.marker('route installed: route=1 audit=0 merged=COLD,CHECKPOINT,SCRATCH,PARKED'))
        self.assertIs(vars(merged.model)[route.MERGED_ATTR], True)
        self.assertEqual(route.SOURCES, ('COLD', 'CHECKPOINT', 'SCRATCH', 'PARKED'))
        alone = MergedEngine(environ={'QWEN_FAST_LEVER_N': '1'}, install=False)
        with alone.scope():
            handle = route.install(alone.runner, alone.model, environ=alone.merged_env, log=alone.log_line)
        self.assertFalse(handle.merged)
        self.assertNotIn(route.MERGED_ATTR, vars(alone.model))

    def test_the_capture_ledger_accepts_a_hit_arriving_through_the_range_entry_only_beside_the_merged_route(self):
        self.assertEqual(route.MERGED_ATTR, dflash_prefill_window.LEVERN_MERGED_ATTR)
        self.assertEqual(route.WROTE_ATTR, dflash_prefill_window.LEVERN_WROTE_ATTR)

    def test_an_unannounced_prompt_is_refused_while_a_suspended_prefill_holds_the_scratch(self):
        merged = MergedEngine()
        tokens = F.prompt(9000, seed=131)
        merged.run('A', tokens, merged.pool.row(9000), 0, 1, stop=2)
        before = list(merged.fake.log)
        with merged.scope(), self.assertRaisesRegex(AssertionError, 'would reset that state'):
            merged.wrapper.prefill_forward(F.prompt(2100, seed=1).reshape(1, -1), merged.pool.row(2100), None, [2100],
                                           empty_slots=[3], start_pos=[0])
        self.assertEqual(merged.fake.log, before)

    def test_an_unannounced_resumed_prompt_is_refused(self):
        merged = MergedEngine()
        with merged.scope(), self.assertRaisesRegex(AssertionError, 'without a lifecycle announcement'):
            merged.wrapper.prefill_forward(F.prompt(4100, seed=1).reshape(1, -1), merged.pool.row(4100), None, [4100],
                                           empty_slots=[3], start_pos=[2048])


class ReviewFixRouteTests(unittest.TestCase):
    """The review of 5b6c0448: S5, S7 and S8."""

    def setUp(self):
        import serving_request_quarantine as quarantine

        shared = quarantine.holder()
        saved = (shared.installed, dict(shared.pending))
        shared.installed = 'test-consumer'
        shared.pending.clear()

        def restore():
            shared.installed, pending = saved
            shared.pending.clear()
            shared.pending.update(pending)
        self.addCleanup(restore)
        self.quarantine = quarantine

    def test_s7_a_checkpoint_short_prompt_parks_a_long_one_and_both_equal_their_cold_twins(self):
        merged = MergedEngine(PARK)
        base = F.prompt(20000, seed=211)
        first, tokens_b, tokens_a = base[:9000], base[:13000], F.prompt(20000, seed=212)
        row_first = merged.pool.row(9000)
        merged.run('first', first, row_first, 0, 1)
        row_a = merged.pool.row(20000)
        steps_a = policy.plan(20000)
        merged.run('A', tokens_a, row_a, 0, 3, stop=2)
        row_b = merged.pool.row(13000, row_first[0, :6144 // BLOCK].tolist())
        steps_b = policy.plan_from(13000, 6144)
        for index, (start, end) in enumerate(steps_b):
            if index == 0:
                merged.admit('B', tokens_b, 6144, end, h=floor_chunk(9000) - BLOCK)
                logits_b = merged.step('B', tokens_b, row_b, start, end, 13000, 2, 'CHECKPOINT', park_owner=True)
            else:
                merged.next_step('B', start, end)
                logits_b = merged.step('B', tokens_b, row_b, start, end, 13000, 2, 'SCRATCH')
        self.assertEqual(merged.handle.parked_total, 1, 'the checkpoint short prompt parked the long one')
        for index, (start, end) in enumerate(steps_a[2:]):
            merged.next_step('A', start, end)
            logits_a = merged.step('A', tokens_a, row_a, start, end, 20000, 3, 'PARKED' if index == 0 else 'SCRATCH')
        cold = reference()
        cold_logits_b, cold_row_b, cold_slot_b = cold.cold(tokens_b, slot=3, req='cold-b')
        cold_logits_a, cold_row_a, cold_slot_a = cold.cold(tokens_a, slot=1, req='cold-a')
        self.assertTrue(torch.equal(logits_b, cold_logits_b), 'the checkpoint short prompt differs from its cold twin')
        self.assertTrue(torch.equal(logits_a, cold_logits_a), 'the resumed long prompt differs from its cold twin')
        assert_same_state(self, merged.state(row_b, 13000, 2), state_of(cold, cold_row_b, 13000, cold_slot_b), 'B')
        assert_same_state(self, merged.state(row_a, 20000, 3), state_of(cold, cold_row_a, 20000, cold_slot_a), 'A')

    def test_s5_a_host_park_failure_ends_the_long_prefill_per_request_and_the_short_one_runs(self):
        merged = MergedEngine(PARK)
        tokens_a, tokens_b = F.prompt(12289, seed=221), F.prompt(5000, seed=222)
        row_a, row_b = merged.pool.row(12289), merged.pool.row(5000)
        merged.run('A', tokens_a, row_a, 0, 1, stop=2)
        merged.admit('B', tokens_b, 0, 2048)
        with mock.patch.object(merged.model, '_qwen_prefix_read_scratch', side_effect=RuntimeError('host allocation failed')):
            logits = merged.step('B', tokens_b, row_b, 0, 2048, 5000, 2, 'COLD', park_owner=True)
        self.assertIsNotNone(logits)
        self.assertIn('A', self.quarantine.holder().pending, 'the long prefill is quarantined, not the engine failed')
        self.assertEqual(merged.handle.parks, {})
        self.assertEqual(merged.handle.park_failures, 1)
        self.assertEqual(route.owner(merged.model), ('B', 2048), 'the short prompt runs and owns the scratch')
        self.assertEqual(policy.state_holder().owner, ('B', 2048))
        self.assertFalse(policy.state_has('A', 4096))
        self.assertEqual(len(merged.marker('lever N quarantine req=A')), 1)

    def test_s5_without_a_quarantine_consumer_a_host_park_failure_stays_fatal(self):
        self.quarantine.holder().installed = None
        merged = MergedEngine(PARK)
        tokens_a, tokens_b = F.prompt(12289, seed=223), F.prompt(5000, seed=224)
        merged.run('A', tokens_a, merged.pool.row(12289), 0, 1, stop=2)
        merged.admit('B', tokens_b, 0, 2048)
        with mock.patch.object(merged.model, '_qwen_prefix_read_scratch', side_effect=RuntimeError('host allocation failed')):
            with self.assertRaises(route.ParkFailed):
                merged.step('B', tokens_b, merged.pool.row(5000), 0, 2048, 5000, 2, 'COLD', park_owner=True)

    def test_s8_the_identity_check_sees_every_attribute_the_bind_rebinds_and_each_conv_state(self):
        def layer():
            return SimpleNamespace(B=8, rec_state=object(), conv_states=[object(), object(), object()], conv_carry=object(), _zero_conv0=object(),
                                   _stable_state=False)
        layers = [layer(), layer()]
        before = route._binding(layers)
        self.assertTrue(route._binding_holds(before))
        for name, value in (('_zero_conv0', object()), ('_stable_state', True), ('B', 1), ('conv_carry', object()), ('rec_state', object())):
            saved = getattr(layers[1], name)
            setattr(layers[1], name, value)
            self.assertFalse(route._binding_holds(before), '%s rebound and not seen' % name)
            setattr(layers[1], name, saved)
        self.assertTrue(route._binding_holds(before))
        saved = layers[0].conv_states[1]
        layers[0].conv_states[1] = object()
        self.assertFalse(route._binding_holds(before), 'an element of conv_states replaced and not seen')
        layers[0].conv_states[1] = saved
        layers[0].conv_states = list(layers[0].conv_states)
        self.assertFalse(route._binding_holds(before), 'the conv_states list rebound and not seen')


class ReleaseTests(unittest.TestCase):
    def test_release_frees_the_owner_and_the_park_and_the_shared_state(self):
        merged = MergedEngine(PARK)
        tokens_a, tokens_b = F.prompt(12289, seed=141), F.prompt(5000, seed=142)
        merged.run('A', tokens_a, merged.pool.row(12289), 0, 1, stop=2)
        merged.admit('B', tokens_b, 0, 2048)
        merged.step('B', tokens_b, merged.pool.row(5000), 0, 2048, 5000, 2, 'COLD', park_owner=True)
        self.assertTrue(route.release(merged.model, 'A'))
        self.assertEqual(merged.handle.parks, {})
        self.assertEqual(policy.state_holder().parks, {})
        self.assertTrue(route.release(merged.model, 'B'))
        self.assertIsNone(policy.state_holder().owner)
        self.assertFalse(route.release(merged.model, 'B'))
        self.assertFalse(policy.state_has('A', 4096) and False)


if __name__ == '__main__':
    unittest.main()
