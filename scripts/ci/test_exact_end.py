"""tp4/exact-end (docs/tp4-exact-end.md): the arithmetic model and the chunk-grid proof, a toy of the arithmetic that shows which properties make a resume byte-exact, exact resume P0 beside the tile-aligned
drafter window, and the prefix gate serving another checkpoint.

  * GridTests / PolicyTests / SavingTests: exact_end_model, checked against the engine's own rules (prefix_judge.Oracle for the resume points, tp4_split_probe for the probe's arms).
  * GridEquivalenceTests: a byte-exact resume runs a SUFFIX of the cold prefill's own program list; a resume at a 128-aligned position inside a 2048 cell runs programs the cold prefill never runs.
  * ToyNumericsTests: a toy prefill with the two properties the analysis rests on (the matmul blocking is chosen by the bucket; the state is stored in bf16 between programs). It is a model of the
    CLAIM, not of the hardware: the card decides which of its four cells the real engine is in (tp4_split_probe --attribute, the SP1 job); the toy shows what each answer would mean.
  * AlignedCarryTests: the carried drafter window joins the tile-aligned window as its first part (QWEN_FAST_TAP_WINDOW_ALIGNED beside QWEN_PREFIX_WINDOW_CARRY), the cold window row for row, in the
    bounded set of shapes the aligned join exists to keep; the flags off change nothing.
  * PrefixGateCheckpointTests: C2_MODEL_REPO / C2_MODEL_REVISION reach every arm's container of the prefix gate and every arm's log is read for them.
"""

import contextlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_prefix_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
import dflash_prefill_window as window  # noqa: E402
import exact_end_model as model  # noqa: E402
import prefix_judge as judge  # noqa: E402
import prefix_window_carry as carry_module  # noqa: E402
import qwen_prefix_registry as prefix_registry  # noqa: E402
import tap_window_aligned as twa  # noqa: E402
import test_c2_prefix_gate as gate_tests  # noqa: E402
import test_exact_resume as base  # noqa: E402
import test_prefix_replay as replay_fakes  # noqa: E402
import test_qwen_c2_checkpoint as checkpoint_tests  # noqa: E402
import tp4_split_probe as probe  # noqa: E402

PRODUCTION = (164887, 165056, 165225, 165394)           # the production prompts of the brief: the resume point stays at 161,792 for each returning turn
CHUNK = model.CHUNK


# ------------------------------------------------------------------------------------------------
# The grid and the policies
# ------------------------------------------------------------------------------------------------
class GridTests(unittest.TestCase):
    def test_the_buckets_are_the_models(self):
        self.assertEqual(model.BUCKETS, (128, 256, 512, 1024, 2048))
        for rows, bucket in ((1, 128), (128, 128), (129, 256), (512, 512), (513, 1024), (1024, 1024), (1047, 2048), (2047, 2048), (2048, 2048)):
            self.assertEqual(model.bucket_for(rows), bucket, rows)
        for rows in (0, -1, 1.5, None):
            with self.assertRaises(ValueError):
                model.bucket_for(rows)

    def test_a_prefill_of_the_first_production_prompt(self):
        programs = model.cold_programs(PRODUCTION[0])
        self.assertEqual(len(programs), 81)
        self.assertEqual(programs[0], (0, CHUNK, CHUNK))
        self.assertEqual(programs[-2], (161792, CHUNK, CHUNK))
        self.assertEqual(programs[-1], (163840, 2048, 1047), 'the tail of 1,047 rows runs the 2048 bucket: the padded rows are computed')
        self.assertEqual(sum(program.valid for program in programs), PRODUCTION[0])

    def test_a_prompt_on_a_chunk_boundary_has_no_tail_and_a_short_one_only_a_tail(self):
        self.assertEqual(model.cold_programs(4096), [(0, CHUNK, CHUNK), (2048, CHUNK, CHUNK)])
        self.assertEqual(model.cold_programs(300), [(0, 512, 300)])

    def test_a_resume_off_the_grid_is_refused(self):
        for start in (1, 64, 128, 2047, 2049, 4100, -2048, 10000, 20000):
            with self.subTest(start=start), self.assertRaises(ValueError):
                model.cold_programs(10000, start)

    def test_the_split_programs_are_the_probes_arms(self):
        for length in (5000, 9000, 41000):
            for name, split in probe.plan_arms(length, attribute=True):
                if split is None or split % CHUNK == 0:
                    continue
                with self.subTest(length=length, arm=name):
                    programs = model.split_programs(length, split)
                    head_end = min(length, probe.ceil_to(split, CHUNK))
                    self.assertEqual(programs[0], (split, model.bucket_for(head_end - split), head_end - split))
                    self.assertEqual(programs[1:], model.cold_programs(length, head_end) if head_end < length else [])
        self.assertEqual(probe.FORCED_BUCKET, model.CHUNK)
        self.assertEqual(model.split_programs(9000, 4096), model.cold_programs(9000, 4096), 'a split on the grid is the cold run')

    def test_the_model_and_the_probe_share_the_grid_constants(self):
        self.assertEqual((probe.CHUNK, probe.BLOCK, probe.QCHUNK), (model.CHUNK, model.BLOCK, model.SUBCHUNK))
        self.assertEqual((judge.CHUNK, judge.BLOCK), (model.CHUNK, model.BLOCK))


class PolicyTests(unittest.TestCase):
    def test_the_production_sequence_today_and_under_p0(self):
        """The brief: 164,887 -> 165,056 -> 165,225 -> 165,394, the resume point staying at 161,792 = floor2048(P_prev) - 2048."""
        today = [model.evaluate(prev, prompt, 'today') for prev, prompt in zip(PRODUCTION, PRODUCTION[1:])]
        self.assertEqual([item['q'] for item in today], [161792] * 3)
        self.assertEqual([item['old'] for item in today], [3095, 3264, 3433], 'the old tokens each turn re-prefills')
        self.assertEqual([PRODUCTION[i + 1] - item['q'] for i, item in enumerate(today)], [3264, 3433, 3602])
        self.assertEqual([item['forwards'] for item in today], [2, 2, 2])
        self.assertEqual([item['rows'] for item in today], [4096] * 3, 'a full chunk and a tail that runs the 2048 bucket')
        p0 = [model.evaluate(prev, prompt, 'p0') for prev, prompt in zip(PRODUCTION, PRODUCTION[1:])]
        self.assertEqual([item['q'] for item in p0], [163840] * 3)
        self.assertEqual([item['old'] for item in p0], [1047, 1216, 1385])
        self.assertEqual([PRODUCTION[i + 1] - item['q'] for i, item in enumerate(p0)], [1216, 1385, 1554])
        self.assertEqual([item['forwards'] for item in p0], [1, 1, 1])
        self.assertEqual([item['rows'] for item in p0], [2048] * 3)
        self.assertTrue(all(item['new'] == 169 for item in today + p0))
        exact = [model.evaluate(prev, prompt, 'exact_end') for prev, prompt in zip(PRODUCTION, PRODUCTION[1:])]
        self.assertEqual([item['q'] for item in exact], [164864, 164992, 165120])
        self.assertTrue(all(item['old'] < model.SUBCHUNK and item['forwards'] == 1 and item['rows'] <= 512 for item in exact))

    def test_the_oracle_agrees_with_the_closed_forms(self):
        """prefix_judge.Oracle is the gates' prediction of the graft's grants: today's rule and P0's are the closed forms of the model."""
        rng = random.Random(20261019)
        tokens = list(range(1, 90000))
        for _ in range(60):
            start = rng.randrange(6000, 60000)
            delta = rng.choice((1, 63, 169, 500, 1400, 2047, 2048, 3000))
            prompts = [start + delta * index for index in range(5)]
            for policy, kwargs in (('today', dict(sticky=True)), ('p0', dict(sticky=True, undrop=True, carry=True))):
                oracle = judge.Oracle(capacity=100, **kwargs)
                for index, prompt in enumerate(prompts):
                    admitted = oracle.admit('salt', tokens[:prompt])
                    if index:
                        want = model.POLICIES[policy](prompts[index - 1], prompt)[0]
                        self.assertEqual(admitted['q'], want, (policy, prompts, index))

    def test_every_policy_resumes_below_the_prompt_and_p0_never_below_today(self):
        rng = random.Random(7)
        for _ in range(500):
            prev = rng.randrange(4200, 250000)
            prompt = prev + rng.choice((1, 100, 169, 700, 2048, 5000))
            today, p0, exact = (model.evaluate(prev, prompt, name) for name in ('today', 'p0', 'exact_end'))
            self.assertLess(p0['q'], prompt)
            self.assertEqual(p0['q'] - today['q'], CHUNK, 'P0 is exactly one chunk higher')
            self.assertLessEqual(p0['q'], prev)
            self.assertLessEqual(exact['q'], prev)
            self.assertGreaterEqual(exact['q'], p0['q'])
            self.assertLessEqual(p0['rows'], today['rows'])
            self.assertLessEqual(exact['rows'], p0['rows'])


class GridEquivalenceTests(unittest.TestCase):
    """A byte-exact resume is the cold prefill's own programs on the cold prefill's own bytes: a suffix of its program list."""

    def test_p0_and_today_run_a_suffix_of_the_cold_prefills_programs(self):
        rng = random.Random(11)
        for _ in range(400):
            prev = rng.randrange(4200, 260000)
            prompt = prev + rng.choice((1, 64, 169, 700, 2047, 2048, 4000))
            cold = model.cold_programs(prompt)
            for policy in ('today', 'p0'):
                item = model.evaluate(prev, prompt, policy)
                self.assertEqual(item['programs'], cold[item['q'] // CHUNK:], (policy, prev, prompt))
                self.assertTrue(all(program in cold for program in item['programs']))

    def test_a_128_aligned_resume_off_the_grid_runs_a_program_the_cold_prefill_never_runs(self):
        rng = random.Random(13)
        checked = 0
        for _ in range(400):
            prev = rng.randrange(4200, 260000)
            prompt = prev + rng.choice((1, 64, 169, 700, 2047, 4000))
            item = model.evaluate(prev, prompt, 'exact_end')
            if item['q'] % CHUNK == 0:
                self.assertEqual(item['programs'], model.cold_programs(prompt)[item['q'] // CHUNK:], 'on the grid it is the cold run')
                continue
            checked += 1
            cold = model.cold_programs(prompt)
            self.assertEqual(item['programs'][0].chunk_start, item['q'])
            self.assertNotIn(item['programs'][0].chunk_start, [program.chunk_start for program in cold], 'a program starting inside a cell')
            self.assertNotIn(item['programs'][0], cold)
        self.assertGreater(checked, 300)

    def test_the_only_exact_way_to_a_cheaper_tail_is_a_smaller_bucket_than_the_cold_run_uses(self):
        """The cost of a program is its bucket. For the tail of the cold prefill (bucket b) any resume that runs the cold programs costs the cold tail's own
        bucket; a resume that saves rows has to run a SMALLER bucket than the cold run does for the same rows. P0 already runs exactly the cold tail."""
        rng = random.Random(17)
        for _ in range(400):
            prev = rng.randrange(4200, 260000)
            prompt = prev + rng.choice((1, 169, 700, 2047))
            if prompt // CHUNK != prev // CHUNK:
                continue                # a prompt that crossed into the next cell has the full chunk of the old one to run first
            cold_tail = model.cold_programs(prompt)[-1]
            p0 = model.evaluate(prev, prompt, 'p0')
            self.assertEqual(p0['programs'], [cold_tail])
            exact = model.evaluate(prev, prompt, 'exact_end')
            self.assertLessEqual(exact['programs'][-1].bucket, cold_tail.bucket)
            if exact['programs'][-1].bucket < cold_tail.bucket:
                self.assertEqual(model.bucket_for(exact['programs'][-1].valid), exact['programs'][-1].bucket)


class SavingTests(unittest.TestCase):
    def test_p0_never_adds_a_forward_and_the_session_table_has_the_cell_crossing(self):
        table = model.session_table(165901, 169, 5)
        self.assertEqual([item['forwards'] for item in table['today']], [2, 2, 2, 2])
        self.assertEqual([item['forwards'] for item in table['p0']], [1, 1, 1, 1])
        crossing = model.session_table(164887 + 169 * 5, 169, 3)          # 165,732 -> 165,901 crosses 165,888
        crossed = [item for item in crossing['p0'] if item['prompt'] == 165901][0]
        self.assertEqual(crossed['programs'], [(163840, 2048, 2048), (165888, 128, 13)], 'the turn that crosses a cell boundary runs the full chunk and a small tail')
        after = model.evaluate(165901, 165901 + 169, 'p0')
        self.assertEqual((after['q'], after['programs']), (165888, [(165888, 256, 182)]), 'and the next turn finds the checkpoint it stored at the boundary')

    def test_the_totals_order_the_policies(self):
        totals = model.simulate(150, 170, seed=3)
        per_turn = dict((name, item['seconds'] / item['turns']) for name, item in totals.items())
        old = dict((name, item['old'] / item['turns']) for name, item in totals.items())
        self.assertAlmostEqual(old['today'], 2048 + 1024, delta=150, msg='the brief measured 2.95 million re-prefilled tokens over 996 turns: 2,962 each')
        self.assertAlmostEqual(old['p0'], 1024, delta=120)
        self.assertLess(old['exact_end'], 128)
        self.assertLess(per_turn['p0'], per_turn['today'] * 0.62, 'P0 removes a forward from almost every turn')
        self.assertLess(per_turn['exact_end'], per_turn['p0'] * 0.85, 'the 128-aligned resume would save about a fifth of what is left')
        self.assertGreater(per_turn['fine'], per_turn['p0'] * 0.95, 'a prefix-stable tail grid (Option C) is no better than P0 under the measured per-forward floor')
        for name, item in totals.items():
            self.assertEqual(item['turns'], 150 * 40)

    def test_the_cost_model_is_the_measured_medians(self):
        self.assertAlmostEqual(model.final_forward_seconds(128, 32768), 0.36)
        self.assertAlmostEqual(model.final_forward_seconds(2048, 131072), 0.83)
        self.assertAlmostEqual(model.final_forward_seconds(2048, 253952), 1.34)
        self.assertAlmostEqual(model.chunk_forward_seconds(128000), 0.71)
        self.assertGreater(model.final_forward_seconds(1024, 200000), model.final_forward_seconds(256, 200000))
        # the floor is most of a small program: a 128-row program costs more than half of the 2048-row one at 128k
        self.assertGreater(model.final_forward_seconds(128, 131072), 0.5 * model.final_forward_seconds(2048, 131072))

    def test_the_command_line_prints_the_tables(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(model.main(['--start', '164887', '--delta', '169', '--turns', '3']), 0)
            self.assertEqual(model.main(['--simulate', '5']), 0)
        text = out.getvalue()
        self.assertIn('P=165056 Q=161792', text)
        self.assertIn('P=165056 Q=163840', text)
        self.assertIn('vs today', text)


# ------------------------------------------------------------------------------------------------
# A toy of the arithmetic
# ------------------------------------------------------------------------------------------------
def bf16(tensor):
    return tensor.to(torch.bfloat16).to(torch.float32)


def same_bits(a, b):
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


class Toy(object):
    """A prefill with the two properties of the engine the analysis rests on, and nothing else.

    1. THE PROGRAM: the projections are matmuls whose accumulation is cut into K blocks of `blocking(bucket)` columns, each partial sum rounded to bf16 (the intermediate circular buffer of
       a program config chosen by the bucket: graft/model_config.py prefill_progcfg(seq_len, k, n)). Two buckets with different blocking are different arithmetic.
    2. THE STORE: the recurrent state is kept in fp32 inside a program and stored in bf16 between programs (QWEN35_GDN_STATE_BF16=1); `internal_bf16` says the scan also rounds at its own 128-row
       sub-chunk boundaries (False: it does not, so a store anywhere but a program's end is a rounding the cold run does not make).
    The recurrence is token by token, so the sub-chunk structure matters only through those rounding points, counted from the program's start."""

    SUB = 128

    def __init__(self, blocking, internal_bf16, dim=8, vocab=61, seed=5):
        generator = torch.Generator().manual_seed(seed)
        self.embedding = torch.randn(vocab, dim, generator=generator)
        self.weights = torch.randn(3, dim, dim, generator=generator) * 0.5
        self.blocking, self.internal_bf16, self.dim = blocking, internal_bf16, dim

    def project(self, rows, bucket):
        block = self.blocking(bucket)
        outputs = []
        for weight in self.weights:
            accumulated = torch.zeros(rows.shape[0], self.dim)
            for k in range(0, self.dim, block):
                accumulated = bf16(accumulated + rows[:, k:k + block] @ weight[k:k + block])
            outputs.append(accumulated)
        return outputs

    def run(self, tokens, programs, state=None):
        """({position: output}, state): the programs in order from `state` (None: zeros), the state stored in bf16 at the end of each."""
        state = torch.zeros(self.dim, self.dim) if state is None else state.clone()
        outputs = {}
        for chunk_start, bucket, valid in programs:
            q, k, v = self.project(self.embedding[tokens[chunk_start:chunk_start + valid]], bucket)
            for index in range(valid):
                if index and index % self.SUB == 0 and self.internal_bf16:
                    state = bf16(state)
                state = 0.97 * state + torch.outer(k[index], v[index])
                outputs[chunk_start + index] = bf16(q[index] @ state)
            state = bf16(state)
        return outputs, state


INVARIANT = lambda bucket: 4                                    # noqa: E731 - every bucket accumulates in blocks of 4
VARYING = lambda bucket: 2 if bucket == CHUNK else 8             # noqa: E731 - the 2048 bucket blocks differently from the smaller ones


class ToyNumericsTests(unittest.TestCase):
    PROMPT = CHUNK + 1500                           # one full chunk and a tail of 1,500 rows: the cold tail runs the 2048 bucket

    def tokens(self):
        return torch.randint(0, 61, (self.PROMPT + 8,), generator=torch.Generator().manual_seed(9))

    def compare(self, cold, got, first):
        outputs = all(same_bits(cold[0][position], got[0][position]) for position in range(first, self.PROMPT))
        return outputs and same_bits(cold[1], got[1])

    def cold(self, toy, tokens):
        return toy.run(tokens, model.cold_programs(self.PROMPT))

    def resume_at_the_grid(self, toy, tokens):
        stored = toy.run(tokens, model.cold_programs(CHUNK))[1]                      # the checkpoint a longer prompt's chunk loop stores at 2048
        return toy.run(tokens, model.cold_programs(self.PROMPT, CHUNK), stored)

    def resume_inside_the_cell(self, toy, tokens, resume, forced):
        """The probe's B arms (forced False) and F arms (forced True): [0, resume) as a prefill of its own (its tail masked, in the natural bucket or forced to 2048), the stored state, then the head."""
        if forced:
            first = [(0, CHUNK, CHUNK), (CHUNK, CHUNK, resume - CHUNK)]
            head = [(resume, CHUNK, self.PROMPT - resume)]
        else:
            first = model.cold_programs(resume)
            head = model.split_programs(self.PROMPT, resume)
        outputs, state = toy.run(tokens, first)
        more, state = toy.run(tokens, head, state)
        outputs.update(more)
        return outputs, state

    def test_a_resume_on_the_grid_is_bit_equal_whatever_the_arithmetic(self):
        tokens = self.tokens()
        for blocking in (INVARIANT, VARYING):
            for internal in (True, False):
                with self.subTest(varying=blocking is VARYING, internal_bf16=internal):
                    toy = Toy(blocking, internal)
                    cold = self.cold(toy, tokens)
                    self.assertTrue(self.compare(cold, self.resume_at_the_grid(toy, tokens), CHUNK))

    def test_the_toy_is_deterministic_and_the_programs_are_the_models(self):
        tokens = self.tokens()
        toy = Toy(VARYING, True)
        first, second = self.cold(toy, tokens), self.cold(toy, tokens)
        self.assertTrue(self.compare(first, second, 0))
        self.assertEqual(model.cold_programs(self.PROMPT), [(0, 2048, 2048), (2048, 2048, 1500)])

    def test_a_128_aligned_resume_in_the_natural_buckets_is_exact_only_if_both_the_programs_and_the_store_are(self):
        tokens, resume = self.tokens(), CHUNK + 640
        expected = {(True, True): True, (True, False): False, (False, True): False, (False, False): False}
        for (invariant, internal), want in expected.items():
            with self.subTest(blocking_invariant=invariant, internal_bf16=internal):
                toy = Toy(INVARIANT if invariant else VARYING, internal)
                got = self.resume_inside_the_cell(toy, tokens, resume, forced=False)
                self.assertEqual(self.compare(self.cold(toy, tokens), got, CHUNK), want)

    def test_forcing_the_cold_bucket_removes_the_programs_from_the_question(self):
        """The probe's F arms: both segments in the cold run's own 2048 bucket, so what is left is the store (and the mask). It does not matter whether the buckets block differently."""
        tokens, resume = self.tokens(), CHUNK + 640
        for blocking in (INVARIANT, VARYING):
            for internal, want in ((True, True), (False, False)):
                with self.subTest(varying=blocking is VARYING, internal_bf16=internal):
                    toy = Toy(blocking, internal)
                    got = self.resume_inside_the_cell(toy, tokens, resume, forced=True)
                    self.assertEqual(self.compare(self.cold(toy, tokens), got, CHUNK), want)

    def test_the_attribution_the_probe_would_print_for_each_cell(self):
        """The SP1 decision table, derived from the toy: cause=bucket when only the natural arm differs, cause=handoff when the forced arm does too, none when neither."""
        tokens, resume = self.tokens(), CHUNK + 640
        causes = {}
        for invariant in (True, False):
            for internal in (True, False):
                toy = Toy(INVARIANT if invariant else VARYING, internal)
                cold = self.cold(toy, tokens)
                natural = self.compare(cold, self.resume_inside_the_cell(toy, tokens, resume, forced=False), CHUNK)
                forced = self.compare(cold, self.resume_inside_the_cell(toy, tokens, resume, forced=True), CHUNK)
                causes[(invariant, internal)] = probe.attribution([dict(length=self.PROMPT, arms=[dict(name='B+640', split=resume, identical=natural),
                                                                                                dict(name='F+640', split=resume, identical=forced)])])[0][-1]
        self.assertEqual(causes, {(True, True): 'none', (True, False): 'handoff', (False, True): 'bucket', (False, False): 'handoff'})

    def test_resuming_at_the_prefill_end_state_of_an_unaligned_prompt_is_never_exact(self):
        """The previous prompt's end (decode start) is at an arbitrary position: its state is stored where the cold run of the next prompt makes no store, and the scan's sub-chunk phase is another."""
        tokens, previous = self.tokens(), CHUNK + 1047
        for blocking in (INVARIANT, VARYING):
            for internal in (True, False):
                with self.subTest(varying=blocking is VARYING, internal_bf16=internal):
                    toy = Toy(blocking, internal)
                    outputs, state = toy.run(tokens, model.cold_programs(previous))
                    more, final = toy.run(tokens, [(previous, CHUNK, self.PROMPT - previous)], state)
                    outputs.update(more)
                    self.assertFalse(self.compare(self.cold(toy, tokens), (outputs, final), CHUNK))


# ------------------------------------------------------------------------------------------------
# Exact resume P0 beside the tile-aligned drafter window
# ------------------------------------------------------------------------------------------------
class Shard(object):
    def __init__(self, owner, chip):
        self.owner, self.chip = owner, chip

    def buffer_address(self):
        return id(self.owner) * 8 + self.chip


class RecordingOps(base.Ops):
    """test_exact_resume's fake mesh, recording the shape of every slice and concat, with two addressable shards per tensor (what the aligned join and its audit read)."""

    def __init__(self):
        super(RecordingOps, self).__init__()
        self.log = []

    def slice(self, value, begins, ends):
        self.log.append(('slice', value.shape[2], begins[2], ends[2]))
        return super(RecordingOps, self).slice(value, begins, ends)

    def concat(self, values, dim, memory_config=None):
        self.log.append(('concat', tuple(value.shape[2] for value in values)))
        return super(RecordingOps, self).concat(values, dim, memory_config)

    def get_device_tensors(self, value):
        return [Shard(value, 0), Shard(value, 1)]

    def to_torch(self, value, mesh_composer=None):
        if isinstance(value, Shard):
            return value.owner.data[value.chip].clone()
        return super(RecordingOps, self).to_torch(value, mesh_composer)


class AlignedCarryTests(unittest.TestCase):
    FIRST = 9000
    BOUNDARY = 8192
    PROMPTS = (9000, 9217, 9500, 9999, 10100, 10112, 10239, 10240 - 1)

    setUp_base = base.WindowTests.setUp
    prefill = base.WindowTests.prefill
    cold = base.WindowTests.cold

    def setUp(self):
        self.setUp_base()
        patcher = mock.patch.dict(os.environ, {twa.FLAG: '1'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def stored(self, ops):
        first = self.prefill(ops, self.FIRST)
        self.assertTrue(first.aligned)
        start, payload, nbytes, sha = carry_module.export(ops, 'mesh', first.outputs(), self.FIRST, self.BOUNDARY)
        first.close()
        return prefix_registry.Window(start, self.BOUNDARY, payload, nbytes, sha)

    def test_a_carried_aligned_window_is_the_cold_window_row_for_row(self):
        ops = RecordingOps()
        stored = self.stored(ops)
        self.assertEqual((stored.start, stored.end), (6952, 8192), 'the stored rows are exactly the window rows: no padding is stored')
        for prompt in self.PROMPTS[1:]:
            with self.subTest(prompt=prompt):
                carried = carry_module.Carried(stored, prompt, self.BOUNDARY)
                second = self.prefill(ops, prompt, start=self.BOUNDARY, carry=carried, audit=True)
                self.assertTrue(second.aligned, 'the carried capture keeps the aligned join')
                lead = (prompt - CHUNK) % 32
                self.assertEqual((carried.lead, carried.begin % 32, carried.begin), (lead, 0, prompt - CHUNK - lead))
                outputs = second.outputs()
                self.assertEqual(carried.transport, 'exact', 'the uploaded bytes, padding included, are read back as stored')
                for got, want in zip(outputs, self.cold(prompt)):
                    self.assertEqual(got.shape[2], CHUNK)
                    self.assertTrue(torch.equal(got.data, want), 'the spliced window is the cold window')
                second.close()

    def test_the_join_sees_the_bounded_set_of_shapes_the_aligned_window_exists_for(self):
        """Every concat input is a whole number of tiles and the concat is 2048 rows tall (2080 when the window starts off a tile); the one slice that cuts the window out sees one input height."""
        ops = RecordingOps()
        stored = self.stored(ops)
        rng = random.Random(23)
        heights, slices = set(), set()
        for prompt in sorted(set(rng.randrange(9000, 10240) for _ in range(40)) | {9000, 10239, 9248, 10112}):
            ops.log.clear()
            carried = carry_module.Carried(stored, prompt, self.BOUNDARY)
            second = self.prefill(ops, prompt, start=self.BOUNDARY, carry=carried)
            outputs = second.outputs()
            for got, want in zip(outputs, self.cold(prompt)):
                self.assertTrue(torch.equal(got.data, want), prompt)
            joins = [entry for entry in ops.log if entry[0] == 'concat']
            self.assertEqual(len(joins), len(outputs), 'one join per tap')
            for _, inputs in joins:
                self.assertEqual([rows % 32 for rows in inputs], [0, 0], prompt)
                self.assertEqual(len(inputs), 2, 'the carried part and the capture\'s own part')
                heights.add(sum(inputs))
            slices |= {(entry[1], entry[2]) for entry in ops.log if entry[0] == 'slice' and entry[1] > CHUNK}
            second.close()
        self.assertEqual(heights, {CHUNK, CHUNK + 32})
        self.assertEqual({rows for rows, _ in slices}, {CHUNK + 32})
        self.assertLessEqual(len(slices), 31, 'one fixed input height at one of 31 offsets')

    def test_the_padding_in_front_of_the_carried_rows_is_never_read(self):
        class Junk(carry_module.Carried):
            def host_rows(self):
                rows = super(Junk, self).host_rows()
                for tensor in rows:
                    tensor[:, :, :self.lead, :] = 7
                return rows

        ops = RecordingOps()
        stored = self.stored(ops)
        prompt = 10100
        carried = Junk(stored, prompt, self.BOUNDARY)
        second = self.prefill(ops, prompt, start=self.BOUNDARY, carry=carried)
        self.assertEqual(carried.lead, 20)
        for got, want in zip(second.outputs(), self.cold(prompt)):
            self.assertTrue(torch.equal(got.data, want))
        second.close()

    def test_the_aligned_audit_compares_with_todays_composition_and_agrees(self):
        ops = RecordingOps()
        stored = self.stored(ops)
        prompt = 10100
        with mock.patch.dict(os.environ, {twa.AUDIT_FLAG: '1'}), mock.patch.object(window, '_log') as log:
            carried = carry_module.Carried(stored, prompt, self.BOUNDARY)
            second = self.prefill(ops, prompt, start=self.BOUNDARY, carry=carried)
            outputs = second.outputs()
        said = [call.args[0] for call in log.call_args_list]
        audits = [call for call in log.call_args_list if call.args[0].startswith(twa.AUDIT_MARKER)]
        self.assertEqual(len(audits), len(outputs))
        self.assertTrue(all(call.args[1] is True for call in audits), 'exact=True on every tap')
        self.assertTrue(any(text.startswith(twa.ENGAGED_MARKER) for text in said))
        self.assertFalse(any(text.startswith(twa.FELL_BACK_MARKER) for text in said), 'the carried part does not make the join fall back')
        engaged = [call for call in log.call_args_list if call.args[0].startswith(twa.ENGAGED_MARKER)][0]
        self.assertEqual(engaged.args[2], 2, 'two parts: the carried rows and the capture\'s own')
        for got, want in zip(outputs, self.cold(prompt)):
            self.assertTrue(torch.equal(got.data, want))
        second.close()

    def test_without_the_aligned_flag_a_carried_window_is_joined_exactly_as_before(self):
        ops = RecordingOps()
        stored = self.stored(ops)
        prompt = 10100
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(twa.FLAG, None)
            carried = carry_module.Carried(stored, prompt, self.BOUNDARY)
            ops.log.clear()
            second = self.prefill(ops, prompt, start=self.BOUNDARY, carry=carried)
            self.assertFalse(second.aligned)
            outputs = second.outputs()
        self.assertEqual(carried.lead, 0, 'align() was never called')
        self.assertEqual([entry for entry in ops.log if entry[0] == 'concat'], [('concat', (140 + 0, 1908))] * len(outputs))
        self.assertFalse([entry for entry in ops.log if entry[0] == 'slice' and entry[1] > CHUNK])
        for got, want in zip(outputs, self.cold(prompt)):
            self.assertTrue(torch.equal(got.data, want))
        second.close()

    def test_the_flags_off_leave_the_uncarried_capture_as_it_was(self):
        for environ in ({twa.FLAG: '1'}, {}):
            with self.subTest(environ=environ), mock.patch.dict(os.environ, environ):
                if not environ:
                    os.environ.pop(twa.FLAG, None)
                ops = RecordingOps()
                capture = self.prefill(ops, 9000)
                self.assertEqual(capture.aligned, bool(environ))
                self.assertIsNone(capture.carried)
                for got, want in zip(capture.outputs(), self.cold(9000)):
                    self.assertTrue(torch.equal(got.data, want))
                capture.close()


# ------------------------------------------------------------------------------------------------
# The prefix gate serving another checkpoint
# ------------------------------------------------------------------------------------------------
PAIR = (checkpoint_tests.SWIFT['repo'], checkpoint_tests.SWIFT['revision'])
CHECKPOINT_LINES = [checkpoint_tests.LogReadTests.HEAD, checkpoint_tests.LogReadTests.BOUND]


class CheckpointEngine(replay_fakes.FakeEngine):
    """The fake engine of test_prefix_replay, whose log carries the contract's two checkpoint lines (what the real container logs when the pair crossed into it)."""

    def __init__(self, *args, **kwargs):
        super(CheckpointEngine, self).__init__(*args, **kwargs)
        self.lines.extend(CHECKPOINT_LINES)


class PrefixGateCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.results = tempfile.mkdtemp()
        self.addCleanup(__import__('shutil').rmtree, self.results, True)

    def run_bringup(self, engine_class, model_pair):
        harness = gate_tests.Harness(engine_class)
        runner = harness.runner(self.results)
        runner.model = model_pair
        arms = gate.plan_arms('bringup', 'general-prefix', 'general', gate_tests.profiles())
        return gate.run_plan('bringup', arms, runner, gate_tests.GOOD_ANCHOR), harness

    def test_the_argv_is_untouched_without_a_pair_and_carries_it_with_one(self):
        plain = gate.server_run('img', 'n', 'general-prefix', ['/a', '/b'])
        self.assertEqual(plain, gate.server_run('img', 'n', 'general-prefix', ['/a', '/b'], model=None))
        self.assertEqual(plain[plain.index('--model') + 1], 'Qwen/Qwen3.8-27B')
        self.assertFalse([token for token in plain if 'QWEN_C2_MODEL' in token])
        swift = gate.server_run('img', 'n', 'general-prefix', ['/a', '/b'], model=PAIR)
        self.assertEqual(swift[swift.index('--model') + 1], PAIR[0], 'the platform asks for the checkpoint it is served')
        self.assertIn('QWEN_C2_MODEL_REPO=%s' % PAIR[0], swift)
        self.assertIn('QWEN_C2_MODEL_REVISION=%s' % PAIR[1], swift)
        self.assertEqual(sorted(token for token in swift if token not in plain),
                         sorted(['QWEN_C2_MODEL_REPO=%s' % PAIR[0], 'QWEN_C2_MODEL_REVISION=%s' % PAIR[1], PAIR[0]]),
                         'nothing else about the container changes')

    def test_the_dry_run_forwards_the_pair_to_every_arm_and_half_a_pair_is_refused(self):
        path = os.path.join(self.results, 'profiles.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(gate_tests.sticky_profiles(), handle)

        def main(*extra):
            lines = []
            code = gate.main(['--image', 'img', '--profiles', path, '--results', os.path.join(self.results, 'out'), '--profile', 'c2-packed-prefix',
                              '--baseline', 'c2-packed', '--plan', 'bringup', '--dry-run'] + list(extra), devices=['/a', '/b'], log=lines.append)
            return code, lines

        code, lines = main('--model-repo', PAIR[0], '--model-revision', PAIR[1])
        self.assertEqual(code, 0, lines)
        arms = [json.loads(line) for line in lines[1:]]
        self.assertTrue(arms)
        for arm in arms:
            self.assertIn('QWEN_C2_MODEL_REPO=%s' % PAIR[0], arm['docker'])
            self.assertIn('QWEN_C2_MODEL_REVISION=%s' % PAIR[1], arm['docker'])
        code, lines = main()
        self.assertEqual(code, 0)
        self.assertFalse([line for line in lines[1:] if 'QWEN_C2_MODEL' in line])
        for half in (('--model-repo', PAIR[0]), ('--model-revision', PAIR[1])):
            with contextlib.redirect_stderr(io.StringIO()):
                code, _ = main(*half)
            self.assertEqual(code, 2, half)

    def test_findings_read_the_two_lines_and_refuse_a_log_without_them(self):
        problems, lines = gate.checkpoint_findings(PAIR, '\n'.join(['x'] + CHECKPOINT_LINES))
        self.assertEqual((problems, lines), ([], ['checkpoint %s@%s: read' % (PAIR[0], PAIR[1][:12])]))
        problems, lines = gate.checkpoint_findings(PAIR, 'a base run logs no checkpoint line')
        self.assertTrue(problems and all(item.startswith('checkpoint: ') for item in problems))
        self.assertIn('NOT READ', lines[0])
        problems, _ = gate.checkpoint_findings(PAIR, CHECKPOINT_LINES[0])
        self.assertTrue(any('never logged' in item for item in problems), 'the worker\'s bound line is read too')
        problems, _ = gate.checkpoint_findings(PAIR, '\n'.join(CHECKPOINT_LINES + [CHECKPOINT_LINES[0].replace(PAIR[0], 'org/other')]))
        self.assertTrue(any('another selection' in item for item in problems))

    def test_an_arm_whose_log_never_names_the_checkpoint_fails_and_one_that_does_passes(self):
        result, harness = self.run_bringup(None, PAIR)
        runs = [call for call in harness.calls if call[:3] == ['docker', 'run', '-d']]
        self.assertEqual(len(runs), 2)
        for call in runs:
            self.assertIn('QWEN_C2_MODEL_REPO=%s' % PAIR[0], call)
        self.assertEqual(result['verdict'], 'FAIL')
        for arm in result['arms'].values():
            self.assertEqual(arm['verdict'], 'FAIL')
            self.assertTrue(any(item.startswith('checkpoint: ') and 'never names the asked checkpoint' in item for item in arm['problems']), arm['problems'])
        result, _ = self.run_bringup(CheckpointEngine, PAIR)
        for arm in result['arms'].values():
            self.assertFalse([item for item in arm['problems'] if item.startswith('checkpoint: ')], arm['problems'])
            self.assertIn('checkpoint %s@%s: read' % (PAIR[0], PAIR[1][:12]), arm['lines'])

    def test_a_run_without_a_pair_reads_no_checkpoint_line(self):
        result, harness = self.run_bringup(None, None)
        runs = [call for call in harness.calls if call[:3] == ['docker', 'run', '-d']]
        self.assertFalse([call for call in runs if any('QWEN_C2_MODEL' in token for token in call)])
        for arm in result['arms'].values():
            self.assertFalse([line for line in arm['lines'] if line.startswith('checkpoint ')])
            self.assertFalse([item for item in arm['problems'] if item.startswith('checkpoint: ')])

    def test_the_workflow_hands_the_pair_to_the_prefix_step_only_when_it_is_named(self):
        text = (HERE.parent.parent / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
        step = text[text.index('- name: Prefix-reuse gates in the agent\'s container shape'):text.index('- name: Tau lab (W-T1)')]
        self.assertIn('MODEL_REPO: ${{ steps.job.outputs.model_repo }}', step)
        self.assertIn('MODEL_REVISION: ${{ steps.job.outputs.model_revision }}', step)
        self.assertIn('${MODEL_REPO:+--model-repo "$MODEL_REPO" --model-revision "$MODEL_REVISION"}', step)

    def test_the_job_reader_carries_the_pair_to_a_prefix_step(self):
        values = dict(C2_IMAGE_TAG='tp4-xe-1', C2_CARDS='quad', C2_ACTIONS='reset prefix', C2_PREFIX_PLAN='exact-resume',
                      C2_PREFIX_PROFILE='c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-s4-xe-audit', C2_PREFIX_BASELINE='none',
                      C2_MODEL_REPO=PAIR[0], C2_MODEL_REVISION=PAIR[1])
        profiles = job.profile_names(str(HERE / 'qwen_c2_profiles.json'))
        outputs = job.read_job(values, profiles, envs=job.profile_envs(str(HERE / 'qwen_c2_profiles.json')))
        self.assertEqual((outputs['model_repo'], outputs['model_revision']), PAIR)


if __name__ == '__main__':
    unittest.main()
