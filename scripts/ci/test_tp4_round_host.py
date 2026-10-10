"""tp4/round-host: the host work inside a verified round (QWEN_FAST_TP4_ROUND_HOST_*; every flag default off, host only).

Measured on the integration image (a [PACKED-PHASE] log of the Lever N traffic profile): an eight-live round is 157 ms, two 49.7 ms verify traces,
7.2 ms of verify host work, the 42.9 ms early draft (two quads) and 6.2 ms between. Where the host work sits on the critical path (the device idle),
and what this branch takes out of it, is docs/tp4-round-host.md. This module pins what the levers do and, as important, what they cannot do:

  - the flags are 0 or 1 (read strictly at the attach), the audit needs a lever, and with every flag off nothing is logged and no counter moves;
  - SELECT: the FP64 draft selection with the codebook checks made once is the reference's (tokens, scores) BIT FOR BIT over random operands, ties
    included; any failed check hands the whole call to the reference, which raises its own message; the codebook scan runs once per tensor and
    again after an in-place edit;
  - READ: the batched merge of the quad readback is the reference's bit for bit and declines to it on every failed check (and on a mixed dtype
    it accepts); the replicated-feature guard runs on the first reads and every 64th and reads chip 0 alone between; the audit's flag-off selection
    is always the reference;
  - KEYED, on the real four-user block over the fake device model: over random users, page tables, idle sets and T2 on or off, the staged device
    state after every round is the full stage's and the predictions are the flag-off run's; the keyed write is the tokens alone; a moved position,
    an appended page, a replaced reader or a K/V conflict takes today's diff (and raises today's error); the audit sees a tampered snapshot;
  - the ledger line: its fields, its pairing keys, its life cycle across a step, an open round written at the next step;
  - LEAN: the phases and lines nobody parses are not written, every line a reader parses stays;
  - the smoke rules, the three profiles (the parent plus the flags and nothing else, the generator reproducing them), the job pack, the
    attribution report, and the shipping lists."""

import contextlib
import json
import os
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import c2_serving_job as job
import c2_smoke_check
import dflash_packed_proposal
import draft_selector
import make_round_host_profiles as generator
import packed_verifier
import profile_twins
import round_host
import round_host_report
import serving_packed_bridge
import serving_packed_step
import serving_worker_hook
import test_dflash_round_b1 as tb1
import test_early_draft as ted
import test_quad_draft_tp4 as tq
import test_serving_packed_bridge as tbridge
import test_serving_packed_step as tstep
import test_tp4_hostgap as thg
import test_verify_prestage as tvp
import verify_prestage as vp
from tp_test_support import four_cards

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
FOLDER = HERE / 'references' / 'tp4-round-host-jobs'
IMAGE = 'tp4-rh-1'
PARENT = generator.PARENT
CONTROL, ARM, AUDITED = PARENT + '-roundhost-log', PARENT + '-roundhost', PARENT + '-roundhost-audit'
LOG, SELECT, READ, KEYED, LEAN, AUDIT = round_host.FLAGS
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.')
LEVERS = {SELECT: '1', READ: '1', KEYED: '1'}


class RoundHostCase(unittest.TestCase):
    """No QWEN_FAST_ flag from the host, the flags re-read around every test, the counters and the codebook cache clean, every line captured."""

    def setUp(self):
        environ = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
        self.addCleanup(round_host.refresh)             # runs after the environment is back
        patcher = patch.dict(os.environ, environ, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.log = []
        logger = patch.object(round_host, 'log_line', side_effect=self.log.append)
        logger.start()
        self.addCleanup(logger.stop)
        round_host.refresh()
        round_host.reset_counters()
        round_host.forget_codebooks()
        round_host.ledger.clear()
        round_host.ledger.open = False
        round_host.ledger.last_end = None
        round_host.ledger.rounds = 0

    def arm(self, **flags):
        for name in round_host.FLAGS:
            os.environ.pop(name, None)
        os.environ.update(flags)
        return round_host.refresh(strict=True)

    def lines(self, marker):
        return [line for line in self.log if line.startswith(marker)]


# ---------------------------------------------------------------------------------------------
# The flags.
# ---------------------------------------------------------------------------------------------

class FlagTests(RoundHostCase):
    def test_every_flag_is_off_by_default_and_strictly_zero_or_one(self):
        self.assertEqual(round_host.parse({}), {name: False for name in round_host.FLAGS})
        self.assertEqual(round_host.refresh(), {name: False for name in round_host.FLAGS})
        self.assertFalse(round_host.requested())
        for name in round_host.FLAGS:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    round_host.parse({name: 'yes'})
                self.assertTrue(round_host.parse({name: '1', SELECT: '1'})[name])
                self.assertEqual(round_host.parse({name: '0'})[name], False)

    def test_the_audit_needs_a_lever(self):
        for flags in ({AUDIT: '1'}, {AUDIT: '1', LOG: '1', LEAN: '1'}):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                round_host.parse(flags)
        for lever in LEVERS:
            self.assertTrue(round_host.parse({AUDIT: '1', lever: '1'})[AUDIT])

    def test_a_malformed_flag_leaves_the_import_off_and_fails_the_attach(self):
        os.environ[SELECT] = 'maybe'
        self.assertEqual(round_host.refresh(), {name: False for name in round_host.FLAGS})
        self.assertFalse(round_host.select_enabled() or round_host.ledger.on)
        with self.assertRaises(ValueError):
            round_host.engage()
        os.environ.pop(SELECT)
        os.environ[AUDIT] = '1'
        with self.assertRaises(ValueError):
            round_host.engage()

    def test_the_attach_logs_nothing_with_every_flag_off_and_one_line_naming_the_flags_otherwise(self):
        round_host.engage()
        self.assertEqual(self.log, [])
        os.environ.update({LOG: '1', SELECT: '1', AUDIT: '1'})
        round_host.engage()
        self.assertEqual(self.log, ['[PINDIAG] round host engaged log=1 select=1 read=0 keyed=0 lean=0 audit=1'])
        self.assertTrue(round_host.ledger.on and round_host.select_enabled() and round_host.audit_enabled())
        self.assertFalse(round_host.read_enabled() or round_host.keyed_enabled() or round_host.lean_enabled())

    def test_the_lean_flag_alone_turns_the_ledger_on_because_it_replaces_the_lines_it_drops(self):
        self.arm(**{LEAN: '1'})
        self.assertTrue(round_host.ledger.on and round_host.log_enabled())

    def test_a_packed_step_engages_at_its_attach_and_a_misset_flag_fails_it(self):
        block = thg.stub_block('A')
        serving_packed_step.PackedStep([block])
        self.assertEqual(self.log, [])
        os.environ[KEYED] = '1'
        serving_packed_step.PackedStep([block])
        self.assertEqual(len(self.lines(round_host.ENGAGED_MARKER)), 1)
        os.environ[KEYED] = '2'
        with self.assertRaises(ValueError):
            serving_packed_step.PackedStep([block])

    def test_the_flag_names_are_the_ones_the_profiles_and_the_smoke_check_read(self):
        self.assertEqual(set(round_host.FLAGS), set(generator.FLAGS))
        self.assertEqual({name.rsplit('_', 1)[-1].lower() for name in round_host.FLAGS},
                         {'log', 'select', 'read', 'keyed', 'lean', 'audit'})
        for name in round_host.FLAGS:
            self.assertTrue(name.startswith('QWEN_FAST_TP4_ROUND_HOST_'))


# ---------------------------------------------------------------------------------------------
# SELECT.
# ---------------------------------------------------------------------------------------------

def selector_case(seed, users=4, positions=15, k=16, vocab=700, rank=8, ties=False):
    generator_ = torch.Generator().manual_seed(seed)
    predecessors = torch.randn(vocab, rank, generator=generator_)
    successors = torch.randn(vocab, rank, generator=generator_)
    hidden = torch.randn(users, positions, rank, generator=generator_)
    candidates = torch.stack([torch.stack([torch.randperm(vocab, generator=generator_)[:k] for _ in range(positions)])
                              for _ in range(users)])
    unary = torch.randn(users, positions, k, generator=generator_)
    anchors = torch.randint(0, vocab, (users,), generator=generator_)
    if ties:
        hidden, unary = torch.zeros_like(hidden), torch.zeros_like(unary)
    return hidden, candidates, unary, predecessors, successors, anchors


def same_tensors(first, second):
    return all(a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b)) for a, b in zip(first, second, strict=True))


class SelectTests(RoundHostCase):
    def test_the_fast_selection_is_the_references_bit_for_bit_over_random_operands(self):
        for seed, positions in enumerate((1, 2, 7, 8, 9, 15, 16, 17, 24, 31)):
            for ties in (False, True):
                with self.subTest(seed=seed, positions=positions, ties=ties):
                    operands = selector_case(seed, users=1 + seed % 8, positions=positions, ties=ties)
                    before = round_host.COUNTS['select_fast']
                    fast = round_host.select_active_candidates(*operands)
                    self.assertEqual(round_host.COUNTS['select_fast'], before + 1, 'the fast path answered')
                    self.assertTrue(same_tensors(fast, draft_selector.select_active_candidates(*operands)))
        self.assertEqual(round_host.COUNTS['select_declined'], 0)

    def test_the_codebook_is_scanned_once_per_tensor_and_again_after_an_in_place_edit(self):
        operands = selector_case(3)
        round_host.select_active_candidates(*operands)
        entries = dict(round_host._FINITE)
        self.assertEqual(len(entries), 2)
        round_host.select_active_candidates(*selector_case(3))
        self.assertEqual(len(round_host._FINITE), 4, 'another tensor object is another scan')
        round_host.select_active_candidates(*operands)
        self.assertTrue(all(round_host._FINITE[key] is entry for key, entry in entries.items()), 'the same objects are not scanned again')
        operands[3][0, 0] = float('nan')                    # an ungathered row, usually; either way the cache must notice
        self.assertFalse(round_host.codebook_finite(operands[3]))
        self.assertIsNot(round_host._FINITE[id(operands[3])], entries[id(operands[3])])

    def test_a_codebook_with_a_non_finite_row_nobody_gathers_is_answered_by_the_reference_with_its_result(self):
        operands = list(selector_case(4, users=2, positions=9))
        gathered = set(operands[1].flatten().tolist()) | set(operands[5].tolist())
        spare = next(row for row in range(operands[3].shape[0]) if row not in gathered)
        operands[3][spare, 2] = float('inf')
        fast = round_host.select_active_candidates(*operands)
        self.assertTrue(same_tensors(fast, draft_selector.select_active_candidates(*operands)))
        self.assertEqual((round_host.COUNTS['select_fast'], round_host.COUNTS['select_declined']), (0, 1))
        self.assertEqual(len(self.lines(round_host.DECLINED_MARKER)), 1)

    def refusal(self, operands):
        """(the reference's message, the fast path's message): a refusal must be the reference's own."""
        with self.assertRaises(ValueError) as reference:
            draft_selector.select_active_candidates(*operands)
        with self.assertRaises(ValueError) as fast:
            round_host.select_active_candidates(*operands)
        return str(reference.exception), str(fast.exception)

    def test_every_refusal_is_the_references_own_message(self):
        cases = {}
        hidden, candidates, unary, predecessors, successors, anchors = selector_case(5, users=2, positions=12)
        gathered = candidates[0, 1, 0].item()
        bad = successors.clone()
        bad[gathered, 0] = float('nan')
        cases['non-finite gathered row'] = (hidden, candidates, unary, predecessors, bad, anchors)
        cases['non-finite hidden'] = (torch.where(torch.arange(hidden.shape[2]) == 0, float('inf'), hidden), candidates, unary, predecessors, successors, anchors)
        duplicated = candidates.clone()
        duplicated[1, 9, 3] = duplicated[1, 9, 4]
        cases['duplicate candidates in the second chunk'] = (hidden, duplicated, unary, predecessors, successors, anchors)
        cases['int32 candidates'] = (hidden, candidates.to(torch.int32), unary, predecessors, successors, anchors)
        cases['an id outside the vocabulary'] = (hidden, candidates + predecessors.shape[0], unary, predecessors, successors, anchors)
        cases['negative anchor'] = (hidden, candidates, unary, predecessors, successors, anchors - 10_000)
        wide = selector_case(5, users=2, positions=32)
        cases['32 positions'] = wide
        cases['mismatched codebooks'] = (hidden, candidates, unary, predecessors, successors[:, :4], anchors)
        cases['a score overflow'] = (hidden, candidates, unary, predecessors * 1e200, successors * 1e200, anchors)
        for name, operands in cases.items():
            with self.subTest(case=name):
                reference, fast = self.refusal(operands)
                self.assertEqual(fast, reference)
        self.assertGreaterEqual(round_host.COUNTS['select_declined'], len(cases))

    def test_the_audit_returns_the_references_bytes_and_logs_when_the_fast_path_differs(self):
        self.arm(**{SELECT: '1', AUDIT: '1'})
        operands = selector_case(6)
        round_host.select_active_candidates(*operands)
        self.assertEqual(self.lines(round_host.AUDIT_MARKER), ['[ROUND-HOST-AUDIT] kind=select equal=1'])
        expected = draft_selector.select_active_candidates(*operands)
        real = round_host._fast_select

        def wrong(*arguments):
            tokens, scores = real(*arguments)
            return tokens + 1, scores

        with patch.object(round_host, '_fast_select', wrong):
            returned = round_host.select_active_candidates(*operands)
        self.assertTrue(same_tensors(returned, expected), 'the reference stands')
        self.assertEqual(self.lines(round_host.AUDIT_MARKER)[-1], '[ROUND-HOST-AUDIT] kind=select equal=0')
        self.assertEqual((round_host.COUNTS['audit_equal'], round_host.COUNTS['audit_unequal']), (1, 1))

    def test_select_packed_batched_takes_the_fast_path_only_with_the_flag_and_answers_the_same(self):
        users, rank, vocab = 4, 8, 700
        hidden, candidates, unary, predecessors, successors, anchors = selector_case(7, users=users, positions=15, vocab=vocab, rank=rank)
        parts = [dict(hidden=hidden[u:u + 1], candidates=candidates[u:u + 1], unary=unary[u:u + 1]) for u in range(users)]
        seeds, counts = [int(value) for value in anchors], (15, 15, 9, 1)
        off = dflash_packed_proposal.select_packed_batched(parts, seeds, counts, predecessors, successors)
        self.assertEqual(round_host.COUNTS['select_fast'], 0)
        self.arm(**{SELECT: '1'})
        on = dflash_packed_proposal.select_packed_batched(parts, seeds, counts, predecessors, successors)
        self.assertEqual(round_host.COUNTS['select_fast'], 1)
        self.assertEqual(off, on)
        reference = dflash_packed_proposal.select_packed(parts, seeds, counts, predecessors, successors)
        self.assertEqual(on, reference, 'and the per-user selection (the C1 audit\'s reference)')
        self.assertEqual(round_host.levers_taken()['select'], 1)


# ---------------------------------------------------------------------------------------------
# READ.
# ---------------------------------------------------------------------------------------------

def merge_chunks(seed, rows=32, ties=False, dtype=torch.float32, index_dtype=torch.int64):
    import tp_shapes
    from draft_shared_head_tp import candidate_chunks

    generator_ = torch.Generator().manual_seed(seed)
    chunks = []
    for chip in range(tp_shapes.chip_count()):
        for start, stop in candidate_chunks():
            if ties:
                values = torch.randint(0, 4, (rows, 16), generator=generator_).to(dtype)
            else:
                values = torch.sort(torch.randn(rows, 16, generator=generator_), dim=-1, descending=True).values.to(dtype)
            indices = torch.stack([torch.randperm(stop - start, generator=generator_)[:16] for _ in range(rows)]).to(index_dtype)
            chunks.append(dict(chip=chip, start=start, stop=stop, values=values, indices=indices))
    return chunks


class ReadTests(RoundHostCase):
    def merges(self, chunks, rows):
        from draft_shared_head_tp import merge_chunk_candidates

        return round_host.merge_chunk_candidates(chunks, block_rows=rows), merge_chunk_candidates(chunks, block_rows=rows)

    def test_the_batched_merge_is_the_references_bit_for_bit(self):
        with four_cards():
            for seed in range(12):
                for rows in (8, 16, 32):
                    with self.subTest(seed=seed, rows=rows):
                        chunks = merge_chunks(seed, rows, ties=seed % 2 == 1, dtype=torch.bfloat16 if seed % 3 == 0 else torch.float32,
                                              index_dtype=torch.int32 if seed % 4 == 0 else torch.int64)
                        before = round_host.COUNTS['read_fast']
                        fast, reference = self.merges(chunks, rows)
                        self.assertEqual(round_host.COUNTS['read_fast'], before + 1)
                        self.assertTrue(same_tensors(fast, reference))
        self.assertEqual(round_host.COUNTS['read_declined'], 0)

    def test_a_mixed_index_dtype_the_reference_accepts_is_declined_to_it_and_answers_the_same(self):
        with four_cards():
            chunks = merge_chunks(1)
            chunks[2]['indices'] = chunks[2]['indices'].to(torch.int32)
            fast, reference = self.merges(chunks, 32)
            self.assertTrue(same_tensors(fast, reference))
            self.assertEqual((round_host.COUNTS['read_fast'], round_host.COUNTS['read_declined']), (0, 1))

    def test_every_refusal_is_the_references_own_message(self):
        from draft_shared_head_tp import merge_chunk_candidates

        with four_cards():
            cases = {}
            base = merge_chunks(2)
            repeated = [dict(chunk) for chunk in base]
            repeated[3] = dict(repeated[2])
            cases['a repeated chunk identity'] = (repeated, 32)
            cases['a missing chunk'] = (base[:-1], 32)
            infinite = [dict(chunk) for chunk in base]
            infinite[1]['values'] = infinite[1]['values'].clone()
            infinite[1]['values'][5, 5] = float('inf')
            cases['a non-finite value'] = (infinite, 32)
            outside = [dict(chunk) for chunk in base]
            outside[4]['indices'] = outside[4]['indices'].clone()
            outside[4]['indices'][0, 0] = outside[4]['stop'] - outside[4]['start']
            cases['an index past its chunk'] = (outside, 32)
            negative = [dict(chunk) for chunk in base]
            negative[0]['indices'] = negative[0]['indices'].clone()
            negative[0]['indices'][3, 3] = -1
            cases['a negative index'] = (negative, 32)
            twice = [dict(chunk) for chunk in base]
            twice[6]['indices'] = twice[6]['indices'].clone()
            twice[6]['indices'][7, 1] = twice[6]['indices'][7, 0]
            cases['a duplicate index'] = (twice, 32)
            cases['a block of 7 rows'] = (base, 7)
            short = [dict(chunk) for chunk in base]
            short[2]['values'] = short[2]['values'][:16]
            cases['a short chunk'] = (short, 32)
            cases['a float index'] = ([dict(chunk, indices=chunk['indices'].float()) for chunk in base], 32)
            for name, (chunks, rows) in cases.items():
                with self.subTest(case=name):
                    with self.assertRaises(ValueError) as reference:
                        merge_chunk_candidates(chunks, block_rows=rows)
                    with self.assertRaises(ValueError) as fast:
                        round_host.merge_chunk_candidates(chunks, block_rows=rows)
                    self.assertEqual(str(fast.exception), str(reference.exception))
            self.assertEqual(round_host.COUNTS['read_declined'], len(cases))

    def test_the_audit_compares_the_merge_with_the_reference_and_logs_it(self):
        self.arm(**{READ: '1', AUDIT: '1'})
        with four_cards():
            round_host.merge_chunk_candidates(merge_chunks(3), block_rows=32)
        self.assertEqual(self.lines(round_host.AUDIT_MARKER), ['[ROUND-HOST-AUDIT] kind=merge equal=1'])

    def test_the_feature_guard_runs_on_the_first_reads_and_every_64th_and_always_with_the_flag_off_or_the_audit_on(self):
        self.assertTrue(all(round_host.guard_full() for _ in range(200)), 'READ off: every read')
        self.assertEqual(round_host.COUNTS['guard_skipped'], 0)
        self.arm(**{READ: '1'})
        round_host.reset_counters()
        seen = [round_host.guard_full() for _ in range(260)]
        self.assertEqual([number + 1 for number, full in enumerate(seen) if not full and number < 70], list(range(65, 71)))
        self.assertTrue(all(seen[:64]))
        self.assertEqual([number + 1 for number, full in enumerate(seen) if full and number >= 64], [128, 192, 256])
        self.assertEqual(round_host.COUNTS['guard_skipped'], 260 - 64 - 3)
        self.arm(**{READ: '1', AUDIT: '1'})
        round_host.reset_counters()
        self.assertTrue(all(round_host.guard_full() for _ in range(200)))
        self.assertEqual(round_host.COUNTS['guard_skipped'], 0, 'the audited ledger carries guard=0 on every step: c2_smoke_check expects that')

    def quad(self):
        generator_ = torch.Generator().manual_seed(21)
        _, quad = tq.four_chip_users(generator_)
        return quad

    def read(self, quad, reads=None, **kwargs):
        import quad_draft_tp

        operations = tq.HostOps4()
        if reads is not None:
            operations.to_torch = reads
        return quad_draft_tp.read_quad_outputs(SimpleNamespace(operations=operations), quad, **kwargs)

    def test_the_quad_readback_is_the_same_bytes_with_the_flag_and_reads_fewer_chips_between_guards(self):
        with four_cards():
            quad = self.quad()
            reads = Mock(wraps=tq.HostOps4().to_torch)
            off = self.read(quad, reads)
            self.assertEqual(reads.call_count, 2 * 4 * 2 + 4)
            self.arm(**{READ: '1'})
            round_host.reset_counters()
            first = Mock(wraps=tq.HostOps4().to_torch)
            on = self.read(quad, first)
            self.assertEqual(first.call_count, 2 * 4 * 2 + 4, 'the first reads carry the guard')
            round_host._GUARD['reads'] = round_host.GUARD_FIRST
            skipped = Mock(wraps=tq.HostOps4().to_torch)
            between = self.read(quad, skipped)
            self.assertEqual(skipped.call_count, 2 * 4 * 2 + 1, 'chip 0\'s features alone')
            for parts in (on, between):
                for user in range(4):
                    for key in ('hidden', 'candidates', 'unary'):
                        self.assertTrue(torch.equal(parts[user][key], off[user][key]) and parts[user][key].dtype == off[user][key].dtype)
            self.assertEqual(round_host.COUNTS['read_fast'], 2 * 2, 'two halves a readback')

    def test_the_guard_still_refuses_replicated_features_that_differ_while_it_runs_and_the_reference_path_always_does(self):
        with four_cards():
            quad = self.quad()
            quad.projected.chips[3][0, 0, 5, 3] += 1
            self.arm(**{READ: '1'})
            with self.assertRaises(AssertionError):
                self.read(quad)
            round_host._GUARD['reads'] = round_host.GUARD_FIRST
            self.read(quad)                                 # the sampled-out read does not look
            with patch.object(round_host, 'merge_chunk_candidates', side_effect=AssertionError('the reference path never merges fast')):
                with self.assertRaises(AssertionError) as raised:
                    self.read(quad, reference=True)
            self.assertEqual(str(raised.exception), 'Replicated learned selector features differ')

    def test_a_refused_block_is_reported_per_chip_and_raises_the_same_error_with_the_flag(self):
        with four_cards():
            self.arm(**{READ: '1'})
            quad = self.quad()
            quad.chunks[1]['values'].chips[3][0, 0, 40:, :] = float('-inf')
            lines, loguru = tq.base.logged()
            with loguru, self.assertRaises(ValueError) as raised:
                import quad_draft_tp

                quad_draft_tp.read_quad_outputs(SimpleNamespace(operations=tq.HostOps4(), pool_slot=SimpleNamespace(index=0)), quad)
        self.assertEqual(str(raised.exception), 'Finite complete-block top16 values and in-range integer indices required')
        self.assertEqual(len([line for line in lines if line.startswith('[PINDIAG] draft outputs rejected')]), 4)


# ---------------------------------------------------------------------------------------------
# KEYED, on the real block.
# ---------------------------------------------------------------------------------------------

class _Equivalence(tvp.EquivalenceTests):
    """test_verify_prestage's random schedules, without its own tests (they run in their module)."""


for _name in [name for name in dir(tvp.EquivalenceTests) if name.startswith('test_')]:
    setattr(_Equivalence, _name, None)


class KeyedEquivalenceTests(_Equivalence):
    def setUp(self):
        super().setUp()
        environ = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_TP4_ROUND_HOST')}
        patcher = patch.dict(os.environ, environ, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(round_host.refresh)
        self.round_host_log = []
        logger = patch.object(round_host, 'log_line', side_effect=self.round_host_log.append)
        logger.start()
        self.addCleanup(logger.stop)
        round_host.refresh()
        round_host.reset_counters()

    def arm(self, **flags):
        for name in round_host.FLAGS:
            os.environ.pop(name, None)
        os.environ.update(flags)
        round_host.refresh(strict=True)
        round_host.reset_counters()

    @staticmethod
    def expected_keyed(plan):
        """The rounds whose users' start and table at the verify are the window's (a vLLM append after the window moves a table)."""
        out = []
        for number, spec in enumerate(plan):
            same = all(spec['users'][name][0] == spec['window'][name][0] and torch.equal(spec['users'][name][1], spec['window'][name][1])
                       for name in spec['live'])
            out.append(number > 0 and same)
        return out

    def test_random_rounds_stage_the_full_stages_state_and_predict_the_control_with_the_keyed_write_on(self):
        taken = declined = 0
        for seed in range(8):
            padded, kv_chains = seed % 2 == 1, seed % 4 >= 2
            with self.subTest(seed=seed, padded=padded, kv_chains=kv_chains):
                plan = self.schedule(random.Random(seed), rounds=6, padded=padded)
                self.arm(**{KEYED: '1'})
                self.h1a.clear()
                on = self.run_schedule(plan, True, padded, kv_chains)         # asserts, every round, that the device holds the full stage's bytes
                paths = self.paths()
                keyed, missed = round_host.COUNTS['keyed'], round_host.COUNTS['keyed_declined']
                self.arm()
                off = self.run_schedule(plan, False, padded, kv_chains)
                self.assertEqual(on, off, 'the predictions are the flag-off run\'s')
                self.assertEqual(paths, ['full'] + ['diff'] * (len(plan) - 1))
                self.assertEqual(keyed, sum(self.expected_keyed(plan)), 'the keyed write ran exactly where the key stood')
                self.assertEqual(keyed + missed, len(plan) - 1, 'every other snapshot took the diff, counted')
                taken += keyed
                declined += missed
        self.assertGreater(taken, 8)
        self.assertGreater(declined, 3, 'the schedules append pages after the window: the diff stays the answer there')

    def test_the_keyed_write_is_the_tokens_buffer_alone_and_the_same_write_the_diff_makes(self):
        block = self.open_block()
        users = tvp.base_users()
        self.round(block, users)
        nxt = tvp.advanced(users, 3)
        self.window(block, nxt)
        diff_written = self.written(lambda: self.round(block, nxt))              # keyed off: the value diff
        self.assertEqual(diff_written, [block.fixture.tokens])
        self.arm(**{KEYED: '1'})
        block = self.open_block()
        self.round(block, users)
        self.window(block, nxt)
        self.assertIsNotNone(block.prestaged.snapshot.key)
        written = self.written(lambda: self.round(block, nxt))
        self.assertEqual(written, [block.fixture.tokens])
        self.assertEqual(self.paths()[-1], 'diff')
        self.assertIn('path=diff buffers=1 ', self.marked(vp.MARKER)[-1])
        self.assertEqual((block.prestaged.counts['keyed'], round_host.COUNTS['keyed']), (1, 1))
        self.assertEqual(round_host.levers_taken()['keyed'], 1)

    def test_a_snapshot_built_without_the_flag_has_no_key_and_the_flag_off_block_counts_no_keyed_round(self):
        block = self.open_block()
        users = tvp.base_users()
        self.round(block, users)
        self.window(block, tvp.advanced(users, 2))
        self.assertIsNone(block.prestaged.snapshot.key)
        self.round(block, tvp.advanced(users, 2))
        self.assertNotIn('keyed', block.prestaged.counts)
        self.assertNotIn('keyed', block.prestaged.last)
        self.assertEqual(round_host.COUNTS['keyed'] + round_host.COUNTS['keyed_declined'], 0)
        self.assertEqual(self.round_host_log, [])

    def declined_round(self, change):
        """A keyed arm whose verify finds `change(block, users)` different from the window: the write is the diff's, counted as declined."""
        self.arm(**{KEYED: '1'})
        block = self.open_block()
        users = tvp.base_users()
        self.round(block, users)
        nxt = tvp.advanced(users, 3)
        self.window(block, nxt)
        verify_users = change(block, nxt)
        predictions = self.round(block, verify_users)[0]
        self.assertEqual((round_host.COUNTS['keyed'], round_host.COUNTS['keyed_declined']), (0, 1))
        self.assertEqual(self.paths()[-1], 'diff')
        return block, verify_users, predictions

    def test_a_moved_start_takes_the_diff(self):
        block, users, predictions = self.declined_round(lambda block, nxt: tvp.advanced(nxt, 1))
        self.assertIn('path=diff', self.marked(vp.MARKER)[-1])

    def test_an_appended_page_takes_the_diff_and_rewrites_that_users_page_buffers(self):
        def append(block, nxt):
            appended = dict(nxt)
            position, pages = appended['C']
            pages = pages.clone()
            pages[0, (position + 15) // 64 + 1] = 99
            appended['C'] = (position, pages)
            return appended

        block, users, predictions = self.declined_round(append)
        self.assertGreater(int(re.search(r'buffers=(\d+)', self.marked(vp.MARKER)[-1]).group(1)), 1)

    def test_a_replaced_reader_declines_the_key(self):
        self.arm(**{KEYED: '1'})
        block = self.open_block()
        users = tvp.base_users()
        self.round(block, users)
        nxt = tvp.advanced(users, 3)
        self.window(block, nxt)
        snapshot = block.prestaged.snapshot
        staged = self.staged_users(block, self.entries(nxt))
        self.assertTrue(block.prestaged.key_matches(snapshot, staged))
        readers = list(block.fixture.replay_reader.readers)
        block.fixture.replay_reader.readers = [object()] + readers[1:]
        self.assertFalse(block.prestaged.key_matches(snapshot, staged))
        block.fixture.replay_reader.readers = readers
        self.assertTrue(block.prestaged.key_matches(snapshot, staged))
        snapshot.key.tables = (snapshot.key.tables[0] + 1,) + snapshot.key.tables[1:]
        self.assertFalse(block.prestaged.key_matches(snapshot, staged), 'a table that moved')

    def test_a_chained_kv_conflict_declines_the_key_and_the_diff_raises_what_it_always_raised(self):
        def conflicting():
            users = tvp.base_users()
            position, pages = users['A']
            users['C'] = (position, pages.clone())
            return users

        messages = []
        for keyed in (False, True):
            self.arm(**({KEYED: '1'} if keyed else {}))
            block = self.open_block(kv_chains=True)
            first = tvp.base_users()
            self.round(block, first)
            users = tvp.advanced(conflicting(), 3)
            self.window(block, users)
            if keyed:
                self.assertFalse(block.prestaged.snapshot.key.kv_ok)
            with self.assertRaises(ValueError) as raised:
                self.round(block, users)
            messages.append(str(raised.exception))
        self.assertEqual(messages[0], messages[1])
        self.assertIn('disjoint cache tile rows', messages[0])
        self.assertEqual(round_host.COUNTS['keyed'], 0)

    def test_invalid_tokens_raise_the_same_error_before_any_copy_keyed_or_not(self):
        raised = []
        for keyed in (False, True):
            self.arm(**({KEYED: '1'} if keyed else {}))
            block = self.open_block()
            users = tvp.base_users()
            self.round(block, users)
            nxt = tvp.advanced(users, 3)
            self.window(block, nxt)
            entries = self.entries(nxt)
            entries[0]['ticket'].tokens = [10 ** 9] * 16               # past the vocabulary
            copies = len(self.ttnn.host_copies)
            with self.assertRaises(ValueError) as error:
                block.verify(entries)
            raised.append(str(error.exception))
            self.assertEqual(len(self.ttnn.host_copies), copies, 'nothing was written')
        self.assertEqual(raised[0], raised[1])

    def test_the_audit_runs_the_diff_and_logs_the_claim_and_sees_a_tampered_snapshot(self):
        self.arm(**{KEYED: '1', AUDIT: '1'})
        block = self.open_block()
        users = tvp.base_users()
        self.round(block, users)
        nxt = tvp.advanced(users, 3)
        self.window(block, nxt)
        self.round(block, nxt)
        self.assertEqual(round_host.COUNTS['keyed'], 0, 'the audited round is the diff\'s')
        self.assertEqual(round_host.COUNTS['keyed_declined'], 0, 'and no decline: the key stood')
        self.assertEqual(round_host.levers_taken()['keyed'], 1, 'the audited arm\'s ledger counts the rounds whose key stood')
        self.assertEqual(len([line for line in self.round_host_log if re.match(r'\[ROUND-HOST-AUDIT\] kind=keyed equal=1 ', line)]), 1)
        # a snapshot holding a wrong value for a non-token destination: the claim "only the tokens differ" is false there
        nxt2 = tvp.advanced(nxt, 2)
        self.window(block, nxt2)
        snapshot = block.prestaged.snapshot
        index = next(i for i, kept in enumerate(snapshot.values) if kept is not None)
        kept = list(snapshot.values[index])
        kept[0] = kept[0] + 1 if kept[0].dtype != torch.bool else ~kept[0]
        snapshot.values[index] = tuple(kept)
        self.round(block, nxt2)
        self.assertRegex(self.round_host_log[-1], r'^\[ROUND-HOST-AUDIT\] kind=keyed equal=0 checked=\d+ changed=')
        self.assertEqual((round_host.COUNTS['audit_equal'], round_host.COUNTS['audit_unequal']), (1, 1))

    def test_the_host_tokens_are_packed_host_inputs_tokens_and_refuse_what_it_refuses(self):
        block = self.open_block()
        users = self.staged_users(block, self.entries(tvp.base_users()))
        model = block.model
        arguments = (block.shape, model.args.rope_head_dim, model.args.rope_theta, model.args.vocab_size)
        reference = packed_verifier.packed_host_inputs(users, *arguments)[0]
        mine = packed_verifier.packed_host_tokens(users, block.shape, model.args.vocab_size)
        self.assertTrue(mine.dtype == reference.dtype and torch.equal(mine, reference))
        outside = (10 ** 9,) * len(users[0][0])
        cases = {'a token past the vocabulary': [(outside, *users[0][1:])] + users[1:],
                 'a start past the capacity': [(users[0][0], block.shape.capacity, users[0][2])] + users[1:],
                 'a short token list': [(tuple(users[0][0])[:-1], *users[0][1:])] + users[1:],
                 'a missing segment': users[:-1], 'an empty segment': [None] + users[1:]}

        def outcome(call):
            try:
                return ('answer', call())
            except ValueError as error:
                return ('refused', str(error))

        for name, broken in cases.items():
            with self.subTest(case=name):
                expected = outcome(lambda: packed_verifier.packed_host_inputs(broken, *arguments)[0])
                got = outcome(lambda: packed_verifier.packed_host_tokens(broken, block.shape, model.args.vocab_size))
                self.assertEqual(got[0], expected[0])
                if expected[0] == 'refused':
                    self.assertEqual(got[1], expected[1])
                else:
                    self.assertTrue(torch.equal(got[1], expected[1]))
        self.assertGreaterEqual(sum(1 for broken in cases.values() if outcome(lambda: packed_verifier.packed_host_tokens(
            broken, block.shape, model.args.vocab_size))[0] == 'refused'), 3)


class TwoBlockKeyedTests(thg.TwoRealBlockTests):
    """Two real blocks in blocks mode, driven as serving drives them, the keyed write on both: every prediction is the flag-off run's."""

    def setUp(self):
        super().setUp()
        environ = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_TP4_ROUND_HOST')}
        patcher = patch.dict(os.environ, environ, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(round_host.refresh)
        logger = patch.object(round_host, 'log_line')
        logger.start()
        self.addCleanup(logger.stop)

    def test_both_blocks_take_the_keyed_write_after_the_first_round_and_predict_what_the_control_predicts(self):
        os.environ[KEYED] = '1'
        round_host.refresh(strict=True)
        round_host.reset_counters()
        self.run_arm(audit=False)
        self.assertEqual(round_host.COUNTS['keyed'], 2 * 2, 'both blocks, the two rounds after the first')
        self.assertEqual(round_host.COUNTS['keyed_declined'], 0)

    def test_the_full_read_back_audit_of_the_hostgap_branch_sees_zero_mismatches_on_the_keyed_rounds_too(self):
        os.environ[KEYED] = '1'
        round_host.refresh(strict=True)
        round_host.reset_counters()
        blocks = self.run_arm(audit=True)
        audits = [re.search(r'block=(\S+) round=\d+ path=(\w+) buffers=(\d+) checked=(\d+) mismatches=(\d+)', line).groups()
                  for line in self.marked(vp.FULL_AUDIT_MARKER)]
        self.assertTrue(audits and all(int(item[4]) == 0 for item in audits), audits)
        self.assertEqual(round_host.COUNTS['keyed'], 2 * 2)
        self.assertTrue(blocks)


for _name in ('test_both_real_blocks_take_the_diff_path_after_the_first_round_and_predict_what_the_control_predicts',
              'test_audited_both_blocks_read_back_every_destination_with_zero_mismatches'):
    setattr(TwoBlockKeyedTests, _name, None)


# ---------------------------------------------------------------------------------------------
# The ledger.
# ---------------------------------------------------------------------------------------------

class LedgerTests(RoundHostCase):
    def test_the_line_has_every_field_in_order_its_pairing_keys_and_the_paths_taken(self):
        values = {'gap': 1.5, 'entry': 2.0, 'v0': 56.0, 'c0': 2.0, 'ed': 42.5, 'select': 2.15}
        line = round_host.format_line(7, 8, values, dict(select=1, read=4, keyed=2, guard=1), {}, 3500.4)
        self.assertEqual(line, '[PACKED-ROUND-HOST] round=7 live=8 pos=3500 gap=1.50 entry=2.00 v0=56.00 v0_in=- v0_tr=- v0_sy=- v0_rb=- c0=2.00 '
                               'v1=- v1_in=- v1_tr=- v1_sy=- v1_rb=- c1=- ed=42.50 ed_pre=- launch=- window=- fence=- collect=- flush=- '
                               'select=2.15 tail=- step=- sel=1 read=4 keyed=2 guard=1')
        self.assertEqual([item.split('=')[0] for item in line.split()[4:-4]], list(round_host.Ledger.FIELDS))
        facts = c2_smoke_check.round_host_facts(line)
        self.assertEqual(len(facts['steps']), 1)
        step = facts['steps'][0]
        self.assertEqual((step['live'], step['position'], step['sel'], step['read'], step['keyed'], step['guard']), (8, 3500, 1, 4, 2, 1))
        self.assertEqual(step['fields']['gap'], 1.5)
        self.assertIsNone(step['fields']['v0_in'])
        self.assertEqual(len(round_host_report.steps_of('x ' + line + ' y', live=8)), 1)

    def test_a_step_is_stamped_at_its_seams_and_written_once(self):
        self.arm(**{LOG: '1'})
        ledger = round_host.ledger
        ledger.begin(now=10.000)
        ledger.set_live(8, 4096.0)
        ledger.mark('step', now=10.002)
        ledger.note_verify(dict(input_ms=2.4, blocking_trace_host_ms=50.4, replay_checks_sync_ms=0.1, output_readback_host_ms=0.8), 54.0)
        ledger.note_commit(2.0)
        ledger.note_verify(dict(input_ms=2.5, blocking_trace_host_ms=50.6, replay_checks_sync_ms=0.2, output_readback_host_ms=0.9), 55.0)
        ledger.note_commit(1.5)
        ledger.mark('stepped', now=10.116)
        ledger.mark('draft0', now=10.117)
        ledger.mark('quads0', now=10.119)
        ledger.mark('quads1', now=10.124)
        ledger.mark('window0', now=10.124)
        ledger.mark('window1', now=10.142)
        ledger.mark('fence0', now=10.142)
        ledger.mark('fence1', now=10.150)
        ledger.mark('collect0', now=10.150)
        ledger.mark('collect1', now=10.1545)
        ledger.mark('flush1', now=10.1546)
        ledger.mark('select1', now=10.1567)
        ledger.mark('draft1', now=10.158)
        line = ledger.emit(now=10.159)
        self.assertEqual(self.log, [line])
        self.assertIsNone(ledger.emit(), 'once')
        facts = c2_smoke_check.round_host_facts(line)['steps'][0]['fields']
        for name, wanted in dict(entry=2.0, step=114.0, v0=54.0, v1=55.0, c0=2.0, c1=1.5, v0_in=2.4, v1_tr=50.6, ed=41.0, ed_pre=2.0,
                                 launch=5.0, window=18.0, fence=8.0, collect=4.5, flush=0.1, select=2.1, tail=1.3).items():
            self.assertAlmostEqual(facts[name], wanted, delta=0.06, msg=name)
        ledger.begin(now=10.200)                                # the next step: the gap since this one's drafts ended
        self.assertAlmostEqual(ledger.values['gap'], 42.0, delta=0.01)

    def test_a_round_still_open_when_the_next_step_enters_is_written_as_it_stands(self):
        self.arm(**{LEAN: '1'})
        ledger = round_host.ledger
        ledger.begin(now=1.0)
        ledger.set_live(4, 100.0)
        ledger.mark('step', now=1.001)
        ledger.begin(now=2.0)
        self.assertEqual(len(self.lines(round_host.LEDGER_MARKER)), 1)
        self.assertIn('live=4 pos=100 ', self.log[0])
        self.assertIn('ed=- ', self.log[0])
        self.assertTrue(ledger.open)

    def test_off_every_method_returns_at_once_and_writes_nothing(self):
        ledger = round_host.ledger
        self.assertFalse(ledger.on)
        ledger.begin()
        ledger.mark('step')
        ledger.add('x', 1.0)
        ledger.set_live(8, 1.0)
        ledger.note_verify({}, 1.0)
        ledger.note_commit(1.0)
        self.assertIsNone(ledger.emit())
        self.assertEqual((self.log, ledger.open, ledger.marks, ledger.values), ([], False, {}, {}))

    def fixture(self):
        return tbridge.PackedBridgeTests('test_the_device_step_runs_once_and_update_states_runs_once').fixture()

    def run_step(self, bridges, scheduled, order):
        def packed_step(entries, *, cancelled):
            order.append('packed-step')
            return [SimpleNamespace(request_id=entry['request_id'], token_ids=[1]) for entry in entries]

        reserve = patch('serving_vllm_state.validate_runner_reservation', side_effect=lambda *args: order.append('reserve'))
        with patch('serving_vllm_state.apply_committed_output'), reserve, \
                patch('serving_packed_bridge.packed_model_runner_output', side_effect=lambda values: [v.request_id for v in values]):
            return serving_packed_bridge.execute_packed_decode(bridges, scheduled, cancelled=lambda: False, packed_step=packed_step)

    def test_the_bridge_opens_the_ledger_with_the_live_users_and_their_mean_frontier_and_leaves_the_call_order_alone(self):
        orders = []
        for flags in ({}, {LOG: '1', LEAN: '1'}):
            runner, bridges, scheduled, _ = self.fixture()
            order = []
            runner._update_states = Mock(side_effect=lambda value: order.append('update'))
            for name, item in bridges.items():
                item.page_binding.refresh = Mock(side_effect=lambda *a, name=name, **k: order.append('refresh-' + name))
                item.validate_storage = None
            self.arm(**flags)
            orders.append((self.run_step(bridges, scheduled, order), order))
        self.assertEqual(orders[0], orders[1])
        self.assertEqual(orders[0][1], ['update', 'reserve', 'refresh-B', 'reserve', 'refresh-A', 'packed-step'])
        ledger = round_host.ledger
        self.assertTrue(ledger.open)
        self.assertEqual(ledger.live, 2)
        self.assertTrue({'begin', 'step', 'stepped'} <= set(ledger.marks))
        self.assertIsNotNone(ledger.position)


class SeamTests(RoundHostCase):
    """The ledger's seams in the real code: the coordinator's select_round (through the real pair trace) and the early draft's step."""

    def selected_round(self, **flags):
        generator_ = torch.Generator().manual_seed(31)
        predecessors, successors = tb1.codebooks(generator_)
        self.arm(**flags)
        from dflash_packed_proposal_coordinator import select_round

        with tb1.fake_trace_runtime(), tb1.round_b1(True):
            pairs = [tb1.build_pair(seed, predecessors=predecessors, successors=successors, generator=torch.Generator().manual_seed(100 + seed))
                     for seed in (40, 50)]
            round_host.ledger.begin()
            after = Mock()
            select_round([([0, 1], pairs[0]), ([2, 3], pairs[1])], [], 1, after_collect=after)
            self.assertEqual(after.call_count, 1)
            return [trace.finish(which, 15) for trace in pairs for which in ('a', 'b')]

    def test_select_round_stamps_collect_flush_and_select_and_the_fast_selection_gives_the_same_tokens(self):
        off = self.selected_round()
        self.assertFalse(round_host.ledger.open)
        on = self.selected_round(**{LOG: '1', SELECT: '1'})
        self.assertEqual(on, off)
        self.assertEqual(round_host.COUNTS['select_fast'], 1, 'one batched call for both pairs')
        self.assertEqual(round_host.COUNTS['select_declined'], 0)
        ledger = round_host.ledger
        self.assertTrue({'collect0', 'collect1', 'flush1', 'select1'} <= set(ledger.marks))
        self.assertLessEqual(ledger.marks['collect0'], ledger.marks['collect1'])
        self.assertLessEqual(ledger.marks['collect1'], ledger.marks['flush1'])
        self.assertLessEqual(ledger.marks['flush1'], ledger.marks['select1'])

    def test_the_early_draft_marks_its_draft_and_writes_the_steps_line_after_it(self):
        for flags, lines in (({}, 0), ({LOG: '1'}, 1)):
            self.arm(**flags)
            self.log.clear()
            harness = ted.Harness(self, ted.GDN)
            harness.on_step = lambda bridges: (round_host.ledger.begin(), round_host.ledger.set_live(2, 4096.0), round_host.ledger.mark('step'),
                                               round_host.ledger.mark('stepped'))
            with patch('early_draft.log_line'):
                harness.execute()
            written = self.lines(round_host.LEDGER_MARKER)
            self.assertEqual(len(written), lines, flags)
            if lines:
                fields = c2_smoke_check.round_host_facts(written[0])['steps'][0]['fields']
                self.assertIsNotNone(fields['ed'])
                self.assertIsNotNone(fields['step'])
                self.assertFalse(round_host.ledger.open)


# ---------------------------------------------------------------------------------------------
# LEAN.
# ---------------------------------------------------------------------------------------------

class _Publish(tstep.PublishInstrumentationTests):
    """test_serving_packed_step's two-user round with a stage-timed runtime, without its own tests."""


for _name in [name for name in dir(tstep.PublishInstrumentationTests) if name.startswith('test_')]:
    setattr(_Publish, _name, None)


class LeanTests(_Publish):
    def setUp(self):
        super().setUp()
        environ = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
        patcher = patch.dict(os.environ, environ, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(round_host.refresh)
        round_host.refresh()
        self.written = []
        logger = patch.object(round_host, 'log_line', side_effect=self.written.append)
        logger.start()
        self.addCleanup(logger.stop)

    def audited_round(self, **flags):
        os.environ.update(flags)
        os.environ['QWEN_FAST_PACKED_AUDIT'] = '1'
        round_host.refresh(strict=True)
        with patch('serving_packed_step.audit_log') as log:
            self.step(self.two())
        return [item.args[0].split(' ')[0] for item in log.call_args_list]

    def test_the_commit_host_line_is_written_without_the_flag_and_not_with_it_and_the_publish_line_stays(self):
        off = self.audited_round()
        self.assertIn('[PACKED-COMMIT-HOST]', off)
        self.assertIn('[PACKED-PUBLISH]', off)
        on = self.audited_round(**{LEAN: '1'})
        self.assertNotIn('[PACKED-COMMIT-HOST]', on)
        self.assertEqual([line for line in on if line != '[PACKED-COMMIT-HOST]'], [line for line in off if line != '[PACKED-COMMIT-HOST]'],
                         'every other audit line of the round is the same lines in the same order')

    def test_the_publication_splits_are_neither_timed_nor_written_with_the_flag(self):
        from dflash_traced_publish import PUBLICATION_SPLITS

        sinks = []
        entries = self.two()
        for item in entries:
            drafter = item['request'].runtime.drafter
            publish = drafter.prepare_publication

            def recording(features, prefix, *, position, publish=publish, **options):
                sinks.append(PUBLICATION_SPLITS.get())
                return publish(features, prefix, position=position, **options)
            drafter.prepare_publication = recording
        os.environ.update({'QWEN_FAST_PACKED_AUDIT': '1', 'QWEN_FAST_ROUND_B1': '1', LEAN: '1'})
        round_host.refresh(strict=True)
        with patch('serving_packed_step.audit_log') as log:
            self.step(entries)
        self.assertEqual(sinks, [None, None])
        self.assertFalse(any(call.args[0].startswith('[PACKED-PUBLISH-SPLIT]') for call in log.call_args_list))

    def test_the_ledger_sees_each_verify_and_its_commits_with_the_flag_and_nothing_without_it(self):
        self.audited_round()
        self.assertEqual((self.written, round_host.ledger.values), ([], {}))
        os.environ.update({LEAN: '1', 'QWEN_FAST_PACKED_AUDIT': '1'})
        round_host.refresh(strict=True)
        round_host.ledger.begin()
        with patch('serving_packed_step.audit_log'):
            self.step(self.two())
        values = round_host.ledger.values
        self.assertIn('v0', values)
        self.assertIn('c0', values)
        self.assertGreaterEqual(values['v0'], 0.0)
        self.assertNotIn('v1', values, 'one block, one verify')

    def test_phase_does_not_write_the_dropped_phases_and_still_writes_the_others(self):
        written = []

        class Logger:
            def info(self, message, *args):
                written.append(message.format(*args))

        fake = SimpleNamespace(logger=Logger())
        with patch.dict(sys.modules, {'loguru': fake}), patch.object(serving_worker_hook, 'PHASE_LOG', True):
            for lean in (False, True):
                written.clear()
                os.environ.pop(LEAN, None)
                if lean:
                    os.environ[LEAN] = '1'
                round_host.refresh(strict=True)
                for name in sorted(round_host.LEAN_PHASES) + ['packed_verify']:
                    self.assertEqual(serving_worker_hook.phase(name, 'x', lambda: 7), 7)
                if lean:
                    self.assertEqual(len(written), 2)
                    self.assertTrue(written[0] == '[PHASE] packed_verify x begin' and written[1].startswith('[PHASE] packed_verify x end '))
                else:
                    self.assertEqual(len(written), 2 * (len(round_host.LEAN_PHASES) + 1))

    def test_the_phases_the_gates_and_the_timing_reader_parse_are_not_dropped(self):
        self.assertNotIn('packed_verify', round_host.LEAN_PHASES)
        parsed = c2_smoke_check.ROUND_HOST_LEAN_ABSENT
        for phase in round_host.LEAN_PHASES:
            self.assertTrue(any(pattern.search('[PHASE] %s id begin' % phase) for pattern in parsed), phase)
        self.assertFalse(any(pattern.search('[PHASE] packed_verify id begin') or pattern.search('[PHASE] execute total=1 new=0')
                             or pattern.search('[PACKED] request=x segment=0') or pattern.search('[PACKED-PHASE] round=1')
                             or pattern.search('[PACKED-PUBLISH] round=1') or pattern.search('[PACKED-FENCES] round=1') for pattern in parsed))


# ---------------------------------------------------------------------------------------------
# The smoke rules.
# ---------------------------------------------------------------------------------------------

def engaged_line(**flags):
    state = {name.rsplit('_', 1)[-1].lower(): int(flags.get(name, '0') == '1') for name in round_host.FLAGS}
    return '[PINDIAG] round host engaged ' + ' '.join('%s=%d' % item for item in state.items())


def ledger_lines(count, live=8, sel=1, read=4, keyed=2, guard=0, start=1000, drop=()):
    out = []
    for number in range(count):
        values = {name: 1.0 for name in round_host.Ledger.FIELDS if name not in drop}
        out.append(round_host.format_line(number + 1, live, values, dict(select=sel, read=read, keyed=keyed, guard=guard), {}, start + number))
    return out


def smoke_text(*groups):
    return '\n'.join(line for group in groups for line in group) + '\n'


ARM_ENV = {LOG: '1', SELECT: '1', READ: '1', KEYED: '1', LEAN: '1'}


class SmokeRuleTests(unittest.TestCase):
    def problems(self, env, text, steady_eight=True):
        return c2_smoke_check.round_host_problems(env, text, steady_eight)[0]

    def clean(self, env=None, **kwargs):
        env = ARM_ENV if env is None else env
        return smoke_text([engaged_line(**env)], ledger_lines(120, guard=1, **kwargs))

    def test_a_profile_without_the_flags_logs_none_of_the_lines(self):
        self.assertEqual(self.problems({}, 'nothing here\n'), [])
        self.assertTrue(self.problems({}, smoke_text(ledger_lines(2))))
        self.assertTrue(self.problems({}, engaged_line() + '\n'))

    def test_a_clean_arm_passes(self):
        self.assertEqual(self.problems(ARM_ENV, self.clean()), [])

    def test_a_clean_control_with_the_ledger_alone_passes_and_needs_no_lever_counters(self):
        env = {LOG: '1'}
        text = smoke_text([engaged_line(**env)], ledger_lines(120, sel=0, read=0, keyed=0))
        self.assertEqual(self.problems(env, text), [])

    def test_the_engaged_line_is_once_and_names_the_profiles_flags(self):
        ledger = ledger_lines(120, guard=1)
        self.assertTrue(any('0 "[PINDIAG] round host engaged" lines' in p for p in self.problems(ARM_ENV, smoke_text(ledger))))
        self.assertTrue(any('2 "[PINDIAG] round host engaged" lines' in p for p in self.problems(ARM_ENV, smoke_text([engaged_line(**ARM_ENV)] * 2, ledger))))
        wrong = engaged_line(**dict(ARM_ENV, **{KEYED: '0'}))
        self.assertTrue(any('keyed=0' in p for p in self.problems(ARM_ENV, smoke_text([wrong], ledger))))

    def test_a_declined_or_refused_line_fails_the_arm(self):
        text = self.clean() + '[PINDIAG] round host declined kind=select reason=check_line_341\n'
        self.assertTrue(any('declined to the reference' in p for p in self.problems(ARM_ENV, text)))
        self.assertTrue(any('refusal' in p for p in self.problems(ARM_ENV, self.clean() + '[PINDIAG] round host refused x\n')))

    def test_the_ledger_must_run_and_carry_every_field(self):
        text = smoke_text([engaged_line(**ARM_ENV)])
        self.assertTrue(any('ledger never ran' in p for p in self.problems(ARM_ENV, text)))
        short = smoke_text([engaged_line(**ARM_ENV)], [line.replace(' collect=1.00', '') for line in ledger_lines(120, guard=1)])
        self.assertTrue(any('lacks the fields collect' in p for p in self.problems(ARM_ENV, short)))
        self.assertTrue(any('wrote lines' in p for p in self.problems({SELECT: '1'}, smoke_text([engaged_line(**{SELECT: '1'})], ledger_lines(3)))))

    def test_each_lever_must_have_run_on_the_steps_that_drafted(self):
        self.assertTrue(any('fast selection' in p for p in self.problems(ARM_ENV, self.clean(sel=0))))
        self.assertTrue(any('fast quad merge' in p for p in self.problems(ARM_ENV, self.clean(read=0))))
        self.assertTrue(any('keyed write' in p for p in self.problems(ARM_ENV, self.clean(keyed=0))))
        self.assertTrue(any('never skipped' in p for p in self.problems(ARM_ENV, smoke_text([engaged_line(**ARM_ENV)], ledger_lines(150, guard=0)))))
        # Under AUDIT the guard runs on every read (round_host.guard_full), so the rule reverses: no skip is the expected ledger (the H1 log of
        # run 38018599272: 375 guards, 0 skips) and a skip fails the arm.
        audited = dict(ARM_ENV, **{AUDIT: '1'})
        audits = ['[ROUND-HOST-AUDIT] kind=select equal=1', '[ROUND-HOST-AUDIT] kind=merge equal=1', '[ROUND-HOST-AUDIT] kind=keyed equal=1 checked=149 changed=0']
        trail = '\n'.join(audits) + '\n'
        self.assertEqual(self.problems(audited, smoke_text([engaged_line(**audited)], ledger_lines(150, guard=0)) + trail), [])
        skipped = self.problems(audited, smoke_text([engaged_line(**audited)], ledger_lines(150, guard=1)) + trail)
        self.assertTrue(any('the audit runs it on every read' in p for p in skipped), skipped)
        self.assertFalse(any('never skipped' in p for p in skipped), skipped)
        facts = c2_smoke_check.round_host_problems(ARM_ENV, self.clean(live=4, keyed=1), True)[1]
        self.assertEqual(facts['steps'][0]['live'], 4)

    def test_a_ramp_up_or_tail_step_with_fewer_live_users_says_nothing_about_the_quad_levers(self):
        mixed = smoke_text([engaged_line(**ARM_ENV)], ledger_lines(120, guard=1), ledger_lines(40, live=6, sel=1, read=0, keyed=1),
                           ledger_lines(40, live=4, sel=0, read=0, keyed=1))
        self.assertEqual(self.problems(ARM_ENV, mixed), [])

    def test_the_run_length_rules_wait_for_the_eight_user_steady_mix_and_a_long_enough_run(self):
        self.assertEqual(self.problems(ARM_ENV, self.clean(sel=0, read=0, keyed=0), steady_eight=False), [])
        short = smoke_text([engaged_line(**ARM_ENV)], ledger_lines(10, sel=0, read=0, keyed=0))
        self.assertEqual(self.problems(ARM_ENV, short), [])

    def test_lean_must_have_dropped_its_lines(self):
        for line in ('[PHASE] propose chatcmpl-1 begin', '[PHASE] packed_commit chatcmpl-1 end 0.4 ms', '[PHASE] early_draft round=1 begin',
                     '[PACKED-PUBLISH-SPLIT] round=1 entry=0 proj=0.00', '[PACKED-COMMIT-HOST] round=1 adopt_ms=[0.01]',
                     '[PACKED-COMMIT] round=1 mode=deferred commit_ms=[0.00] sync_ms=0.01'):
            with self.subTest(line=line):
                self.assertTrue(any('a line it drops' in p for p in self.problems(ARM_ENV, self.clean() + line + '\n')))
        kept = '[PHASE] packed_verify chatcmpl-1 begin\n[PHASE] execute total=128 new=0 cached=8 spec=8\n[PACKED] request=a segment=0\n[PACKED-PUBLISH] round=1\n'
        self.assertEqual(self.problems(ARM_ENV, self.clean() + kept), [])

    def test_the_audit_needs_an_equal_line_for_each_lever_and_no_unequal_one(self):
        env = dict(ARM_ENV, **{AUDIT: '1'})
        good = ['[ROUND-HOST-AUDIT] kind=select equal=1', '[ROUND-HOST-AUDIT] kind=merge equal=1', '[ROUND-HOST-AUDIT] kind=keyed equal=1 checked=149 changed=0']
        base = smoke_text([engaged_line(**env)], ledger_lines(120, guard=0))
        self.assertEqual(self.problems(env, base + '\n'.join(good) + '\n'), [])
        for missing in range(3):
            lines = [line for number, line in enumerate(good) if number != missing]
            self.assertTrue(any('nothing was compared' in p for p in self.problems(env, base + '\n'.join(lines) + '\n')), missing)
        bad = base + '\n'.join(good + ['[ROUND-HOST-AUDIT] kind=merge equal=0']) + '\n'
        self.assertTrue(any('equal=0' in p for p in self.problems(env, bad)))
        self.assertTrue(any('is not set and the log holds' in p for p in self.problems(ARM_ENV, self.clean() + good[0] + '\n')))

    def test_the_check_reads_the_profile_env_and_files_the_facts(self):
        env = dict(ARM_ENV)
        entry = dict(env=env)
        facts = c2_smoke_check.check('{}', self.clean(), False, env=env, entry=None)[1]
        self.assertEqual(facts['round_host']['steps'], 120)

    def test_the_real_lines_are_the_ones_the_rules_read(self):
        self.assertTrue(c2_smoke_check.ROUND_HOST_ENGAGED_LINE.search(engaged_line(**ARM_ENV)))
        with patch.object(round_host, 'log_line') as write:
            os.environ.update({name: '1' for name in (LOG, SELECT)})
            try:
                round_host.engage()
            finally:
                for name in (LOG, SELECT):
                    os.environ.pop(name, None)
                round_host.refresh()
        self.assertTrue(c2_smoke_check.ROUND_HOST_ENGAGED_LINE.search(write.call_args.args[0]))


# ---------------------------------------------------------------------------------------------
# The profiles.
# ---------------------------------------------------------------------------------------------

def load_profiles():
    with open(PROFILES_PATH, encoding='utf-8') as handle:
        return json.load(handle)['profiles']


class ProfileTests(unittest.TestCase):
    ADDED = {CONTROL: {LOG: '1'}, ARM: ARM_ENV, AUDITED: dict(ARM_ENV, **{AUDIT: '1'})}

    def test_each_twin_is_the_traffic_profile_plus_its_flags_and_nothing_else_and_gate_only(self):
        found = load_profiles()
        parent = found[PARENT]
        for name, flags in self.ADDED.items():
            with self.subTest(name=name):
                profile = found[name]
                self.assertEqual({key: value for key, value in profile['env'].items() if key not in flags}, parent['env'])
                self.assertEqual({key: profile['env'][key] for key in flags}, flags)
                self.assertEqual({key: value for key, value in profile.items() if key not in ('description', 'env', 'gate_only')},
                                 {key: value for key, value in parent.items() if key not in ('description', 'env', 'gate_only')})
                self.assertIs(profile['gate_only'], True)
                self.assertEqual(profile['env']['QWEN_FAST_VERIFY_T1_AUDIT'], '0', 'the audits stay off')
                self.assertEqual(profile['engine'], parent['engine'])
                self.assertNotIn('gate_only', parent)
                self.assertIsNone(BANNED.search(profile['description']))

    def test_the_control_differs_from_the_arm_by_the_four_levers_alone_and_the_audited_arm_by_the_audit(self):
        found = load_profiles()
        self.assertEqual({key: found[ARM]['env'][key] for key in found[ARM]['env'] if found[CONTROL]['env'].get(key) != found[ARM]['env'][key]},
                         {SELECT: '1', READ: '1', KEYED: '1', LEAN: '1'})
        self.assertEqual({key for key in found[AUDITED]['env'] if found[ARM]['env'].get(key) != found[AUDITED]['env'][key]}, {AUDIT})

    def test_the_parent_and_every_other_profile_carry_no_round_host_flag_and_the_default_stays_production(self):
        with open(PROFILES_PATH, encoding='utf-8') as handle:
            data = json.load(handle)
        self.assertEqual(data['default'], 'c2-packed-tp4')
        import make_fusion_profiles as fusion

        # the op-fusion programme's two combined twins (gate only) carry the four levers on purpose, and the audit flag on the audited one
        combined = {fusion.NAMESPACE + 'all': {SELECT, READ, KEYED, LEAN}, fusion.NAMESPACE + 'all-audit': {SELECT, READ, KEYED, LEAN, AUDIT}}
        # the host-gap package WPH's composition twins (-fx-wph-*) are the combined timed arm plus their own flags, so they carry the four levers (never the audit flag)
        combined.update({name: {SELECT, READ, KEYED, LEAN} for name in data['profiles'] if name.startswith(fusion.NAMESPACE + 'wph-')})
        for name, profile in data['profiles'].items():
            if name in self.ADDED:
                continue
            with self.subTest(name=name):
                carried = set(round_host.FLAGS) & set(profile.get('env', {}))
                self.assertEqual(carried, combined.get(name, set()))
                if name in combined:
                    self.assertIs(profile.get('gate_only'), True)

    def test_the_checked_in_twins_are_what_the_generator_makes_from_the_parent(self):
        self.assertEqual(generator.main(['--check']), 0)
        self.assertEqual(set(generator.twin_names()), set(self.ADDED))
        self.assertTrue(set(generator.twin_names()) <= set(profile_twins.twin_names()))

    def test_the_generator_refuses_a_parent_that_already_carries_a_flag_or_is_gate_only(self):
        data = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
        data['profiles'][PARENT]['env'][SELECT] = '1'
        with self.assertRaises(ValueError):
            generator.generate(data)
        data = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
        data['profiles'][PARENT]['gate_only'] = True
        with self.assertRaises(ValueError):
            generator.generate(data)

    def test_the_twins_sit_right_after_their_parent_and_nothing_else_moved(self):
        names = list(load_profiles())
        at = names.index(PARENT)
        self.assertEqual(names[at + 1:at + 4], [CONTROL, ARM, AUDITED])


# ---------------------------------------------------------------------------------------------
# The job pack.
# ---------------------------------------------------------------------------------------------

EXPECTED = {
    'X0-status-rescan-reset': ('status rescan reset', None, 'stop'), 'B0-build': ('build', 'c2-packed-tp4', 'stop'),
    'E1-exactness-control-smoke': ('reset smoke', CONTROL, 'stop'), 'A1-audited-attach-smoke': ('reset smoke', AUDITED, 'stop'),
    'E2-exactness-arm-smoke': ('reset smoke', ARM, 'stop'),
    'T1-timed8-A-control': ('reset smoke', CONTROL, 'soft'), 'T2-timed8-B-arm': ('reset smoke', ARM, 'soft'),
    'T3-timed8-A-control': ('reset smoke', CONTROL, 'soft'), 'T4-timed8-B-arm': ('reset smoke', ARM, 'soft'),
    'T5-timed4-A-control': ('reset smoke', CONTROL, 'soft'), 'T6-timed4-B-arm': ('reset smoke', ARM, 'soft'),
    'T7-timed4-A-control': ('reset smoke', CONTROL, 'soft'), 'T8-timed4-B-arm': ('reset smoke', ARM, 'soft'),
    'Z-reset': ('status reset', None, 'soft'),
}
AGENT_ACTIONS = {'agentstop', 'agentstart', 'unserve', 'platform', 'replay', 'priority', 'cardm'}


def pack_text(name):
    return (FOLDER / (name + '.env')).read_text(encoding='utf-8')


def pack_job(name):
    return job.read_job(job.parse_env(pack_text(name)), sorted(load_profiles()), root=ROOT)


def tests_of(name):
    return pack_job(name)['tests'].replace(' ', ',').split(',')


def order_lines():
    return [line.split() for line in (FOLDER / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]


class PackTests(unittest.TestCase):
    def test_order_lists_exactly_the_templates_in_the_asked_order(self):
        lines = order_lines()
        self.assertEqual([line[0] for line in lines], list(EXPECTED))
        self.assertEqual(sorted(path.stem for path in FOLDER.glob('*.env')), sorted(EXPECTED))
        for name, mode, image, minutes in lines:
            self.assertEqual((mode, image), (EXPECTED[name][2], IMAGE), name)
            self.assertTrue(minutes.isdigit() and int(minutes) > 0, name)

    def test_every_template_parses_with_its_actions_profile_and_the_one_image(self):
        for name, (actions, profile, _mode) in EXPECTED.items():
            with self.subTest(name=name):
                result = pack_job(name)
                self.assertEqual((result['actions'], result['cards'], result['tag']), (actions, 'quad', IMAGE))
                if profile:
                    self.assertEqual(result['profile'], profile)

    def test_the_first_quad_job_is_status_rescan_reset_and_no_job_touches_the_agent_or_production(self):
        self.assertEqual(order_lines()[0][0], 'X0-status-rescan-reset')
        self.assertEqual(order_lines()[-1][0], 'Z-reset')
        for name in EXPECTED:
            with self.subTest(name=name):
                result = pack_job(name)
                self.assertFalse(AGENT_ACTIONS & set(result['actions'].split()))
                self.assertEqual(result['bake_default_profile'] or '', '')
        self.assertNotIn(IMAGE, job.PROTECTED)
        self.assertFalse(IMAGE.startswith(job.PROTECTED_PREFIXES))

    def test_every_eight_seat_quad_smoke_runs_concurrent8_steady_because_the_smoke_rule_fails_it_otherwise(self):
        for name, (actions, profile, _mode) in EXPECTED.items():
            if profile and 'smoke' in actions:
                with self.subTest(name=name):
                    self.assertIn('concurrent8_steady', tests_of(name))
                    self.assertEqual(load_profiles()[profile]['env']['QWEN_FAST_QUAD_DRAFT_BLOCKS'], '2')
        # and the rule really does fail a job without it
        problems = c2_smoke_check.blocks_problems(dict(quad_markers=0, quad_marker_slots=(), quads_rounds=0, quad_fallbacks=0, quad_disabled=0),
                                                  load_profiles()[ARM]['env'], False)
        self.assertTrue(any('concurrent8_steady' in problem for problem in problems))

    def test_exactness_runs_the_same_unaudited_tests_on_the_control_and_the_arm_on_the_same_image(self):
        control, arm = tests_of('E1-exactness-control-smoke'), tests_of('E2-exactness-arm-smoke')
        self.assertEqual(control, arm)
        self.assertTrue({'concurrent4_steady', 'concurrent8_steady', 'concurrent8_code_equal', 'concurrent8_code_32k', 'replay_concurrent8',
                         'concurrent5_split', 'concurrent8_drain'} <= set(control))
        env = load_profiles()
        for profile in (CONTROL, ARM):
            self.assertEqual((env[profile]['env']['QWEN_FAST_VERIFY_T1_AUDIT'], env[profile]['env']['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'))
            self.assertNotIn('QWEN_FAST_PACKED_AUDIT', env[profile]['env'], 'audits off in the profile: the exactness runs are the cheap ones')

    def test_the_audited_attach_is_the_audited_profile_and_the_host_audit_only(self):
        self.assertEqual(pack_job('A1-audited-attach-smoke')['profile'], AUDITED)
        self.assertTrue({'concurrent4_steady', 'concurrent8_steady', 'concurrent8_code_32k'} <= set(tests_of('A1-audited-attach-smoke')))

    def test_the_timing_jobs_alternate_abab_at_eight_live_then_at_four_on_the_same_tests(self):
        eight = ['T1-timed8-A-control', 'T2-timed8-B-arm', 'T3-timed8-A-control', 'T4-timed8-B-arm']
        four = ['T5-timed4-A-control', 'T6-timed4-B-arm', 'T7-timed4-A-control', 'T8-timed4-B-arm']
        self.assertEqual([pack_job(name)['profile'] for name in eight], [CONTROL, ARM, CONTROL, ARM])
        self.assertEqual([pack_job(name)['profile'] for name in four], [CONTROL, ARM, CONTROL, ARM])
        self.assertEqual(len({pack_job(name)['tests'] for name in eight}), 1)
        self.assertEqual(len({pack_job(name)['tests'] for name in four}), 1)
        self.assertTrue({'concurrent8_steady', 'concurrent8_code_32k', 'concurrent8_code_128k'} <= set(tests_of(eight[0])))
        self.assertTrue({'concurrent4_steady', 'concurrent4_code_32k'} <= set(tests_of(four[0])))
        self.assertEqual([line[0] for line in order_lines()[5:13]], eight + four)

    def test_the_order_carries_the_dependencies_the_pause_the_load_rule_and_the_read_rules(self):
        order = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        for text in ('# NEEDS B0 <- X0', '# NEEDS E1 <- X0 B0', '# NEEDS A1 E2 <- E1', '# NEEDS T1 T2 T3 T4 T5 T6 T7 T8 <- A1 E2', 'CI PAUSED',
                     'ARC runner sets to zero', 'load-limit', 'PAIRED', 'round_host_report.py pair', 'w2ln_timing_compare.py pair',
                     'speed_window_compare.py', 'STRICT exactness', 'five consecutive', 'NO-GO'):
            self.assertIn(text, order)

    def test_no_hostname_address_registry_digest_or_path_and_lf_endings(self):
        for path in FOLDER.iterdir():
            text = path.read_text(encoding='utf-8')
            self.assertIsNone(BANNED.search(text), path.name)
            self.assertNotIn('\r', text, path.name)


# ---------------------------------------------------------------------------------------------
# The attribution report.
# ---------------------------------------------------------------------------------------------

def report_log(count, delta=0.0, live=8, start=1000, step=10):
    lines = []
    for number in range(count):
        values = {name: 2.0 + (number % 3) for name in round_host.Ledger.FIELDS}
        values['select'] += delta
        lines.append(round_host.format_line(number + 1, live, values, dict(select=1, read=2, keyed=1, guard=0), {}, start + step * number))
    return '\n'.join('(EngineCore pid=1) 2026-10-10 00:00:00.000 | INFO | round_host:log_line:63 - ' + line for line in lines) + '\n'


class ReportTests(unittest.TestCase):
    def test_the_summary_reads_the_live_steps_and_the_paths_taken(self):
        found = round_host_report.summary(report_log(300) + report_log(40, live=4), live=8)
        self.assertEqual(found['steps'], 300)
        self.assertEqual(found['fields']['select']['n'], 300)
        self.assertEqual(found['paths'], dict(sel=1.0, read=1.0, keyed=1.0, guard=0.0))
        self.assertEqual(round_host_report.summary(report_log(40, live=4), live=4)['steps'], 40)
        self.assertEqual(round_host_report.summary('nothing', live=8)['steps'], 0)

    def test_the_pair_reports_each_phase_b_minus_a_by_mean_position_bucket(self):
        found = round_host_report.pair(report_log(400), report_log(400, delta=-0.9, start=1500), live=8, min_matched=100)
        self.assertEqual(found['fields']['select']['verdict'], 'MEASURED')
        self.assertAlmostEqual(found['fields']['select']['delta_ms'], -0.9, places=3)
        self.assertAlmostEqual(found['fields']['entry']['delta_ms'], 0.0, places=3)
        self.assertIn('select', round_host_report.render_pair(found))

    def test_a_pair_with_too_few_matched_steps_is_void_and_a_window_reads_one_shape(self):
        found = round_host_report.pair(report_log(50), report_log(50, delta=-1.0), live=8, min_matched=100)
        self.assertTrue(all(item['verdict'] == 'VOID' for item in found['fields'].values()))
        windowed = round_host_report.steps_of(report_log(300, start=28000, step=100), live=8, window='32k')
        self.assertTrue(0 < len(windowed) < 300)
        self.assertTrue(all(28672 <= step['position'] < 40960 for step in windowed))

    def test_the_command_line_runs_on_files(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / 'a.log', Path(folder) / 'b.log'
            first.write_text(report_log(300), encoding='utf-8')
            second.write_text(report_log(300, delta=-0.5), encoding='utf-8')
            with contextlib.redirect_stdout(Mock()):
                self.assertEqual(round_host_report.main(['summary', str(first)]), 0)
                self.assertEqual(round_host_report.main(['pair', str(first), str(second), '--json']), 0)
                self.assertEqual(round_host_report.main(['summary', str(first), '--live', '4']), 1)


# ---------------------------------------------------------------------------------------------
# Flag-off identity and shipping.
# ---------------------------------------------------------------------------------------------

class FlagOffIdentityTests(KeyedEquivalenceTests):
    def test_with_every_flag_off_or_set_to_zero_nothing_is_logged_counted_or_kept_and_the_rounds_are_the_same(self):
        results = []
        for flags in ({}, {name: '0' for name in round_host.FLAGS}):
            os.environ.update(flags)
            round_host.refresh(strict=True)
            round_host.reset_counters()
            plan = self.schedule(random.Random(3), rounds=6, padded=True)
            self.h1a.clear()
            served = self.run_schedule(plan, True, True, True)
            results.append((served, self.paths(), list(self.h1a)))
            self.assertEqual(self.round_host_log, [])
            self.assertEqual(sum(round_host.COUNTS.values()), 0)
            self.assertFalse(round_host.ledger.on or round_host.ledger.open)
        self.assertEqual(results[0][0], results[1][0])
        self.assertEqual(results[0][1], results[1][1])
        self.assertEqual([re.sub(r'ms=[0-9.]+', 'ms=X', line) for line in results[0][2]],
                         [re.sub(r'ms=[0-9.]+', 'ms=X', line) for line in results[1][2]], 'the same lines, the same order')

    def test_no_snapshot_of_a_flag_off_run_carries_a_key(self):
        block = self.open_block()
        users = tvp.base_users()
        self.round(block, users)
        self.window(block, tvp.advanced(users, 1))
        self.assertIsNone(block.prestaged.snapshot.key)


for _name in [name for name in dir(KeyedEquivalenceTests) if name.startswith('test_') and name not in (
        'test_with_every_flag_off_or_set_to_zero_nothing_is_logged_counted_or_kept_and_the_rounds_are_the_same',
        'test_no_snapshot_of_a_flag_off_run_carries_a_key')]:
    setattr(FlagOffIdentityTests, _name, None)


class ShippingTests(unittest.TestCase):
    RUNTIME = ('round_host.py', 'verify_prestage.py', 'packed_verifier.py', 'serving_packed_step.py', 'serving_packed_bridge.py',
               'serving_worker_hook.py', 'early_draft.py', 'dflash_packed_proposal.py', 'dflash_packed_proposal_coordinator.py', 'quad_draft_tp.py')

    def test_every_module_the_levers_touch_is_in_both_image_copy_lists(self):
        from test_serving_image_copy_closure import context_modules, dockerfile_modules, dockerfile_text

        docker, context = dockerfile_modules(dockerfile_text()), context_modules()
        for name in self.RUNTIME:
            with self.subTest(module=name):
                self.assertIn(name, docker)
                self.assertIn(name, context)

    def test_the_overlay_manifest_names_them_all_so_the_image_runs_this_commits_copies(self):
        listed = {line.split()[0] for line in (ROOT / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8').splitlines()
                  if line.strip() and not line.startswith('#')}
        for name in self.RUNTIME:
            with self.subTest(module=name):
                self.assertIn('scripts/ci/' + name, listed)

    def test_the_suite_runs_in_the_cpu_workflow_and_the_host_scripts_are_not_shipped(self):
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        self.assertRegex(workflow, r'python -B -m unittest [^\n]*\btest_tp4_round_host\b')
        from test_serving_image_copy_closure import dockerfile_modules, dockerfile_text

        self.assertNotIn('round_host_report.py', dockerfile_modules(dockerfile_text()))

    def test_the_new_and_edited_files_are_lf(self):
        for name in self.RUNTIME + ('test_tp4_round_host.py', 'round_host_report.py', 'make_round_host_profiles.py', 'profile_twins.py',
                                    'c2_smoke_check.py', 'qwen_c2_profiles.json'):
            with self.subTest(name=name):
                self.assertNotIn(b'\r\n', (HERE / name).read_bytes())

    def test_round_host_imports_only_the_standard_library_at_module_level(self):
        source = (HERE / 'round_host.py').read_text(encoding='utf-8')
        top = [line for line in source.splitlines() if re.match(r'^(import|from) ', line)]
        self.assertEqual(top, ['import os', 'import sys', 'import time', 'import weakref'])


if __name__ == '__main__':
    unittest.main()
