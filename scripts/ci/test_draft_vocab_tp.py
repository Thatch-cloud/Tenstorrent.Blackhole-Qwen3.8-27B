"""QWEN_FAST_DRAFT_VOCAB (draft_vocab_tp.py, docs/tp4-draft-vocab.md): the drafter's head over a coding shortlist of the vocabulary, gate only, default off.

Held here, on the CPU (nothing runs on a card):
  - the list: the committed one is ascending, pinned by sha256, holds every special and single-byte token, and splits into whole 32-row tiles on four chips; a bad list,
    name, width or chip count is refused by name;
  - the mapping: a chip's local index maps to the right global token id, and the merge keeps the full-vocabulary merge's order and its tie rule (the lowest global id);
  - selection equivalence: when the full vocabulary's top-16 lies inside the list, the shortlist run returns the same tokens and scores, and the FP64 candidate selector
    (draft_selector.select_active_candidates) returns the same proposals; when it does not, every candidate is inside the list and is the top-16 of the restricted logits;
  - the head's op sequence (one linear over the SLICED weight, one padded top-16, one chunk) and the build of the sliced weight from the target head's shards;
  - flag-off identity: unset, empty or '0' the twin's functions are the bodies they were and draft_vocab_tp is not imported;
  - the contract, the smoke rule, the markers and the image copy lists."""

import contextlib
import io
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import c2_smoke_check
import draft_selector
import draft_shared_head_tp
import draft_vocab_tp
import round_host
import serving_c2_contract
import tp_shapes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FLAG = 'QWEN_FAST_DRAFT_VOCAB'
TOPK = 'QWEN_FAST_DRAFT_VOCAB_TOPK'
CODING = 'coding-40960'


def four(**extra):
    environment = {'QWEN_FAST_TP': '4'}
    environment.update(extra)
    return patch.dict(os.environ, environment, clear=True)


def write_ids(directory, ids, name='list.ids'):
    path = os.path.join(directory, name)
    with open(path, 'wb') as handle:
        handle.write(struct.pack('<%dI' % len(ids), *ids))
    return path


def spread_ids(rows, seed=1):
    """`rows` ascending ids spread over the whole vocabulary (all four chips' natural shards)."""
    generator = torch.Generator().manual_seed(seed)
    return tuple(sorted(torch.randperm(248077, generator=generator)[:rows].tolist()))


class Isolated(unittest.TestCase):
    def setUp(self):
        draft_vocab_tp._PLANS.clear()
        draft_vocab_tp._HEADS.clear()
        del draft_vocab_tp._ENGAGED[:]
        self.addCleanup(draft_vocab_tp._PLANS.clear)
        self.addCleanup(draft_vocab_tp._HEADS.clear)
        self.addCleanup(lambda: draft_vocab_tp._ENGAGED.clear())
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)


class CommittedListTests(Isolated):
    def test_the_named_list_is_pinned_ascending_and_four_chips_wide(self):
        with four(**{FLAG: CODING}):
            plan = draft_vocab_tp.active_plan()
        stem, rows, digest = draft_vocab_tp.NAMED[CODING]
        self.assertEqual((plan.rows, plan.per_chip, plan.chips), (40960, 10240, 4))
        self.assertEqual(plan.sha256, digest)
        self.assertEqual(plan.topk_width, 32768)
        self.assertEqual(plan.rows, rows)
        self.assertTrue(all(a < b for a, b in zip(plan.ids, plan.ids[1:])))
        self.assertLess(plan.ids[-1], 248077)
        self.assertEqual(plan.rows % (4 * 32), 0)

    def test_the_sidecar_agrees_with_the_list_and_the_pin(self):
        import json

        stem, rows, digest = draft_vocab_tp.NAMED[CODING]
        meta = json.loads((HERE / (stem + '.json')).read_text(encoding='utf-8'))
        self.assertEqual((meta['rows'], meta['ids_sha256'], meta['ids_file']), (rows, digest, stem + '.ids'))
        self.assertEqual(meta['ids_bytes'], (HERE / (stem + '.ids')).stat().st_size)

    def test_every_special_and_single_byte_token_is_proposable(self):
        import json

        stem = draft_vocab_tp.NAMED[CODING][0]
        meta = json.loads((HERE / (stem + '.json')).read_text(encoding='utf-8'))
        with four(**{FLAG: CODING}):
            chosen = set(draft_vocab_tp.active_plan().ids)
        added, single = meta['required']['added_tokens'], meta['required']['single_byte_tokens']
        self.assertEqual(len(added), 33)
        self.assertEqual(len(single), 256)
        # the chat, vision, FIM, tool-call and think markers are the tokenizer's added tokens, ids 248044..248076
        self.assertEqual(added, list(range(248044, 248077)))
        self.assertTrue(set(added) <= chosen)
        self.assertTrue(set(single) <= chosen)
        # the markers the parser and the agent loop act on, by id (the served tokenizer's: <tool_call>, </tool_call>, <think>, </think>, <|im_end|>, <|endoftext|>)
        for marker in (248058, 248059, 248068, 248069, 248046, 248044):
            self.assertIn(marker, chosen)

    def test_no_padding_row_is_proposable(self):
        # the vocabulary has 248,320 rows but the tokenizer defines 248,077: the rest is padding the model never emits
        with four(**{FLAG: CODING}):
            self.assertLess(draft_vocab_tp.active_plan().ids[-1], 248077)

    def test_the_coverage_the_sidecar_states_meets_its_bar(self):
        import json

        meta = json.loads((HERE / (draft_vocab_tp.NAMED[CODING][0] + '.json')).read_text(encoding='utf-8'))
        coverage = meta['coverage']
        self.assertGreaterEqual(coverage['heldout_code_weighted'], 0.96)
        self.assertGreaterEqual(coverage['heldout_code_worst_category'], 0.96)
        self.assertEqual(meta['rows'], 40960)


class FlagTests(Isolated):
    def test_requested_is_strict_about_off(self):
        for environment, expected in (({}, False), ({FLAG: ''}, False), ({FLAG: '0'}, False), ({FLAG: CODING}, True), ({FLAG: '1'}, True)):
            with self.subTest(environment=environment):
                self.assertEqual(draft_vocab_tp.requested(environment), expected)

    def test_the_twin_and_the_module_agree_on_what_off_means(self):
        for value in (None, '', '0', CODING, '/x/y.ids', '1'):
            environment = {} if value is None else {FLAG: value}
            with patch.dict(os.environ, environment, clear=True):
                self.assertEqual(draft_shared_head_tp.vocab_requested(), draft_vocab_tp.requested(), value)

    def test_a_name_that_is_not_a_list_is_refused(self):
        for value in ('1', 'yes', 'coding', 'relative/path.ids', '/dir/'):
            with self.subTest(value=value), four(**{FLAG: value}):
                with self.assertRaisesRegex(ValueError, 'neither a shortlist name'):
                    draft_vocab_tp.active_plan()

    def test_the_pair_is_refused(self):
        with patch.dict(os.environ, {FLAG: CODING}, clear=True):
            with self.assertRaisesRegex(ValueError, 'four-card lever'):
                draft_vocab_tp.active_plan()
        with patch.dict(os.environ, {FLAG: CODING, 'QWEN_FAST_TP': '2'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'four-card lever'):
                draft_vocab_tp.active_plan()

    def test_the_width_flag_is_strict_and_needs_the_list(self):
        for value in ('0', '8192', '65536', 'wide', '', '16384.0'):
            with self.subTest(value=value), four(**{FLAG: CODING, TOPK: value}):
                with self.assertRaisesRegex(ValueError, TOPK):
                    draft_vocab_tp.active_plan()
        with four(**{TOPK: '16384'}):
            with self.assertRaisesRegex(ValueError, 'needs ' + FLAG):
                draft_vocab_tp.active_plan()
        with four(**{FLAG: CODING, TOPK: '16384'}):
            self.assertEqual(draft_vocab_tp.active_plan().topk_width, 16384)

    def test_off_is_none_and_reads_nothing(self):
        with four():
            self.assertIsNone(draft_vocab_tp.active_plan())
            self.assertIsNone(draft_vocab_tp.admission())
        with four(**{FLAG: '0'}):
            self.assertIsNone(draft_vocab_tp.active_plan())


class ListValidationTests(Isolated):
    def plan(self, ids, name='list.ids'):
        path = write_ids(self.directory.name, ids, name)
        with four(**{FLAG: path}):
            return draft_vocab_tp.active_plan()

    def test_a_path_to_a_good_list_loads(self):
        ids = spread_ids(1280)
        plan = self.plan(ids)
        self.assertEqual((plan.ids, plan.rows, plan.per_chip), (ids, 1280, 320))

    def test_a_list_that_is_not_whole_tiles_on_each_chip_is_refused(self):
        for rows in (100, 130, 4 * 32 + 32):
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(ValueError, 'whole 32-row tile'):
                    self.plan(spread_ids(rows), 'rows%d.ids' % rows)

    def test_unordered_duplicate_and_out_of_range_ids_are_refused(self):
        good = list(spread_ids(256))
        for name, ids in (('unordered', [good[1], good[0]] + good[2:]), ('duplicate', [good[0]] + good), ('range', good[:-1] + [248320]),
                          ('huge', good[:-1] + [2 ** 31])):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, 'strictly ascending'):
                    self.plan(ids, name + '.ids')

    def test_an_empty_or_ragged_file_is_refused(self):
        for name, data in (('empty', b''), ('ragged', b'\x01\x00\x00')):
            path = os.path.join(self.directory.name, name)
            Path(path).write_bytes(data)
            with self.subTest(name=name), four(**{FLAG: path}):
                with self.assertRaisesRegex(ValueError, 'whole number of uint32'):
                    draft_vocab_tp.active_plan()

    def test_a_missing_file_is_refused_without_its_path(self):
        with four(**{FLAG: os.path.join(self.directory.name, 'absent.ids')}):
            with self.assertRaisesRegex(ValueError, 'cannot be read') as raised:
                draft_vocab_tp.active_plan()
        self.assertNotIn(self.directory.name, str(raised.exception))

    def test_a_list_wider_than_the_launch_is_refused(self):
        # 4 chips x 16,512 columns do not fit a 16,384-wide top-16 launch
        path = write_ids(self.directory.name, spread_ids(4 * 16512), 'wide.ids')
        with self.assertRaisesRegex(ValueError, 'do not fit'):
            draft_vocab_tp.load(path, 4, 16384)
        self.assertEqual(draft_vocab_tp.load(path, 4, 32768).per_chip, 16512)

    def test_a_named_list_whose_bytes_changed_is_refused(self):
        stem, rows, digest = draft_vocab_tp.NAMED[CODING]
        with patch.dict(draft_vocab_tp.NAMED, {CODING: (stem, rows, '0' * 64)}), four(**{FLAG: CODING}):
            with self.assertRaisesRegex(ValueError, 'not the pinned one'):
                draft_vocab_tp.active_plan()

    def test_the_named_rows_are_held(self):
        stem, rows, digest = draft_vocab_tp.NAMED[CODING]
        with patch.dict(draft_vocab_tp.NAMED, {CODING: (stem, rows + 128, digest)}), four(**{FLAG: CODING}):
            with self.assertRaisesRegex(ValueError, 'not the named'):
                draft_vocab_tp.active_plan()


class MappingTests(Isolated):
    def setUp(self):
        super().setUp()
        self.ids = spread_ids(1280, seed=3)
        self.path = write_ids(self.directory.name, self.ids)

    def environment(self, **extra):
        return four(**{FLAG: self.path, **extra})

    def test_local_indices_map_to_the_chips_slice_of_the_list(self):
        with self.environment():
            plan = draft_vocab_tp.active_plan()
            for chip in range(4):
                local = torch.arange(0, 320, 7)
                mapped = draft_vocab_tp.map_tokens(plan, chip, 0, local)
                self.assertEqual(mapped.tolist(), [self.ids[chip * 320 + index] for index in local.tolist()])

    def test_the_chunk_is_the_whole_shard_and_one_per_chip(self):
        with self.environment():
            self.assertEqual(draft_shared_head_tp.candidate_chunks(), ((0, 320),))
            self.assertEqual(draft_vocab_tp.candidate_chunks(), ((0, 320),))
        with four():
            self.assertEqual(draft_shared_head_tp.candidate_chunks(), ((0, 32768), (32768, 62080)))

    def chunks(self, logits, plan):
        found = []
        for chip in range(4):
            scores, indices = logits[:, chip * plan.per_chip:(chip + 1) * plan.per_chip].topk(16, dim=-1)
            found.append(dict(chip=chip, start=0, stop=plan.per_chip, values=scores, indices=indices))
        return found

    def test_the_merge_returns_global_ids_in_any_arrival_order(self):
        generator = torch.Generator().manual_seed(5)
        with self.environment():
            plan = draft_vocab_tp.active_plan()
            logits = torch.randn(8, 1280, generator=generator)
            expected_scores, expected_local = logits.topk(16, dim=-1)
            expected_tokens = torch.tensor(self.ids)[expected_local]
            chunks = self.chunks(logits, plan)
            for order in (chunks, list(reversed(chunks))):
                tokens, scores = draft_shared_head_tp.merge_chunk_candidates(order)
                self.assertTrue(torch.equal(tokens, expected_tokens[None, 1:8]))
                self.assertTrue(torch.equal(scores, expected_scores[None, 1:8]))

    def test_ties_go_to_the_lowest_global_id_across_chips(self):
        with self.environment():
            plan = draft_vocab_tp.active_plan()
            # every score a tie: each chip offers local rows 0..15 at the same value
            chunks = [dict(chip=chip, start=0, stop=plan.per_chip, values=torch.zeros(8, 16), indices=torch.arange(16).repeat(8, 1)) for chip in range(4)]
            tokens, scores = draft_shared_head_tp.merge_chunk_candidates(chunks)
            self.assertTrue(torch.all(tokens[0, :, 1:] > tokens[0, :, :-1]))
            # every one of the 16 is a zero, and the 16 are the 16 lowest global ids that were offered (each chip offers its own first 16)
            offered = sorted(self.ids[chip * 320 + index] for chip in range(4) for index in range(16))
            self.assertEqual(tokens[0, 0].tolist(), offered[:16])

    def test_a_malformed_readback_is_refused_as_the_full_merge_refuses_it(self):
        with self.environment():
            plan = draft_vocab_tp.active_plan()
            generator = torch.Generator().manual_seed(6)
            chunks = self.chunks(torch.randn(8, 1280, generator=generator), plan)
            for broken in (chunks[:-1], chunks + [chunks[0]]):
                with self.assertRaises(ValueError):
                    draft_shared_head_tp.merge_chunk_candidates(broken)
            chunks[-1]['indices'][0, 0] = 320
            with self.assertRaisesRegex(ValueError, 'in-range'):
                draft_shared_head_tp.merge_chunk_candidates(chunks)
            with self.assertRaisesRegex(ValueError, 'eight/16/32'):
                draft_shared_head_tp.merge_chunk_candidates(chunks, block_rows=7)

    def test_the_batched_round_host_merge_equals_the_reference_bit_for_bit(self):
        generator = torch.Generator().manual_seed(7)
        with self.environment():
            plan = draft_vocab_tp.active_plan()
            for rows in (8, 16, 32):
                chunks = self.chunks(torch.randn(rows, 1280, generator=generator), plan)
                fast = round_host._fast_merge(chunks, rows)
                reference = draft_shared_head_tp.merge_chunk_candidates(chunks, block_rows=rows)
                self.assertTrue(torch.equal(fast[0], reference[0]))
                self.assertTrue(torch.equal(fast[1], reference[1]))

    def test_the_batched_merge_without_the_flag_is_the_full_vocabulary_one(self):
        generator = torch.Generator().manual_seed(8)
        with four():
            logits = torch.randn(8, 248320, generator=generator)
            chunks = []
            for chip in range(4):
                for start, stop in draft_shared_head_tp.candidate_chunks():
                    scores, indices = logits[:, chip * 62080 + start:chip * 62080 + stop].topk(16, dim=-1)
                    chunks.append(dict(chip=chip, start=start, stop=stop, values=scores, indices=indices))
            fast = round_host._fast_merge(chunks, 8)
            reference = draft_shared_head_tp.merge_chunk_candidates(chunks)
            expected_scores, expected_tokens = logits.topk(16, dim=-1)
            self.assertTrue(torch.equal(fast[0], reference[0]) and torch.equal(fast[1], reference[1]))
            self.assertTrue(torch.equal(reference[0], expected_tokens[None, 1:8]))


class SelectionEquivalenceTests(Isolated):
    """The shortlist changes WHAT the head offers, never how the offer is selected from."""

    def setUp(self):
        super().setUp()
        self.ids = spread_ids(2560, seed=11)
        self.path = write_ids(self.directory.name, self.ids)
        generator = torch.Generator().manual_seed(12)
        self.rank = 16
        self.predecessors = torch.randn(248320, self.rank, generator=generator, dtype=torch.float64)
        self.successors = torch.randn(248320, self.rank, generator=generator, dtype=torch.float64)
        self.hidden = torch.randn(1, 7, self.rank, generator=generator, dtype=torch.float64)

    def full_candidates(self, logits):
        chunks = []
        for chip in range(4):
            for start, stop in draft_shared_head_tp.candidate_chunks():
                scores, indices = logits[:, chip * 62080 + start:chip * 62080 + stop].topk(16, dim=-1)
                chunks.append(dict(chip=chip, start=start, stop=stop, values=scores, indices=indices))
        return draft_shared_head_tp.merge_chunk_candidates(chunks)

    def shortlist_candidates(self, logits):
        plan = draft_vocab_tp.active_plan()
        restricted = logits[:, torch.tensor(self.ids)]
        chunks = []
        for chip in range(4):
            scores, indices = restricted[:, chip * plan.per_chip:(chip + 1) * plan.per_chip].topk(16, dim=-1)
            chunks.append(dict(chip=chip, start=0, stop=plan.per_chip, values=scores, indices=indices))
        return draft_shared_head_tp.merge_chunk_candidates(chunks)

    def logits_with_winners_inside(self, seed):
        generator = torch.Generator().manual_seed(seed)
        logits = torch.randn(8, 248320, generator=generator)
        inside = torch.tensor(self.ids)
        for row in range(8):
            picks = inside[torch.randperm(len(self.ids), generator=generator)[:16]]
            logits[row, picks] = 100.0 + torch.arange(16, dtype=torch.float32) * 0.25 + row
        return logits

    def select(self, tokens, scores):
        selected, unused = draft_selector.select_active_candidates(self.hidden, tokens, scores.double(), self.predecessors, self.successors,
                                                                  torch.tensor([self.ids[5]], dtype=torch.int64))
        return selected

    def test_when_the_top_16_is_inside_the_list_the_candidates_and_the_proposals_are_the_full_vocabularys(self):
        for seed in (21, 22, 23):
            logits = self.logits_with_winners_inside(seed)
            with four():
                full_tokens, full_scores = self.full_candidates(logits)
            with four(**{FLAG: self.path}):
                short_tokens, short_scores = self.shortlist_candidates(logits)
            self.assertTrue(torch.equal(full_tokens, short_tokens), seed)
            self.assertTrue(torch.equal(full_scores, short_scores), seed)
            self.assertTrue(torch.equal(self.select(full_tokens, full_scores), self.select(short_tokens, short_scores)), seed)

    def test_a_winner_outside_the_list_costs_a_candidate_never_correctness(self):
        generator = torch.Generator().manual_seed(31)
        logits = torch.randn(8, 248320, generator=generator)
        outside = sorted(set(range(248077)) - set(self.ids))[:5]
        logits[:, outside] = 500.0 + torch.arange(5, dtype=torch.float32)
        with four():
            full_tokens, unused = self.full_candidates(logits)
        with four(**{FLAG: self.path}):
            short_tokens, short_scores = self.shortlist_candidates(logits)
        chosen = set(self.ids)
        self.assertTrue(all(token in chosen for token in short_tokens.flatten().tolist()))
        self.assertTrue(any(token in outside for token in full_tokens.flatten().tolist()))
        # what it offers is the top-16 of the logits restricted to the list
        restricted_scores, restricted_local = logits[:, torch.tensor(self.ids)].topk(16, dim=-1)
        self.assertTrue(torch.equal(short_tokens, torch.tensor(self.ids)[restricted_local][None, 1:8]))
        self.assertTrue(torch.equal(short_scores, restricted_scores[None, 1:8]))
        # and the selector still returns a token the list can offer, from the codebooks by global id
        self.assertTrue(all(token in chosen for token in self.select(short_tokens, short_scores).flatten().tolist()))

    def test_the_selection_semantics_are_the_selectors_own(self):
        # the same candidates under the same selector give the same proposals however they were produced: the selector never sees the list
        generator = torch.Generator().manual_seed(41)
        tokens = torch.stack([torch.tensor(sorted(torch.randperm(2560, generator=generator)[:16].tolist())) for unused in range(7)])[None]
        tokens = torch.tensor(self.ids)[tokens]
        scores = torch.randn(1, 7, 16, generator=generator)
        first = self.select(tokens, scores)
        self.assertTrue(torch.equal(first, self.select(tokens.clone(), scores.clone())))
        self.assertEqual(first.shape, (1, 7))


class FakeTensor(object):
    def __init__(self, shape, dtype='bf16', layout='tile', memory='dram', name=''):
        self.shape, self.dtype, self.layout, self._memory, self.name = tuple(shape), dtype, layout, memory, name

    def memory_config(self):
        return self._memory


class FakeOperations(object):
    """The ttnn surface the head uses, recording every call."""

    bfloat16, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'tile', 'dram'
    bfloat8_b = 'bf8'

    def __init__(self):
        self.calls = []

    def linear(self, hidden, weight):
        self.calls.append(('linear', weight))
        return FakeTensor((1, 1, hidden.shape[2], weight.shape[-1]), name='logits')

    def slice(self, value, start, stop, stride):
        self.calls.append(('slice',))
        return FakeTensor((1, 1, value.shape[2], stop[-1] - start[-1]))

    def pad(self, value, padding, fill):
        self.calls.append(('pad', value.shape[-1] + padding[-1][1], fill))
        return FakeTensor((1, 1, value.shape[2], value.shape[-1] + padding[-1][1]))

    def topk(self, value, **options):
        self.calls.append(('topk', value.shape[-1], options))
        return FakeTensor((1, 1, value.shape[2], 16), name='values'), FakeTensor((1, 1, value.shape[2], 16), 'u32', name='indices')

    def deallocate(self, value):
        self.calls.append(('deallocate', value))


class HeadOpsTests(Isolated):
    def setUp(self):
        super().setUp()
        self.ids = spread_ids(1280, seed=3)
        self.path = write_ids(self.directory.name, self.ids)
        self.operations = FakeOperations()
        self.full_weight = FakeTensor((1, 1, 5120, 62080), 'bf8', name='full')
        self.model = SimpleNamespace(num_devices=4, vocab_size=248320, _lmhead_vocab_sharded=True, lm_head_weight=self.full_weight)
        self.sliced = FakeTensor((1, 1, 5120, 320), 'bf8', name='sliced')   # a chip's shard: the shape a mesh tensor reports

    def install_head(self, plan):
        draft_vocab_tp._HEADS[id(self.model)] = draft_vocab_tp.Head(self.sliced, plan, 'bf8', 1)

    def normalized(self, rows):
        return FakeTensor((1, 1, rows, 5120))

    def test_one_linear_over_the_sliced_weight_one_padded_top16_one_chunk(self):
        for rows in (8, 16, 32):
            with self.subTest(rows=rows), four(**{FLAG: self.path}):
                draft_vocab_tp._PLANS.clear()
                self.operations.calls.clear()
                self.install_head(draft_vocab_tp.active_plan())
                owned = []
                outputs = draft_shared_head_tp.shared_head_candidates(self.operations, self.model, self.normalized(rows), owned)
                self.assertEqual(len(outputs), 1)
                self.assertEqual((outputs[0]['start'], outputs[0]['stop']), (0, 320))
                self.assertEqual(outputs[0]['values'].shape, (1, 1, rows, 16))
                self.assertIs(self.operations.calls[0][1], self.sliced)
                self.assertIsNot(self.operations.calls[0][1], self.full_weight)
                self.assertEqual([call[0] for call in self.operations.calls], ['linear', 'pad', 'topk'])
                self.assertEqual(self.operations.calls[1], ('pad', 32768, float('-inf')))
                self.assertEqual(self.operations.calls[2][1], 32768)
                self.assertEqual(self.operations.calls[2][2], dict(k=16, dim=-1, largest=True, sorted=True))
                # logits, the padded copy, the values and the indices are the temporaries to free
                self.assertEqual(len(owned), 4)

    def test_the_narrow_launch_pads_to_16384_and_a_full_width_list_pads_nothing(self):
        with four(**{FLAG: self.path, TOPK: '16384'}):
            self.install_head(draft_vocab_tp.active_plan())
            draft_shared_head_tp.shared_head_candidates(self.operations, self.model, self.normalized(8), [])
            self.assertEqual(self.operations.calls[1], ('pad', 16384, float('-inf')))
            self.assertEqual(self.operations.calls[2][1], 16384)
        wide = write_ids(self.directory.name, spread_ids(4 * 16384, seed=9), 'wide.ids')
        self.operations.calls.clear()
        draft_vocab_tp._PLANS.clear()
        draft_vocab_tp._HEADS.clear()
        with four(**{FLAG: wide, TOPK: '16384'}):
            plan = draft_vocab_tp.active_plan()
            draft_vocab_tp._HEADS[id(self.model)] = draft_vocab_tp.Head(FakeTensor((1, 1, 5120, 16384)), plan, 'bf8', 1)
            outputs = draft_shared_head_tp.shared_head_candidates(self.operations, self.model, self.normalized(8), [])
            self.assertEqual([call[0] for call in self.operations.calls], ['linear', 'topk'])
            self.assertEqual((outputs[0]['start'], outputs[0]['stop']), (0, 16384))

    def test_the_engaged_marker_is_logged_once_with_the_rows(self):
        lines = []
        with four(**{FLAG: self.path}), patch.object(draft_vocab_tp, 'log_line', lines.append):
            self.install_head(draft_vocab_tp.active_plan())
            for unused in range(3):
                draft_shared_head_tp.shared_head_candidates(self.operations, self.model, self.normalized(8), [])
        engaged = [line for line in lines if line.startswith(draft_vocab_tp.ENGAGED)]
        self.assertEqual(len(engaged), 1)
        self.assertIn('rows=1280 per_chip=320 chips=4 topk_width=32768', engaged[0])

    def test_a_head_that_was_not_built_at_the_attach_is_an_error_not_a_fallback(self):
        with four(**{FLAG: self.path}):
            with self.assertRaisesRegex(ValueError, 'no shortlist head was built'):
                draft_shared_head_tp.shared_head_candidates(self.operations, self.model, self.normalized(8), [])
        self.assertEqual(self.operations.calls, [])

    def test_a_head_cut_for_another_list_is_an_error(self):
        other = write_ids(self.directory.name, spread_ids(1280, seed=77), 'other.ids')
        with four(**{FLAG: other}):
            plan = draft_vocab_tp.active_plan()
        draft_vocab_tp._PLANS.clear()
        self.install_head(plan)
        with four(**{FLAG: self.path}):
            with self.assertRaisesRegex(ValueError, 'another list or width'):
                draft_shared_head_tp.shared_head_candidates(self.operations, self.model, self.normalized(8), [])

    def test_the_input_checks_are_the_full_heads(self):
        with four(**{FLAG: self.path}):
            self.install_head(draft_vocab_tp.active_plan())
            for shape in ((1, 1, 24, 5120), (1, 1, 8, 4096), (1, 8, 5120)):
                with self.assertRaisesRegex(ValueError, 'Replicated eight/16/32-row'):
                    draft_shared_head_tp.shared_head_candidates(self.operations, self.model, FakeTensor(shape), [])
            for wrong in (dict(dtype='bf8'), dict(layout='row'), dict(memory='l1')):
                with self.assertRaisesRegex(ValueError, 'Replicated eight/16/32-row'):
                    draft_shared_head_tp.shared_head_candidates(self.operations, self.model, FakeTensor((1, 1, 8, 5120), **wrong), [])
            pair = SimpleNamespace(num_devices=2, vocab_size=248320, _lmhead_vocab_sharded=True, lm_head_weight=self.full_weight)
            with self.assertRaisesRegex(ValueError, 'Replicated eight/16/32-row'):
                draft_shared_head_tp.shared_head_candidates(self.operations, pair, self.normalized(8), [])

    def test_local_logits_of_the_wrong_width_are_refused(self):
        with four(**{FLAG: self.path}):
            with self.assertRaisesRegex(ValueError, 'Expected local shortlist shards'):
                draft_shared_head_tp.local_head_candidates(self.operations, FakeTensor((1, 1, 8, 62080)), [])

    def test_the_ops_without_the_flag_are_the_full_heads_two_chunks(self):
        with four():
            owned = []
            outputs = draft_shared_head_tp.shared_head_candidates(self.operations, self.model, self.normalized(8), owned)
        self.assertEqual(len(outputs), 2)
        self.assertEqual([(chunk['start'], chunk['stop']) for chunk in outputs], [(0, 32768), (32768, 62080)])
        self.assertIs(self.operations.calls[0][1], self.full_weight)
        self.assertEqual(sum(call[0] == 'pad' for call in self.operations.calls), 1)
        self.assertEqual([call[1] for call in self.operations.calls if call[0] == 'topk'], [32768, 32768])


class BuildHeadTests(Isolated):
    HIDDEN = 64
    WIDTH = 62080

    def setUp(self):
        super().setUp()
        self.ids = spread_ids(1280, seed=3)
        self.path = write_ids(self.directory.name, self.ids)
        generator = torch.Generator().manual_seed(13)
        self.full = torch.randn(self.HIDDEN, 4 * self.WIDTH, generator=generator).bfloat16()
        self.uploads = []
        self.released = []
        test = self

        class Operations(object):
            bfloat16, bfloat8_b, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'bf8', 'tile', 'dram'

            @staticmethod
            def get_device_tensors(tensor):
                return [('shard', chip) for chip in range(4)]

            @staticmethod
            def to_torch(shard):
                chip = shard[1]
                return test.full[:, chip * test.WIDTH:(chip + 1) * test.WIDTH].reshape(1, 1, test.HIDDEN, test.WIDTH).clone()

            @staticmethod
            def ShardTensorToMesh(mesh, dim):
                return ('shard', mesh, dim)

            @staticmethod
            def from_torch(value, **options):
                test.uploads.append((value, options))
                return SimpleNamespace(uploaded=value)

            @staticmethod
            def deallocate(tensor):
                test.released.append(tensor)

        self.operations = Operations
        self.mesh = object()
        self.model = SimpleNamespace(num_devices=4, vocab_size=248320, _lmhead_vocab_sharded=True, mesh_device=self.mesh,
                                     lm_head_weight=SimpleNamespace(dtype='bf8'))

    def build(self, **extra):
        lines = []
        with four(**{FLAG: self.path, **extra}):
            head = draft_vocab_tp.build_head(self.operations, self.model, log=lambda template, text: lines.append(text), hidden=self.HIDDEN)
        return head, lines

    def test_the_uploaded_weight_is_the_lists_columns_of_the_target_head_in_the_target_dtype(self):
        head, lines = self.build()
        value, options = self.uploads[0]
        self.assertEqual(tuple(value.shape), (1, 1, self.HIDDEN, 1280))
        self.assertEqual(value.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(value.reshape(self.HIDDEN, 1280), self.full[:, torch.tensor(self.ids)]))
        self.assertEqual(options['dtype'], 'bf8')
        self.assertEqual((options['layout'], options['memory_config'], options['device']), ('tile', 'dram', self.mesh))
        self.assertEqual(options['mesh_mapper'], ('shard', self.mesh, 3))
        self.assertIs(head, draft_vocab_tp.head_for(self.model))

    def test_chip_c_holds_the_c_th_quarter_of_the_list_whatever_chips_the_columns_came_from(self):
        self.build()
        columns = self.uploads[0][0].reshape(self.HIDDEN, 1280)
        for chip in range(4):
            chunk_ids = self.ids[chip * 320:(chip + 1) * 320]
            self.assertTrue(torch.equal(columns[:, chip * 320:(chip + 1) * 320], self.full[:, torch.tensor(chunk_ids)]))
        # the natural shards are lopsided (a coding list is mostly low ids) yet the chips' loads are equal
        natural = [sum(1 for token in self.ids if token // self.WIDTH == chip) for chip in range(4)]
        self.assertEqual(sum(natural), 1280)

    def test_the_build_line_names_rows_dtype_and_bytes(self):
        head, lines = self.build()
        self.assertEqual(len(lines), 1)
        self.assertIn(draft_vocab_tp.BUILT, lines[0])
        self.assertIn('rows=1280 per_chip=320 chips=4', lines[0])
        self.assertIn('dtype=bf8', lines[0])
        self.assertEqual(head.bytes_per_chip, int(self.HIDDEN * 320 * 1.0625))
        self.assertEqual(head.dtype_name, 'bf8')

    def test_a_second_build_for_the_same_model_is_refused(self):
        self.build()
        with four(**{FLAG: self.path}), self.assertRaisesRegex(ValueError, 'already built'):
            draft_vocab_tp.build_head(self.operations, self.model, hidden=self.HIDDEN)

    def test_the_flag_off_builds_nothing(self):
        with four(), self.assertRaisesRegex(ValueError, 'is off'):
            draft_vocab_tp.build_head(self.operations, self.model, hidden=self.HIDDEN)
        self.assertEqual(self.uploads, [])

    def test_a_pair_model_or_a_wrong_shard_count_is_refused_before_any_read(self):
        pair = SimpleNamespace(num_devices=2, vocab_size=248320, _lmhead_vocab_sharded=True, mesh_device=self.mesh, lm_head_weight=object())
        with four(**{FLAG: self.path}), self.assertRaisesRegex(ValueError, 'four-card target head'):
            draft_vocab_tp.build_head(self.operations, pair, hidden=self.HIDDEN)
        self.operations.get_device_tensors = staticmethod(lambda tensor: [1, 2])
        with four(**{FLAG: self.path}), self.assertRaisesRegex(ValueError, 'chips required'):
            draft_vocab_tp.build_head(self.operations, self.model, hidden=self.HIDDEN)

    def test_a_shard_of_the_wrong_size_is_refused(self):
        self.operations.to_torch = staticmethod(lambda shard: torch.zeros(self.HIDDEN, 100))
        with four(**{FLAG: self.path}), self.assertRaisesRegex(ValueError, 'elements'):
            draft_vocab_tp.build_head(self.operations, self.model, hidden=self.HIDDEN)

    def test_head_for_a_model_without_one_and_release(self):
        with self.assertRaisesRegex(ValueError, 'no shortlist head'):
            draft_vocab_tp.head_for(self.model)
        head, unused = self.build()
        draft_vocab_tp.release(self.operations, self.model)
        draft_vocab_tp.release(self.operations, self.model)
        self.assertEqual(self.released, [head.weight])
        with self.assertRaisesRegex(ValueError, 'no shortlist head'):
            draft_vocab_tp.head_for(self.model)

    def test_cut_columns_finds_every_column_in_exactly_one_shard(self):
        with four(**{FLAG: self.path}):
            plan = draft_vocab_tp.active_plan()
        read = []

        def shards(source):
            read.append(source)
            return self.full[:, source * self.WIDTH:(source + 1) * self.WIDTH]

        columns = draft_vocab_tp.cut_columns(shards, plan, self.WIDTH, self.HIDDEN)
        self.assertTrue(torch.equal(columns, self.full[:, torch.tensor(self.ids)]))
        # one shard in memory at a time, each read once
        self.assertEqual(sorted(read), sorted(set(read)))
        # an id past the last shard has no column
        with self.assertRaises(ValueError):
            draft_vocab_tp.cut_columns(shards, SimpleNamespace(tensor=lambda: torch.tensor([4 * self.WIDTH]), rows=1, chips=4), self.WIDTH, self.HIDDEN)


class FlagOffIdentityTests(Isolated):
    def test_the_twin_does_not_import_the_module_with_the_flag_off(self):
        saved = sys.modules.pop('draft_vocab_tp')
        try:
            with four():
                draft_shared_head_tp.candidate_chunks()
                operations = FakeOperations()
                weight = FakeTensor((1, 1, 5120, 62080))
                model = SimpleNamespace(num_devices=4, vocab_size=248320, _lmhead_vocab_sharded=True, lm_head_weight=weight)
                draft_shared_head_tp.shared_head_candidates(operations, model, FakeTensor((1, 1, 8, 5120)), [])
                values = torch.zeros(8, 16)
                chunks = [dict(chip=chip, start=start, stop=stop, values=values.clone(), indices=torch.arange(16).repeat(8, 1))
                          for chip in range(4) for start, stop in draft_shared_head_tp.candidate_chunks()]
                draft_shared_head_tp.merge_chunk_candidates(chunks)
                round_host._fast_merge(chunks, 8)
            self.assertNotIn('draft_vocab_tp', sys.modules)
        finally:
            sys.modules['draft_vocab_tp'] = saved

    def test_the_merge_with_the_flag_off_is_the_full_vocabulary_one(self):
        generator = torch.Generator().manual_seed(51)
        logits = torch.randn(16, 248320, generator=generator)
        with four():
            chunks = []
            for chip in range(4):
                for start, stop in draft_shared_head_tp.candidate_chunks():
                    scores, indices = logits[:, chip * 62080 + start:chip * 62080 + stop].topk(16, dim=-1)
                    chunks.append(dict(chip=chip, start=start, stop=stop, values=scores, indices=indices))
            tokens, scores = draft_shared_head_tp.merge_chunk_candidates(chunks, block_rows=16)
        expected_scores, expected_tokens = logits.topk(16, dim=-1)
        self.assertTrue(torch.equal(tokens, expected_tokens[None, 1:16]))
        self.assertTrue(torch.equal(scores, expected_scores[None, 1:16]))

    def test_off_values_leave_the_twin_on_its_own_path(self):
        for value in ('', '0'):
            with four(**{FLAG: value}):
                self.assertFalse(draft_shared_head_tp.vocab_requested())
                self.assertEqual(draft_shared_head_tp.candidate_chunks(), ((0, 32768), (32768, 62080)))


class ContractTests(Isolated):
    def profile(self, name='c2-packed-tp4-x-gate', gate_only=True, **env):
        base = {'QWEN_FAST_TP': '4'}
        base.update(env)
        return {'name': name, 'gate_only': gate_only, 'env': base}

    def test_the_names_are_the_modules(self):
        self.assertEqual(serving_c2_contract.DRAFT_VOCAB_NAMES, draft_vocab_tp.NAMES)
        self.assertEqual(serving_c2_contract.DRAFT_VOCAB_PREFIX, draft_vocab_tp.PREFIX)
        self.assertEqual(c2_smoke_check.DRAFT_VOCAB_FLAG, draft_vocab_tp.FLAG)
        self.assertEqual(c2_smoke_check.DRAFT_VOCAB_TOPK_FLAG, draft_vocab_tp.TOPK_FLAG)
        self.assertEqual(c2_smoke_check.DRAFT_VOCAB_ADMITTED, draft_vocab_tp.ADMITTED)
        self.assertEqual(c2_smoke_check.DRAFT_VOCAB_BUILT, draft_vocab_tp.BUILT)
        self.assertEqual(c2_smoke_check.DRAFT_VOCAB_ENGAGED, draft_vocab_tp.ENGAGED)
        self.assertEqual(c2_smoke_check.DRAFT_VOCAB_DEFAULT_TOPK, draft_vocab_tp.TOPK_DEFAULT)
        self.assertEqual(c2_smoke_check.DRAFT_VOCAB_NAMED_ROWS, {name: row[1] for name, row in draft_vocab_tp.NAMED.items()})

    def test_a_gate_profile_may_name_the_list(self):
        self.assertEqual(serving_c2_contract.draft_vocab_problems(self.profile(**{FLAG: CODING})), [])
        self.assertEqual(serving_c2_contract.draft_vocab_problems(self.profile(**{FLAG: CODING, TOPK: '16384'})), [])
        self.assertEqual(serving_c2_contract.draft_vocab_problems(self.profile()), [])

    def test_the_tau_labs_derived_arm_may_and_a_traffic_profile_may_not(self):
        lab = self.profile(name='c2-packed-tp4+taulab+dvocab', gate_only=False, **{FLAG: CODING})
        self.assertEqual(serving_c2_contract.draft_vocab_problems(lab), [])
        traffic = self.profile(name='c2-packed-tp4', gate_only=False, **{FLAG: CODING})
        problems = serving_c2_contract.draft_vocab_problems(traffic)
        self.assertEqual(len(problems), 1)
        self.assertIn('gate-only profile or the tau lab', problems[0])

    def test_unknown_names_the_pair_and_bad_values_are_refused(self):
        cases = (self.profile(QWEN_FAST_DRAFT_VOCAB_X='1', **{FLAG: CODING}), self.profile(**{FLAG: '1'}), self.profile(**{FLAG: CODING, TOPK: '99'}),
                 self.profile(**{TOPK: '16384'}), {'name': 'p-gate', 'gate_only': True, 'env': {FLAG: CODING}},
                 {'name': 'p-gate', 'gate_only': True, 'env': {FLAG: CODING, 'QWEN_FAST_TP': '2'}})
        for profile in cases:
            with self.subTest(env=profile['env']):
                self.assertTrue(serving_c2_contract.draft_vocab_problems(profile))

    def test_an_inherited_process_value_is_refused_outside_a_gate_and_dropped_inside_one(self):
        traffic = self.profile(name='c2-packed-tp4', gate_only=False)
        self.assertTrue(serving_c2_contract.draft_vocab_problems(traffic, {FLAG: CODING}))
        self.assertEqual(serving_c2_contract.draft_vocab_problems(traffic, {}), [])
        gate = self.profile(**{'QWEN_FAST_TP4_X': '1'})
        gate['mesh_graph_descriptor'] = '/x'
        environ = {FLAG: CODING, TOPK: '16384', 'OTHER': 'kept'}
        serving_c2_contract.apply_environment(gate, environ)
        self.assertNotIn(FLAG, environ)
        self.assertNotIn(TOPK, environ)
        self.assertEqual(environ['OTHER'], 'kept')
        named = self.profile(**{FLAG: CODING})
        named['mesh_graph_descriptor'] = '/x'
        environ = {}
        serving_c2_contract.apply_environment(named, environ)
        self.assertEqual(environ[FLAG], CODING)


class SmokeRuleTests(Isolated):
    LINES = ('[PINDIAG] draft vocab admitted: rows=40960 per_chip=10240 chips=4 topk_width=32768 sha256=94fbd2a832f6 (GATE ONLY, UNQUALIFIED; ...)',
             '[PINDIAG] draft vocab head built: rows=40960 per_chip=10240 chips=4 topk_width=32768 sha256=94fbd2a832f6 dtype=bf8 bytes_per_chip=55705600 ...',
             '[PINDIAG] draft vocab engaged rows=40960 per_chip=10240 chips=4 topk_width=32768 sha256=94fbd2a832f6 dtype=bf8 chunks=1 (was 2) rows_per_launch=32')

    def problems(self, env, lines):
        return c2_smoke_check.draft_vocab_problems(env, '\n'.join(lines))

    def test_the_markers_are_what_the_module_logs(self):
        captured = []
        ids = spread_ids(1280)
        path = write_ids(self.directory.name, ids)
        operations = SimpleNamespace()
        with four(**{FLAG: path}), patch.object(draft_vocab_tp, 'log_line', captured.append):
            draft_vocab_tp.admission()
            plan = draft_vocab_tp.active_plan()
            captured.append('%s: %s dtype=bf8 bytes_per_chip=1' % (draft_vocab_tp.BUILT, plan.describe()))
            draft_vocab_tp._HEADS[1] = None
            model = SimpleNamespace(num_devices=4, vocab_size=248320, _lmhead_vocab_sharded=True)
            draft_vocab_tp._HEADS[id(model)] = draft_vocab_tp.Head(FakeTensor((1, 1, 5120, 320)), plan, 'bf8', 1)
            draft_vocab_tp.shared_head_candidates(FakeOperations(), model, FakeTensor((1, 1, 8, 5120)), [])
        self.assertEqual(self.problems({FLAG: path}, captured), [])

    def test_flag_off_means_no_line_at_all(self):
        self.assertEqual(self.problems({}, ['nothing here']), [])
        self.assertEqual(self.problems({FLAG: '0'}, ['nothing here']), [])
        self.assertTrue(self.problems({}, self.LINES))
        self.assertTrue(self.problems({FLAG: '0'}, self.LINES[:1]))

    def test_the_flag_on_needs_each_line_exactly_once_in_order(self):
        env = {FLAG: CODING}
        self.assertEqual(self.problems(env, self.LINES), [])
        for index in range(3):
            self.assertTrue(self.problems(env, self.LINES[:index] + self.LINES[index + 1:]), index)
            self.assertTrue(self.problems(env, self.LINES + (self.LINES[index],)), index)
        self.assertTrue(self.problems(env, (self.LINES[1], self.LINES[0], self.LINES[2])))
        self.assertTrue(self.problems(env, ()))

    def test_the_lines_must_name_the_same_whole_tile_shortlist(self):
        env = {FLAG: CODING}
        changed = self.LINES[2].replace('rows=40960', 'rows=40832')
        self.assertTrue(self.problems(env, self.LINES[:2] + (changed,)))
        ragged = [line.replace('per_chip=10240', 'per_chip=10241') for line in self.LINES]
        self.assertTrue(self.problems(env, ragged))
        pair = [line.replace('chips=4', 'chips=2') for line in self.LINES]
        self.assertTrue(self.problems(env, pair))
        self.assertTrue(self.problems({FLAG: CODING}, [line.replace('rows=40960', 'rows=20480').replace('per_chip=10240', 'per_chip=5120') for line in self.LINES]))

    def test_the_width_flag_is_held(self):
        narrow = [line.replace('topk_width=32768', 'topk_width=16384') for line in self.LINES]
        self.assertEqual(self.problems({FLAG: CODING, TOPK: '16384'}, narrow), [])
        self.assertTrue(self.problems({FLAG: CODING}, narrow))
        self.assertTrue(self.problems({FLAG: CODING, TOPK: '16384'}, self.LINES))

    def test_check_and_the_lever_rule_call_it(self):
        env = {FLAG: CODING}
        self.assertTrue(any('draft-vocabulary' in problem for problem in c2_smoke_check.lever_engagement_problems(env, 'no lines')))
        self.assertFalse(any('draft-vocabulary' in problem for problem in c2_smoke_check.lever_engagement_problems(env, '\n'.join(self.LINES))))
        self.assertFalse(any('draft-vocabulary' in problem for problem in c2_smoke_check.lever_engagement_problems({}, 'no lines')))


class CopyListTests(unittest.TestCase):
    def test_every_runtime_file_is_in_all_three_copy_lists(self):
        overlay = (ROOT / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8').splitlines()
        dockerfile = (ROOT / 'docker' / 'qwen-fast-serving.Dockerfile').read_text(encoding='utf-8')
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-fast-serving-image.yml').read_text(encoding='utf-8')
        for name in draft_vocab_tp.RUNTIME_FILES:
            with self.subTest(name=name):
                self.assertTrue((HERE / name).is_file())
                self.assertIn('scripts/ci/' + name, overlay)
                self.assertIn('scripts/ci/' + name, dockerfile)
                self.assertIn(name, workflow)

    def test_the_named_files_are_the_runtime_files(self):
        stem = draft_vocab_tp.NAMED[CODING][0]
        self.assertEqual(set(draft_vocab_tp.RUNTIME_FILES), {'draft_vocab_tp.py', stem + '.ids', stem + '.json'})

    def test_the_generator_is_not_shipped(self):
        # the builder reads corpora and the tokenizer: it never belongs in the serving image
        for path in ((ROOT / 'docker' / 'qwen-c2-overlay.txt'), (ROOT / 'docker' / 'qwen-fast-serving.Dockerfile'),
                     (ROOT / '.github' / 'workflows' / 'qwen-fast-serving-image.yml')):
            self.assertNotIn('draft_vocab_build', path.read_text(encoding='utf-8'))

    def test_both_test_modules_are_on_the_cpu_allowlist(self):
        text = (ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        for name in ('test_draft_vocab_tp', 'test_draft_vocab_build', 'test_tp4_draft_vocab_jobs'):
            self.assertRegex(text, r'\b%s\b' % name)


if __name__ == '__main__':
    unittest.main()
