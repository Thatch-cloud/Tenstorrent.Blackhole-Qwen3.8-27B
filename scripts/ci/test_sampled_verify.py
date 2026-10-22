"""sampled_verify (QWEN_FAST_SAMPLED, docs/tp4-sampled-verify.md): the serving side of sampled verification, over a fake four-chip mesh (sampled_test_support), pure CPU.

Held here:
  - the switch is strict and profile-owned; a request's Sampling is read from what the contract left (seed in extra_args, not in the plugin's seed manager; top_k inside the
    slice; no penalties) and anything else is refused; the engine-side gate stays closed until the attach proved every block can draw;
  - the device slice: stock ttnn.topk, k = 64, on the 32,768-column chunks of the 62,080-wide shard and 32-row pieces (the shapes the drafter's head runs), intermediates freed,
    outputs held; a build that refuses k=64 refuses with the reason (fail closed); one fence in the batched reads' async mode;
  - the host merge of the chips' and chunks' top-64 is the full-vocabulary top-64 (ties at the cut aside, which the audit states), whatever order the device breaks ties in;
  - packed_predictions: each sampled entry's rows are the keyed draws at ticket.position + 1 + row from its own spec and seed, greedy entries keep the argmax rows, idle segments
    are never read; the audit compares the slice and the draw with the reference path and catches a corrupted slice or draw;
  - the sequential engine's host slice gives the same draws; the first token is the keyed draw at len(prompt) from the logits as the plugin's sampler would have seen them.
"""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / 'speculative-decoding' / 'harness'))

import sampled_accept as sa  # noqa: E402
import sampled_test_support as support  # noqa: E402
import sampled_verify as sv  # noqa: E402
from packed_shapes import m3_shape  # noqa: E402
from tp_test_support import four_cards  # noqa: E402

SWITCH = {sv.FLAG: '1'}


def params(**values):
    base = dict(temperature=0.7, top_p=0.9, top_k=8, min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, seed=None,
                extra_args={sv.SEED_KEY: 1234})
    base.update(values)
    return SimpleNamespace(**base)


class Clean(unittest.TestCase):
    def setUp(self):
        sv.reset()
        self.addCleanup(sv.reset)
        patcher = mock.patch.dict(os.environ, support.clean_environment(), clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.log = []
        logger = mock.patch.object(sv, 'log_line', side_effect=self.log.append)
        logger.start()
        self.addCleanup(logger.stop)
        context = four_cards()
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def lines(self, marker):
        return [line for line in self.log if marker in line]


class SwitchAndRequestTests(Clean):
    def test_the_switch_is_strict(self):
        self.assertFalse(sv.enabled())
        for value, expected in (('0', False), ('1', True)):
            self.assertEqual(sv.enabled({sv.FLAG: value}), expected)
        for bad in ('', 'yes', '2', 'true'):
            with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                sv.enabled({sv.FLAG: bad})

    def test_the_audit_needs_the_switch(self):
        self.assertFalse(sv.audit_enabled({}))
        self.assertTrue(sv.audit_enabled({sv.FLAG: '1', sv.AUDIT_FLAG: '1'}))
        with self.assertRaisesRegex(ValueError, 'needs'):
            sv.audit_enabled({sv.AUDIT_FLAG: '1'})

    def test_a_greedy_request_has_no_sampling(self):
        self.assertIsNone(sv.of_params(params(temperature=0.0, top_p=1.0, top_k=0, extra_args=None)))

    def test_a_sampled_request_is_read_from_what_the_contract_left(self):
        sampling = sv.of_params(params())
        self.assertEqual(sampling, sv.Sampling(sa.SamplingSpec(0.7, 0.9, 8), 1234))
        self.assertEqual(sv.of_params(params(extra_args={sv.SEED_KEY: 2 ** 64 - 1})).seed, 2 ** 64 - 1)

    def test_what_the_verifier_cannot_serve_is_refused(self):
        for broken in (dict(extra_args=None), dict(extra_args={}), dict(extra_args={sv.SEED_KEY: -1}), dict(extra_args={sv.SEED_KEY: 2 ** 64}),
                       dict(extra_args={sv.SEED_KEY: True}), dict(extra_args={sv.SEED_KEY: '3'}),
                       dict(seed=5), dict(top_k=0), dict(top_k=65), dict(top_k=-1), dict(top_p=None), dict(top_p=0.0), dict(top_p=1.5),
                       dict(min_p=0.1), dict(presence_penalty=0.5), dict(frequency_penalty=-1.0), dict(repetition_penalty=1.1),
                       dict(temperature=-1.0), dict(temperature=float('inf')), dict(temperature='hot')):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                sv.of_params(params(**broken))

    def test_the_seed_key_is_the_contracts(self):
        import serving_c2_contract as contract

        self.assertEqual(sv.SEED_KEY, contract.SAMPLED_SEED_KEY)
        self.assertEqual(sv.K, contract.SAMPLED_K_MAX)
        self.assertEqual(sv.FLAG, contract.SAMPLED_SWITCH)
        self.assertEqual(sv.AUDIT_FLAG, contract.SAMPLED_AUDIT_SWITCH)

    def test_the_gate_is_closed_until_the_attach_and_only_with_the_switch(self):
        sampled = params()
        with mock.patch.dict(os.environ, SWITCH):
            self.assertFalse(sv.accepts(sampled), 'no attach yet')
            blocks = [SimpleNamespace(sampled=True), SimpleNamespace(sampled=True)]
            self.assertEqual(sv.attach(blocks), 2)
            self.assertTrue(sv.accepts(sampled))
            self.assertFalse(sv.accepts(params(extra_args=None)), 'a request without its key')
            self.assertFalse(sv.accepts(params(temperature=0.0, top_p=1.0, top_k=0)), 'a greedy request is not the sampled gate\'s')
            self.assertEqual(len(self.lines(sv.ATTACH_MARKER)), 1)
            self.assertIn('blocks=2', self.lines(sv.ATTACH_MARKER)[0])
        self.assertFalse(sv.accepts(sampled), 'the switch is off')

    def test_the_attach_refuses_a_block_that_did_not_engage_and_does_nothing_with_the_switch_off(self):
        with mock.patch.dict(os.environ, SWITCH):
            with self.assertRaisesRegex(sv.SampledRefused, r'block\(s\) \[1\]'):
                sv.attach([SimpleNamespace(sampled=True), SimpleNamespace(sampled=False)])
            self.assertFalse(sv.STATE.ready)
            with self.assertRaises(sv.SampledRefused):
                sv.attach([SimpleNamespace()])
        self.assertEqual(sv.attach([SimpleNamespace(sampled=False)]), 0)
        self.assertFalse(sv.STATE.ready)


class LayoutTests(Clean):
    def test_a_64_row_block_is_two_pieces_of_two_chunks(self):
        plan = sv.layout(64, 62080)
        self.assertEqual(plan, ((0, 32, 0, 0, 32768), (0, 32, 1, 32768, 62080), (32, 32, 0, 0, 32768), (32, 32, 1, 32768, 62080)))

    def test_smaller_blocks_are_one_piece(self):
        for rows in (8, 16, 32):
            self.assertEqual(sv.layout(rows, 62080), ((0, rows, 0, 0, 32768), (0, rows, 1, 32768, 62080)))

    def test_other_row_counts_are_refused(self):
        for rows in (0, 1, 4, 24, 48, 128):
            with self.assertRaises(sv.SampledRefused):
                sv.layout(rows, 62080)

    def test_the_chunks_are_the_drafter_heads(self):
        import draft_shared_head_tp

        self.assertEqual(sv.chunks(62080), draft_shared_head_tp.candidate_chunks())
        self.assertEqual(sv.CHUNK, 32768)


def random_rows(rows, seed, scale=2.0):
    """bf16-valued logits rows over the whole vocabulary with ties (a coarse grid) - the worst case for the merge."""
    rng = np.random.default_rng(seed)
    return (np.round(rng.normal(size=(rows, support.VOCAB)) * scale * 8) / 8).astype(np.float32)


def gather(ops, rows, array):
    """The slice tensors the trace produces for `array` and what the host reads back and merges: (outputs, merged values, merged ids)."""
    logits = support.logits_tensor(array)
    outputs = sv.device_slice(ops, logits, rows)
    per_chip = sv.read_tensors(ops, 'mesh', outputs)
    chip_values, chip_ids = sv.chip_slices(per_chip, rows, support.SHARD)
    values, ids = sv.merged_slice(chip_values, chip_ids, np.arange(rows), support.SHARD)
    return logits, outputs, values, ids


class DeviceSliceTests(Clean):
    def test_stock_topk_k64_on_the_drafter_heads_chunks_and_pieces(self):
        ops = support.Ops()
        logits, outputs, _, _ = gather(ops, 64, random_rows(64, 1))
        self.assertEqual(len(outputs), 8)
        self.assertEqual(ops.topks, [((1, 1, 32, 32768), 64)] * 4)
        self.assertEqual(ops.slices, [((0, 0, 0, 0), (1, 1, 32, 32768)), ((0, 0, 0, 32768), (1, 1, 32, 62080)),
                                     ((0, 0, 32, 0), (1, 1, 64, 32768)), ((0, 0, 32, 32768), (1, 1, 64, 62080))])
        self.assertEqual([padding for padding, value in ops.pads], [((0, 0), (0, 0), (0, 0), (0, 3456))] * 2)
        self.assertTrue(all(value == float('-inf') for _, value in ops.pads))

    def test_intermediates_are_freed_and_the_outputs_are_held(self):
        ops = support.Ops()
        logits, outputs, _, _ = gather(ops, 32, random_rows(32, 2))
        freed = {id(tensor) for tensor in ops.freed}
        self.assertTrue(all(id(tensor) in freed for tensor in ops.live if tensor not in outputs and tensor is not logits and tensor.shape[3] in (32768, 29312)))
        for tensor in outputs:
            self.assertFalse(tensor.freed)
            self.assertEqual(tensor.shape[3], 64)
        self.assertFalse(logits.freed)

    def test_a_refused_k_fails_closed_with_the_reason_and_frees_what_it_made(self):
        ops = support.Ops(k_limit=32)
        with self.assertRaisesRegex(sv.SampledRefused, r'ttnn\.topk refused k=64 .*at most 32'):
            sv.device_slice(ops, support.logits_tensor(random_rows(32, 3)), 32)
        self.assertTrue(all(tensor.freed for tensor in ops.live if tensor.shape[3] in (32768, 29312, 64)))

    def test_a_second_launch_refusing_frees_the_first_launchs_outputs(self):
        ops = support.Ops()
        real, calls = ops.topk, []

        def second_refuses(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise support.RefusedTopk('TT_FATAL')
            return real(*args, **kwargs)

        ops.topk = second_refuses
        with self.assertRaises(sv.SampledRefused):
            sv.device_slice(ops, support.logits_tensor(random_rows(16, 4)), 16)
        first = [tensor for tensor in ops.live if tensor.shape[3] == 64]
        self.assertEqual(len(first), 2)
        self.assertTrue(all(tensor.freed for tensor in first))

    def test_logits_it_cannot_take_are_refused(self):
        ops = support.Ops()
        good = support.logits_tensor(random_rows(32, 5))
        wrong_width = support.Tensor([torch.zeros(1, 1, 32, 100)] * 4)
        wrong_dtype = support.Tensor(good.chips, dtype='fp32')
        wrong_layout = support.Tensor(good.chips, layout='row_major')
        for tensor, rows in ((wrong_width, 32), (wrong_dtype, 32), (wrong_layout, 32), (good, 16)):
            with self.subTest(shape=tensor.shape, dtype=tensor.dtype), self.assertRaises(sv.SampledRefused):
                sv.device_slice(ops, tensor, rows)

    def test_the_merged_slice_is_the_full_vocabulary_slice_and_its_first_candidate_the_argmax(self):
        for rows, seed in ((16, 11), (32, 12), (64, 13)):
            array = random_rows(rows, seed)
            ops = support.Ops()
            logits, _, values, ids = gather(ops, rows, array)
            bf16 = torch.from_numpy(array).bfloat16().float().numpy()
            reference_values, reference_ids = sa.topk_slice(bf16, sa.K_MAX)
            for row in range(rows):
                self.assertIsNone(sv.slice_difference(values[row], ids[row], reference_values[row].astype(np.float64), reference_ids[row], bf16[row]), (rows, row))
            self.assertTrue(np.array_equal(ids[:, 0], bf16.argmax(axis=1)), 'first occurrence on ties')

    def test_the_merge_does_not_depend_on_the_devices_tie_order(self):
        array = random_rows(32, 21, scale=0.3)           # a very coarse grid: many ties inside the slice
        first = support.Ops()
        _, _, values_a, ids_a = gather(first, 32, array)
        second = support.Ops()
        real = second.topk

        def reversed_ties(tensor, k, dim=-1, largest=True, sorted=True):
            values, indices = real(tensor, k, dim, largest, sorted)
            for part in (values, indices):
                part.chips = [chip.flip(-1) for chip in part.chips]
            return values, indices

        second.topk = reversed_ties
        _, _, values_b, ids_b = gather(second, 32, array)
        self.assertTrue(np.array_equal(values_a, values_b))
        bf16 = torch.from_numpy(array).bfloat16().float().numpy()
        for row in range(32):
            self.assertIsNone(sv.slice_difference(values_b[row], ids_b[row], values_a[row], sa.topk_slice(bf16[row][None], 64)[1][0], bf16[row]))

    def test_an_index_outside_its_chunk_is_a_device_fault(self):
        ops = support.Ops()
        logits = support.logits_tensor(random_rows(16, 6))
        outputs = sv.device_slice(ops, logits, 16)
        outputs[3].chips[0][0, 0, 0, 0] = 29312              # the second chunk is 29,312 columns wide: this index points into the -inf pad
        per_chip = sv.read_tensors(ops, 'mesh', outputs)
        with self.assertRaisesRegex(sv.SampledRefused, 'outside its'):
            sv.chip_slices(per_chip, 16, support.SHARD)

    def test_the_reads_follow_the_batched_reads_flag(self):
        import batched_reads_tp as reads

        array = random_rows(64, 7)
        results = {}
        for mode, single, composed, copies, fences in (('0', 32, 0, 0, 0), ('1', 0, 8, 0, 0), ('async', 0, 0, 8, 1)):
            with self.subTest(mode=mode):
                reads.reset()
                self.addCleanup(reads.reset)
                with mock.patch.dict(os.environ, {'QWEN_FAST_BATCHED_READS': mode}), mock.patch('verify_prestage.log_line'):
                    ops = support.Ops()
                    _, _, values, ids = gather(ops, 64, array)
                self.assertEqual((ops.single_reads, ops.composed_reads, ops.copies), (single, composed, copies), mode)
                self.assertEqual(ops.fences, fences, mode)
                results[mode] = (values, ids)
        for mode in ('1', 'async'):
            self.assertTrue(np.array_equal(results[mode][0], results['0'][0]) and np.array_equal(results[mode][1], results['0'][1]), mode)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the packed block's predictions
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

SHAPE = m3_shape(1024)


def make_entry(segment_tokens, position, sampling, request_id='r'):
    ticket = SimpleNamespace(tokens=tuple(segment_tokens), position=position, request_id=request_id)
    session = SimpleNamespace(sampling=sampling)
    return dict(request=SimpleNamespace(session=session), ticket=ticket, request_id=request_id)


class PackedPredictionsTests(Clean):
    def build(self, ops, rows_array, entries, segments, **block_options):
        logits = support.logits_tensor(rows_array)
        outputs = sv.device_slice(ops, logits, 64)
        block = SimpleNamespace(output=(logits, None, None, None, *outputs), operations=ops, mesh='mesh', block_rows=64, rows_per_user=16, shape=SHAPE, users=4,
                                shard_audit=False, **block_options)
        host = [int(token) for token in torch.from_numpy(rows_array).bfloat16().float().numpy().argmax(axis=1)]
        predictions = [host[16 * segment:16 * segment + 16] for segment in segments]
        return block, host, predictions

    def test_each_sampled_entry_draws_its_rows_with_its_own_spec_seed_and_positions(self):
        array = random_rows(64, 31)
        entries = [make_entry(range(16), 100, sv.Sampling(sa.SamplingSpec(0.7, 0.9, 8), 5)), make_entry(range(16), 4000, sv.Sampling(sa.SamplingSpec(1.0, 1.0, 64), 2 ** 63 + 9)),
                   make_entry(range(16), 77, None)]
        segments = [2, 0, 3]                                  # not the entries' order; segment 1 idle
        array[16:32] = np.nan                                  # the idle segment's rows are never read
        ops = support.Ops()
        block, host, predictions = self.build(ops, array, entries, segments)
        out, metrics = sv.packed_predictions(block, entries, segments, host, predictions, None)
        bf16 = torch.from_numpy(array).bfloat16().float().numpy()
        for number, (sampling, position) in enumerate(((entries[0]['request'].session.sampling, 100), (entries[1]['request'].session.sampling, 4000))):
            segment = segments[number]
            expected = [sa.sample_logits(bf16[16 * segment + row], sampling.spec, sampling.seed, position + 1 + row) for row in range(16)]
            self.assertEqual(out[number], expected, number)
        self.assertEqual(out[2], predictions[2], 'a greedy entry keeps its argmax rows')
        self.assertEqual(metrics['entries'], 2)
        self.assertEqual(metrics['rows'], 32)
        self.assertEqual(block.sampled_rounds, 1)
        self.assertEqual(len(self.lines(sv.ROUND_MARKER)), 1)

    def test_a_block_of_greedy_entries_reads_nothing_and_changes_nothing(self):
        array = random_rows(64, 32)
        entries = [make_entry(range(16), 100, None), make_entry(range(16), 50, None)]
        ops = support.Ops()
        block, host, predictions = self.build(ops, array, entries, [0, 1])
        reads_before = (ops.single_reads, ops.composed_reads, ops.copies)
        out, metrics = sv.packed_predictions(block, entries, [0, 1], host, predictions, 1)
        self.assertIs(out, predictions)
        self.assertEqual(metrics, dict(entries=0))
        self.assertEqual((ops.single_reads, ops.composed_reads, ops.copies), reads_before)

    def test_a_row_does_not_depend_on_what_else_is_in_the_block(self):
        array = random_rows(64, 33)
        spec = sv.Sampling(sa.SamplingSpec(0.9, 0.95, 20), 41)
        alone = [make_entry(range(16), 200, spec)]
        crowd = [make_entry(range(16), 9, sv.Sampling(sa.SamplingSpec(0.3, 1.0, 5), 3)), make_entry(range(16), 200, spec), make_entry(range(16), 31, None)]
        first_ops = support.Ops()
        block, host, predictions = self.build(first_ops, array, alone, [1])
        out_alone, _ = sv.packed_predictions(block, alone, [1], host, predictions, None)
        second_ops = support.Ops()
        block, host, predictions = self.build(second_ops, array, crowd, [3, 1, 0])
        out_crowd, _ = sv.packed_predictions(block, crowd, [3, 1, 0], host, predictions, None)
        self.assertEqual(out_alone[0], out_crowd[1])

    def test_the_noise_is_keyed_by_the_position_so_a_shifted_ticket_shifts_the_draws(self):
        array = random_rows(64, 34)
        spec = sv.Sampling(sa.SamplingSpec(1.0, 1.0, 64), 8)
        entries = [make_entry(range(16), 100, spec)]
        ops = support.Ops()
        block, host, predictions = self.build(ops, array, entries, [0])
        base, _ = sv.packed_predictions(block, entries, [0], host, predictions, None)
        shifted = [make_entry(range(16), 101, spec)]
        shifted_out, _ = sv.packed_predictions(block, shifted, [0], host, predictions, None)
        bf16 = torch.from_numpy(array).bfloat16().float().numpy()
        self.assertEqual(shifted_out[0], [sa.sample_logits(bf16[row], spec.spec, spec.seed, 102 + row) for row in range(16)])
        self.assertNotEqual(base[0], shifted_out[0])

    def test_a_ticket_of_the_wrong_width_is_refused(self):
        array = random_rows(64, 35)
        entries = [make_entry(range(8), 100, sv.Sampling(sa.SamplingSpec(), 1))]
        ops = support.Ops()
        block, host, predictions = self.build(ops, array, entries, [0])
        with self.assertRaisesRegex(sv.SampledRefused, '8 rows in a block of 16'):
            sv.packed_predictions(block, entries, [0], host, predictions, None)

    def test_a_slice_whose_first_candidate_is_not_the_argmax_is_logged_and_raises_when_audited(self):
        array = random_rows(64, 36)
        entries = [make_entry(range(16), 100, sv.Sampling(sa.SamplingSpec(), 1))]
        ops = support.Ops()
        block, host, predictions = self.build(ops, array, entries, [0])
        host[3] = (host[3] + 1) % support.VOCAB
        out, _ = sv.packed_predictions(block, entries, [0], host, predictions, None)
        self.assertEqual(len(out[0]), 16)
        self.assertEqual(len(self.lines(sv.AUDIT_MISMATCH)), 1)
        self.assertIn('top1 rows=[3]', self.lines(sv.AUDIT_MISMATCH)[0])
        block.shard_audit = True
        with self.assertRaisesRegex(AssertionError, 'audit mismatch'):
            sv.packed_predictions(block, entries, [0], host, predictions, None)

    def audited(self, ops, array, entries, segments):
        block, host, predictions = self.build(ops, array, entries, segments)
        with mock.patch.dict(os.environ, {sv.FLAG: '1', sv.AUDIT_FLAG: '1'}):
            return sv.packed_predictions(block, entries, segments, host, predictions, 1), block

    def test_an_audited_round_reads_the_logits_back_and_compares_the_slice_and_the_draw(self):
        array = random_rows(64, 37)
        entries = [make_entry(range(16), 100, sv.Sampling(sa.SamplingSpec(0.8, 0.95, 20), 17)), make_entry(range(16), 2000, sv.Sampling(sa.SamplingSpec(1.0, 1.0, 64), 18))]
        ops = support.Ops()
        (out, metrics), block = self.audited(ops, array, entries, [1, 2])
        passing = self.lines(sv.AUDIT_MARKER + ' site=packed_verify round=1')
        self.assertEqual(len(passing), 1, self.log)
        self.assertIn('slice=exact draws=exact', passing[0])
        self.assertEqual(self.lines(sv.AUDIT_MISMATCH), [])
        self.assertGreater(ops.single_reads, 32, 'the audited round read the logits back too')

    def test_an_unaudited_round_does_not_read_the_logits_back(self):
        array = random_rows(64, 38)
        entries = [make_entry(range(16), 100, sv.Sampling(sa.SamplingSpec(0.8, 0.95, 20), 17))]
        ops = support.Ops()
        block, host, predictions = self.build(ops, array, entries, [0])
        sv.packed_predictions(block, entries, [0], host, predictions, 1)
        self.assertEqual(ops.single_reads, 32)
        self.assertEqual(self.lines(sv.AUDIT_MARKER + ' site'), [])

    def test_the_audit_catches_a_corrupted_slice(self):
        array = random_rows(64, 39)
        entries = [make_entry(range(16), 100, sv.Sampling(sa.SamplingSpec(1.0, 1.0, 64), 19))]
        ops = support.Ops()
        block, host, predictions = self.build(ops, array, entries, [0])
        block.output[-8].chips[1][0, 0, 5, 7] += 1.0       # one value of one chip's first chunk, bumped by a whole step
        with mock.patch.dict(os.environ, {sv.FLAG: '1', sv.AUDIT_FLAG: '1'}), self.assertRaisesRegex(AssertionError, 'audit mismatch'):
            sv.packed_predictions(block, entries, [0], host, predictions, 1)
        self.assertEqual(len(self.lines(sv.AUDIT_MISMATCH)), 1)

    def test_the_audit_catches_a_wrong_draw(self):
        array = random_rows(64, 40)
        entries = [make_entry(range(16), 100, sv.Sampling(sa.SamplingSpec(1.0, 1.0, 64), 20))]
        ops = support.Ops()
        block, host, predictions = self.build(ops, array, entries, [0])
        real = sa.draw_batch
        calls = []

        def wrong_first(*args, **kwargs):
            result = real(*args, **kwargs)
            calls.append(1)
            if len(calls) == 1:
                result = result.copy()
                result[2] = (result[2] + 1) % support.VOCAB
            return result

        with mock.patch.dict(os.environ, {sv.FLAG: '1', sv.AUDIT_FLAG: '1'}), mock.patch.object(sa, 'draw_batch', wrong_first), \
                self.assertRaisesRegex(AssertionError, 'draws rows=\\[2\\]'):
            sv.packed_predictions(block, entries, [0], host, predictions, 1)

    def test_a_tie_at_the_cut_is_the_devices_to_break_and_the_audit_says_so(self):
        values_a = np.array([5.0, 4.0, 3.0, 3.0, 3.0], dtype=np.float64)
        row = np.array([5.0, 4.0, 3.0, 3.0, 3.0, 3.0, 1.0], dtype=np.float32)
        # the reference keeps the lowest ids of the tie at the cut; a device may hold any ids of that value
        self.assertIsNone(sv.slice_difference(values_a, np.array([0, 1, 2, 3, 5]), values_a, np.array([0, 1, 2, 3, 4]), row))
        self.assertIn('tie group', sv.slice_difference(values_a, np.array([0, 1, 2, 3, 6]), values_a, np.array([0, 1, 2, 3, 4]), row))
        self.assertIn('tie group', sv.slice_difference(values_a, np.array([0, 1, 2, 3, 3]), values_a, np.array([0, 1, 2, 3, 4]), row))
        self.assertIn('above the cut', sv.slice_difference(values_a, np.array([1, 0, 2, 3, 4]), values_a, np.array([0, 1, 2, 3, 4]), row))
        self.assertIn('values differ', sv.slice_difference(values_a + 0.5, np.array([0, 1, 2, 3, 4]), values_a, np.array([0, 1, 2, 3, 4]), row))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the sequential engine
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class SequentialTests(Clean):
    def engine(self, array, replicated=False):
        ops = support.Ops()
        if replicated:
            logits = support.Tensor([torch.from_numpy(array).reshape(1, 1, *array.shape).bfloat16()] * 4)
        else:
            logits = support.logits_tensor(array)
        return SimpleNamespace(operations=ops, model=SimpleNamespace(args=SimpleNamespace(vocab_size=support.VOCAB))), (logits, None)

    def test_the_rows_are_the_keyed_draws_at_the_tickets_positions(self):
        for rows in (1, 2, 4):
            array = random_rows(rows, 50 + rows)
            engine, output = self.engine(array)
            sampling = sv.Sampling(sa.SamplingSpec(0.8, 0.95, 20), 99)
            ticket = SimpleNamespace(tokens=tuple(range(rows)), position=310)
            got = sv.sequential_predictions(engine, output, ticket, sampling)
            bf16 = torch.from_numpy(array).bfloat16().float().numpy()
            self.assertEqual(got, [sa.sample_logits(bf16[row], sampling.spec, sampling.seed, 311 + row) for row in range(rows)], rows)

    def test_the_first_calls_are_logged_so_a_card_run_can_see_the_path_ran(self):
        engine, output = self.engine(random_rows(2, 62))
        ticket = SimpleNamespace(tokens=(1, 2), position=40)
        for _ in range(6):
            sv.sequential_predictions(engine, output, ticket, sv.Sampling(sa.SamplingSpec(), 1))
        lines = self.lines(sv.SEQUENTIAL_MARKER)
        self.assertEqual(len(lines), sv.FIRST_LOGGED)
        self.assertIn('n=1 rows=2 position=40', lines[0])

    def test_the_replicated_full_row_gives_the_same_draws(self):
        array = random_rows(4, 60)
        sampling = sv.Sampling(sa.SamplingSpec(1.0, 1.0, 64), 7)
        ticket = SimpleNamespace(tokens=(1, 2, 3, 4), position=5)
        sharded = sv.sequential_predictions(*self.engine(array), ticket, sampling)
        full = sv.sequential_predictions(*self.engine(array, replicated=True), ticket, sampling)
        self.assertEqual(sharded, full)

    def test_an_unexpected_width_is_refused(self):
        engine, _ = self.engine(random_rows(1, 61))
        output = (support.Tensor([torch.zeros(1, 1, 1, 123)] * 4), None)
        with self.assertRaisesRegex(sv.SampledRefused, 'neither'):
            sv.sequential_predictions(engine, output, SimpleNamespace(tokens=(1,), position=0), sv.Sampling(sa.SamplingSpec(), 1))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the first token
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class FakeSampler(object):
    """The plugin's host sampler: records what it saw and rescales the logits in place the way vLLM's sampler may; returns the token it 'sampled' unkeyed."""

    def __init__(self, token=11):
        self.token, self.calls = token, []

    def __call__(self, logits=None, sampling_metadata=None):
        self.calls.append(logits.clone())
        logits.div_(0.5)
        return SimpleNamespace(sampled_token_ids=torch.full((logits.shape[0], 1), self.token, dtype=torch.int32), logprobs_tensors=None)


class FirstTokenTests(Clean):
    def runner(self, **options):
        return SimpleNamespace(host_sampler=FakeSampler(), sample_on_device_mode=options.get('mode'))

    def draw(self, wrapper, request_id, array, **kwargs):
        with wrapper.sampling_for(request_id):
            return wrapper(logits=torch.from_numpy(array).reshape(1, support.VOCAB).clone(), **kwargs)

    def test_the_wrapper_draws_token_zero_at_the_prompt_length_from_the_pristine_logits(self):
        runner = self.runner()
        inner = runner.host_sampler
        wrapper = sv.FirstToken.install(runner)
        self.assertIs(runner.host_sampler, wrapper)
        sampling = sv.of_params(params())
        self.assertEqual(wrapper.arm('req-1', params(), 1000), sampling)
        array = random_rows(1, 70)
        output = self.draw(wrapper, 'req-1', array, sampling_metadata=object())
        expected = sa.sample_logits(array[0], sampling.spec, sampling.seed, 1000)
        self.assertEqual(int(output.sampled_token_ids[0, 0]), expected)
        self.assertEqual(output.sampled_token_ids.dtype, torch.int32)
        self.assertTrue(torch.equal(inner.calls[0], torch.from_numpy(array).reshape(1, support.VOCAB)), 'the plugin\'s sampler still ran, on the same logits')
        result = SimpleNamespace(sampled_token_ids=[[expected]])
        self.assertEqual(wrapper.finish('req-1', result), expected)
        self.assertEqual(wrapper.arms, {}, 'an arm is spent')
        self.assertEqual(len(self.lines(sv.FIRST_TOKEN_MARKER)), 1)
        self.assertIn('position=1000', self.lines(sv.FIRST_TOKEN_MARKER)[0])
        wrapper.uninstall(runner)
        self.assertIs(runner.host_sampler, inner)

    def test_the_draw_is_a_function_of_seed_and_position_only(self):
        runner = self.runner()
        wrapper = sv.FirstToken.install(runner)
        array = random_rows(1, 71)
        tokens = []
        for position in (50, 50, 51):
            wrapper.arm('r', params(), position)
            tokens.append(int(self.draw(wrapper, 'r', array).sampled_token_ids[0, 0]))
            wrapper.forget('r')
        self.assertEqual(tokens[0], tokens[1])
        spec = sv.of_params(params()).spec
        self.assertEqual(tokens[2], sa.sample_logits(array[0], spec, 1234, 51))

    def test_a_call_outside_the_scope_of_an_armed_request_passes_through_untouched(self):
        runner = self.runner()
        inner = runner.host_sampler
        wrapper = sv.FirstToken.install(runner)
        wrapper.arm('r', params(), 5)                              # armed, but the plugin is not sampling its final chunk (a mid-prompt step, another request's prefill)
        output = wrapper(logits=torch.ones(3, 10))
        self.assertEqual(output.sampled_token_ids.shape, (3, 1))
        self.assertEqual(int(output.sampled_token_ids[0, 0]), inner.token)
        with wrapper.sampling_for('someone-else'):
            wrapper(logits=torch.ones(1, 10))
        self.assertEqual(self.lines(sv.FIRST_TOKEN_MARKER), [])
        self.assertIn('r', wrapper.arms, 'and its arm is still there')
        self.assertIsNone(wrapper.finish('unarmed', None), 'nothing armed, nothing to check')

    def test_arms_are_per_request_so_a_parked_prefill_that_resumes_is_still_drawn(self):
        runner = self.runner()
        wrapper = sv.FirstToken.install(runner)
        long_array, short_array = random_rows(1, 80), random_rows(1, 81)
        wrapper.arm('long', params(extra_args={sv.SEED_KEY: 11}), 9000)       # admitted first, then parked while a short one ran
        wrapper.arm('short', params(extra_args={sv.SEED_KEY: 22}), 100)
        short = int(self.draw(wrapper, 'short', short_array).sampled_token_ids[0, 0])
        self.assertEqual(wrapper.finish('short', SimpleNamespace(sampled_token_ids=[[short]])), short)
        long = int(self.draw(wrapper, 'long', long_array).sampled_token_ids[0, 0])
        spec = sv.of_params(params()).spec
        self.assertEqual(short, sa.sample_logits(short_array[0], spec, 22, 100))
        self.assertEqual(long, sa.sample_logits(long_array[0], spec, 11, 9000))
        self.assertEqual(wrapper.finish('long', SimpleNamespace(sampled_token_ids=[[long]])), long)

    def test_a_greedy_request_holds_no_arm_and_clears_a_stale_one_of_its_own_id(self):
        runner = self.runner()
        wrapper = sv.FirstToken.install(runner)
        wrapper.arm('a', params(), 10)
        self.assertIsNone(wrapper.arm('b', params(temperature=0.0, top_p=1.0, top_k=0, extra_args=None), 20))
        self.assertEqual(sorted(wrapper.arms), ['a'])
        self.assertIsNone(wrapper.arm('a', params(temperature=0.0, top_p=1.0, top_k=0, extra_args=None), 20))
        self.assertEqual(wrapper.arms, {})
        with wrapper.sampling_for('b'):
            output = wrapper(logits=torch.ones(1, 10))
        self.assertEqual(int(output.sampled_token_ids[0, 0]), 11)

    def test_the_oldest_arm_goes_when_too_many_requests_were_armed_and_never_finished(self):
        wrapper = sv.FirstToken.install(self.runner())
        for number in range(sv.FirstToken.MAX_ARMS + 5):
            wrapper.arm('r%d' % number, params(), number)
        self.assertEqual(len(wrapper.arms), sv.FirstToken.MAX_ARMS)
        self.assertNotIn('r0', wrapper.arms)
        self.assertIn('r%d' % (sv.FirstToken.MAX_ARMS + 4), wrapper.arms)

    def test_an_armed_request_that_was_not_drawn_fails_closed(self):
        runner = self.runner()
        wrapper = sv.FirstToken.install(runner)
        wrapper.arm('r', params(), 5)
        with wrapper.sampling_for('r'):
            pass                                                   # the plugin never called the host sampler for it
        with self.assertRaisesRegex(sv.SampledRefused, 'without the keyed draw'):
            wrapper.finish('r', SimpleNamespace(sampled_token_ids=[[3]]))
        wrapper.arm('r', params(), 5)
        self.draw(wrapper, 'r', random_rows(1, 72))
        with self.assertRaisesRegex(sv.SampledRefused, 'is not the keyed draw'):
            wrapper.finish('r', SimpleNamespace(sampled_token_ids=[[3]]))
        wrapper.arm('r', params(), 5)
        self.draw(wrapper, 'r', random_rows(1, 72))
        with self.assertRaisesRegex(sv.SampledRefused, 'first token None is not the keyed draw'):
            wrapper.finish('r', None)

    def test_a_draw_for_another_request_is_not_this_ones(self):
        wrapper = sv.FirstToken.install(self.runner())
        wrapper.arm('a', params(), 5)
        wrapper.arm('b', params(), 6)
        self.draw(wrapper, 'a', random_rows(1, 73))
        with self.assertRaisesRegex(sv.SampledRefused, 'drawn for a'):
            wrapper.finish('b', SimpleNamespace(sampled_token_ids=[[3]]))

    def test_one_logits_row_is_required(self):
        runner = self.runner()
        wrapper = sv.FirstToken.install(runner)
        wrapper.arm('r', params(), 5)
        for logits in (torch.ones(2, 10), torch.ones(10), None):
            with self.subTest(shape=None if logits is None else tuple(logits.shape)), self.assertRaises(sv.SampledRefused), wrapper.sampling_for('r'):
                wrapper(logits=logits)

    def test_the_prefill_must_be_sampled_on_the_host(self):
        with self.assertRaisesRegex(sv.SampledRefused, 'host'):
            sv.FirstToken.install(self.runner(mode='all'))
        self.assertIsNotNone(sv.FirstToken.install(self.runner(mode='decode_only')))
        with self.assertRaisesRegex(sv.SampledRefused, 'host_sampler'):
            sv.FirstToken.install(SimpleNamespace())

    def test_the_wrapper_forwards_other_attributes_to_the_plugins_sampler(self):
        runner = self.runner()
        runner.host_sampler.extra = 'x'
        wrapper = sv.FirstToken.install(runner)
        self.assertEqual(wrapper.extra, 'x')


if __name__ == '__main__':
    unittest.main()
