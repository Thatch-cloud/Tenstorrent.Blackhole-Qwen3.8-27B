from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import torch

from dflash_device import DFlashDevice
from dflash_prefill_window import BATCHED_PREFILL_ENTRIES, PrefillWindowCapture, chunk_window, prefill_window, snapshot_prefill_tail, validate_prefill_chunks


class PrefillWindowTests(unittest.TestCase):
    def operations(self):
        return SimpleNamespace(bfloat16=torch.bfloat16, DRAM_MEMORY_CONFIG='dram',
            slice=lambda value, start, end: value[..., start[2]:end[2], :],
            clone=lambda value, **kwargs: value.clone(), deallocate=Mock(),
            concat=lambda values, dim, **kwargs: torch.cat(values, dim=dim),
            get_device_tensors=lambda value: [value, value], to_torch=lambda value: value)

    def test_absolute_frontier_and_window_are_distinct(self):
        for position in (170, 2047, 2048, 2049, 4093, 4096, 64504):
            window = prefill_window(position)
            self.assertEqual(window['end'], position)
            self.assertEqual(window['rows'], min(position, 2048))
            self.assertEqual(window['start'] + window['rows'], position)
        for position in (0, -1, True, 1.5, 65505):
            with self.assertRaises(ValueError):
                prefill_window(position)

    def test_tail_excludes_prefix_and_padding_and_owns_its_storage(self):
        operations = self.operations()
        for position in (170, 2049, 4093):
            value = (torch.arange(position + 32).reshape(1, 1, -1, 1) % 97).expand(1, 1, -1, 2560).bfloat16()
            window = prefill_window(position)
            expected = value[..., window['start']:position, :].clone()
            checks = []
            with patch('dflash_prefill_window.addresses', side_effect=lambda operations, tensor: tensor.untyped_storage().data_ptr()):
                actual = snapshot_prefill_tail(operations, value, position, checks=checks)
            value.zero_()
            self.assertTrue(torch.equal(actual.view(torch.int16), expected.view(torch.int16)))
            self.assertEqual(checks, [dict(chip=chip, **window, exact=True) for chip in range(2)])
        operations.deallocate.assert_not_called()

    def test_measured_snapshot_never_reads_to_host(self):
        operations = self.operations()
        operations.to_torch = Mock(side_effect=AssertionError('Timed capture must not audit'))
        value = torch.zeros((1, 1, 4096, 2560), dtype=torch.bfloat16)
        with patch('dflash_prefill_window.addresses', return_value=(1, 2)):
            self.assertEqual(snapshot_prefill_tail(operations, value, 4096).shape, (1, 1, 2048, 2560))
        operations.to_torch.assert_not_called()

    def test_chunk_or_wrong_precision_cannot_masquerade_as_complete_prefill(self):
        operations = self.operations()
        for value in (torch.zeros((1, 1, 2048, 2560), dtype=torch.bfloat16),
                torch.zeros((1, 1, 4096, 2560)), torch.zeros((1, 1, 4096, 128), dtype=torch.bfloat16)):
            with self.assertRaises(ValueError):
                snapshot_prefill_tail(operations, value, 4096)

    def test_constructor_requires_explicit_tail_origin_and_exact_window_length(self):
        model = SimpleNamespace(num_devices=2, vocab_size=248320, _lmhead_vocab_sharded=True)
        for rows, start in ((4096, 0), (2048, 0), (2048, 2047), (2049, 2048)):
            features = [SimpleNamespace(shape=(1, 1, rows, 2560))] * 5
            with self.subTest(rows=rows, start=start), self.assertRaises(ValueError):
                DFlashDevice(Mock(), model, Mock(), [None] * 5, {}, {}, features,
                    position=4096, feature_start=start)

    def model(self):
        model = SimpleNamespace(layers=[SimpleNamespace(forward=lambda value: value) for index in range(4)])
        def chunk(token_buf, valid_len, chunk_start, page_table, bucket, **kwargs):
            for layer in model.layers:
                token_buf = layer.forward(token_buf)
            return token_buf
        model._forward_prefill_chunk_masked_tp = chunk
        return model

    def host_rows(self, position):
        return (torch.arange(position).reshape(1, 1, -1, 1) % 97).expand(1, 1, -1, 2560).bfloat16()

    def storage_addresses(self):
        return patch('dflash_prefill_window.addresses',
                     side_effect=lambda operations, tensor: (tensor.untyped_storage().data_ptr(),) * 2)

    def test_native_chunks_stitch_only_the_valid_tail_after_sources_are_reused(self):
        for position in (170, 2048, 2049, 4093, 4096, 6144):
            operations, model, checks = self.operations(), self.model(), []
            original = model._forward_prefill_chunk_masked_tp
            capture = PrefillWindowCapture(operations, model, position, (1, 3), checks=checks)
            host = (torch.arange(position).reshape(1, 1, -1, 1) % 97).expand(1, 1, -1, 2560).bfloat16()
            with patch('dflash_prefill_window.addresses', side_effect=lambda operations, tensor: (tensor.untyped_storage().data_ptr(),) * 2):
                with capture.capture():
                    for start in range(0, position, 2048):
                        valid = min(2048, position - start)
                        bucket = ((valid + 31) // 32) * 32
                        value = torch.full((1, 1, bucket, 2560), -500, dtype=torch.bfloat16)
                        value[..., :valid, :] = host[..., start:start + valid, :]
                        model._forward_prefill_chunk_masked_tp(value, valid, start, None, bucket)
                        value.zero_()
                pieces = validate_prefill_chunks(position, capture.chunks)
                self.assertEqual(len(checks), 4 * len(pieces))
                self.assertEqual(checks, [dict(chip=chip, **piece, exact=True)
                    for piece in pieces for tap in range(2) for chip in range(2)])
                for output in capture.outputs():
                    self.assertTrue(torch.equal(output, host[..., -2048:, :]))
                self.assertIs(capture.outputs(), capture.outputs())
                capture.close()
                calls = operations.deallocate.call_count
                capture.close()
                self.assertEqual(operations.deallocate.call_count, calls)
            self.assertIs(model._forward_prefill_chunk_masked_tp, original)
            self.assertFalse(hasattr(model, '_qwen_dflash_prefill_capture'))

    def test_missing_replayed_or_out_of_order_chunks_restore_hooks_and_fail(self):
        for starts in ((), (0,), (2048,), (0, 0)):
            operations, model = self.operations(), self.model()
            original = model._forward_prefill_chunk_masked_tp
            capture = PrefillWindowCapture(operations, model, 4096, (1, 3))
            with self.subTest(starts=starts), self.assertRaises(ValueError):
                with capture.capture():
                    for start in starts:
                        value = torch.ones((1, 1, 2048, 2560), dtype=torch.bfloat16)
                        model._forward_prefill_chunk_masked_tp(value, 2048, start, None, 2048)
            self.assertTrue(capture.closed)
            self.assertIs(model._forward_prefill_chunk_masked_tp, original)
            self.assertFalse(hasattr(model, '_qwen_dflash_prefill_capture'))

    def drive(self, capture, model, host, start, valid):
        """One chunk, the way the plugin delivers it: bucket-padded, then reused."""
        bucket = ((valid + 31) // 32) * 32
        value = torch.full((1, 1, bucket, 2560), -500, dtype=torch.bfloat16)
        value[..., :valid, :] = host[..., start:start + valid, :]
        model._forward_prefill_chunk_masked_tp(value, valid, start, None, bucket)
        value.zero_()

    def test_a_prompt_captured_across_segments_matches_the_one_shot_result(self):
        """Lever N suspends a prefill between chunks so a decode round can run, so one
        prompt's chunks arrive across several execute_model calls. The stitched draft
        tail must be identical to the unsuspended capture's."""
        for position in (4096, 4093, 6144, 65504 % 8192 + 4096):
            with self.subTest(position=position):
                model, host = self.model(), self.host_rows(position)
                one_shot = PrefillWindowCapture(self.operations(), self.model(), position, (1, 3))
                spanning = PrefillWindowCapture(self.operations(), model, position, (1, 3))
                with self.storage_addresses():
                    reference_model = one_shot.model
                    with one_shot.capture():
                        for start in range(0, position, 2048):
                            self.drive(one_shot, reference_model, host, start, min(2048, position - start))
                    # The same chunks, one segment each.
                    for start in range(0, position, 2048):
                        with spanning.segment():
                            self.drive(spanning, model, host, start, min(2048, position - start))
                self.assertTrue(spanning.complete)
                self.assertEqual(spanning.segments, len(range(0, position, 2048)))
                for left, right in zip(spanning.outputs(), one_shot.outputs()):
                    self.assertTrue(torch.equal(left, right))

    def test_the_draft_tail_may_straddle_two_segments(self):
        """The case that makes 'do it on the last chunk' wrong: at a position just past a
        chunk boundary the 2048-row window is split across two calls, so both must
        contribute or the stitched tail is short."""
        position = 4096 + 32
        model, host = self.model(), self.host_rows(position)
        capture = PrefillWindowCapture(self.operations(), model, position, (1, 3))
        with self.storage_addresses():
            for start in range(0, position, 2048):
                with capture.segment():
                    self.drive(capture, model, host, start, min(2048, position - start))
        retained = [chunk['retained_rows'] for chunk in capture.chunks]
        self.assertEqual(sum(retained), 2048)
        self.assertEqual(retained, [0, 2016, 32])          # both of the last two contribute
        for output in capture.outputs():
            self.assertTrue(torch.equal(output, host[..., -2048:, :]))

    def test_the_cursor_and_its_refusals_span_the_suspension(self):
        model, host = self.model(), self.host_rows(4096)
        capture = PrefillWindowCapture(self.operations(), model, 4096, (1, 3))
        with self.storage_addresses():
            with capture.segment():
                self.drive(capture, model, host, 0, 2048)
            self.assertEqual(capture.cursor, 2048)
            self.assertFalse(capture.complete)             # NOT complete at the first exit
            with self.assertRaises(ValueError):            # replayed chunk, across the gap
                with capture.segment():
                    self.drive(capture, model, host, 0, 2048)
        self.assertTrue(capture.closed)

    def test_the_model_carries_no_capture_attributes_while_suspended(self):
        """What keeps the one-non-nested-capture guard meaningful mid-prefill: between
        segments the model is clean, so a second capture would be refused on entry but
        nothing is left installed to confuse an unrelated decode step."""
        model, host = self.model(), self.host_rows(4096)
        original = model._forward_prefill_chunk_masked_tp
        capture = PrefillWindowCapture(self.operations(), model, 4096, (1, 3))
        with self.storage_addresses():
            with capture.segment():
                self.drive(capture, model, host, 0, 2048)
                self.assertTrue(hasattr(model, '_qwen_dflash_prefill_capture'))
            self.assertFalse(hasattr(model, '_qwen_dflash_prefill_capture'))
            self.assertIs(model._forward_prefill_chunk_masked_tp, original)
            with capture.segment():
                self.drive(capture, model, host, 2048, 2048)
        self.assertTrue(capture.complete)

    def test_a_finished_capture_refuses_another_segment(self):
        model, host = self.model(), self.host_rows(2048)
        capture = PrefillWindowCapture(self.operations(), model, 2048, (1, 3))
        with self.storage_addresses():
            with capture.segment():
                self.drive(capture, model, host, 0, 2048)
            self.assertTrue(capture.complete)
            with self.assertRaisesRegex(ValueError, 'already covered its prompt'):
                with capture.segment():
                    pass

    def test_one_shot_capture_still_refuses_a_second_entry(self):
        model, host = self.model(), self.host_rows(2048)
        capture = PrefillWindowCapture(self.operations(), model, 2048, (1, 3))
        with self.storage_addresses():
            with capture.capture():
                self.drive(capture, model, host, 0, 2048)
            with self.assertRaisesRegex(ValueError, 'One non-nested native prefill capture required'):
                with capture.capture():
                    pass

    def test_straddling_window_coordinates_are_absolute_not_chunk_local(self):
        self.assertEqual(chunk_window(4093, 0, 2048), dict(start=2045, end=2048, rows=3))
        self.assertEqual(chunk_window(4093, 2048, 2045), dict(start=2048, end=4093, rows=2045))
        self.assertEqual(chunk_window(4096, 0, 2048)['rows'], 0)
        for start, valid in ((True, 2048), (0, True), (2048, 2048), (-1, 2048)):
            with self.assertRaises(ValueError):
                chunk_window(4093, start, valid)

    def storage_addresses(self):
        # Snapshots must own storage distinct from the layer output they were cut from.
        return patch('dflash_prefill_window.addresses',
                     side_effect=lambda operations, tensor: (tensor.untyped_storage().data_ptr(),) * 2)

    def batched_model(self):
        # The plugin's batched entry: prefill_paged_slots runs the chunks and writes the
        # result into the user's decode slot (empty_slots) on the way out.
        model, calls = self.model(), []

        def prefill_paged_slots(token_ids_list, page_table, empty_slots, valid_lens=None):
            calls.append((token_ids_list, page_table, empty_slots, valid_lens))
            value = torch.ones((1, 1, 2048, 2560), dtype=torch.bfloat16)
            model._forward_prefill_chunk_masked_tp(value, 2048, 0, page_table, 2048)
            return 'logits'

        model.prefill_paged_slots = prefill_paged_slots
        return model, calls

    def test_a_continuation_may_re_enter_the_batched_entry_with_the_same_slot(self):
        """The relaxation step 4 promised in a comment and did not make.

        A resumed prompt re-enters prefill_paged_slots_range on every chunk. The guard
        said one CALL per capture, so run 35688313093 - the first run ever to reach
        prefill_paged_slots_range, with starts=[1992] - was refused on its continuation
        with 'One batched prefill per capture required'. Three 2048-token chunks of a
        6144-token prompt, so the capture is still open for the second and third.

        Driven through the entries the plugin really calls (plugin_model): the first
        chunk through prefill_paged_slots, each resumed one through
        prefill_paged_slots_range at the capture's cursor. This used to re-enter the
        one-shot entry with a stub that advanced by itself, which the capture now
        refuses before the native call (check_resumed_call) - the plugin never sends a
        nonzero start there.
        """
        operations, (model, calls) = self.operations(), self.plugin_model()
        capture = PrefillWindowCapture(operations, model, 6144, (1, 3))
        with self.storage_addresses():
            for index in range(3):
                with capture.segment():
                    self.plugin_chunk(model, index, 1)
        self.assertEqual(capture.prefill_slot, 1)
        self.assertEqual(len(calls), 3, 'every chunk reaches the native entry')
        self.assertEqual([entry for entry, *_ in calls], ['one-shot', 'range', 'range'])
        self.assertEqual(capture.segments, 3)
        self.assertTrue(capture.complete)
        capture.close()

    def plugin_model(self):
        """The two entries the plugin's _prefill_forward_tp_batched really calls
        (qwen36_vllm.py:303-319): prefill_paged_slots for a prompt's first chunk, and
        prefill_paged_slots_range(token_ids_list, pt, empty_slots, starts, plens,
        valid_lens=plens) for every chunk with a nonzero start, where starts is the
        absolute offset and plens the absolute end."""
        model, calls = self.model(), []

        def run(start, end):
            value = torch.ones((1, 1, 2048, 2560), dtype=torch.bfloat16)
            model._forward_prefill_chunk_masked_tp(value, end - start, start, None, 2048)

        def prefill_paged_slots(token_ids_list, page_table, empty_slots, valid_lens=None):
            calls.append(('one-shot', list(empty_slots), 0, valid_lens[0]))
            run(0, valid_lens[0])
            return 'logits'

        def prefill_paged_slots_range(token_ids_list, page_table, empty_slots, starts, ends,
                                      valid_lens=None):
            calls.append(('range', list(empty_slots), starts[0], ends[0]))
            run(starts[0], ends[0])
            return 'logits'

        model.prefill_paged_slots = prefill_paged_slots
        model.prefill_paged_slots_range = prefill_paged_slots_range
        return model, calls

    def plugin_chunk(self, model, index, slot, start=None):
        """Chunk `index` the way the plugin sends it, into plugin slot `slot`."""
        start = 2048 * index if start is None else start
        if start == 0:
            return model.prefill_paged_slots('tokens', 'pages', [slot], valid_lens=[2048])
        return model.prefill_paged_slots_range('tokens', 'pages', [slot], [start], [start + 2048],
                                               valid_lens=[start + 2048])

    def plugin_sequence(self, slots):
        """A prompt of len(slots) 2048-token chunks, one segment (engine step) each, the
        plugin choosing slots[i] for chunk i. Returns the capture, the native calls, each
        segment's recorded slot, and the moved-slot markers logged."""
        operations, (model, calls) = self.operations(), self.plugin_model()
        capture = PrefillWindowCapture(operations, model, 2048 * len(slots), (1, 3))
        segment_slots = []
        with self.storage_addresses(), patch('dflash_prefill_window._log') as log:
            for index, slot in enumerate(slots):
                with capture.segment():
                    self.plugin_chunk(model, index, slot)
                segment_slots.append(getattr(capture, 'segment_slot', None))
        moved = [entry.args for entry in log.call_args_list if 'prefill slot moved' in entry.args[0]]
        return capture, calls, segment_slots, moved

    def test_a_resumed_prompt_follows_its_slot_when_the_plugin_moves_it(self):
        """Run v121, user 3 at 4 x 131k: chunks 1-2 into plugin slot 2 while users 1 and
        2 held 0 and 1; user 1 finished; chunk 3 was handed [0], because the plugin
        re-allocates a prefill's slot on every prompt step and a lone prefill is row 0.
        The capture refused it ('One prefill slot per capture required; 2 then 0') and
        the engine died with three users on it.

        That refusal encoded a wrong assumption - a different slot meant another prompt
        had taken the capture over. It did not: it is the same prompt at the next
        cursor. The prompt completes, and the capture records the LAST slot written,
        which is the one holding the finished state and the one admission adopts."""
        capture, calls, segment_slots, moved = self.plugin_sequence([2, 2, 0, 0])
        self.assertTrue(capture.complete)
        self.assertFalse(capture.closed)
        self.assertEqual(capture.prefill_slot, 0, 'the final chunk landed in slot 0')
        self.assertEqual(segment_slots, [2, 2, 0, 0])
        self.assertEqual(calls, [('one-shot', [2], 0, 2048), ('range', [2], 2048, 4096),
                                 ('range', [0], 4096, 6144), ('range', [0], 6144, 8192)],
                         'every chunk reaches the native entry with the slot the plugin chose')
        self.assertEqual(len(moved), 1, 'one marker, at the move and not after it')
        message, old, new, cursor, position, segment = moved[0]
        self.assertTrue(message.startswith('[PINDIAG] prefill slot moved: {} -> {}'))
        self.assertEqual((old, new, cursor, position, segment), (2, 0, 4096, 8192, 3))
        self.assertEqual(validate_prefill_chunks(8192, capture.chunks)[0]['start'], 6144)
        capture.close()

    def test_the_last_slot_wins_whichever_way_the_slot_moves(self):
        """Admission adopts capture.prefill_slot, so it must be the final chunk's slot:
        keeping the first would copy a stale snapshot over the finished state."""
        for slots, last in (([1, 3, 3], 3), ([0, 2], 2), ([2, 0, 1], 1)):
            with self.subTest(slots=slots):
                capture, _, segment_slots, moved = self.plugin_sequence(slots)
                self.assertTrue(capture.complete)
                self.assertEqual(capture.prefill_slot, last)
                self.assertEqual(segment_slots, slots)
                self.assertEqual(len(moved), sum(a != b for a, b in zip(slots, slots[1:])))
                capture.close()

    def test_a_slot_that_never_moves_behaves_exactly_as_before(self):
        """No move, no marker, and the recorded slot is the one slot used throughout -
        the shape v118 (4 x 32k, slots 0..3 stable) ran token-exact."""
        for slot in (0, 1, 3):
            with self.subTest(slot=slot):
                capture, calls, segment_slots, moved = self.plugin_sequence([slot] * 4)
                self.assertTrue(capture.complete)
                self.assertEqual(capture.prefill_slot, slot)
                self.assertEqual(segment_slots, [slot] * 4)
                self.assertEqual([call[1] for call in calls], [[slot]] * 4)
                self.assertEqual(moved, [])
                capture.close()

    def test_a_moved_slot_is_refused_unless_it_resumes_this_prompt_at_its_cursor(self):
        """What the relaxed guard still refuses, before any device work: a moved slot on
        a call that does not continue this prompt from exactly where it stopped. A fresh
        prompt (start 0, or the one-shot entry), a replayed chunk and a skipped chunk
        are all foreign to the capture's cursor."""
        cases = (
            ('fresh prompt through the range entry', dict(start=0)),
            ('fresh prompt through the one-shot entry', dict(start=0, one_shot=True)),
            # The entry check alone: the one-shot entry's first positional argument
            # (valid_lens) happens to equal the cursor, which a start-only check would
            # take for a resumed start.
            ('one-shot entry naming the cursor positionally', dict(start=4096, one_shot=True)),
            ('replayed chunk', dict(start=2048)),
            ('skipped chunk', dict(start=6144)),
        )
        for label, case in cases:
            with self.subTest(label):
                operations, (model, calls) = self.operations(), self.plugin_model()
                original = model.prefill_paged_slots_range
                capture = PrefillWindowCapture(operations, model, 8192, (1, 3))
                with self.storage_addresses(), patch('dflash_prefill_window._log') as log:
                    for index in range(2):
                        with capture.segment():
                            self.plugin_chunk(model, index, 2)
                    self.assertEqual(capture.cursor, 4096)
                    with self.assertRaisesRegex(ValueError, 'may move only on a resumed chunk'):
                        with capture.segment():
                            if case.get('one_shot') and case['start']:
                                model.prefill_paged_slots('tokens', 'pages', [0], [case['start']])
                            elif case.get('one_shot'):
                                model.prefill_paged_slots('tokens', 'pages', [0], valid_lens=[2048])
                            else:
                                start = case['start']
                                model.prefill_paged_slots_range('tokens', 'pages', [0], [start],
                                    [start + 2048], valid_lens=[start + 2048])
                self.assertEqual(len(calls), 2, 'refused before the native prefill ran')
                self.assertEqual(capture.prefill_slot, 2, 'the refused slot is not recorded')
                self.assertTrue(capture.closed)
                self.assertIs(model.prefill_paged_slots_range, original)
                log.assert_not_called()

    def test_a_foreign_call_keeping_the_slot_is_refused_before_the_native_prefill(self):
        """With the slot unchanged, a call that does not resume this prompt used to be
        refused only by the chunk ledger in wrap(), INSIDE the native call - after
        prefill_paged_slots had bound and, at start 0, reset the persistent GDN scratch.
        Loud either way, but asymmetric with the moved-slot path. Once the cursor has
        moved, every call must be the resumable entry at the cursor whatever slot it
        names, so each of these is now refused before any device work."""
        cases = (
            ('fresh prompt through the one-shot entry', dict(start=0, one_shot=True)),
            ('one-shot entry naming the cursor positionally', dict(start=4096, one_shot=True)),
            ('fresh prompt through the range entry', dict(start=0)),
            ('replayed chunk', dict(start=2048)),
            ('skipped chunk', dict(start=6144)),
        )
        for label, case in cases:
            with self.subTest(label):
                operations, (model, calls) = self.operations(), self.plugin_model()
                capture = PrefillWindowCapture(operations, model, 8192, (1, 3))
                with self.storage_addresses():
                    for index in range(2):
                        with capture.segment():
                            self.plugin_chunk(model, index, 2)
                    with self.assertRaisesRegex(ValueError, 'must resume this capture at its cursor'):
                        with capture.segment():
                            if case.get('one_shot') and case['start']:
                                model.prefill_paged_slots('tokens', 'pages', [2], [case['start']])
                            elif case.get('one_shot'):
                                model.prefill_paged_slots('tokens', 'pages', [2], valid_lens=[2048])
                            else:
                                start = case['start']
                                model.prefill_paged_slots_range('tokens', 'pages', [2], [start],
                                    [start + 2048], valid_lens=[start + 2048])
                self.assertEqual(len(calls), 2, 'refused before the native prefill ran')
                self.assertEqual((capture.cursor, capture.prefill_slot), (4096, 2))
                self.assertTrue(capture.closed)

    def test_a_moved_slot_before_any_chunk_ran_is_refused(self):
        """The cursor clause on its own. A batched call that advanced no chunk leaves the
        cursor at 0 with a slot recorded; a later call in another slot naming start 0
        matches that cursor, but it resumes nothing, so it is refused."""
        operations, model, calls = self.operations(), self.model(), []

        def prefill_paged_slots(token_ids_list, page_table, empty_slots, valid_lens=None):
            calls.append(list(empty_slots))
            return 'logits'

        def prefill_paged_slots_range(token_ids_list, page_table, empty_slots, starts, ends,
                                      valid_lens=None):
            calls.append(list(empty_slots))
            return 'logits'

        model.prefill_paged_slots = prefill_paged_slots
        model.prefill_paged_slots_range = prefill_paged_slots_range
        capture = PrefillWindowCapture(operations, model, 4096, (1, 3))
        with self.storage_addresses():
            with capture.segment():
                model.prefill_paged_slots('tokens', 'pages', [2], valid_lens=[2048])
            self.assertEqual((capture.cursor, capture.prefill_slot), (0, 2))
            with self.assertRaisesRegex(ValueError, 'may move only on a resumed chunk'):
                with capture.segment():
                    model.prefill_paged_slots_range('tokens', 'pages', [0], [0], [2048],
                                                    valid_lens=[2048])
        self.assertEqual(calls, [[2]])
        self.assertTrue(capture.closed)

    def test_a_move_inside_one_segment_is_still_one_call_too_many(self):
        """Relaxing the slot did not relax one batched call per segment."""
        operations, (model, calls) = self.operations(), self.plugin_model()
        capture = PrefillWindowCapture(operations, model, 8192, (1, 3))
        with self.storage_addresses(), patch('dflash_prefill_window._log'):
            with self.assertRaisesRegex(ValueError, 'One batched prefill per capture required'):
                with capture.segment():
                    self.plugin_chunk(model, 0, 2)
                    self.plugin_chunk(model, 1, 0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(capture.closed)

    def test_each_segment_reports_only_its_own_slot(self):
        """segment_slot is what the lifecycle reads after a continuation to decide
        whether the chunk wrote slot 0; a segment with no batched call reports None,
        never the previous segment's slot."""
        operations, (model, _) = self.operations(), self.plugin_model()
        capture = PrefillWindowCapture(operations, model, 6144, (1, 3))
        self.assertIsNone(capture.segment_slot)
        with self.storage_addresses():
            with capture.segment():
                self.plugin_chunk(model, 0, 2)
            self.assertEqual(capture.segment_slot, 2)
            with capture.segment():
                self.assertIsNone(capture.segment_slot, 'reset on entry')
                value = torch.ones((1, 1, 2048, 2560), dtype=torch.bfloat16)
                model._forward_prefill_chunk_masked_tp(value, 2048, 2048, None, 2048)
            self.assertIsNone(capture.segment_slot)
            self.assertEqual(capture.prefill_slot, 2)
            with capture.segment():
                self.plugin_chunk(model, 2, 2)
            self.assertEqual(capture.segment_slot, 2)
        self.assertTrue(capture.complete)
        capture.close()

    def test_a_prompt_finished_without_a_batched_call_is_not_left_adoptable(self):
        """prefill_slot carries over between segments while segment_slot does not, and
        admission adopts prefill_slot as the FINAL chunk's slot. A final segment that
        reached position without a batched call would hand admission an earlier chunk's
        partial snapshot, silently - so completion refuses it. (The plugin always
        prefills through a batched entry on TP; this is the broken-contract case.)"""
        operations, (model, calls) = self.operations(), self.plugin_model()
        capture = PrefillWindowCapture(operations, model, 4096, (1, 3))
        with self.storage_addresses():
            with capture.segment():
                self.plugin_chunk(model, 0, 2)
            with self.assertRaisesRegex(ValueError, 'final prefill segment made no batched call'):
                with capture.segment():
                    value = torch.ones((1, 1, 2048, 2560), dtype=torch.bfloat16)
                    model._forward_prefill_chunk_masked_tp(value, 2048, 2048, None, 2048)
        self.assertFalse(capture.complete)
        self.assertTrue(capture.closed)
        self.assertEqual(len(calls), 1)

    def test_admission_adopts_the_slot_the_final_chunk_wrote(self):
        """Run v121's user 3: the plugin wrote chunks 1-2 into slot 2 and, after user 1
        finished, the rest into slot 0. Every chunk writes a complete snapshot of the
        prefill scratch, so only the final chunk's slot holds the finished state. A real
        capture driven through that sequence must hand admission the final slot: 0 is
        'nothing to adopt' (the finished state already sits in the working row), and a
        final nonzero slot is copied from there, never from the stale earlier one.

        Kept HERE rather than in test_serving_request_factory: that suite also runs
        inside the serving image build (qwen-fast-serving.Dockerfile), where this module
        is not overlaid and arrives as the frozen bundle's copy, which has no
        plugin_sequence - importing it from there broke the build."""
        from serving_request_factory import adopt_prefill_slot

        for slots, adopted in (([2, 2, 0, 0], None), ([2, 0, 1], 1), ([1, 3, 3], 3)):
            with self.subTest(slots=slots):
                capture = self.plugin_sequence(slots)[0]
                self.assertTrue(capture.complete)
                helpers = [Mock(spec=['adopt_slot'], **{'adopt_slot.return_value': 2})
                           for _ in range(48)]
                with patch('serving_request_factory._log') as log:
                    adopt_prefill_slot(helpers, capture, 'cmpl-ae24a631')
                if adopted is None:
                    self.assertEqual([helper.adopt_slot.call_count for helper in helpers], [0] * 48)
                    log.assert_called_once_with('[PINDIAG] prefill slot {}, nothing to adopt for request {}',
                                                0, 'cmpl-ae24a631')
                else:
                    self.assertEqual([helper.adopt_slot.call_args_list for helper in helpers],
                                     [[call(adopted, layer=layer)] for layer in range(48)])
                capture.close()

    def test_the_lifecycle_displaces_on_the_slot_a_real_capture_reports(self):
        """The seam between the two halves of the v121 fix, which the lifecycle suite
        covers only with a stand-in capture: _displace_after_continuation reads
        segment_slot off the real PrefillWindowCapture after each continuation. Through
        v121's sequence it displaces the resident engine after the chunks that wrote
        slot 0, and after no other."""
        from serving_lifecycle import FastServingLifecycle

        operations, (model, _) = self.operations(), self.plugin_model()
        capture = PrefillWindowCapture(operations, model, 8192, (1, 3))
        lifecycle = SimpleNamespace(capture=capture, request_id='cmpl-ae24a631')
        displaced = []
        with self.storage_addresses(), patch('dflash_prefill_window._log'), \
                patch('serving_lifecycle.note_prefill',
                      side_effect=lambda: displaced.append(capture.segments)):
            for index, slot in enumerate([2, 2, 0, 0]):
                with capture.segment():
                    self.plugin_chunk(model, index, slot)
                if index:              # continuations only; admission displaces itself
                    FastServingLifecycle._displace_after_continuation(lifecycle)
        self.assertEqual(displaced, [3, 4])
        self.assertTrue(capture.complete)
        capture.close()

    def test_more_than_one_slot_in_a_call_is_still_refused(self):
        """prefill_slot, not the call counter, is what catches two prompts batched into
        one step - which is why relaxing the counter costs nothing real."""
        operations, (model, _) = self.operations(), self.batched_model()
        capture = PrefillWindowCapture(operations, model, 6144, (1, 3))
        with self.storage_addresses():
            with self.assertRaisesRegex(ValueError, 'exactly one user per call'):
                with capture.segment():
                    model.prefill_paged_slots('tokens', 'pages', [0, 1], valid_lens=[2048])
        capture.close()

    def test_batched_prefill_records_its_one_slot_and_restores_the_hook(self):
        # vLLM hands the request's decode slot as a one-element list - [0] for the first
        # user, [1] for the second - and a tensor or tuple element records as a plain int.
        for slots, expected in (([0], 0), ([1], 1), ((3,), 3), (torch.tensor([2]), 2)):
            operations, (model, calls) = self.operations(), self.batched_model()
            original = model.prefill_paged_slots
            capture = PrefillWindowCapture(operations, model, 2048, (1, 3))
            self.assertIsNone(capture.prefill_slot)
            with self.subTest(slots=slots), self.storage_addresses():
                with capture.capture():
                    self.assertEqual(model.prefill_paged_slots('tokens', 'pages', slots, valid_lens=[2048]), 'logits')
                self.assertEqual(capture.prefill_slot, expected)
                self.assertIs(type(capture.prefill_slot), int)
                self.assertEqual(len(calls), 1)
                self.assertIs(calls[0][2], slots, 'the native prefill sees its own arguments')
                self.assertEqual(calls[0][3], [2048])
                self.assertTrue(capture.complete)
            self.assertIs(model.prefill_paged_slots, original)
            capture.close()

    def test_single_sequence_prefill_leaves_the_slot_unrecorded(self):
        # No batched entry point on the model: nothing is wrapped and nothing recorded.
        operations, model = self.operations(), self.model()
        capture = PrefillWindowCapture(operations, model, 2048, (1, 3))
        with self.storage_addresses(), capture.capture():
            self.assertFalse(hasattr(model, 'prefill_paged_slots'))
            model._forward_prefill_chunk_masked_tp(torch.ones((1, 1, 2048, 2560), dtype=torch.bfloat16), 2048, 0, None, 2048)
        self.assertIsNone(capture.prefill_slot)
        capture.close()
        # The entry point exists but the single-sequence prefill_traced_chunked path ran.
        operations, (model, calls) = self.operations(), self.batched_model()
        capture = PrefillWindowCapture(operations, model, 2048, (1, 3))
        with self.storage_addresses(), capture.capture():
            model._forward_prefill_chunk_masked_tp(torch.ones((1, 1, 2048, 2560), dtype=torch.bfloat16), 2048, 0, None, 2048)
        self.assertIsNone(capture.prefill_slot)
        self.assertEqual(calls, [])
        capture.close()

    def test_more_or_fewer_than_one_user_per_prefill_is_refused(self):
        # The lifecycle admits one fresh prompt per prefill; anything else is a broken
        # contract, refused before the native prefill runs, with the hooks restored.
        for slots in ([], [0, 1], [True], ['1'], [None], [-1], 5, None):
            operations, (model, calls) = self.operations(), self.batched_model()
            original = model.prefill_paged_slots
            capture = PrefillWindowCapture(operations, model, 2048, (1, 3))
            with self.subTest(slots=slots), self.assertRaises(ValueError):
                with capture.capture():
                    model.prefill_paged_slots('tokens', 'pages', slots)
            self.assertIsNone(capture.prefill_slot)
            self.assertEqual(calls, [])
            self.assertTrue(capture.closed)
            self.assertIs(model.prefill_paged_slots, original)
            self.assertFalse(hasattr(model, '_qwen_dflash_prefill_capture'))

    def test_every_batched_prefill_entry_point_is_wrapped_not_just_the_known_one(self):
        """The hazard this closes: M1 adds prefill_paged_slots_range, the capture bound
        only prefill_paged_slots, so a resumed prompt left prefill_slot None - and None
        is a LEGITIMATE value meaning 'single-sequence path, nothing to adopt', so
        adopt_prefill_slot logged and returned and the next user decoded from this
        one's recurrent state. Wrong output, no error. The capture now wraps every
        entry point the model exposes."""
        model = self.model()
        calls = []

        def paged_slots_range(token_ids_list, page_table, empty_slots, starts, ends, valid_lens=None):
            calls.append(('range', list(empty_slots)))
            return None

        model.prefill_paged_slots_range = paged_slots_range
        capture = PrefillWindowCapture(self.operations(), model, 64, (1, 3))
        with patch('dflash_prefill_window.addresses',
                   side_effect=lambda operations, tensor: (tensor.untyped_storage().data_ptr(),) * 2):
            with capture.capture():
                model.prefill_paged_slots_range(['t'], 'pt', [3], [0], [64])
                value = torch.full((1, 1, 64, 2560), 0, dtype=torch.bfloat16)
                model._forward_prefill_chunk_masked_tp(value, 64, 0, None, 64)
        self.assertEqual(capture.prefill_slot, 3)
        self.assertEqual(calls, [('range', [3])])

    def test_an_unknown_batched_entry_point_is_refused_loudly_at_capture_time(self):
        """A method added later must fail here, where it is obvious, rather than there,
        where it is silent. The refusal is at capture entry, before any device work."""
        model = self.model()
        model.prefill_paged_slots_elsewhere = lambda *a, **k: None
        capture = PrefillWindowCapture(self.operations(), model, 64, (1, 3))
        with self.assertRaisesRegex(ValueError, 'Unrecognised batched prefill entry point'):
            with capture.capture():
                pass

    def test_the_allowlist_names_exactly_the_two_known_entry_points(self):
        self.assertEqual(set(BATCHED_PREFILL_ENTRIES),
                         {'prefill_paged_slots', 'prefill_paged_slots_range'})

    def test_a_model_with_no_batched_entry_point_still_captures(self):
        """The single-sequence path through prefill_traced_chunked, whose state already
        lands where the fast path reads it. prefill_slot stays None and that is correct."""
        model = self.model()
        # The fixture model has no batched entry point to begin with, which IS the
        # single-sequence shape; drop one only if a future fixture adds it.
        if hasattr(model, 'prefill_paged_slots'):
            del model.prefill_paged_slots
        capture = PrefillWindowCapture(self.operations(), model, 64, (1, 3))
        with patch('dflash_prefill_window.addresses',
                   side_effect=lambda operations, tensor: (tensor.untyped_storage().data_ptr(),) * 2):
            with capture.capture():
                value = torch.full((1, 1, 64, 2560), 0, dtype=torch.bfloat16)
                model._forward_prefill_chunk_masked_tp(value, 64, 0, None, 64)
        self.assertIsNone(capture.prefill_slot)

    def test_a_second_batched_prefill_in_one_capture_is_refused(self):
        operations, (model, calls) = self.operations(), self.batched_model()
        capture = PrefillWindowCapture(operations, model, 2048, (1, 3))
        with self.assertRaises(ValueError), self.storage_addresses():
            with capture.capture():
                model.prefill_paged_slots('tokens', 'pages', [1])
                model.prefill_paged_slots('tokens', 'pages', [1])
        self.assertEqual(len(calls), 1)
        self.assertTrue(capture.closed)


if __name__ == '__main__':
    unittest.main()
