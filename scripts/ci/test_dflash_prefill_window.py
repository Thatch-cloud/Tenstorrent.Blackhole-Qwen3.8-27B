from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

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
