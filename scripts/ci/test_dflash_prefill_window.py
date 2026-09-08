from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from dflash_device import DFlashDevice
from dflash_prefill_window import PrefillWindowCapture, chunk_window, prefill_window, snapshot_prefill_tail, validate_prefill_chunks


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


if __name__ == '__main__':
    unittest.main()
