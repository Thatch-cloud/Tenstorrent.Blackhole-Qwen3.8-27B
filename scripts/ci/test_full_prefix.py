import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch


spec = importlib.util.spec_from_file_location("full_prefix", Path(__file__).with_name("full-prefix.py"))
full_prefix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full_prefix)


class FullPrefixTests(unittest.TestCase):
    def test_tree_attention_requires_parallel_and_compact_native_scratch(self):
        for flags, message in ((['--attention-tree'], 'requires parallel'),
                (['--attention-tree', '--attention-parallel'], 'process-fixed compact')):
            with patch.dict('os.environ', {'QWEN_HARDWARE_TESTS': '1', 'QWEN_CARDS_ALLOCATED': '1'}, clear=True), patch(
                    'sys.argv', ['full-prefix.py', *flags]):
                with self.assertRaisesRegex(ValueError, message):
                    full_prefix.main()

    def test_attention_engine_requires_matched_request_mode(self):
        with patch.dict('os.environ', {'QWEN_HARDWARE_TESTS': '1', 'QWEN_CARDS_ALLOCATED': '1'}, clear=True), patch(
                'sys.argv', ['full-prefix.py', '--attention-engine']):
            with self.assertRaisesRegex(ValueError, 'matched norm-batch request'):
                full_prefix.main()

    def test_replay_attention_rejects_uncertified_request_modes(self):
        for flag in ('--request-pilot', '--device-selection', '--attribution', '--grouped-attention'):
            with patch.dict('os.environ', {'QWEN_HARDWARE_TESTS': '1', 'QWEN_CARDS_ALLOCATED': '1'}, clear=True), patch(
                'sys.argv', ['full-prefix.py', '--attention-replay', '--norm-batch', '--replay-inputs', flag]):
                with self.assertRaisesRegex(ValueError, 'limited to retained'):
                    full_prefix.main()

    def test_parallel_attention_requires_dma_before_hardware(self):
        with patch.dict('os.environ', {'QWEN_HARDWARE_TESTS': '1', 'QWEN_CARDS_ALLOCATED': '1'}, clear=True), patch(
                'sys.argv', ['full-prefix.py', '--attention-parallel']):
            with self.assertRaisesRegex(ValueError, 'requires DMA'):
                full_prefix.main()

    def test_grouped_attention_rejects_dynamic_modes_before_hardware(self):
        for flag in ('--deferred-commit', '--replay-inputs', '--device-selection', '--request-pilot', '--attribution'):
            with patch.dict('os.environ', {'QWEN_HARDWARE_TESTS': '1', 'QWEN_CARDS_ALLOCATED': '1'}, clear=True), patch(
                'sys.argv', ['full-prefix.py', '--grouped-attention', '--norm-batch', flag]):
                with self.assertRaisesRegex(ValueError, 'limited to static'):
                    full_prefix.main()

    def test_t32_requires_packed_history_control(self):
        flags = dict(packed_checkpoints=True, ordered_cache=True, deferred_commit=False, attribution=False)
        self.assertEqual(full_prefix.verification_widths(16, **flags), (1, 2, 4, 8, 16))
        self.assertEqual(full_prefix.verification_widths(32, **flags), (1, 2, 4, 8, 16, 32))
        flags['deferred_commit'] = True
        self.assertEqual(full_prefix.verification_widths(32, **flags), (1, 2, 4, 8, 16, 32))
        for name in ('packed_checkpoints', 'ordered_cache', 'attribution'):
            invalid = dict(flags)
            invalid[name] = not flags[name]
            with self.assertRaises(ValueError):
                full_prefix.verification_widths(32, **invalid)
        for width in (True, 16.0, 64):
            with self.assertRaises(ValueError):
                full_prefix.verification_widths(width, **flags)

    def test_chunked_long_prefix_covers_all_heads_and_pages(self):
        values = torch.arange(257 * 2 * 64 * 3).reshape(257, 2, 64, 3)
        for length in (4095, 4096, 4097, 16383, 16401):
            chunks = [full_prefix.logical_kv_chunk(values[start:min(start + 64, 257)], start, length)
                      for start in range(0, (length + 63) // 64, 64)]
            actual = torch.cat(chunks, dim=1)
            expected = values.permute(1, 0, 2, 3).reshape(2, -1, 3)[:, :length]
            self.assertTrue(torch.equal(actual, expected))

    def test_long_prefix_masks_only_future_tail(self):
        chunk = torch.zeros(1, 2, 64, 3)
        before = full_prefix.logical_kv_chunk(chunk, 256, 16401).clone()
        chunk[:, :, 17:] = 99
        self.assertTrue(torch.equal(before, full_prefix.logical_kv_chunk(chunk, 256, 16401)))
        chunk[:, :, 16] = 99
        self.assertFalse(torch.equal(before, full_prefix.logical_kv_chunk(chunk, 256, 16401)))
        with self.assertRaises(ValueError):
            full_prefix.logical_kv_chunk(chunk, 257, 16401)

    def test_native_api_padding_is_not_an_active_token(self):
        values = torch.arange(8 * 12).reshape(1, 8, 12)
        expected = values[:, 0].clone()
        self.assertTrue(torch.equal(full_prefix.active_serial_logits(values, 12), expected))
        values[:, 1:] = -999
        self.assertTrue(torch.equal(full_prefix.active_serial_logits(values, 12), expected))
        values[:, 0] += 1
        self.assertFalse(torch.equal(full_prefix.active_serial_logits(values, 12), expected))
        self.assertTrue(torch.equal(full_prefix.active_serial_logits(expected, 12), expected))

    def test_invalid_logit_geometry_is_not_silently_trimmed(self):
        for shape in ((2, 12), (1, 24), (32, 12)):
            with self.assertRaises(ValueError):
                full_prefix.active_serial_logits(torch.zeros(shape), 12)

    def test_native_shape_requires_integer_indexing(self):
        class NativeShape:
            def __len__(self):
                return 4

            def __getitem__(self, index):
                if type(index) is not int:
                    raise TypeError("Native Shape does not accept slices")
                return (8200, 2, 64, 256)[index]

        self.assertEqual(full_prefix.cache_geometry(NativeShape()), (2, 64, 256))

    def test_page_boundary_and_head_order(self):
        values = torch.arange(2 * 3 * 64 * 2).reshape(2, 3, 64, 2)
        for count in (63, 64, 65, 128):
            actual = full_prefix.logical_kv_prefix(values, count)
            for head in range(3):
                expected = torch.cat([values[0, head], values[1, head]])[:count]
                self.assertTrue(torch.equal(actual[head], expected))

    def test_future_page_data_is_excluded(self):
        values = torch.zeros(2, 1, 64, 2)
        before = full_prefix.logical_kv_prefix(values, 65).clone()
        values[1, :, 1:] = 42
        self.assertTrue(torch.equal(before, full_prefix.logical_kv_prefix(values, 65)))
        values[1, :, 0] = 42
        self.assertFalse(torch.equal(before, full_prefix.logical_kv_prefix(values, 65)))

    def test_rejects_invalid_layout_or_length(self):
        for count in (0, 129, True):
            with self.assertRaises(ValueError):
                full_prefix.logical_kv_prefix(torch.zeros(2, 1, 64, 2), count)
        with self.assertRaises(ValueError):
            full_prefix.logical_kv_prefix(torch.zeros(1, 1, 128, 2), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
