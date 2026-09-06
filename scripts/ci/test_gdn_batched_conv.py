import unittest
from pathlib import Path

from gdn_batched_conv import history_windows


class BatchedConvolutionTests(unittest.TestCase):
    def test_dma_stages_full_tiles_before_rearranging_rows(self):
        source = Path(__file__).with_name('gdn_conv_windows.cpp').read_text()
        self.assertEqual(source.count('noc_async_read_tile('), 5)
        self.assertNotIn('noc_async_read(', source)
        self.assertIn('scratch + 4 * 2048', source)
        self.assertIn('source_tiles + 5 * 2048', source)

    def test_parallel_windows_equal_serial_shift_for_every_prefix(self):
        for rows in (1, 2, 4, 8, 16):
            history = ['entry0', 'entry1', 'entry2', 'entry3'] + list(range(rows))
            windows = [history[start:end] for start, end in history_windows(rows)]
            serial = history[:4]
            for token in range(rows):
                self.assertEqual([window[token] for window in windows], serial)
                serial = serial[1:] + [token]
                shifted = [windows[1][token], windows[2][token], windows[3][token], token]
                self.assertEqual(shifted, serial)

    def test_invalid_widths(self):
        for rows in (0, 3, 32, True, 2.0):
            with self.assertRaises(ValueError):
                history_windows(rows)
