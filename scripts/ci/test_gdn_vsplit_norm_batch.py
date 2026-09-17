"""Row-parallel norm layout and unchanged arithmetic source contracts."""

import os
from pathlib import Path
import struct
import unittest

import gdn_vsplit as split
import gdn_vsplit_norm_batch as batch


ROOT = Path(os.environ.get('GDN_VSPLIT_SOURCE_ROOT', str(
    Path(__file__).resolve().parents[2] / 'hardware-evidence.local/34009341359'
    / 'qwen-hardware-inventory-34009341359/gdn-source')))


class LayoutTests(unittest.TestCase):
    def test_fp32_bridge_rows_preserve_every_partition_bit(self):
        for rows in (1, 2, 4, 8, 16, 32):
            tiles = [bytearray(4096) for partition in range(4)]
            for token in range(rows):
                offset = (512 * (token // 16) + 16 * (token % 16)) * 4
                for partition in range(4):
                    words = [0x3f800001 + token * 128 + partition * 32 + column for column in range(32)]
                    stick = struct.pack('<32I', *words)
                    tiles[partition][offset:offset + 64] = stick[:64]
                    tiles[partition][offset + 1024:offset + 1088] = stick[64:]
            for token in range(32):
                for column in range(128):
                    actual = struct.unpack_from('<I', tiles[column // 32], batch.tile_element(token, column % 32) * 4)[0]
                    self.assertEqual(actual, 0x3f800001 + token * 128 + column if token < rows else 0)

    def test_weight_rows_replicate_without_clobbering_source(self):
        tile = bytearray(struct.pack('<1024H', *([0x7fc0] * 1024)))
        for column in range(32):
            struct.pack_into('<H', tile, batch.tile_element(0, column) * 2, 0x3f80 + column)
        for row in range(1, 32):
            offset = (512 * (row // 16) + 16 * (row % 16)) * 2
            tile[offset:offset + 32] = tile[:32]
            tile[offset + 512:offset + 544] = tile[512:544]
        for row in range(32):
            for column in range(32):
                self.assertEqual(struct.unpack_from('<H', tile, batch.tile_element(row, column) * 2)[0], 0x3f80 + column)

    def test_output_zero_padding_preserves_all_active_values(self):
        for rows in (1, 2, 4, 8, 16, 32):
            tile = bytearray(struct.pack('<1024H', *range(1024)))
            for row in range(rows, 32):
                offset = (512 * (row // 16) + 16 * (row % 16)) * 2
                tile[offset:offset + 32] = bytes(32)
                tile[offset + 512:offset + 544] = bytes(32)
            for row in range(32):
                for column in range(32):
                    element = batch.tile_element(row, column)
                    self.assertEqual(struct.unpack_from('<H', tile, element * 2)[0], element if row < rows else 0)

    def test_invalid_coordinates_fail_closed(self):
        for token, column in ((32, 0), (-1, 0), (0, 32), (True, 0)):
            with self.assertRaises(ValueError):
                batch.tile_element(token, column)


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / split.native.KERNEL_ROOT).exists():
            raise unittest.SkipTest('Pinned source unavailable')
        cls.control = split.load_kernels(ROOT)
        cls.candidate = batch.load_kernels(ROOT)

    def test_recurrence_sources_are_identical(self):
        self.assertEqual(self.control['recurrence'], self.candidate['recurrence'])

    def test_only_norm_loop_count_changes_in_compute(self):
        expected = self.control['norm_gate']['compute'].replace(split.LOOP,
            '    for (uint32_t it = 0; it < 1; ++it) {', 1)
        self.assertEqual(self.candidate['norm_gate']['compute'], expected)

    def test_one_full_tile_block_and_weight_ring_handoff(self):
        reader = self.candidate['norm_gate']['reader']
        self.assertEqual(reader.count('pre.reserve_back(4);'), 1)
        self.assertEqual(reader.count('pre.push_back(4);'), 1)
        self.assertGreater(reader.index('pre.push_back(4);'), reader.index(split.READER_LOOP))
        self.assertIn('weights.reserve_back(2 * Vt);', reader)
        self.assertIn('weights.push_back(2 * Vt);', reader)
        self.assertIn('gates.push_back(Vt);', reader)

    def test_writer_waits_before_padding_and_fences_before_pop(self):
        writer = self.candidate['norm_gate']['writer']
        self.assertLess(writer.index('output.wait_front(Vt);'), writer.index('for (uint32_t row = n_inst;'))
        self.assertLess(writer.index('noc.async_write_barrier();'), writer.index('output.pop_front(Vt);'))
        self.assertIn('{.page_id = bh_start * Vt + tile}', writer)
        self.assertNotIn(split.READER_LOOP, writer)

    def test_weight_anchor_drift_is_rejected(self):
        with self.assertRaises(ValueError):
            batch.before_weight_load('missing')
        with self.assertRaises(ValueError):
            batch.before_weight_load('    if constexpr (FNG) {' * 2)


if __name__ == '__main__':
    unittest.main()
