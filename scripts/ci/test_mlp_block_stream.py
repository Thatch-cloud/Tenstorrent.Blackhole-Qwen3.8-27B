from collections import Counter
from pathlib import Path
import unittest

from mlp_block_stream import BLOCK_BYTES, BLOCK_TILES, TILE_BYTES, geometry, source_pages, reader_source


class BlockStreamTests(unittest.TestCase):
    def test_reader_keeps_output_publication_and_buffer_capacity(self):
        original = Path(__file__).with_name('fused_1d_weights.cpp').read_text()
        changed = reader_source(original)
        output = '    for (uint32_t pair = 0; pair < pairs_per_worker; ++pair) {\n'
        self.assertEqual(original[original.index(output):], changed[changed.index(output):])
        self.assertIn('cb_reserve_back(1, 48)', changed)
        self.assertIn('cb_push_back(1, 48)', changed)
        self.assertIn('block * 91 + first_pair / 3', changed)
        self.assertNotIn('noc_async_read_tile', changed)
        with self.assertRaises(ValueError):
            reader_source(changed)

    def test_full_native_weight_coverage_without_requantization(self):
        shape = geometry()
        counts = Counter(page for block in range(shape['blocks']) for worker in range(shape['workers'])
            for page in source_pages(worker, block) if page is not None)
        self.assertEqual(counts, Counter(range(160 * 544)))
        self.assertEqual(shape['workers'], 91)
        self.assertEqual(shape['padding_tiles'], 320)
        self.assertEqual(shape['stream_bytes'], 50319360)
        self.assertEqual(BLOCK_BYTES, BLOCK_TILES * TILE_BYTES)

    def test_reader_order_matches_every_original_worker_block(self):
        for pairs, blocks in ((1, 1), (3, 2), (8, 2), (272, 20)):
            shape = geometry(pairs, blocks)
            for worker in range(shape['workers']):
                first_pair = worker * 3
                valid_pairs = min(3, pairs - first_pair)
                for block in range(blocks):
                    expected = tuple((block * 8 + inner) * pairs * 2 + first_pair * 2 + column
                        if column < valid_pairs * 2 else None for inner in range(8) for column in range(6))
                    self.assertEqual(source_pages(worker, block, pairs=pairs, blocks=blocks), expected)

    def test_block_major_pages_spread_first_blocks_across_banks(self):
        shape = geometry()
        for banks in (8, 12):
            for block in range(shape['blocks']):
                counts = Counter((block * shape['workers'] + worker) % banks for worker in range(shape['workers']))
                self.assertEqual(len(counts), banks)
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_invalid_geometry_fails_closed(self):
        for pairs, blocks in ((True, 20), (272, False), (0, 20), (273, 20), (272, 21)):
            with self.assertRaises(ValueError):
                geometry(pairs, blocks)
        for worker, block in ((91, 0), (0, 20), (-1, 0), (True, 0)):
            with self.assertRaises(ValueError):
                source_pages(worker, block)


if __name__ == '__main__':
    unittest.main()
