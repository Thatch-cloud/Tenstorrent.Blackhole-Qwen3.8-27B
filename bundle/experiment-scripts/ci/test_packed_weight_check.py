from types import SimpleNamespace
import unittest

import torch

from packed_weight_check import comparison_geometry, read_comparison


class PackedWeightCheckTests(unittest.TestCase):
    def test_native_rank_two_weights_have_identical_tile_geometry(self):
        for packed in ((5120, 17408), (1, 1, 5120, 17408)):
            for separate in ((5120, 8704), (1, 1, 5120, 8704)):
                for offset in (0, 1):
                    self.assertEqual(comparison_geometry(packed, separate, offset), (64, 272, 43520))
        for shape in ((2, 1, 5120, 8704), (1, 2, 5120, 8704), (5120, 8736)):
            with self.assertRaises(ValueError):
                comparison_geometry((5120, 17408), shape, 0)

    def test_geometry_and_complete_tile_mapping(self):
        for rows, columns in ((32, 32), (64, 96), (320, 256), (5120, 8704)):
            for offset in (0, 1):
                workers, tiles, pages = comparison_geometry((1, 1, rows, 2 * columns), (1, 1, rows, columns), offset)
                visited = [page for worker in range(workers) for page in range(worker, pages, workers)]
                self.assertEqual(sorted(visited), list(range(pages)))
                paired = [(page // tiles) * tiles * 2 + (page % tiles) * 2 + offset for page in visited]
                self.assertEqual(sorted(paired), list(range(offset, pages * 2, 2)))
        for shape, offset in (((1, 1, 31, 32), 0), ((1, 1, 5120, 8736), 0), ((1, 1, 32, 32), True)):
            with self.assertRaises(ValueError):
                comparison_geometry((*shape[:3], shape[-1] * 2), shape, offset)

    def test_coverage_sentinel_and_output_padding_are_required(self):
        pages, workers = 80, 64
        value = torch.zeros((1, 1, workers * 32, 32), dtype=torch.int64)
        tiles = value.reshape(workers, 32, 32)
        tiles[:, 0, 4] = 0x514B5631
        tiles[:, 0, 5] = 0xFFFFFFFF
        for worker in range(workers):
            tiles[worker, 0, 1] = len(range(worker, pages, workers))
        operations = SimpleNamespace(get_device_tensors=lambda result: (result, result), to_torch=lambda result: result)
        checks = read_comparison(operations, value, pages)
        self.assertTrue(all(check['exact'] and check['pages'] == pages for check in checks))
        tiles[0, 0, 0] = 1
        with self.assertRaisesRegex(AssertionError, 'integrity'):
            read_comparison(operations, value, pages)
        tiles[0, 0, 5] = 0xFFFFFFFE
        self.assertTrue(all(not check['exact'] for check in read_comparison(operations, value, pages)))
        tiles[0, 0, 1] = 0
        with self.assertRaisesRegex(AssertionError, 'coverage'):
            read_comparison(operations, value, pages)
        tiles[0, 0, 1] = 2
        tiles[0, 1, 0] = 1
        with self.assertRaisesRegex(AssertionError, 'canary'):
            read_comparison(operations, value, pages)


if __name__ == '__main__':
    unittest.main()
