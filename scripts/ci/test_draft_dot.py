import unittest

from draft_dot import dot_geometry, dot_buffer_tiles


class DraftDotTests(unittest.TestCase):
    def test_cache_budget_matches_reduction_width(self):
        self.assertEqual(sum(count for _, count in dot_buffer_tiles(65, False)) * 4096, 28 * 1024)
        self.assertEqual(sum(count for _, count in dot_buffer_tiles(4, True)) * 4096, 48 * 1024)
        self.assertEqual(sum(count for _, count in dot_buffer_tiles(65, True)) * 4096, 536 * 1024)
        for width, cache in ((66, True), (4, 1), (0, False)):
            with self.assertRaises(ValueError):
                dot_buffer_tiles(width, cache)

    def test_bounded_worker_assignment_for_qk_and_pv(self):
        self.assertEqual(dot_geometry((1, 16, 32, 128), (1, 16, 32, 128)), (16, 1, 4))
        self.assertEqual(dot_geometry((1, 16, 32, 128), (1, 16, 2080, 128)), (64, 65, 4))
        self.assertEqual(dot_geometry((1, 16, 32, 2080), (1, 16, 128, 2080)), (64, 4, 65))

    def test_invalid_shapes_rejected(self):
        for right in ((1, 4, 32, 128), (1, 16, 31, 128), (1, 16, 32, 64), (1, 16, 2112, 128)):
            with self.assertRaises(ValueError):
                dot_geometry((1, 16, 32, 128), right)
