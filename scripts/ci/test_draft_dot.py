import unittest

from draft_dot import dot_geometry, dot_buffer_tiles, dot_core_layout


class DraftDotTests(unittest.TestCase):
    def test_wider_workers_cover_tasks_without_exceeding_grid(self):
        for limit in (64, 80, 110):
            workers, key_tiles, _ = dot_geometry((1, 16, 32, 128), (1, 16, 2080, 128), limit)
            columns, rows, tail = dot_core_layout(workers)
            self.assertEqual(columns * rows + tail, workers)
            self.assertLessEqual(columns, 11)
            self.assertLessEqual(rows + bool(tail), 10)
            tasks = [task for worker in range(workers) for task in range(worker, 16 * key_tiles, workers)]
            self.assertEqual(sorted(tasks), list(range(16 * key_tiles)))
        self.assertEqual(dot_core_layout(96), (11, 8, 8))
        self.assertEqual(dot_geometry((1, 16, 32, 2080), (1, 16, 128, 2080), 110)[0], 64)
        with self.assertRaises(ValueError):
            dot_geometry((1, 16, 32, 128), (1, 16, 2080, 128), 111)

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
