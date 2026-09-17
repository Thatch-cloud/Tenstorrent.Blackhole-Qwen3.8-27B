import unittest

from dspark_score_layout import geometry


class ScoreLayoutTests(unittest.TestCase):
    def test_full_vocabulary_and_single_worker_cover_every_tile(self):
        for vocabulary in (64, 248320):
            for limit in (1, 64, 110):
                workers, tiles = geometry((1, 1, 15, vocabulary), (1, 1, 1, vocabulary), 14, limit)
                tasks = [tile for worker in range(workers) for tile in range(worker, tiles, workers)]
                self.assertEqual(sorted(tasks), list(range(vocabulary // 32)))

    def test_invalid_geometry_rejected(self):
        for base, bias, step, workers in (
                ((1, 1, 15, 63), (1, 1, 1, 63), 0, 110),
                ((1, 1, 15, 64), (1, 1, 2, 64), 0, 110),
                ((1, 1, 15, 64), (1, 1, 1, 64), 15, 110),
                ((1, 1, 15, 64), (1, 1, 1, 64), True, 110),
                ((1, 1, 15, 64), (1, 1, 1, 64), 0, 111)):
            with self.assertRaises(ValueError):
                geometry(base, bias, step, workers)


if __name__ == '__main__':
    unittest.main()
