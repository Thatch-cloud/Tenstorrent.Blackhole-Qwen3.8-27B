import unittest

from dspark_markov_sfpu import geometry


class MarkovSfpuGeometryTests(unittest.TestCase):
    def test_full_rank_and_complete_vocabulary_with_all_proposal_steps(self):
        for vocabulary in (64, 248320):
            for rows in (1, 3, 7, 15):
                for step in range(rows):
                    for workers in (1, 64, 80, 110):
                        self.assertEqual(geometry((1, 1, 1, 256), (1, 1, vocabulary, 256), (1, 1, rows, vocabulary), step, workers),
                            (min(workers, vocabulary // 32), vocabulary // 32, vocabulary))

    def test_incomplete_or_ambiguous_operands_and_worker_ranges_fail(self):
        valid = [(1, 1, 1, 256), (1, 1, 64, 256), (1, 1, 7, 64), 6, 110]
        for index, value in ((0, (1, 1, 7, 256)), (1, (1, 1, 256, 64)), (2, (1, 1, 31, 64)),
                (2, (1, 1, 7, 32)), (3, 7), (3, -1), (3, True), (4, 111), (4, True)):
            arguments = list(valid)
            arguments[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                geometry(*arguments)


if __name__ == '__main__':
    unittest.main()
