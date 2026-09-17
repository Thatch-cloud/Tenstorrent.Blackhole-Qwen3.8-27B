import unittest

from dspark_attention_8k_gate import require_matrix


class MatrixTests(unittest.TestCase):
    def test_exact_coordinate_matrix(self):
        require_matrix([dict(chip=0, passed=True), dict(chip=1, passed=True)],
            ('chip',), {(0,), (1,)}, 'passed')

    def test_duplicate_missing_and_failed_coordinates_rejected(self):
        for records in ([dict(chip=0, passed=True)] * 2,
                        [dict(chip=0, passed=True)],
                        [dict(chip=0, passed=True), dict(chip=1, passed=False)]):
            with self.assertRaises(ValueError):
                require_matrix(records, ('chip',), {(0,), (1,)}, 'passed')


if __name__ == '__main__':
    unittest.main()
