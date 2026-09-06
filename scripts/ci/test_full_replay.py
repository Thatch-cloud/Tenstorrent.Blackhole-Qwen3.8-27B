import unittest

from full_replay import validate_fixture


class FullReplayTests(unittest.TestCase):
    def test_all_replay_shapes_include_both_correction_tokens(self):
        for rows in (2, 16, 32):
            for first in (0, 1, rows):
                for second in (1, rows):
                    validate_fixture(rows, first, second, first + rows + 3)
                    with self.assertRaises(ValueError):
                        validate_fixture(rows, first, second, first + rows + 2)

    def test_rejects_unsupported_or_ambiguous_replay_geometry(self):
        for values in ((64, 1, 1, 100), (2, -1, 1, 100), (2, 1, 0, 100),
                       (16, 8, 1, 100), (16, 1, True, 100), (2.0, 1, 1, 100)):
            with self.assertRaises(ValueError):
                validate_fixture(*values)
