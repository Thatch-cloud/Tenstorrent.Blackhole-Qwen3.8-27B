import unittest

from gdn_commit_dma import validate_shapes


class CommitDmaTests(unittest.TestCase):
    def fixture(self, rows=16):
        compact = [(1, 24, 128, 128)] + [(1, 1, 5120)] * 4
        return compact + [(rows, 24, 128, 128)] + [(1, rows, 5120)] * 4 + [(8, 24, 128, 128)] + [(1, 8, 5120)] * 4 + compact

    def test_all_widths_prefixes_and_layer_counts(self):
        for rows in (2, 4, 8, 16):
            for prefix in range(rows + 1):
                for count in (1, 2, 48):
                    self.assertEqual(validate_shapes([self.fixture(rows)] * count, prefix), rows)

    def test_rejects_invalid_or_mixed_layers_and_prefixes(self):
        for layers, prefix in (([], 0), ([self.fixture()] * 49, 0), ([self.fixture()], True),
                               ([self.fixture()], -1), ([self.fixture()], 17),
                               ([self.fixture(), self.fixture(8)], 0), ([self.fixture()[:-1]], 0)):
            with self.assertRaises(ValueError):
                validate_shapes(layers, prefix)


if __name__ == '__main__':
    unittest.main()
