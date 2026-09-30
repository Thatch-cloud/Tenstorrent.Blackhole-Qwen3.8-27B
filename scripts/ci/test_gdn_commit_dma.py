import unittest
from unittest.mock import Mock, patch

from gdn_commit_dma import publish, validate_shapes


class CommitDmaTests(unittest.TestCase):
    def test_eager_publication_executes_prepared_program_once(self):
        operation = Mock()
        with patch('gdn_commit_dma.prepare', return_value=operation) as prepare:
            publish('mesh', 'layers', 8)
        prepare.assert_called_once_with('mesh', 'layers', 8)
        operation.assert_called_once_with()

    def fixture(self, rows=16):
        compact = [(1, 24, 128, 128)] + [(1, 1, 5120)] * 4
        return compact + [(rows, 24, 128, 128)] + [(1, rows, 5120)] * 4 + [(8, 24, 128, 128)] + [(1, 8, 5120)] * 4 + compact

    def test_all_widths_prefixes_and_layer_counts(self):
        for rows in (2, 4, 8, 16, 32):
            for prefix in range(rows + 1):
                for count in (1, 2, 48):
                    self.assertEqual(validate_shapes([self.fixture(rows)] * count, prefix), rows)

    def test_sixty_four_row_histories_are_refused_because_the_kernel_reads_one_tile_row(self):
        """gdn_commit_dma.cpp addresses the convolution history row inside tile row zero
        (offset from token / 16 and token % 16 over pages 0..159), so a history taller
        than 32 rows would silently read the wrong row. M3 never needs it: each packed
        user commits its own 16-row histories (gdn_records.RetainedGDNBlock.segment_layers)."""
        for prefix in (0, 1, 33, 64):
            with self.assertRaises(ValueError):
                validate_shapes([self.fixture(64)] * 48, prefix)
        self.assertEqual(validate_shapes([self.fixture(16)] * 48, 16), 16)

    def test_rejects_invalid_or_mixed_layers_and_prefixes(self):
        for layers, prefix in (([], 0), ([self.fixture()] * 49, 0), ([self.fixture()], True),
                               ([self.fixture()], -1), ([self.fixture()], 17),
                               ([self.fixture(), self.fixture(8)], 0), ([self.fixture()[:-1]], 0)):
            with self.assertRaises(ValueError):
                validate_shapes(layers, prefix)


if __name__ == '__main__':
    unittest.main()
