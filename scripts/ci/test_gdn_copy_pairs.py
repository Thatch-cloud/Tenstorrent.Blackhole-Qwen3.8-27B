"""Host checks for the scoped, fail-closed recurrence transformation."""

import unittest
from unittest.mock import patch

import gdn_copy_pairs as candidate


class CopyPairsTests(unittest.TestCase):
    def source(self):
        return ('prefix\nvoid copy_tiles(uint32_t in, uint32_t o, uint32_t n) {\n'
                '    cb_reserve_back(o, n);\n' + candidate.ORIGINAL +
                '\n    cb_push_back(o, n);\n}\nsuffix')

    def test_only_helper_loop_changes(self):
        source = self.source()
        self.assertEqual(candidate.transform(source),
                         source.replace(candidate.ORIGINAL, candidate.PAIRED))

    def test_changed_or_duplicate_anchors_fail(self):
        for source in (self.source().replace('copy_tiles(', 'other('),
                       self.source() * 2, self.source().replace('copy_tile(in, i, 0);', 'changed();')):
            with self.assertRaises(ValueError):
                candidate.transform(source)

    def test_norm_and_dataflow_unchanged(self):
        kernels = dict(recurrence=dict(compute=self.source(), reader='reader', writer='writer'),
                       norm_gate=dict(compute='norm', reader='reader', writer='writer'))
        with patch.object(candidate, 'baseline_kernels', return_value=kernels):
            result = candidate.load_kernels('pinned-root')
        self.assertEqual(result['norm_gate'], dict(compute='norm', reader='reader', writer='writer'))
        self.assertEqual(result['recurrence']['reader'], 'reader')
        self.assertEqual(result['recurrence']['writer'], 'writer')
        self.assertIn(candidate.PAIRED, result['recurrence']['compute'])

    def test_pair_schedule_preserves_tiles_and_tail(self):
        for count in (1, 2, 3, 4, 8, 16):
            tiles = [tile + register for tile in range(0, count, 2)
                     for register in range(min(2, count - tile))]
            self.assertEqual(tiles, list(range(count)))


if __name__ == '__main__':
    unittest.main()
