"""Fail-closed source transformation tests; numerical admission requires TTsim."""

import unittest
from unittest.mock import patch

import gdn_outer_add as candidate


class OuterAddTests(unittest.TestCase):
    def source(self):
        return '#include "api/compute/eltwise_binary.h"\nvoid kernel_main() {\n' + candidate.ORIGINAL + '\n}'

    def test_replaces_only_explicit_chain_and_adds_helper(self):
        result = candidate.transform(self.source())
        self.assertIn(candidate.FUSED, result)
        self.assertIn(candidate.HELPER, result)
        self.assertNotIn('POP(cb_outer, kv)', result)
        self.assertIn('add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(state, first + slot, slot)', result)
        self.assertNotIn('add_binary_tile', result)
        self.assertEqual(result.count('tile_regs_acquire();'), 1)
        self.assertEqual(result.count('pack_tile(slot, output, first + slot);'), 1)

    def test_pair_schedule_preserves_outer_coordinates_and_tail(self):
        for key_tiles, value_tiles in ((4, 1), (3, 1), (4, 4)):
            count = key_tiles * value_tiles
            tiles = [first + slot for first in range(0, count, 2)
                     for slot in range(min(2, count - first))]
            self.assertEqual([(tile // value_tiles, tile % value_tiles) for tile in tiles],
                             [(key, value) for key in range(key_tiles) for value in range(value_tiles)])

    def test_changed_or_duplicate_chain_fails(self):
        for source in (self.source() + candidate.ORIGINAL,
                       self.source().replace('WAIT(cb_outer, kv);', 'changed();')):
            with self.assertRaises(ValueError):
                candidate.transform(source)

    def test_only_recurrence_compute_changes(self):
        baseline = dict(recurrence=dict(compute=self.source(), reader='reader', writer='writer'),
                        norm_gate=dict(compute='norm'))
        with patch.object(candidate, 'baseline_kernels', return_value=baseline):
            result = candidate.load_kernels('root')
        self.assertEqual(result['norm_gate'], dict(compute='norm'))
        self.assertEqual(result['recurrence']['reader'], 'reader')
        self.assertEqual(result['recurrence']['writer'], 'writer')


if __name__ == '__main__':
    unittest.main()
