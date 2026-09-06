import unittest

from gdn_multitoken import cb_plan, replace_once, transform, validate_geometry


class MultiTokenTests(unittest.TestCase):
    def test_frozen_geometry_is_a_single_sequence_not_independent_batch(self):
        for rows in (1, 2, 4, 8, 16):
            self.assertEqual(validate_geometry((1, rows, 5120), (1, rows, 24), (1, rows, 24), (1, 24, 128, 128)), rows)
        for initial in ((16, 24, 128, 128), (1, 12, 128, 128)):
            with self.assertRaises(ValueError):
                validate_geometry((1, 16, 5120), (1, 16, 24), (1, 16, 24), initial)

    def test_feedback_is_io_bf16_and_has_own_ring(self):
        io, fp32 = cb_plan()
        self.assertEqual(io[30], 16)
        self.assertNotIn(30, fp32)
        self.assertFalse(set(io) & set(fp32))
        self.assertEqual(io[5], 16)
        self.assertEqual(io[18], 16)

    def test_ambiguous_or_missing_native_anchor_fails(self):
        for source in ('missing', 'anchor anchor'):
            with self.assertRaises(ValueError):
                replace_once(source, 'anchor', 'new')

    def test_reader_loads_initial_state_only_once_per_head(self):
        source = '''for (uint32_t bh = bh_start; bh < bh_start + n_inst; ++bh) {
CircularBuffer cbs(cb_state);
cbs.push_back(kv);
}'''
        result = transform('reader', source)
        self.assertIn('bh_start + token * H', result)
        self.assertIn('if (token == 0)', result)
        self.assertEqual(result.count('cbs.push_back(kv);'), 1)

    def test_compute_feedback_rounds_before_next_token(self):
        source = '''WAIT(cb_state, kv);
copy_tiles(cb_state, cb_sf, kv);
POP(cb_state, kv);
copy_tiles(cb_snew, cb_sout, kv);'''
        result = transform('compute', source)
        self.assertIn('copy_tiles(it == 0 ? cb_state : 30, cb_sf, kv);', result)
        self.assertIn('if (it + 1 < n_inst) { copy_tiles(cb_snew, 30, kv); }', result)
        self.assertIn('copy_tiles(cb_snew, cb_sout, kv);', result)


if __name__ == '__main__':
    unittest.main()
