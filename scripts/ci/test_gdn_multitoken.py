import unittest

from gdn_multitoken import cb_plan, replace_once, replace_section, transform, validate_geometry


class MultiTokenTests(unittest.TestCase):
    def test_norm_gate_feedback_reuses_initial_ring_not_norm_scratch(self):
        source = 'WAIT(cb_state, kv);\nPOP(cb_state, kv);\ncopy_tiles(cb_snew, cb_sout, kv);'
        result = transform('compute', source, True)
        self.assertIn('if (it + 1 < n_inst) { copy_tiles(cb_snew, cb_state, kv); }', result)
        self.assertNotIn('30', result)
        self.assertIn('copy_tiles(cb_snew, cb_sout, kv);', result)
        io, fp32 = cb_plan(True)
        self.assertEqual(io[5], 16)
        self.assertEqual(io[30], 8)
        self.assertEqual(fp32[31], 4)
        self.assertEqual(io[2], 8)
        self.assertEqual(io[27], 4)
        self.assertFalse(set(io) & set(fp32))

    def test_section_replacement_fails_closed(self):
        self.assertEqual(replace_section('aSTARToldENDz', 'START', 'END', 'new'), 'anewENDz')
        for source in ('STARTSTARTEND', 'ENDSTART', 'STARTmissing'):
            with self.assertRaises(ValueError):
                replace_section(source, 'START', 'END', 'new')

    def test_norm_writer_local_assembly_has_no_remote_semaphores(self):
        source = '''    if constexpr (FNG) {
        // Rows at or beyond B old
    // Per-instance loop:
    for (uint32_t bh = bh_start; bh < bh_start + n_inst; ++bh) {
        if constexpr (FNG) {
            // gated row old
            noc_semaphore_inc();
        } else {
        // o: stage unchanged
        }
    }
    if constexpr (FNG) {
        // Assembler duty: old
        noc_semaphore_wait();
    }
}
'''
        result = transform('writer', source, True)
        self.assertNotIn('noc_semaphore', result)
        self.assertIn('const uint32_t row = token;', result)
        self.assertIn('zero(asm_base, Vt * tb_io / 4)', result)
        self.assertIn('.page_id = bh_start * Vt + tile', result)
        self.assertIn('noc.async_write_barrier();', result)

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
