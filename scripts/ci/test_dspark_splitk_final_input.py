import unittest

from dspark_splitk_final_input import transform


class FinalInputTests(unittest.TestCase):
    def test_only_root_finalization_changes(self):
        source = '''            /* OUT_ACC *= 1/SUM */
            mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_prev_sum);
            move_block<true>(cb_out_accumulate_im, cb_out_final, out_chunk_tiles);
        } else if (has_parent) {
            move_block<true>(cb_out_accumulate_im, cb_out_o, out_chunk_tiles);'''
        result = transform(source)
        self.assertIn('mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_o, cb_prev_sum);', result)
        self.assertIn('move_block<true>(cb_out_o, cb_out_final, out_chunk_tiles);', result)
        self.assertTrue(result.endswith(source[source.index('        } else if'):]))
        self.assertEqual(result.count('pack_reconfig_data_format(cb_out_o);'), 1)

    def test_rejects_source_drift(self):
        with self.assertRaisesRegex(ValueError, 'boundaries'):
            transform('')


if __name__ == '__main__':
    unittest.main()
