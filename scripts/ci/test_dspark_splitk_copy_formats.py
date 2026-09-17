import unittest

from dspark_splitk_copy_formats import transform


class CopyFormatsTests(unittest.TestCase):
    def test_moves_configure_both_formats(self):
        source = '\n'.join('    move_block<true>(cb_out_o, cb_out_accumulate_im_2, out_chunk_tiles);'
            for unused in range(11))
        result = transform(source)
        self.assertEqual(result.count('reconfig_data_format_srca(cb_out_o);'), 11)
        self.assertEqual(result.count('pack_reconfig_data_format(cb_out_accumulate_im_2);'), 11)
        self.assertEqual(result.count('move_block<true>'), 11)

    def test_changed_call_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Expected ten native moves'):
            transform('    move_block<true>(cb_out_o, cb_out_accumulate_im_2, out_chunk_tiles);')


if __name__ == '__main__':
    unittest.main()
