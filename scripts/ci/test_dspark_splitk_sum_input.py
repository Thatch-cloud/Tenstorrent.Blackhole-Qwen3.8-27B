import unittest

from dspark_splitk_sum_input import transform


SOURCE = '''    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
                reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                pack_reconfig_data_format(cb_cur_sum);
                reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
                    cb_cur_sum, cb_cur_sum, Sk_chunk_t_dynamic, false);
                matmul_blocks(cb_qk_im, cb_v_in, cb_out_mm);
                CircularBuffer(cb_qk_im).pop_front(qk_chunk_tiles_dynamic);'''


class SumInputTests(unittest.TestCase):
    def test_preserves_original_for_value_matmul(self):
        result = transform(SOURCE)
        self.assertIn('qwen_splitk_copy_fp32(cb_qk_im, row * Sk_chunk_t_dynamic + column, 1);', result)
        self.assertIn('row < Sq_chunk_t', result)
        self.assertIn('CircularBuffer(cb_cur_sum).reserve_back(1)', result)
        self.assertIn('matmul_blocks(cb_qk_im, cb_v_in, cb_out_mm);', result)
        self.assertNotIn('CircularBuffer(cb_exponent_sum)', result)
        self.assertEqual(result.count('CircularBuffer(cb_qk_im).pop_front'), 1)
        self.assertIn('sfpu::calculate_reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,', result)

    def test_rejects_source_drift(self):
        with self.assertRaisesRegex(ValueError, 'Exact decode sum-input'):
            transform(SOURCE.replace('Sk_chunk_t_dynamic, false', 'Sk_chunk_t_dynamic, true'))


if __name__ == '__main__':
    unittest.main()
