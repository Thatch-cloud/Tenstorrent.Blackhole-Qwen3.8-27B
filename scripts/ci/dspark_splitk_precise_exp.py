"""Simulator-only precise exponent control with explicit FP32 scaling."""


BEFORE = '''                sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, scale_fp32, true, false, vector_mode>(
                    cb_cur_max, cb_cur_sum, Sk_chunk_t_dynamic);'''
AFTER = '''                {
                    reconfig_data_format(cb_qk_im, cb_cur_max);
                    sub_bcast_cols_init(cb_qk_im, cb_cur_max);
                    pack_reconfig_data_format(cb_qk_im);
                    binop_with_scalar_tile_init();
                    exp_tile_init<false>();
                    CircularBuffer(cb_qk_im).wait_front(qk_chunk_tiles_dynamic);
                    CircularBuffer(cb_cur_max).wait_front(Sq_chunk_t);
                    for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
                        for (uint32_t column = 0; column < Sk_chunk_t_dynamic; ++column) {
                            tile_regs_acquire();
                            sub_tiles_bcast_cols(cb_qk_im, cb_cur_max, 0, row, 0);
                            mul_unary_tile(0, scale_fp32);
                            exp_tile<false>(0);
                            tile_regs_commit();
                            CircularBuffer(cb_qk_im).pop_front(1);
                            CircularBuffer(cb_qk_im).reserve_back(1);
                            tile_regs_wait();
                            pack_tile(0, cb_qk_im);
                            tile_regs_release();
                            CircularBuffer(cb_qk_im).push_back(1);
                        }
                    }
                }'''


def transform(source):
    include = '#include "api/compute/eltwise_unary/recip.h"'
    if source.count(BEFORE) != 1 or source.count(include) != 1:
        raise ValueError('Exact decode exponent anchors required')
    return source.replace(BEFORE, AFTER).replace(include,
        include + '\n#include "api/compute/eltwise_unary/binop_with_scalar.h"\n'
        '#include "api/compute/eltwise_unary/typecast.h"')
