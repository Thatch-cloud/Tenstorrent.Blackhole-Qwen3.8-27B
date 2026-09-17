"""Preserve FP32 probabilities through SFPU row reduction in the simulator."""


def transform(source):
    declaration = '    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;'
    before = '''                reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                pack_reconfig_data_format(cb_cur_sum);'''
    after = '''                CircularBuffer(cb_qk_im).wait_front(qk_chunk_tiles_dynamic);'''
    reduce = '''                reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
                    cb_cur_sum, cb_cur_sum, Sk_chunk_t_dynamic, false);'''
    for anchor in (declaration, before, reduce):
        if source.count(anchor) != 1:
            raise ValueError('Exact decode sum-input anchors required')
    replacement = '''                for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
                    tile_regs_acquire();
                    qwen_splitk_copy_fp32_init(cb_qk_im);
                    qwen_splitk_copy_fp32(cb_qk_im, row * Sk_chunk_t_dynamic, 0);
                    for (uint32_t column = 1; column < Sk_chunk_t_dynamic; ++column) {
                        qwen_splitk_copy_fp32_init(cb_qk_im);
                        qwen_splitk_copy_fp32(cb_qk_im, row * Sk_chunk_t_dynamic + column, 1);
                        add_binary_tile_init();
                        add_binary_tile(0, 1, 0);
                    }
                    MATH((sfpu::init_reduce<PoolType::SUM, DataFormat::Float32, true>()));
                    MATH((_llk_math_eltwise_sfpu_start_(0)));
                    MATH((sfpu::calculate_reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
                        DataFormat::Float32, true, DataFormat::Float32>(1, 1)));
                    MATH((_llk_math_eltwise_sfpu_done_()));
                    tile_regs_commit();
                    CircularBuffer(cb_cur_sum).reserve_back(1);
                    tile_regs_wait();
                    pack_reconfig_data_format(cb_cur_sum);
                    pack_tile(0, cb_cur_sum);
                    tile_regs_release();
                    CircularBuffer(cb_cur_sum).push_back(1);
                }'''
    return source.replace(declaration, declaration + '\n    constexpr uint32_t cb_exponent_sum = tt::CBIndex::c_32;').replace(
        before, after).replace(reduce, replacement).replace('#include "api/compute/eltwise_unary/recip.h"',
        '#include "api/compute/eltwise_unary/recip.h"\n#include "llk_sfpu/ckernel_sfpu_reduce.h"')
