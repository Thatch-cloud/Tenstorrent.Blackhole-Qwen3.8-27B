"""Scoped SFPU row reduction, adapted from draft_row_sum_compute.cpp."""


BEFORE = '        matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);'
AFTER = '''#if defined(QWEN_DRAFT_EXP_APPROX)
        if constexpr (!QWEN_DRAFT_EXP_APPROX) {
            reconfig_data_format_srca(alias_prev_sum);
            copy_tile_to_dst_init_short(alias_prev_sum);
            pack_reconfig_data_format(alias_prev_sum);
            for (uint32_t row_tile = 0; row_tile < Sq_chunk_t; ++row_tile) {
                CircularBuffer(alias_prev_sum).wait_front(1);
                tile_regs_acquire();
                copy_tile(alias_prev_sum, 0, 0);
                MATH((sfpu::init_reduce<PoolType::SUM, DataFormat::Float32, true>()));
                MATH((_llk_math_eltwise_sfpu_start_(0)));
                MATH((sfpu::calculate_reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
                    DataFormat::Float32, true, DataFormat::Float32>(1, 1)));
                MATH((_llk_math_eltwise_sfpu_done_()));
                tile_regs_commit();
                CircularBuffer(alias_prev_sum).pop_front(1);
                CircularBuffer(alias_prev_sum).reserve_back(1);
                tile_regs_wait();
                pack_tile(0, alias_prev_sum);
                tile_regs_release();
                CircularBuffer(alias_prev_sum).push_back(1);
            }
        } else {
''' + BEFORE + '''
        }
#else
''' + BEFORE + '''
#endif'''


def substitutions():
    include = '#include "api/compute/eltwise_unary/recip.h"'
    return ((include, include + '\n#include "llk_sfpu/ckernel_sfpu_reduce.h"'), (BEFORE, AFTER))
