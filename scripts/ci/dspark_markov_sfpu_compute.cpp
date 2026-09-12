#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "llk_sfpu/ckernel_sfpu_reduce.h"

void kernel_main() {
    const uint32_t worker = get_arg_val<uint32_t>(4);
    const uint32_t workers = get_arg_val<uint32_t>(5);
    const uint32_t tiles = get_arg_val<uint32_t>(6);
    cb_wait_front(1, 8);
    for (uint32_t task = worker; task < tiles; task += workers) {
        init_sfpu(0, 16);
        sfpu_mul_bcast_row_init();
        tile_regs_acquire();
        for (uint32_t column = 0; column < 8; column++) {
            cb_wait_front(0, 1);
            const uint32_t slot = column == 0 ? 0 : 1;
            copy_tile_to_dst_init_short(0);
            copy_tile(0, 0, slot);
            copy_tile_to_dst_init_short(1);
            copy_tile(1, column, 2);
            sfpu_mul_bcast_row(slot, 2);
            if (column != 0) {
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
                sfpu_mul_bcast_row_init();
            }
            cb_pop_front(0, 1);
        }
        MATH((sfpu::init_reduce<PoolType::SUM, DataFormat::Float32, true>()));
        MATH((_llk_math_eltwise_sfpu_start_(0)));
        MATH((sfpu::calculate_reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
            DataFormat::Float32, true, DataFormat::Float32>(1, 1)));
        MATH((_llk_math_eltwise_sfpu_done_()));
        cb_wait_front(2, 1);
        copy_tile_to_dst_init_short(2);
        copy_tile(2, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        cb_pop_front(2, 1);
        tile_regs_commit();
        cb_reserve_back(16, 1);
        tile_regs_wait();
        pack_tile(0, 16);
        tile_regs_release();
        cb_push_back(16, 1);
    }
    cb_pop_front(1, 8);
}
