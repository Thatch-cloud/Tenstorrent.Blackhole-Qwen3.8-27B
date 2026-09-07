#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "llk_sfpu/ckernel_sfpu_reduce.h"

void kernel_main() {
    constexpr uint32_t columns = get_compile_time_arg_val(0);
    init_sfpu(0, 16);
    tile_regs_acquire();
    cb_wait_front(0, 1);
    copy_tile(0, 0, 0);
    cb_pop_front(0, 1);
    for (uint32_t column = 1; column < columns; column++) {
        cb_wait_front(0, 1);
        copy_tile(0, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        cb_pop_front(0, 1);
    }
    MATH((sfpu::init_reduce<PoolType::SUM, DataFormat::Float32, true>()));
    MATH((_llk_math_eltwise_sfpu_start_(0)));
    MATH((sfpu::calculate_reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
        DataFormat::Float32, true, DataFormat::Float32>(1, 1)));
    MATH((_llk_math_eltwise_sfpu_done_()));
    tile_regs_commit();
    cb_reserve_back(16, 1);
    tile_regs_wait();
    pack_tile(0, 16);
    tile_regs_release();
    cb_push_back(16, 1);
}
