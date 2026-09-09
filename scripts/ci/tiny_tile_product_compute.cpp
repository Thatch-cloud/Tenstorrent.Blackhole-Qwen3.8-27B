#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    const uint32_t pages = get_arg_val<uint32_t>(0);
    unary_op_init_common(0, 16);
    mul_binary_tile_init();
    for (uint32_t page = 0; page < pages; page++) {
        cb_wait_front(0, 1);
        cb_wait_front(1, 1);
        cb_reserve_back(16, 1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(1, 0);
        copy_tile(0, 0, 0);
        copy_tile_to_dst_init_short_with_dt(0, 1);
        copy_tile(1, 0, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 16);
        tile_regs_release();
        cb_push_back(16, 1);
        cb_pop_front(0, 1);
        cb_pop_front(1, 1);
    }
}
