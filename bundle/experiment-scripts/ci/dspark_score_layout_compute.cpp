#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

void kernel_main() {
    const uint32_t worker = get_arg_val<uint32_t>(3);
    const uint32_t workers = get_arg_val<uint32_t>(4);
    const uint32_t tiles = get_arg_val<uint32_t>(5);
    init_sfpu(0, 16);
    for (uint32_t task = worker; task < tiles; task += workers) {
        cb_wait_front(0, 1);
        cb_wait_front(1, 1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(0);
        copy_tile(0, 0, 0);
        copy_tile_to_dst_init_short(1);
        copy_tile(1, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        cb_reserve_back(16, 1);
        tile_regs_wait();
        pack_tile(0, 16);
        tile_regs_release();
        cb_push_back(16, 1);
        cb_pop_front(0, 1);
        cb_pop_front(1, 1);
    }
}
