#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"

void kernel_main() {
    constexpr uint32_t pages = get_compile_time_arg_val(0);
    init_sfpu(0, 16);
    for (uint32_t page = 0; page < pages; ++page) {
        cb_wait_front(0, 7);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(0);
        copy_tile(0, 0, 0);
        copy_tile(0, 1, 1);
        copy_tile(0, 6, 3);
        for (uint32_t term = 2; term < 6; ++term) {
            copy_tile_to_dst_init_short(0);
            copy_tile(0, term, 2);
            mul_binary_tile_init();
            mul_binary_tile(term < 4 ? 0 : 1, 2, 2);
            typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>();
            typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(2);
            add_binary_tile_init();
            add_binary_tile(3, 2, 3);
            typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>();
            typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(3);
        }
        tile_regs_commit();
        cb_reserve_back(16, 1);
        tile_regs_wait();
        pack_tile(3, 16);
        tile_regs_release();
        cb_push_back(16, 1);
        cb_pop_front(0, 7);
    }
}
