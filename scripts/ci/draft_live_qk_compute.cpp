#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "llk_sfpu/ckernel_sfpu_reduce.h"

#ifdef TRISC_MATH
inline void multiply_live_rows(uint32_t data_slot) {
    using namespace ckernel::sfpu;
    constexpr InstrModLoadStore mode = InstrModLoadStore::DEFAULT;
    const uint32_t data_base = data_slot * DEST_TILE_SIZE_RAW;
    constexpr uint32_t broadcast_base = 2 * DEST_TILE_SIZE_RAW + FACE0_BASE;
    _llk_math_eltwise_sfpu_start_(0);
    TT_SFPLOAD(p_sfpu::LREG4, mode, ADDR_MOD_7, broadcast_base + COL_GROUP_OFFSETS[0]);
    TT_SFPLOAD(p_sfpu::LREG5, mode, ADDR_MOD_7, broadcast_base + COL_GROUP_OFFSETS[1]);
    TT_SFPLOAD(p_sfpu::LREG6, mode, ADDR_MOD_7, broadcast_base + COL_GROUP_OFFSETS[2]);
    TT_SFPLOAD(p_sfpu::LREG7, mode, ADDR_MOD_7, broadcast_base + COL_GROUP_OFFSETS[3]);
    for (uint32_t band = 0; band < 2; band++) {
        _process_row_bcast_data_band_<BinaryOp::MUL>(data_base, data_base, FACE0_BASE, band * ROW_BAND_STRIDE);
    }
    _llk_math_eltwise_sfpu_done_();
}
#endif

void kernel_main() {
    const uint32_t worker = get_arg_val<uint32_t>(3);
    const uint32_t workers = get_arg_val<uint32_t>(4);
    const uint32_t key_tiles = get_arg_val<uint32_t>(5);
    for (uint32_t task = worker; task < 16 * key_tiles; task += workers) {
        cb_wait_front(0, 4);
        for (uint32_t key = 0; key < 32; key++) {
            init_sfpu(0, 16);
            sfpu_mul_bcast_row_init();
            tile_regs_acquire();
            for (uint32_t column = 0; column < 4; column++) {
                cb_wait_front(1, 1);
                const uint32_t slot = column == 0 ? 0 : 1;
                copy_tile_to_dst_init_short(0);
                copy_tile(0, column, slot);
                copy_tile_to_dst_init_short(1);
                copy_tile(1, 0, 2);
                MATH((multiply_live_rows(slot)));
                if (column != 0) {
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                    sfpu_mul_bcast_row_init();
                }
                cb_pop_front(1, 1);
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
        cb_pop_front(0, 4);
    }
}
