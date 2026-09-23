// gdn_prefill_conv_exact compute (TRISC). See gdn_prefill_conv_exact.py.
//
// Per output tile, in 16-bit DST (fp32_dest_acc_en=False) and in the served order, operand
// slots and LLK calls:
//   tap 0   binary_ng SFPU row-bcast mul:  x[t-3] -> d0, w0 -> d1, mul_binary_tile(0, 1, 0)
//   taps 1-3  ternary addcmul row-bcast:   acc stays in d0, x[t-3+k] -> d1, w_k -> d2,
//             addcmul_tile<Float16_b>(0, 1, 2, 0, bits(1.0f))
//   silu    unary: silu_tile(0)
// The taps are row-broadcast once per column into c_5 with the same unary_bcast<ROW> + pack the
// served kernels use. The accumulator stays in DST between taps; every SFPU op above already
// stores an exactly-bf16 value, so this changes no value. MIRROR_PACK replays the served
// pack -> unpack of the accumulator between ops literally (c_9).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/bcast.h"
#include "api/compute/reconfig_data_format.h"

namespace pcx {
constexpr uint32_t cb_x3 = 0;
constexpr uint32_t cb_x2 = 1;
constexpr uint32_t cb_x1 = 2;
constexpr uint32_t cb_x0 = 3;
constexpr uint32_t cb_taps = 4;
constexpr uint32_t cb_bcast = 5;
constexpr uint32_t cb_out = 6;
constexpr uint32_t cb_acc = 9;
constexpr uint32_t one_f32 = 0x3F800000u;
}  // namespace pcx

#ifdef NEG_FP32
#define PCX_ADDCMUL_FORMAT DataFormat::Float32
#else
#define PCX_ADDCMUL_FORMAT DataFormat::Float16_b
#endif

void kernel_main() {
    using namespace pcx;
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t strips = get_compile_time_arg_val(3);
    const uint32_t unit_start = get_arg_val<uint32_t>(0);
    const uint32_t unit_count = get_arg_val<uint32_t>(1);
    // Tap k's row-broadcast tile in c_5; NEG_TAPSWAP swaps taps 1 and 2 (negative control).
#ifdef NEG_TAPSWAP
    constexpr uint32_t tap_index[4] = {0, 2, 1, 3};
#else
    constexpr uint32_t tap_index[4] = {0, 1, 2, 3};
#endif
    constexpr uint32_t shifted[4] = {cb_x3, cb_x2, cb_x1, cb_x0};

    init_sfpu(cb_x3, cb_out);
    uint32_t prev_ct = 0xFFFFFFFFu;
    for (uint32_t unit = unit_start; unit < unit_start + unit_count; ++unit) {
        const uint32_t ct = unit / strips;
        if (ct != prev_ct) {
            if (prev_ct != 0xFFFFFFFFu) {
                cb_pop_front(cb_bcast, 4);
            }
            cb_wait_front(cb_taps, 4);
            reconfig_data_format(cb_taps, cb_taps);
            pack_reconfig_data_format(cb_bcast);
            unary_bcast_init<BroadcastType::ROW>(cb_taps);
            for (uint32_t j = 0; j < 4; ++j) {
                cb_reserve_back(cb_bcast, 1);
                tile_regs_acquire();
                unary_bcast<BroadcastType::ROW>(cb_taps, j, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_bcast);
                tile_regs_release();
                cb_push_back(cb_bcast, 1);
            }
            cb_pop_front(cb_taps, 4);
            cb_wait_front(cb_bcast, 4);
            reconfig_data_format(cb_x3, cb_x3);
            pack_reconfig_data_format(cb_out);
            prev_ct = ct;
        }
        for (uint32_t t = 0; t < R; ++t) {
            cb_wait_front(cb_x3, 1);
            cb_wait_front(cb_x2, 1);
            cb_wait_front(cb_x1, 1);
            cb_wait_front(cb_x0, 1);
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_x3);
            copy_tile(cb_x3, 0, 0);
            copy_tile_to_dst_init_short(cb_bcast);
            copy_tile(cb_bcast, tap_index[0], 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            for (uint32_t k = 1; k < 4; ++k) {
#ifdef MIRROR_PACK
                // The served chain packs the accumulator to bf16 and unpacks it for the next op.
                tile_regs_commit();
                cb_reserve_back(cb_acc, 1);
                tile_regs_wait();
                pack_reconfig_data_format(cb_acc);
                pack_tile(0, cb_acc);
                tile_regs_release();
                cb_push_back(cb_acc, 1);
                cb_wait_front(cb_acc, 1);
                tile_regs_acquire();
                copy_tile_to_dst_init_short(cb_acc);
                copy_tile(cb_acc, 0, 0);
                cb_pop_front(cb_acc, 1);
                pack_reconfig_data_format(cb_out);
#endif
                copy_tile_to_dst_init_short(shifted[k]);
                copy_tile(shifted[k], 0, 1);
                copy_tile_to_dst_init_short(cb_bcast);
                copy_tile(cb_bcast, tap_index[k], 2);
                addcmul_tile_init();
                addcmul_tile<PCX_ADDCMUL_FORMAT>(0, 1, 2, 0, one_f32);
            }
#ifdef MIRROR_PACK
            tile_regs_commit();
            cb_reserve_back(cb_acc, 1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_acc);
            pack_tile(0, cb_acc);
            tile_regs_release();
            cb_push_back(cb_acc, 1);
            cb_wait_front(cb_acc, 1);
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_acc);
            copy_tile(cb_acc, 0, 0);
            cb_pop_front(cb_acc, 1);
            pack_reconfig_data_format(cb_out);
#endif
            silu_tile_init();
            silu_tile(0);
            tile_regs_commit();
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
            cb_pop_front(cb_x3, 1);
            cb_pop_front(cb_x2, 1);
            cb_pop_front(cb_x1, 1);
            cb_pop_front(cb_x0, 1);
        }
    }
    if (prev_ct != 0xFFFFFFFFu) {
        cb_pop_front(cb_bcast, 4);
    }
}
