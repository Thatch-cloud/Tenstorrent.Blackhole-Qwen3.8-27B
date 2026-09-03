// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel: fused GDN decode conv + gates, per (batch-tile, channel-tile) instance.
//
//   prod[j] = window[j] * tap[j]      (tap row 0 broadcast over the batch rows; fp32 out)
//   acc     = prod[0] + ... + prod[K-1]
//   out     = silu(acc)               (io dtype)
//   shift   = window                  (pass-through copy so the writer can store the
//                                      advanced shift-register; CBs are single-consumer)
// Gate instances (gate core only):
//   beta = sigmoid(b)
//   sp   = softplus(a + dt_bias)      (beta 1, threshold 20 -- python SOFTPLUS(1.0, 20.0);
//                                      rounded to io dtype like the composed graph)
//   g    = sp * neg_exp_A             (row-0 broadcast)
//
// Every packed CB page is wait_front'ed before it is read back (the packer/unpacker
// ordering rule the recurrence kernel documents).
//
// Compile args: {K, Nvt, ONE_BITS, TWENTY_BITS}. Runtime args: {n_inst, g_n}.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t cb_win = 0, cb_tap = 1, cb_prod = 2, cb_acc = 3, cb_out = 4, cb_shift = 5;
constexpr uint32_t cb_a = 6, cb_b = 7, cb_dtb = 8, cb_nega = 9, cb_sp = 10, cb_beta = 11, cb_g = 12;

inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }
inline void POP(uint32_t cb, uint32_t n) { CircularBuffer(cb).pop_front(n); }

// o = copy of the n tiles of `in` (format conversion on the copy).
void copy_tiles(uint32_t in, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// o (1 tile) = a[ia] + b[ib]
void add2(uint32_t a, uint32_t ia, uint32_t b, uint32_t ib, uint32_t o) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);
    add_init(a, b);
    tile_regs_acquire();
    add_tiles(a, b, ia, ib, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t ONE_BITS = get_compile_time_arg_val(2);
    constexpr uint32_t TWENTY_BITS = get_compile_time_arg_val(3);
    static_assert(K == 4, "compute is written for K == 4 taps");

    const uint32_t n_inst = get_arg_val<uint32_t>(0);
    const uint32_t g_n = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_win, cb_tap, cb_out);

    for (uint32_t it = 0; it < n_inst; ++it) {
        WAIT(cb_win, K);
        WAIT(cb_tap, K);

        // ---- prod[j] = window[j] * tap[j]  (row-0 broadcast of the tap) ----
        cb_reserve_back(cb_prod, K);
        pack_reconfig_data_format(cb_prod);
        reconfig_data_format(cb_win, cb_tap);
        mul_bcast_rows_init(cb_win, cb_tap);
        for (uint32_t j = 0; j < K; j++) {
            tile_regs_acquire();
            mul_tiles_bcast_rows(cb_win, cb_tap, j, j, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_prod, j);
            tile_regs_release();
        }
        cb_push_back(cb_prod, K);

        // ---- pass the window through for the state write-back ----
        copy_tiles(cb_win, cb_shift, K);
        POP(cb_win, K);
        POP(cb_tap, K);
        WAIT(cb_prod, K);

        // ---- acc = sum_j prod[j]  (2-page ring on cb_acc) ----
        add2(cb_prod, 0, cb_prod, 1, cb_acc);
        WAIT(cb_acc, 1);
        for (uint32_t j = 2; j < K; j++) {
            add2(cb_acc, 0, cb_prod, j, cb_acc);
            POP(cb_acc, 1);  // drop the older partial sum; the new one is now front
            WAIT(cb_acc, 1);
        }
        POP(cb_prod, K);

        // ---- out = silu(acc) ----
        cb_reserve_back(cb_out, 1);
        pack_reconfig_data_format(cb_out);
        reconfig_data_format_srca(cb_acc);
        copy_tile_to_dst_init_short(cb_acc);
        silu_tile_init();
        tile_regs_acquire();
        copy_tile(cb_acc, 0, 0);
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out, 0);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        POP(cb_acc, 1);
    }

    for (uint32_t gi = 0; gi < g_n; ++gi) {
        WAIT(cb_a, 1);
        WAIT(cb_b, 1);
        WAIT(cb_dtb, 1);
        WAIT(cb_nega, 1);

        // ---- beta = sigmoid(b) ----
        cb_reserve_back(cb_beta, 1);
        pack_reconfig_data_format(cb_beta);
        reconfig_data_format_srca(cb_b);
        copy_tile_to_dst_init_short(cb_b);
        sigmoid_tile_init<false>();
        tile_regs_acquire();
        copy_tile(cb_b, 0, 0);
        sigmoid_tile<VectorMode::RC, false>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_beta, 0);
        tile_regs_release();
        cb_push_back(cb_beta, 1);

        // ---- sp = softplus(a + dt_bias) ----
        cb_reserve_back(cb_sp, 1);
        pack_reconfig_data_format(cb_sp);
        reconfig_data_format(cb_a, cb_dtb);
        add_bcast_rows_init(cb_a, cb_dtb);
        tile_regs_acquire();
        add_tiles_bcast_rows(cb_a, cb_dtb, 0, 0, 0);
        softplus_tile_init();
        softplus_tile(0, ONE_BITS, ONE_BITS, TWENTY_BITS);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sp, 0);
        tile_regs_release();
        cb_push_back(cb_sp, 1);
        WAIT(cb_sp, 1);

        // ---- g = sp * neg_exp_A ----
        cb_reserve_back(cb_g, 1);
        pack_reconfig_data_format(cb_g);
        reconfig_data_format(cb_sp, cb_nega);
        mul_bcast_rows_init(cb_sp, cb_nega);
        tile_regs_acquire();
        mul_tiles_bcast_rows(cb_sp, cb_nega, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_g, 0);
        tile_regs_release();
        cb_push_back(cb_g, 1);

        POP(cb_sp, 1);
        POP(cb_a, 1);
        POP(cb_b, 1);
        POP(cb_dtb, 1);
        POP(cb_nega, 1);
    }
}
