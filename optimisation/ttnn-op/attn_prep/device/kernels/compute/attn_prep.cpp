// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel: fused attention decode prologue, per (batch row, kind) instance. Only q (kind 0)
// and k (kind 1) reach this kernel; v and gate are copied by the reader straight into cb_out.
// On the 32-row (heads) x HD block:
//
//   xf  = fp32(block)
//   fac = sqrt(HD) * rsqrt(rowsum(xf*xf) + HD*eps)        (rows are heads: per-head rms norm)
//   xn  = xf * fac ;  xw = xn * norm_w (row-0 broadcast) ;  xwr = fp32(bf16(xw))
//   rope on the first RD dims (HF rotate-half, tile-aligned halves, RDh = RD/64 tiles each):
//     out[t]       = xwr[t] * cos[t]       - xwr[RDh+t] * sin[t]
//     out[RDh + t] = xwr[RDh+t] * cos[RDh+t] + xwr[t] * sin[RDh+t]      (cos/sin row-0 broadcast)
//   out[RDt..HDt) = xwr (pass-through)
//
// Every binary op reads index 0.. of two DISTINCT rings (the ring-layout rule from the plan),
// and every packed page is wait_front'ed before it is read back.
//
// Compile args: {HDt, RDt, EPS_BITS, SCALE_BITS}. Runtime args: {inst_start, n_inst}.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t cb_blk = 1, cb_out = 2, cb_cos = 3, cb_sin = 4, cb_wq = 5, cb_wk = 6;
constexpr uint32_t cb_wqf = 7, cb_wkf = 8, cb_ones = 9, cb_xf = 10, cb_sq = 11, cb_sc = 12, cb_fac = 13;
constexpr uint32_t cb_xn = 14, cb_xw = 15, cb_rt = 16, cb_xwr = 17, cb_cosf = 18, cb_sinf = 19, cb_ta = 20, cb_tb = 21;

inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }
inline void POP(uint32_t cb, uint32_t n) { CircularBuffer(cb).pop_front(n); }

void copy_tiles(uint32_t in, uint32_t o, uint32_t n, uint32_t in0 = 0) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in, in0 + i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

void ew_mul(uint32_t a, uint32_t b, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);
    mul_init(a, b);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles(a, b, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// o (1 tile) = a[ia] (op) b[ib]; op 0 add, 1 sub.
void addsub1(uint32_t a, uint32_t ia, uint32_t b, uint32_t ib, uint32_t o, int op) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);
    if (op == 0) {
        add_init(a, b);
    } else {
        sub_init(a, b);
    }
    tile_regs_acquire();
    if (op == 0) {
        add_tiles(a, b, ia, ib, 0);
    } else {
        sub_tiles(a, b, ia, ib, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

void rowsum(uint32_t in, uint32_t o, uint32_t n) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(in, cb_ones);
    matmul_init(in, cb_ones, 0);
    tile_regs_acquire();
    for (uint32_t k = 0; k < n; k++) {
        matmul_tiles(in, cb_ones, k, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

void inv_rms(uint32_t in, uint32_t o, uint32_t eps_bits, uint32_t scale_bits) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    tile_regs_acquire();
    copy_tile(in, 0, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, scale_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

// o[n] = a[i] * col[0] broadcast along columns (row r scaled by col[r,0]).
void bcast_cols_mul(uint32_t a, uint32_t col, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, col);
    mul_bcast_cols_init(a, col);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(a, col, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// o[n] = a[i] * row[i]'s row 0 broadcast down every row.
void bcast_rows_mul(uint32_t a, uint32_t row, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, row);
    mul_bcast_rows_init(a, row);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(a, row, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// o (1 tile) = a[ia] * row[ir]'s row 0 broadcast.
void bcast_rows_mul1(uint32_t a, uint32_t ia, uint32_t row, uint32_t ir, uint32_t o) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, row);
    mul_bcast_rows_init(a, row);
    tile_regs_acquire();
    mul_tiles_bcast_rows(a, row, ia, ir, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t HDt = get_compile_time_arg_val(0);
    constexpr uint32_t RDt = get_compile_time_arg_val(1);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(2);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(3);
    constexpr uint32_t RDh = RDt / 2;

    const uint32_t inst_start = get_arg_val<uint32_t>(0);
    const uint32_t n_inst = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_blk, cb_cos, cb_out);
    WAIT(cb_ones, 1);
    // persistent fp32 norm weights
    WAIT(cb_wq, HDt);
    copy_tiles(cb_wq, cb_wqf, HDt);
    POP(cb_wq, HDt);
    WAIT(cb_wqf, HDt);
    WAIT(cb_wk, HDt);
    copy_tiles(cb_wk, cb_wkf, HDt);
    POP(cb_wk, HDt);
    WAIT(cb_wkf, HDt);

    for (uint32_t inst = inst_start; inst < inst_start + n_inst; ++inst) {
        const uint32_t kind = inst % 4;
        if (kind >= 2) {
            continue;  // v and gate: the reader landed them in cb_out
        }
        const uint32_t cb_wf = (kind == 0) ? cb_wqf : cb_wkf;

        WAIT(cb_blk, HDt);
        copy_tiles(cb_blk, cb_xf, HDt);
        POP(cb_blk, HDt);
        WAIT(cb_xf, HDt);

        // per-row rms factor over HD
        ew_mul(cb_xf, cb_xf, cb_sq, HDt);
        WAIT(cb_sq, HDt);
        rowsum(cb_sq, cb_sc, HDt);
        WAIT(cb_sc, 1);
        POP(cb_sq, HDt);
        inv_rms(cb_sc, cb_fac, EPS_BITS, SCALE_BITS);
        POP(cb_sc, 1);
        WAIT(cb_fac, 1);
        bcast_cols_mul(cb_xf, cb_fac, cb_xn, HDt);
        WAIT(cb_xn, HDt);
        POP(cb_xf, HDt);
        POP(cb_fac, 1);
        bcast_rows_mul(cb_xn, cb_wf, cb_xw, HDt);
        WAIT(cb_xw, HDt);
        POP(cb_xn, HDt);
        // round to bf16 where the composed chain does (rms_norm * w -> bf16), then back to fp32
        copy_tiles(cb_xw, cb_rt, HDt);
        WAIT(cb_rt, HDt);
        POP(cb_xw, HDt);
        copy_tiles(cb_rt, cb_xwr, HDt);
        WAIT(cb_xwr, HDt);
        POP(cb_rt, HDt);

        // rope on the first RDt tiles
        WAIT(cb_cos, RDt);
        WAIT(cb_sin, RDt);
        copy_tiles(cb_cos, cb_cosf, RDt);
        copy_tiles(cb_sin, cb_sinf, RDt);
        POP(cb_cos, RDt);
        POP(cb_sin, RDt);
        WAIT(cb_cosf, RDt);
        WAIT(cb_sinf, RDt);
        cb_reserve_back(cb_out, HDt);
        for (uint32_t t = 0; t < RDh; t++) {
            // out[t] = x1*cos[t] - x2*sin[t]
            bcast_rows_mul1(cb_xwr, t, cb_cosf, t, cb_ta);
            WAIT(cb_ta, 1);
            bcast_rows_mul1(cb_xwr, RDh + t, cb_sinf, t, cb_tb);
            WAIT(cb_tb, 1);
            {
                pack_reconfig_data_format(cb_out);
                reconfig_data_format(cb_ta, cb_tb);
                sub_init(cb_ta, cb_tb);
                tile_regs_acquire();
                sub_tiles(cb_ta, cb_tb, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out, t);
                tile_regs_release();
            }
            POP(cb_ta, 1);
            POP(cb_tb, 1);
            // out[RDh+t] = x2*cos[RDh+t] + x1*sin[RDh+t]
            bcast_rows_mul1(cb_xwr, RDh + t, cb_cosf, RDh + t, cb_ta);
            WAIT(cb_ta, 1);
            bcast_rows_mul1(cb_xwr, t, cb_sinf, RDh + t, cb_tb);
            WAIT(cb_tb, 1);
            {
                pack_reconfig_data_format(cb_out);
                reconfig_data_format(cb_ta, cb_tb);
                add_init(cb_ta, cb_tb);
                tile_regs_acquire();
                add_tiles(cb_ta, cb_tb, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out, RDh + t);
                tile_regs_release();
            }
            POP(cb_ta, 1);
            POP(cb_tb, 1);
        }
        // pass-through tail
        {
            pack_reconfig_data_format(cb_out);
            reconfig_data_format_srca(cb_xwr);
            copy_tile_to_dst_init_short(cb_xwr);
            for (uint32_t t = RDt; t < HDt; t++) {
                tile_regs_acquire();
                copy_tile(cb_xwr, t, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out, t);
                tile_regs_release();
            }
        }
        cb_push_back(cb_out, HDt);
        POP(cb_xwr, HDt);
        POP(cb_cosf, RDt);
        POP(cb_sinf, RDt);
    }
}
