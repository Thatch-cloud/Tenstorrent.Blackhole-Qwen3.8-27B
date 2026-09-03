// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel: fused GDN decode output norm + gate, per (batch-tile, head, column-tile)
// instance. The reader delivers this head's 32 x V block ROW-MAJOR (32 sticks); the unpacker
// tilizes it into Vt tiles (row = batch). The per-row rms factor needs the whole row, so
// the squares and the row-sum run over all Vt tiles; everything after that touches only
// this instance's column tile ct:
//
//   ot  = tilize(orm)                          (Vt tiles, io dtype)
//   sq  = ot * ot                              (io x io -> fp32; uniform operand formats)
//   sc  = rowsum(sq)   (matmul with the all-ones tile: every column of row r = row r's sum)
//   fac = sqrt(V) * rsqrt(sc + V*eps)          (== 1/sqrt(mean + eps), see the factory)
//   of  = fp32(ot[ct]) ; xn = of * fac         (per-row broadcast of the factor)
//   xw  = xn * norm_w                           (row-0 broadcast)
//   zs  = silu(fp32(z)) ; out = xw * zs         -> io dtype of z
//
// Compile args: {Vt, EPS_BITS, SCALE_BITS}. Runtime args: {n_inst, inst_start}.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tilize.h"
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

constexpr uint32_t cb_orm = 0, cb_z = 1, cb_w = 2, cb_ones = 3, cb_ot = 4;
constexpr uint32_t cb_sq = 5, cb_sc = 6, cb_fac = 7, cb_of = 8, cb_xn = 9;
constexpr uint32_t cb_wf = 10, cb_xw = 11, cb_zs = 12, cb_out = 13;

inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }
inline void POP(uint32_t cb, uint32_t n) { CircularBuffer(cb).pop_front(n); }

// o (1 tile) = copy of tile `idx` of `in` (format conversion on the copy).
void copy_one(uint32_t in, uint32_t idx, uint32_t o) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    tile_regs_acquire();
    copy_tile(in, idx, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

// out = A * B elementwise, n tiles (tile i of each).
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

// o[1 tile] = in[1,Vt] @ ones: every element [r,c] holds row r's sum.
void rowsum(uint32_t in, uint32_t o, uint32_t Vt) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(in, cb_ones);
    matmul_init(in, cb_ones, 0);
    tile_regs_acquire();
    for (uint32_t ki = 0; ki < Vt; ki++) {
        matmul_tiles(in, cb_ones, ki, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

// o = scale * rsqrt(in + eps), one tile (in and o distinct CBs).
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

// out (1 tile) = a * col broadcast along columns (row r scaled by col[r,0]).
void bcast_cols_mul1(uint32_t a, uint32_t col, uint32_t o) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, col);
    mul_bcast_cols_init(a, col);
    tile_regs_acquire();
    mul_tiles_bcast_cols(a, col, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

// out (1 tile) = a * row broadcast along rows (row 0 of `row`).
void bcast_rows_mul1(uint32_t a, uint32_t row, uint32_t o) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, row);
    mul_bcast_rows_init(a, row);
    tile_regs_acquire();
    mul_tiles_bcast_rows(a, row, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

// out (1 tile) = silu(in) (io -> fp32 on the copy).
void silu_one(uint32_t in, uint32_t o) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    silu_tile_init();
    tile_regs_acquire();
    copy_tile(in, 0, 0);
    silu_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Vt = get_compile_time_arg_val(0);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(1);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(2);

    const uint32_t n_inst = get_arg_val<uint32_t>(0);
    const uint32_t inst_start = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_orm, cb_z, cb_out);
    WAIT(cb_ones, 1);

    for (uint32_t it = 0; it < n_inst; ++it) {
        const uint32_t ct = (inst_start + it) % Vt;  // instance numbering is ct-fastest

        // ---- tilize the row-major 32 x V block ----
        WAIT(cb_orm, Vt);
        cb_reserve_back(cb_ot, Vt);
        tilize_init(cb_orm, Vt, cb_ot);
        tilize_block(cb_orm, Vt, cb_ot);
        tilize_uninit(cb_orm, cb_ot);
        cb_push_back(cb_ot, Vt);
        POP(cb_orm, Vt);
        WAIT(cb_ot, Vt);

        // ---- per-row rms factor from the whole row ----
        ew_mul(cb_ot, cb_ot, cb_sq, Vt);
        WAIT(cb_sq, Vt);
        rowsum(cb_sq, cb_sc, Vt);
        WAIT(cb_sc, 1);
        POP(cb_sq, Vt);
        inv_rms(cb_sc, cb_fac, EPS_BITS, SCALE_BITS);
        POP(cb_sc, 1);
        WAIT(cb_fac, 1);

        // ---- this column tile only: scale, weight, gate ----
        copy_one(cb_ot, ct, cb_of);
        POP(cb_ot, Vt);
        WAIT(cb_of, 1);
        bcast_cols_mul1(cb_of, cb_fac, cb_xn);
        WAIT(cb_xn, 1);
        POP(cb_of, 1);
        POP(cb_fac, 1);

        WAIT(cb_w, 1);
        copy_one(cb_w, 0, cb_wf);
        POP(cb_w, 1);
        WAIT(cb_wf, 1);
        bcast_rows_mul1(cb_xn, cb_wf, cb_xw);
        WAIT(cb_xw, 1);
        POP(cb_xn, 1);
        POP(cb_wf, 1);

        WAIT(cb_z, 1);
        silu_one(cb_z, cb_zs);
        POP(cb_z, 1);
        WAIT(cb_zs, 1);
        ew_mul(cb_xw, cb_zs, cb_out, 1);
        POP(cb_xw, 1);
        POP(cb_zs, 1);
    }
}
