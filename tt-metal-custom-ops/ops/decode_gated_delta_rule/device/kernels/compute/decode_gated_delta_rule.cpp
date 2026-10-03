// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel: the whole T=1 gated delta rule recurrent step for one head,
// fp32 accumulation throughout. Mirrors the python graph
// `recurrent_gated_delta_rule_decode_ttnn` (bf16/fp32 IO tiles, math in fp32):
//
//   qn = l2norm(q) * scale        (scale == K**-0.5; folded into the norm)
//   kn = l2norm(k)
//   h  = state * exp(g)           (per-head scalar g broadcast)
//   v_read = kn @ h               ([1,K]@[K,V] -> [1,V])
//   delta  = v - v_read           ([1,V])
//   outer  = (kn)^T @ (beta*delta)  ([K,1]@[1,V] rank-1)
//   new_h  = h + outer
//   o      = qn @ new_h           ([1,K]@[K,V] -> [1,V])
//
// All inputs arrive as row-0 tiles (reader gather). Padding rows are exact
// zeros, so the rank-1 outer product and row-broadcast norms are exact.
//
// Compile args: {Kt, Vt, has_s0, EPS_BITS, SCALE_BITS} (eps/scale fp32 bits).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4;
constexpr uint32_t cb_state = 5, cb_ones = 6;
constexpr uint32_t cb_qsq = 7, cb_ksq = 8;
constexpr uint32_t cb_sc = 9;    // 1 tile fp32 rowsum of squares
constexpr uint32_t cb_sc2 = 28;  // 1 tile fp32 q-chain inv-rms factor (dedicated)
constexpr uint32_t cb_sc3 = 29;  // 1 tile fp32 k-chain inv-rms factor (dedicated)
constexpr uint32_t cb_qn = 10, cb_kn = 11, cb_kcol = 12, cb_gexp = 13;
constexpr uint32_t cb_sdec = 14, cb_vread = 15, cb_delta = 16;
constexpr uint32_t cb_outer = 17, cb_sout = 18, cb_out = 19;
// fp32 mirrors of the io-dtype inputs (every math operand must be fp32:
// mixed bf16-srcA x fp32-srcB pairs corrupt the fp32 side) + fp32 new state.
constexpr uint32_t cb_qf = 20, cb_kf = 21, cb_vf = 22, cb_gf = 23;
constexpr uint32_t cb_betaf = 24, cb_sf = 25, cb_snew = 26;

inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }
inline void POP(uint32_t cb, uint32_t n) { CircularBuffer(cb).pop_front(n); }

// out[Mt,Nt] = A[Mt,Kt] @ (tr ? B[Nt,Kt]^T : B[Kt,Nt]). Inputs must be available.
void mm(uint32_t a, uint32_t b, uint32_t o, uint32_t Mt, uint32_t Kt, uint32_t Nt, bool tr) {
    cb_reserve_back(o, Mt * Nt);
    pack_reconfig_data_format(o);
    // matmul_tiles(a,b): in0=a->srcA, in1=b->srcB (reconfig src formats to match).
    reconfig_data_format(a, b);
    matmul_init(a, b, tr ? 1 : 0);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            for (uint32_t ki = 0; ki < Kt; ki++) {
                uint32_t bi = tr ? (ni * Kt + ki) : (ki * Nt + ni);
                matmul_tiles(a, b, mi * Kt + ki, bi, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    cb_push_back(o, Mt * Nt);
}

// out = A (op) B elementwise, n tiles. op: 0 add, 1 sub, 2 mul.
// o1[n] = o2[n] = a + b, packed twice from the same DST tile (o1 and o2 may differ in format).
void add2out(uint32_t a, uint32_t b, uint32_t o1, uint32_t o2, uint32_t n) {
    cb_reserve_back(o1, n);
    cb_reserve_back(o2, n);
    reconfig_data_format(a, b);
    add_init(a, b);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        add_tiles(a, b, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(o1);
        pack_tile(0, o1, i);
        pack_reconfig_data_format(o2);
        pack_tile(0, o2, i);
        tile_regs_release();
    }
    cb_push_back(o1, n);
    cb_push_back(o2, n);
}

void ew(uint32_t a, uint32_t b, uint32_t o, uint32_t n, int op) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);  // binary(a,b): a->srcA, b->srcB
    if (op == 0) {
        add_init(a, b);
    } else if (op == 1) {
        sub_init(a, b);
    } else {
        mul_init(a, b);
    }
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        if (op == 0) {
            add_tiles(a, b, i, i, 0);
        } else if (op == 1) {
            sub_tiles(a, b, i, i, 0);
        } else {
            mul_tiles(a, b, i, i, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// out = A[ia0+i] (op) B[ib0+i], n tiles (operands may be two page ranges of ONE CB).
void ew_off(uint32_t a, uint32_t ia0, uint32_t b, uint32_t ib0, uint32_t o, uint32_t n, int op) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);
    if (op == 0) {
        add_init(a, b);
    } else if (op == 1) {
        sub_init(a, b);
    } else {
        mul_init(a, b);
    }
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        if (op == 0) {
            add_tiles(a, b, ia0 + i, ib0 + i, 0);
        } else if (op == 1) {
            sub_tiles(a, b, ia0 + i, ib0 + i, 0);
        } else {
            mul_tiles(a, b, ia0 + i, ib0 + i, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// One contiguous 2n-tile block in `o`: tiles [0,n) = xn * w (elementwise, row-0 tiles), tiles
// [n,2n) = silu(zf). Reserved and pushed as ONE block so both halves are contiguous and a
// consumer may read tile n+i as operand B; `o` must be a 2n-page ring with aligned pointers.
void xw_zs_block(uint32_t xn, uint32_t w, uint32_t zf, uint32_t o, uint32_t n) {
    cb_reserve_back(o, 2 * n);
    pack_reconfig_data_format(o);
    reconfig_data_format(xn, w);
    mul_init(xn, w);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles(xn, w, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    reconfig_data_format_srca(zf);
    copy_tile_to_dst_init_short(zf);
    silu_tile_init();
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(zf, i, 0);
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, n + i);
        tile_regs_release();
    }
    cb_push_back(o, 2 * n);
}

// out = exp(in), n tiles (copy to fp32 DST, SFPU exp).
void expc(uint32_t in, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    exp_tile_init();
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in, i, 0);
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// out = silu(in), n tiles (copy to fp32 DST, SFPU silu).
void silu_tiles(uint32_t in, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    silu_tile_init();
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in, i, 0);
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// out = S * scalar[0,0], n tiles.
void bcast_scalar_mul(uint32_t a, uint32_t scal, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, scal);  // bcast(a,scal): a->srcA, scal->srcB
    mul_bcast_scalar_init(a, scal);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(a, scal, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// out[Mt,Nt] = A[Mt,Nt] * col[Mt,1] (broadcast col's single column across N).
void bcast_cols_mul(uint32_t a, uint32_t col, uint32_t o, uint32_t Mt, uint32_t Nt) {
    cb_reserve_back(o, Mt * Nt);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, col);  // bcast(a,col): a->srcA, col->srcB
    mul_bcast_cols_init(a, col);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(a, col, mi * Nt + ni, mi, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    cb_push_back(o, Mt * Nt);
}

// o[Vt] : tile j = ones * delta_j with delta's row 0 broadcast down every row, i.e.
// D'_j[i,c] = delta_j[0,c]. Only row 0 of delta is read.
void rowbcast_delta(uint32_t delta, uint32_t o, uint32_t Vt) {
    cb_reserve_back(o, Vt);
    pack_reconfig_data_format(o);
    reconfig_data_format(cb_ones, delta);  // bcast(a=ones, b=delta): a->srcA, b->srcB
    mul_bcast_rows_init(cb_ones, delta);
    for (uint32_t j = 0; j < Vt; j++) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(cb_ones, delta, 0, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, j);
        tile_regs_release();
    }
    cb_push_back(o, Vt);
}

// o[Kt,Vt] : tile (i,j) = D'_j * kcol_i's column 0 broadcast across columns, i.e.
// outer[i*32+r, j*32+c] = delta[0, j*32+c] * kn[0, i*32+r]. Only column 0 of kcol is read.
void outer_bcast(uint32_t dprime, uint32_t kcol, uint32_t o, uint32_t Kt, uint32_t Vt) {
    cb_reserve_back(o, Kt * Vt);
    pack_reconfig_data_format(o);
    reconfig_data_format(dprime, kcol);  // bcast(a=D', col=kcol): a->srcA, col->srcB
    mul_bcast_cols_init(dprime, kcol);
    for (uint32_t i = 0; i < Kt; i++) {
        for (uint32_t j = 0; j < Vt; j++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(dprime, kcol, j, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, i * Vt + j);
            tile_regs_release();
        }
    }
    cb_push_back(o, Kt * Vt);
}

// rowsum_k: o[1 tile] = in[1,Kt] @ ones — every element [r,c] of the output
// holds the row-r sum (row 0 = this head's sum of squares of `in`). Reuses the
// reader-built fp32 ones tile as the contraction operand.
void rowsum_k(uint32_t in, uint32_t o, uint32_t Kt) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(in, cb_ones);  // matmul(in, ones): in->srcA, ones->srcB
    matmul_init(in, cb_ones, 0);
    tile_regs_acquire();
    for (uint32_t ki = 0; ki < Kt; ki++) {
        matmul_tiles(in, cb_ones, ki, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

// inv_rms: o = rsqrt(in + eps) [* scale]. in holds per-row sums of squares
// (rowsum_k output); every column of a row carries the same factor, so the
// result feeds bcast_cols_mul directly. eps/scale arrive as fp32-bit-cast
// uint32 compile args. in (cb_sc) and o (cb_sc2) must be DISTINCT CBs: with
// in==out on the 2-page ring, the second norm chain's bcast consumed the
// first chain's factor (ttsim: gdn_decode_simdiag3.py swap experiment).
void inv_rms(uint32_t in, uint32_t o, uint32_t eps_bits, uint32_t scale_bits, bool do_scale) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    tile_regs_acquire();
    copy_tile(in, 0, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps_bits);  // + eps  (== python rms_norm(x, eps/K))
    rsqrt_tile_init();
    rsqrt_tile(0);  // 1/sqrt(sumsq + eps)
    if (do_scale) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scale_bits);  // * scale (q only; == python q*K**-0.5)
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

// o = copy of the n tiles of `in` (format conversion happens on the copy:
// unpack converts io->fp32 dest, packer converts fp32 dest->io).
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

// o[Kt,1] = transpose(in[1,Kt]): tile ki of o is the transposed tile ki of in
// (row 0 -> column 0), the column form needed for the rank-1 outer product.
void transpose_row(uint32_t in, uint32_t o, uint32_t Kt) {
    cb_reserve_back(o, Kt);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    transpose_init(in);
    for (uint32_t ki = 0; ki < Kt; ki++) {
        tile_regs_acquire();
        transpose_tile(in, ki, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, ki);
        tile_regs_release();
    }
    cb_push_back(o, Kt);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t has_s0 = get_compile_time_arg_val(2);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(3);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(4);
    // Fused norm + gate: gated = rms_norm(o) * norm_w * silu(z), on this head's row-0 tiles,
    // packed to cb_out in place of o. Operand rings reused (all dead once o exists):
    //   of = vread, sq = qsq, sum = sc, fac = sc2, xn = qn, xw = kn, zf = vf, zs = delta,
    //   z arrives in cb_v after v; norm_w in cb_w (30) -> fp32 cb_wf (31), once per core.
    constexpr uint32_t FNG = get_compile_time_arg_val(5);
    constexpr uint32_t NG_EPS_BITS = get_compile_time_arg_val(6);
    constexpr uint32_t NG_SCALE_BITS = get_compile_time_arg_val(7);
    constexpr uint32_t cb_w = 30, cb_wf = 31;
    (void)has_s0;

    constexpr uint32_t kv = Kt * Vt;

    // Runtime arg: instance count for this core (BH = B*H can exceed the grid;
    // each core loops over its contiguous instance chunk — the math below is
    // per-instance, CB-mediated against the reader/writer per instance).
    const uint32_t n_inst = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_q, cb_k, cb_out);

    if constexpr (FNG) {
        // norm_w arrives as a full-ring (2*Vt) block so cb_w's pointers stay block-aligned;
        // only the first Vt tiles carry data.
        WAIT(cb_w, 2 * Vt);
        copy_tiles(cb_w, cb_wf, Vt);
        POP(cb_w, 2 * Vt);
        WAIT(cb_wf, Vt);  // persistent front for the whole kernel
    }

    for (uint32_t it = 0; it < n_inst; ++it) {
        WAIT(cb_q, Kt);
        WAIT(cb_k, Kt);
        WAIT(cb_v, Vt);
        WAIT(cb_g, 1);
        WAIT(cb_beta, 1);
        WAIT(cb_state, kv);
        WAIT(cb_ones, 1);

        // ---- io -> fp32: every later operand is fp32 (mixed bf16 x fp32 srcA/srcB
        // pairs lose the fp32 side; all scratch math must be uniform fp32). ----
        copy_tiles(cb_q, cb_qf, Kt);
        copy_tiles(cb_k, cb_kf, Kt);
        copy_tiles(cb_v, cb_vf, Vt);
        copy_tiles(cb_g, cb_gf, 1);
        copy_tiles(cb_beta, cb_betaf, 1);
        // the state is NOT converted: the decay multiply reads cb_state directly (io x io)
        POP(cb_q, Kt);
        POP(cb_k, Kt);
        POP(cb_v, Vt);
        POP(cb_g, 1);
        POP(cb_beta, 1);

        // Read-back barrier discipline (the canonical layernorm pattern: every CB
        // page this kernel packed is wait_front'ed before this kernel reads it —
        // without the wait the unpacker can run ahead of the packer and read the
        // page's PREVIOUS contents; ttsim showed stale/zero norm factors).
        WAIT(cb_qf, Kt);
        WAIT(cb_kf, Kt);
        WAIT(cb_vf, Vt);
        WAIT(cb_gf, 1);
        WAIT(cb_betaf, 1);

        // ---- L2 norm q (scale folded): qn = q * scale / sqrt(||q||^2 + eps) ----
        ew(cb_qf, cb_qf, cb_qsq, Kt, 2);  // q^2
        WAIT(cb_qsq, Kt);
        rowsum_k(cb_qsq, cb_sc, Kt);  // [0,*] = ||q||^2
        WAIT(cb_sc, 1);
        POP(cb_qsq, Kt);
        inv_rms(cb_sc, cb_sc2, EPS_BITS, SCALE_BITS, true);
        POP(cb_sc, 1);    // drop the sumsq page
        WAIT(cb_sc2, 1);  // packer drain before the read-back
        bcast_cols_mul(cb_qf, cb_sc2, cb_qn, 1, Kt);
        WAIT(cb_qn, Kt);
        POP(cb_qf, Kt);
        POP(cb_sc2, 1);

        // ---- L2 norm k (no scale) ----
        ew(cb_kf, cb_kf, cb_ksq, Kt, 2);
        WAIT(cb_ksq, Kt);
        rowsum_k(cb_ksq, cb_sc, Kt);
        WAIT(cb_sc, 1);
        POP(cb_ksq, Kt);
        inv_rms(cb_sc, cb_sc3, EPS_BITS, SCALE_BITS, false);
        POP(cb_sc, 1);
        WAIT(cb_sc3, 1);  // packer drain before the read-back
        bcast_cols_mul(cb_kf, cb_sc3, cb_kn, 1, Kt);
        WAIT(cb_kn, Kt);
        POP(cb_kf, Kt);
        POP(cb_sc3, 1);

        // ---- decay: h = state * exp(g) ----
        expc(cb_gf, cb_gexp, 1);  // [0,0] = exp(g_h), packed in the io format (cb_gexp is df_io)
        WAIT(cb_gexp, 1);
        POP(cb_gf, 1);
        bcast_scalar_mul(cb_state, cb_gexp, cb_sdec, kv);  // io x io -> fp32 (uniform srcA/srcB)
        WAIT(cb_sdec, kv);
        POP(cb_state, kv);
        POP(cb_gexp, 1);

        // ---- v_read = kn @ h; delta = v - v_read; delta *= beta ----
        mm(cb_kn, cb_sdec, cb_vread, 1, Kt, Vt, false);
        WAIT(cb_vread, Vt);
        ew(cb_vf, cb_vread, cb_delta, Vt, 1);  // delta = v - v_read
        WAIT(cb_delta, Vt);
        POP(cb_vf, Vt);
        POP(cb_vread, Vt);
        bcast_scalar_mul(cb_delta, cb_betaf, cb_delta, Vt);  // in-place (2-page CB)
        POP(cb_betaf, 1);
        POP(cb_delta, Vt);  // drop the pre-beta page

        // ---- rank-1 write: new_h = h + (kn)^T (x) (beta*delta) ----
        // As two broadcasts rather than a [K,1]@[1,V] matmul: the matmul contracts over the
        // 32 rows of its operands and so needed rows 1..31 of the gathered tiles to be exact
        // zeros (17 us of reader zeroing per layer). D'[i,j] = delta[0,j] via a row-0 broadcast
        // against the all-ones tile, then outer[i,j] = D'[i,j] * kcol[i,0] via a column-0
        // broadcast: only the wanted row and column are read. One product per element, so the
        // result is the matmul's to the bit.
        transpose_row(cb_kn, cb_kcol, Kt);  // kn -> column form (column 0 is what matters)
        WAIT(cb_kcol, Kt);
        POP(cb_kn, Kt);
        rowbcast_delta(cb_delta, cb_vread, Vt);  // D'_j = delta row 0, every row (reuses vread's ring)
        WAIT(cb_vread, Vt);
        POP(cb_delta, Vt);
        outer_bcast(cb_vread, cb_kcol, cb_outer, Kt, Vt);  // outer[i,j] = D'_j * kcol_i[:,0]
        WAIT(cb_outer, kv);
        POP(cb_vread, Vt);
        POP(cb_kcol, Kt);
        add2out(cb_sdec, cb_outer, cb_snew, cb_sout, kv);  // new state: fp32 (o matmul) + io (writer)
        WAIT(cb_snew, kv);
        POP(cb_sdec, kv);
        POP(cb_outer, kv);

        if constexpr (FNG) {
            // ---- o (fp32) -> rms_norm * norm_w * silu(z) -> cb_out (io) ----
            mm(cb_qn, cb_snew, cb_vread, 1, Kt, Vt, false);  // of = o, fp32
            POP(cb_qn, Kt);
            WAIT(cb_vread, Vt);
            ew(cb_vread, cb_vread, cb_qsq, Vt, 2);  // o^2
            WAIT(cb_qsq, Vt);
            rowsum_k(cb_qsq, cb_sc, Vt);  // [0,*] = sum o^2
            WAIT(cb_sc, 1);
            POP(cb_qsq, Vt);
            inv_rms(cb_sc, cb_sc2, NG_EPS_BITS, NG_SCALE_BITS, true);  // sqrt(V)/sqrt(sum + V*eps)
            POP(cb_sc, 1);
            WAIT(cb_sc2, 1);
            bcast_cols_mul(cb_vread, cb_sc2, cb_qn, 1, Vt);  // xn = o * factor
            WAIT(cb_qn, Vt);
            POP(cb_vread, Vt);
            POP(cb_sc2, 1);
            // The composed chain rounds to io (bf16) after rms_norm*w, after silu(z) and after the
            // multiply; do the same so the fold reproduces its numerics to accumulation order.
            // Both rounded intermediates live in cb_w (io, 2*Vt pages), which after the one-time
            // norm_w read is produced and consumed by this kernel alone -- NOT in z's ring, where
            // the reader's next-instance v push races the compute's own pushes (a deadlock the
            // first attempt found), and not in cb_out, whose only consumer is the writer.
            WAIT(cb_v, Vt);  // z (gathered after v)
            copy_tiles(cb_v, cb_vf, Vt);  // zf fp32
            POP(cb_v, Vt);
            WAIT(cb_vf, Vt);
            // Round each intermediate to io (bf16) with a copy round-trip through cb_w (one
            // Vt-page block at a time; two per head == the ring, so the pointers stay aligned),
            // then multiply as fp32 x fp32 from two distinct rings exactly as the unrounded fold
            // did. Every binary op reads index 0 of two different CBs -- the proven pattern.
            ew(cb_qn, cb_wf, cb_kn, Vt, 2);  // xw = xn * norm_w (fp32)
            WAIT(cb_kn, Vt);
            POP(cb_qn, Vt);
            copy_tiles(cb_kn, cb_w, Vt);  // -> bf16
            WAIT(cb_w, Vt);
            POP(cb_kn, Vt);
            copy_tiles(cb_w, cb_kn, Vt);  // -> fp32 again, now bf16-valued
            WAIT(cb_kn, Vt);
            POP(cb_w, Vt);
            silu_tiles(cb_vf, cb_delta, Vt);  // zs = silu(z) (fp32)
            WAIT(cb_delta, Vt);
            POP(cb_vf, Vt);
            copy_tiles(cb_delta, cb_w, Vt);  // -> bf16
            WAIT(cb_w, Vt);
            POP(cb_delta, Vt);
            copy_tiles(cb_w, cb_delta, Vt);  // -> fp32 again, bf16-valued
            WAIT(cb_delta, Vt);
            POP(cb_w, Vt);
            ew(cb_kn, cb_delta, cb_out, Vt, 2);  // gated = bf16(xw * zs) -> io
            POP(cb_kn, Vt);
            POP(cb_delta, Vt);
        } else {
            // ---- o = qn @ new_h (fp32 x fp32; packed straight to io) ----
            mm(cb_qn, cb_snew, cb_out, 1, Kt, Vt, false);
            POP(cb_qn, Kt);
        }

        // (new state already packed to cb_sout for the writer)
        POP(cb_snew, kv);
    }
}
