// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// T fused Gated DeltaNet recurrent decode steps in one dispatch.
//
// Per token:
//   h1     = state * exp(g)      c_11, c_1 -> c_2   (g is ONE scalar tile; the exp
//                                                    happens here, on that tile alone)
//   v_read = k @ h1              c_3, c_2  -> c_7
//   delta  = (v - v_read) * beta c_4, c_7, c_5 -> c_8   (beta is ONE scalar tile,
//                                                        SCALAR-broadcast over c_13)
//   kT     = transpose(k)        c_3 -> c_9
//   state' = h1 + kT @ delta     c_9, c_8, c_2 -> c_10 -> c_17 (out) and c_11 (next iteration)
//   o      = q @ state'          c_6, c_11 -> c_16
//
// Every intermediate state is emitted on c_17, which is what speculative rollback needs: verify
// advances by T but commits n.
//
// Two rules this kernel is built around, both learned the hard way:
//   * a CB pushed to a *dataflow* consumer must not be modified afterwards -- hence the separate
//     c_10 for the matmul result, and hence `o = q @ state'` reading c_11 rather than c_17.
//   * c_11 holds two slots so the new state can be reserved while the old is still the front.
//
// Runtime args: 0 Kt, 1 Vt, 2 Mt, 3 T

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/dataflow/circular_buffer.h"

// Matmul maps in0 -> SrcB and in1 -> SrcA, hence the reversed source order.
static inline void matmul_reconfig_and_init(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb) {
    reconfig_data_format<SrcOrder::Reverse>(in0_cb, in1_cb);
    matmul_init(in0_cb, in1_cb);
    pack_reconfig_data_format(out_cb);
}

void kernel_main() {
    const uint32_t Kt = get_arg_val<uint32_t>(0);
    const uint32_t Vt = get_arg_val<uint32_t>(1);
    const uint32_t Mt = get_arg_val<uint32_t>(2);
    const uint32_t T = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_h0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_g = tt::CBIndex::c_1;
    constexpr uint32_t cb_h1 = tt::CBIndex::c_2;
    constexpr uint32_t cb_k = tt::CBIndex::c_3;
    constexpr uint32_t cb_v = tt::CBIndex::c_4;
    constexpr uint32_t cb_b = tt::CBIndex::c_5;
    constexpr uint32_t cb_q = tt::CBIndex::c_6;
    constexpr uint32_t cb_vr = tt::CBIndex::c_7;
    constexpr uint32_t cb_d = tt::CBIndex::c_8;
    constexpr uint32_t cb_kt = tt::CBIndex::c_9;
    constexpr uint32_t cb_upd = tt::CBIndex::c_10;
    constexpr uint32_t cb_st = tt::CBIndex::c_11;
    constexpr uint32_t cb_eg = tt::CBIndex::c_12;
    constexpr uint32_t cb_dt = tt::CBIndex::c_13;
    constexpr uint32_t cb_o = tt::CBIndex::c_16;
    constexpr uint32_t cb_sout = tt::CBIndex::c_17;

    CircularBuffer o_h0(cb_h0), o_g(cb_g), o_h1(cb_h1), o_k(cb_k), o_v(cb_v), o_b(cb_b);
    CircularBuffer o_q(cb_q), o_vr(cb_vr), o_d(cb_d), o_kt(cb_kt), o_upd(cb_upd);
    CircularBuffer o_st(cb_st), o_o(cb_o), o_sout(cb_sout);
    CircularBuffer o_eg(cb_eg), o_dt(cb_dt);

    const uint32_t st = Kt * Vt, ktile = Mt * Kt, vtile = Mt * Vt;

    // Seed the running state from the initial h.
    o_h0.wait_front(st);
    o_st.reserve_back(st);
    unary_op_init_common(cb_h0, cb_st);
    copy_tile_to_dst_init_short(cb_h0);
    for (uint32_t t = 0; t < st; t++) {
        tile_regs_acquire();
        copy_tile(cb_h0, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_st, t);
        tile_regs_release();
    }
    o_st.push_back(st);
    o_h0.pop_front(st);
    o_st.wait_front(st);

    for (uint32_t step = 0; step < T; step++) {
        // ---- h1 = state * exp(g), from ONE scalar tile ----
        // One SFPU exp per token instead of Kt*Vt. exp() turns tilize's zero padding into 1.0
        // in the 1023 lanes we do not use; that is inert because SCALAR broadcast reads only
        // SrcB[0][0] and replicates it (llk_unpack_AB.h issues unpack_srcb once as the MOP
        // start op; llk_math_eltwise_binary.h selects p_elwise::SRCB_BCAST_ALL).
        // c_12 is compute-private -- pushed, waited, used and popped within one iteration --
        // so this does not violate the never-modify-a-CB-after-push_back rule.
        o_g.wait_front(1);
        o_eg.reserve_back(1);
        unary_op_init_common(cb_g, cb_eg);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_g);
        copy_tile(cb_g, 0, 0);
        exp_tile_init<false>();
        exp_tile<false>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_eg, 0);
        tile_regs_release();
        o_eg.push_back(1);
        o_eg.wait_front(1);
        o_h1.reserve_back(st);
        // Full init FIRST (it retargets the packer and does the unpack/math hw_configure),
        // THEN the short-form bcast init, which overwrites the MOP with SCALAR. Reversed,
        // binary_op_init_common's trailing llk_unpack_AB_init<NONE> clobbers it and the
        // result is a full-tile multiply against a mostly-1.0 tile -- a silent wrong answer.
        binary_op_init_common(cb_st, cb_eg, cb_h1);
        mul_bcast_scalar_init(cb_st, cb_eg);
        for (uint32_t t = 0; t < st; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_st, cb_eg, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_h1, t);
            tile_regs_release();
        }
        o_h1.push_back(st);
        o_h1.wait_front(st);
        o_eg.pop_front(1);
        o_g.pop_front(1);

        // ---- v_read = k @ h1 ----
        o_k.wait_front(ktile);
        o_vr.reserve_back(vtile);
        matmul_reconfig_and_init(cb_k, cb_h1, cb_vr);
        for (uint32_t m = 0; m < Mt; m++) {
            for (uint32_t j = 0; j < Vt; j++) {
                tile_regs_acquire();
                for (uint32_t i = 0; i < Kt; i++) {
                    matmul_tiles(cb_k, cb_h1, m * Kt + i, i * Vt + j, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_vr, m * Vt + j);
                tile_regs_release();
            }
        }
        o_vr.push_back(vtile);
        o_vr.wait_front(vtile);

        // ---- delta = (v - v_read) * beta ----
        o_v.wait_front(vtile);
        o_b.wait_front(1);  // beta: one scalar tile
        o_dt.reserve_back(vtile);
        binary_op_init_common(cb_v, cb_vr, cb_dt);
        sub_init(cb_v, cb_vr);
        for (uint32_t t = 0; t < vtile; t++) {
            tile_regs_acquire();
            sub_tiles(cb_v, cb_vr, t, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_dt, t);
            tile_regs_release();
        }
        o_dt.push_back(vtile);
        o_dt.wait_front(vtile);
        o_d.reserve_back(vtile);
        binary_op_init_common(cb_dt, cb_b, cb_d);
        mul_bcast_scalar_init(cb_dt, cb_b);
        for (uint32_t t = 0; t < vtile; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_dt, cb_b, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_d, t);
            tile_regs_release();
        }
        o_d.push_back(vtile);
        o_d.wait_front(vtile);
        o_dt.pop_front(vtile);
        o_v.pop_front(vtile);
        o_b.pop_front(1);
        o_vr.pop_front(vtile);

        // ---- kT = transpose(k) ----
        o_kt.reserve_back(Kt * Mt);
        for (uint32_t i = 0; i < Kt; i++) {
            for (uint32_t m = 0; m < Mt; m++) {
                tile_regs_acquire();
                transpose_wh_init_short(cb_k);
                transpose_wh_tile(cb_k, m * Kt + i, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_kt, i * Mt + m);
                tile_regs_release();
            }
        }
        o_kt.push_back(Kt * Mt);
        o_kt.wait_front(Kt * Mt);
        o_k.pop_front(ktile);

        // ---- upd = kT @ delta ----
        o_upd.reserve_back(st);
        matmul_reconfig_and_init(cb_kt, cb_d, cb_upd);
        for (uint32_t i = 0; i < Kt; i++) {
            for (uint32_t j = 0; j < Vt; j++) {
                tile_regs_acquire();
                for (uint32_t m = 0; m < Mt; m++) {
                    matmul_tiles(cb_kt, cb_d, i * Mt + m, m * Vt + j, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_upd, i * Vt + j);
                tile_regs_release();
            }
        }
        o_upd.push_back(st);
        o_upd.wait_front(st);

        // ---- state' = h1 + upd, packed to BOTH the writer's CB and the running state ----
        o_sout.reserve_back(st);
        o_st.reserve_back(st);
        binary_op_init_common(cb_h1, cb_upd, cb_sout);
        add_init(cb_h1, cb_upd);
        for (uint32_t t = 0; t < st; t++) {
            tile_regs_acquire();
            add_tiles(cb_h1, cb_upd, t, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_sout, t);
            pack_tile(0, cb_st, t);
            tile_regs_release();
        }
        o_sout.push_back(st);
        o_st.push_back(st);
        o_st.pop_front(st);   // drop the previous state; the new one is now the front
        o_st.wait_front(st);
        o_upd.pop_front(st);
        o_h1.pop_front(st);
        o_kt.pop_front(Kt * Mt);
        o_d.pop_front(vtile);

        // ---- o = q @ state'  (from c_11, which is compute-private) ----
        o_q.wait_front(ktile);
        o_o.reserve_back(vtile);
        matmul_reconfig_and_init(cb_q, cb_st, cb_o);
        for (uint32_t m = 0; m < Mt; m++) {
            for (uint32_t j = 0; j < Vt; j++) {
                tile_regs_acquire();
                for (uint32_t i = 0; i < Kt; i++) {
                    matmul_tiles(cb_q, cb_st, m * Kt + i, i * Vt + j, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_o, m * Vt + j);
                tile_regs_release();
            }
        }
        o_o.push_back(vtile);
        o_q.pop_front(ktile);
    }
}
