// gdn_seq_block compute (K5-A). See gdn_seq_block.py and the K5 plan, section 3.
//
// NOT a standalone file. gdn_seq_block.generate() emits the served native compute prefix first
// (decode_gated_delta_rule.cpp up to its entry point, sha256-checked against
// gdn_multitoken.HASHES): its includes, its CB names and its helpers copy_tiles, ew, ew_off, expc,
// silu_tiles, bcast_scalar_mul, bcast_cols_mul, rowsum_k, inv_rms and rowbcast_delta, verbatim.
// This file follows it, with the build line below replaced by the generated build constants.
//
// One core runs one (user, head) for a whole T=16 block. The recurrence chain stays sequential,
// token by token, with the served LLK calls on the served operand values in the served
// srcA/srcB roles and K order; what is batched is the work around it:
//   prologue  (once) the bf16 -> fp32 copies, exp(g) over the whole g tile, and the q/k L2 norms
//             over all 16 rows at once (every one of those ops is row-independent);
//   chain     (16x) T1 state copy, T2 decay, T3 v_read / delta / beta, T4 D', T5 rank-1 update
//             with the state add in DST (DEST_TO_SRCB, gdn_outer_add.py:7-31), T6 one copy packed
//             twice (snapshot + bf16 feedback, gdn_dual_state_copy.py), T7 o = q~ @ h_new;
//   epilogue  (once) rms_norm(o) * w * silu(z) over the 16 rows the writer assembled into O.
// The reader stages each token's row operands (TOKA, TOKB) as bit copies of the fp32 words the
// prologue packed; nothing is re-read from DRAM per token.
//
// State CBs are column-major: tile (i, j) (K tile i, V tile j) sits at page 4j + i, so a column
// is four contiguous pages and SOUT can be pushed and written per column. The served kernel
// holds tile (i, j) at 4i + j; each per-tile op below names the same tile (i, j) it does.
//
// Every CB has ONE producer RISC and ONE consumer RISC (gdn_seq_block.CB_PLAN), with no
// exception: norm_w arrives in W (reader -> compute, read once, in the epilogue), and the
// epilogue's two bf16 round trips go through RT, a ring this kernel alone produces and consumes.
// (The served kernel reuses its norm_w ring for them, compute.cpp:540-552; there the packer's
// local count never saw the reader's push, so the second trip's wait could not block.)
// ONES stays at index 6: rowsum_k and rowbcast_delta hard-code cb_ones.
//
// Compile-time args: {EPS_BITS, SCALE_BITS, NG_EPS_BITS, NG_SCALE_BITS, LEVEL, SRC_TAG}.
// No runtime args are read.

// @@GDN_SEQ_BLOCK_BUILD@@

#if GDN_SEQ_BLOCK_VARIANT == 2
#include "api/compute/eltwise_binary_sfpu.h"
#endif

namespace {

constexpr uint32_t SB_IN_QK = 0, SB_IN_V = 1, SB_IN_Z = 2, SB_IN_GB = 3, SB_S0 = 4, SB_FB = 5;
constexpr uint32_t SB_ONES = 6;
constexpr uint32_t SB_SOUT = 7, SB_W = 8, SB_OUT = 9;
constexpr uint32_t SB_X = 10, SB_QKN = 11, SB_VF = 12, SB_ZF = 13, SB_GF = 14, SB_BF = 15, SB_GEXP = 16;
constexpr uint32_t SB_WF = 17, SB_NSQ = 18, SB_SUM = 19, SB_FAC_Q = 20, SB_FAC_K = 21;
constexpr uint32_t SB_TOKA = 22, SB_TOKB = 23, SB_S = 24, SB_H = 25, SB_UD = 26, SB_DL = 27;
constexpr uint32_t SB_OT = 28, SB_O = 29, SB_RT = 30;
constexpr uint32_t SB_OUTER = 31;  // probe build A0 only
static_assert(SB_ONES == cb_ones, "rowsum_k and rowbcast_delta read the ones tile at cb_ones");

constexpr uint32_t SB_T = 16;   // tokens (rows) per block
constexpr uint32_t SB_KT = 4;   // K tiles
constexpr uint32_t SB_VT = 4;   // V tiles
constexpr uint32_t SB_KV = SB_KT * SB_VT;

// TOKA: [a = exp(g)[t,h] at [0,0], beta[t,h] at [0,0], v row t x4 (row 0), k~ row t x4 (row 0)].
constexpr uint32_t TOKA_A = 0, TOKA_BETA = 1, TOKA_V = 2, TOKA_K = 6, TOKA_PAGES = 10;
// TOKB: [k~ row t as column 0 x4, q~ row t x4 (row 0)].
constexpr uint32_t TOKB_KCOL = 0, TOKB_Q = 4, TOKB_PAGES = 8;
static_assert(TOKA_A == 0, "the served bcast_scalar_mul reads its scalar at page 0");
static_assert(TOKB_KCOL == 0, "the T5 helpers read kcol_i at page i");

constexpr uint32_t SB_VARIANT_A = 0, SB_VARIANT_A0 = 1, SB_VARIANT_N = 2;
constexpr uint32_t SB_DIAG_NONE = 0, SB_DIAG_NOSNAP = 1, SB_DIAG_PASSTHROUGH = 2;
constexpr uint32_t SB_LEVEL = GDN_SEQ_BLOCK_LEVEL;
constexpr uint32_t SB_VARIANT = GDN_SEQ_BLOCK_VARIANT;
constexpr uint32_t SB_DIAG = GDN_SEQ_BLOCK_DIAG;
static_assert(SB_VARIANT <= SB_VARIANT_N && SB_DIAG <= SB_DIAG_PASSTHROUGH, "unknown build");
static_assert(SB_LEVEL == 0, "no A+ increment is implemented; level 0 only");

// mm (compute.cpp:57-77) with an A page offset and a column-major B: o[j] = sum_ki A[a0+ki] @ B(ki, j),
// B(ki, j) at page 4j + ki. One DST per column, the served K order (ki = 0..3). T3a [472], T7 [506].
void mm_cols(uint32_t a, uint32_t a0, uint32_t b, uint32_t o) {
    cb_reserve_back(o, SB_VT);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);
    matmul_init(a, b, 0);
    for (uint32_t j = 0; j < SB_VT; j++) {
        tile_regs_acquire();
        for (uint32_t ki = 0; ki < SB_KT; ki++) {
            matmul_tiles(a, b, a0 + ki, j * SB_KT + ki, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, j);
        tile_regs_release();
    }
    cb_push_back(o, SB_VT);
}

// bcast_scalar_mul (compute.cpp:207-221) with the scalar at page `s` of `scal`. T3c [478].
void bcast_scalar_mul_at(uint32_t a, uint32_t scal, uint32_t s, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, scal);
    mul_bcast_scalar_init(a, scal);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(a, scal, i, s, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// The served L2 norm (compute.cpp:434-446 for q, 448-460 for k), its helpers verbatim, over the
// front SB_KT pages of `x` - all 16 token rows at once: every op in it is row-independent
// (elementwise, rowsum_k's matmul against ones, SFPU rsqrt, the column-0 broadcast). The served
// WAIT on the output is dropped: its consumer is the reader, not this kernel. The caller pops x.
void l2_norm_rows(uint32_t x, uint32_t fac, uint32_t o, uint32_t eps_bits, uint32_t scale_bits, bool do_scale) {
    ew(x, x, SB_NSQ, SB_KT, 2);  // x^2
    WAIT(SB_NSQ, SB_KT);
    rowsum_k(SB_NSQ, SB_SUM, SB_KT);  // [r,*] = ||x_r||^2
    WAIT(SB_SUM, 1);
    POP(SB_NSQ, SB_KT);
    inv_rms(SB_SUM, fac, eps_bits, scale_bits, do_scale);
    POP(SB_SUM, 1);
    WAIT(fac, 1);  // packer drain before the read-back
    bcast_cols_mul(x, fac, o, 1, SB_KT);
    POP(fac, 1);
}

// T5, served [495] + [499] with the add in DST: per column j, windows of two tiles
// (gdn_outer_add.HELPER's arithmetic): DST[s] = D'_j * kcol_i[:,0] (mul_tiles_bcast_cols,
// srcA = D', srcB = kcol), then DST[s] = H(i,j) + DST[s] (DEST_TO_SRCB: DST -> srcB, H -> srcA,
// the served ew(sdec, outer) roles), packed to the new state. Pushed per column.
void outer_add_cols(uint32_t dprime, uint32_t kcol, uint32_t state, uint32_t o) {
    pack_reconfig_data_format(o);
    for (uint32_t j = 0; j < SB_VT; j++) {
        cb_reserve_back(o, SB_KT);
        for (uint32_t first = 0; first < SB_KT; first += 2) {
            reconfig_data_format(dprime, kcol);
            mul_bcast_cols_init(dprime, kcol);
            tile_regs_acquire();
            for (uint32_t slot = 0; slot < 2; ++slot) {
                mul_tiles_bcast_cols(dprime, kcol, j, first + slot, slot);
            }
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(state);
            for (uint32_t slot = 0; slot < 2; ++slot) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(state, j * SB_KT + first + slot, slot);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t slot = 0; slot < 2; ++slot) {
                pack_tile(slot, o, first + slot);
            }
            tile_regs_release();
        }
        cb_push_back(o, SB_KT);
    }
}

// Probe build A0, T5 unfused: outer_bcast (compute.cpp:262-278) in the column-major order, so
// that page 4j + i of `o` is tile (i, j) and ew(H, OUTER) adds matching tiles as [499] does.
void outer_cols(uint32_t dprime, uint32_t kcol, uint32_t o) {
    cb_reserve_back(o, SB_KV);
    pack_reconfig_data_format(o);
    reconfig_data_format(dprime, kcol);
    mul_bcast_cols_init(dprime, kcol);
    for (uint32_t j = 0; j < SB_VT; j++) {
        for (uint32_t i = 0; i < SB_KT; i++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(dprime, kcol, j, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, j * SB_KT + i);
            tile_regs_release();
        }
    }
    cb_push_back(o, SB_KV);
}

#if GDN_SEQ_BLOCK_VARIANT == 2
// Negative control N, T5 with the state add on the SFPU (the rejected candidate of
// docs/gdn-outer-add-experiment.md:6-17): H(i,j) copied into DST beside the product and added
// there in fp32. Known not to reproduce the FPU add; P0 must see it.
void outer_add_sfpu_cols(uint32_t dprime, uint32_t kcol, uint32_t state, uint32_t o) {
    pack_reconfig_data_format(o);
    for (uint32_t j = 0; j < SB_VT; j++) {
        cb_reserve_back(o, SB_KT);
        for (uint32_t i = 0; i < SB_KT; i++) {
            reconfig_data_format(dprime, kcol);
            mul_bcast_cols_init(dprime, kcol);
            tile_regs_acquire();
            mul_tiles_bcast_cols(dprime, kcol, j, i, 0);
            reconfig_data_format_srca(state);
            copy_tile_to_dst_init_short(state);
            copy_tile(state, j * SB_KT + i, 1);
            add_binary_tile_init();
            add_binary_tile(1, 0, 0);  // state + outer, the served operand order
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, i);
            tile_regs_release();
        }
        cb_push_back(o, SB_KT);
    }
}
#endif

// T6: the served copy_tiles(snew, sout) [563] and copy_tiles(snew, feedback)
// (gdn_multitoken.py:81-82) as ONE unpack packed twice (gdn_dual_state_copy.py). Per column.
void state_out(uint32_t s, uint32_t out, uint32_t feedback, bool more) {
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(s);
    copy_tile_to_dst_init_short(s);
    for (uint32_t j = 0; j < SB_VT; j++) {
        cb_reserve_back(out, SB_KT);
        if (more) {
            cb_reserve_back(feedback, SB_KT);
        }
        for (uint32_t i = 0; i < SB_KT; i++) {
            tile_regs_acquire();
            copy_tile(s, j * SB_KT + i, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out, i);
            if (more) {
                pack_tile(0, feedback, i);
            }
            tile_regs_release();
        }
        cb_push_back(out, SB_KT);
        if (more) {
            cb_push_back(feedback, SB_KT);
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(0);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(1);
    constexpr uint32_t NG_EPS_BITS = get_compile_time_arg_val(2);
    constexpr uint32_t NG_SCALE_BITS = get_compile_time_arg_val(3);
    constexpr uint32_t LEVEL = get_compile_time_arg_val(4);
    constexpr uint32_t SRC_TAG = get_compile_time_arg_val(5);  // keys the JIT cache on content
    static_assert(LEVEL == SB_LEVEL, "LEVEL compile arg must equal the generated level");
    (void)SRC_TAG;

    compute_kernel_hw_startup(SB_IN_QK, SB_IN_V, SB_OUT);

    // ---- prologue, once per block (norm_w waits for the epilogue: the reader sends it last) ----
    // g and beta, whole tiles: element (t, h) is token t's scalar for head h.
    WAIT(SB_IN_GB, 2);
    copy_tiles(SB_IN_GB, SB_GF, 1);
    POP(SB_IN_GB, 1);
    copy_tiles(SB_IN_GB, SB_BF, 1);  // -> reader
    POP(SB_IN_GB, 1);
    WAIT(SB_GF, 1);
    expc(SB_GF, SB_GEXP, 1);  // [t,h] = exp(g[t,h]) [463]; -> reader
    POP(SB_GF, 1);
    WAIT(SB_IN_V, SB_VT);
    copy_tiles(SB_IN_V, SB_VF, SB_VT);  // [412]; -> reader
    POP(SB_IN_V, SB_VT);
    WAIT(SB_IN_Z, SB_VT);
    copy_tiles(SB_IN_Z, SB_ZF, SB_VT);  // [528], for the epilogue
    POP(SB_IN_Z, SB_VT);
    WAIT(SB_IN_QK, 2 * SB_KT);
    copy_tiles(SB_IN_QK, SB_X, 2 * SB_KT);  // q | k [410-411]
    POP(SB_IN_QK, 2 * SB_KT);
    WAIT(SB_X, 2 * SB_KT);
    WAIT(SB_ONES, 1);
    l2_norm_rows(SB_X, SB_FAC_Q, SB_QKN, EPS_BITS, SCALE_BITS, true);  // q~ -> QKN pages 0-3 [434-446]
    POP(SB_X, SB_KT);
    l2_norm_rows(SB_X, SB_FAC_K, SB_QKN, EPS_BITS, SCALE_BITS, false);  // k~ -> QKN pages 4-7 [448-460]
    POP(SB_X, SB_KT);

    // ---- the chain, token by token ----
    for (uint32_t t = 0; t < SB_T; ++t) {
        const uint32_t src = t == 0 ? SB_S0 : SB_FB;
        const bool more = t + 1 < SB_T;

        // T1: bf16 state -> fp32 [415].
        WAIT(src, SB_KV);
        copy_tiles(src, SB_S, SB_KV);
        POP(src, SB_KV);
        WAIT(SB_S, SB_KV);
        WAIT(SB_TOKA, TOKA_PAGES);

        if constexpr (SB_DIAG == SB_DIAG_PASSTHROUGH) {
            // Diagnostic only: the chain's math replaced by pass-through (bytes-bound timing).
            POP(SB_TOKA, TOKA_PAGES);
            WAIT(SB_TOKB, TOKB_PAGES);
            state_out(SB_S, SB_SOUT, SB_FB, more);
            copy_tiles(SB_TOKB, SB_OT, SB_VT);
            POP(SB_S, SB_KV);
            POP(SB_TOKB, TOKB_PAGES);
            continue;
        }

        // T2: h = S * exp(g) [466] (TOKA page 0 holds the scalar at [0,0]).
        bcast_scalar_mul(SB_S, SB_TOKA, SB_H, SB_KV);
        WAIT(SB_H, SB_KV);
        POP(SB_S, SB_KV);
        // T3a: v_read = k~ @ h [472].
        mm_cols(SB_TOKA, TOKA_K, SB_H, SB_UD);
        WAIT(SB_UD, SB_VT);
        // T3b: delta = v - v_read [474] (DL pages 0-3).
        ew_off(SB_TOKA, TOKA_V, SB_UD, 0, SB_DL, SB_VT, 1);
        WAIT(SB_DL, SB_VT);
        POP(SB_UD, SB_VT);
        // T3c: delta *= beta [478] (DL pages 4-7, the served in-place ring).
        bcast_scalar_mul_at(SB_DL, SB_TOKA, TOKA_BETA, SB_DL, SB_VT);
        POP(SB_TOKA, TOKA_PAGES);
        POP(SB_DL, SB_VT);  // drop the pre-beta pages
        WAIT(SB_DL, SB_VT);
        // T4: D'_j = delta row 0 broadcast down every row [492].
        rowbcast_delta(SB_DL, SB_UD, SB_VT);
        WAIT(SB_UD, SB_VT);
        POP(SB_DL, SB_VT);
        // T5: h_new = h + k~^T (x) D' [495, 499].
        WAIT(SB_TOKB, TOKB_PAGES);
        if constexpr (SB_VARIANT == SB_VARIANT_A0) {
            outer_cols(SB_UD, SB_TOKB, SB_OUTER);
            WAIT(SB_OUTER, SB_KV);
            ew(SB_H, SB_OUTER, SB_S, SB_KV, 0);
            POP(SB_OUTER, SB_KV);
        } else if constexpr (SB_VARIANT == SB_VARIANT_N) {
#if GDN_SEQ_BLOCK_VARIANT == 2
            outer_add_sfpu_cols(SB_UD, SB_TOKB, SB_H, SB_S);
#endif
        } else {
            outer_add_cols(SB_UD, SB_TOKB, SB_H, SB_S);
        }
        POP(SB_UD, SB_VT);
        POP(SB_H, SB_KV);
        WAIT(SB_S, SB_KV);
        // T6: the bf16 snapshot for the writer and, before the last token, the bf16 feedback.
        if constexpr (SB_VARIANT == SB_VARIANT_A0) {
            copy_tiles(SB_S, SB_SOUT, SB_KV);
            if (more) {
                copy_tiles(SB_S, SB_FB, SB_KV);
            }
        } else {
            state_out(SB_S, SB_SOUT, SB_FB, more);
        }
        // T7: o = q~ @ h_new [506] (fp32, row 0) -> the writer assembles row t of O.
        mm_cols(SB_TOKB, TOKB_Q, SB_S, SB_OT);
        POP(SB_S, SB_KV);
        POP(SB_TOKB, TOKB_PAGES);
    }

    // ---- epilogue, once per block: the served fused norm/gate [507-553] on 16 rows ----
    // Rings as the served chain reuses its own: of = O, sq = NSQ, sum = SUM, fac = FAC_Q, xn = X,
    // xw = H, zf = ZF (the z copy [527-530] happened in the prologue), zs = DL, io round trips in
    // RT (served: cb_w).
    // norm_w -> fp32 [393-396], while the writer still assembles O's last row.
    WAIT(SB_W, SB_VT);
    copy_tiles(SB_W, SB_WF, SB_VT);
    POP(SB_W, SB_VT);
    WAIT(SB_WF, SB_VT);
    WAIT(SB_O, SB_VT);
    ew(SB_O, SB_O, SB_NSQ, SB_VT, 2);  // o^2
    WAIT(SB_NSQ, SB_VT);
    rowsum_k(SB_NSQ, SB_SUM, SB_VT);
    WAIT(SB_SUM, 1);
    POP(SB_NSQ, SB_VT);
    inv_rms(SB_SUM, SB_FAC_Q, NG_EPS_BITS, NG_SCALE_BITS, true);  // sqrt(V)/sqrt(sum + V*eps)
    POP(SB_SUM, 1);
    WAIT(SB_FAC_Q, 1);
    bcast_cols_mul(SB_O, SB_FAC_Q, SB_X, 1, SB_VT);  // xn = o * factor
    WAIT(SB_X, SB_VT);
    POP(SB_O, SB_VT);
    POP(SB_FAC_Q, 1);
    WAIT(SB_ZF, SB_VT);
    ew(SB_X, SB_WF, SB_H, SB_VT, 2);  // xw = xn * norm_w (fp32)
    WAIT(SB_H, SB_VT);
    POP(SB_X, SB_VT);
    POP(SB_WF, SB_VT);
    copy_tiles(SB_H, SB_RT, SB_VT);  // -> bf16
    WAIT(SB_RT, SB_VT);
    POP(SB_H, SB_VT);
    copy_tiles(SB_RT, SB_H, SB_VT);  // -> fp32 again, bf16-valued
    WAIT(SB_H, SB_VT);
    POP(SB_RT, SB_VT);
    silu_tiles(SB_ZF, SB_DL, SB_VT);  // zs = silu(z) (fp32)
    WAIT(SB_DL, SB_VT);
    POP(SB_ZF, SB_VT);
    copy_tiles(SB_DL, SB_RT, SB_VT);  // -> bf16
    WAIT(SB_RT, SB_VT);
    POP(SB_DL, SB_VT);
    copy_tiles(SB_RT, SB_DL, SB_VT);  // -> fp32 again, bf16-valued
    WAIT(SB_DL, SB_VT);
    POP(SB_RT, SB_VT);
    ew(SB_H, SB_DL, SB_OUT, SB_VT, 2);  // gated = bf16(xw * zs) -> io
    POP(SB_H, SB_VT);
    POP(SB_DL, SB_VT);
}
