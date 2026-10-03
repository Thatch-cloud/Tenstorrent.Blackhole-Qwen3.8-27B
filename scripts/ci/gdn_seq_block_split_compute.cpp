// gdn_seq_block_split compute (V5, QWEN_FAST_GDN_SPLIT_V=2). See gdn_seq_block_split.py and docs/tp4-recurrence-split.md.
//
// NOT a standalone file. gdn_seq_block_split.generate() emits the served native compute prefix first
// (decode_gated_delta_rule.cpp up to its entry point, sha256-checked against gdn_multitoken.HASHES),
// exactly as gdn_seq_block does: its includes, its CB names and its helpers copy_tiles, ew, ew_off,
// expc, silu_tiles, bcast_scalar_mul, bcast_cols_mul, rowsum_k, inv_rms and rowbcast_delta, verbatim.
// This file follows it, with the build line below replaced by the generated build constants.
//
// This is gdn_seq_block_compute.cpp (K5-A) with the value columns of one (user, head) split over TWO
// cores. Half 0 (the OWNER) runs value tiles 0..LVt-1, half 1 (the HELPER) runs tiles LVt..2LVt-1.
// Nothing in the 16-token chain couples value columns: every helper below is a loop over the column
// j with its own DST acquire, so the chain is the K5-A chain with the loop bound SB_LVT in place of
// SB_VT, on the same operands, in the same srcA/srcB roles and K order. The prologue runs on both
// halves (the same bits twice); only the value tiles differ.
//
// The one place columns meet is the epilogue's gated RMSNorm: rowsum_k adds the four squared value
// tiles in the order 0, 1, 2, 3 inside one DST, so the split must not change that order. The helper
// therefore ships its fp32 O rows (raw bit copies of its T7 outputs) to the owner's writer, which
// pushes all four O tiles once the helper's have landed; the owner's epilogue is the K5-A epilogue,
// byte for byte, on all four tiles. Data moves; partial sums never do. The helper's compute ends
// after T7 of the last token.
//
// Every CB has ONE producer RISC and ONE consumer RISC (gdn_seq_block_split.CB_PLAN). O is produced by
// the owner's writer alone (its own two tiles by word copies, the helper's two by a remote write the
// helper's writer issues before its semaphore increment); the helper's O ring is only an address.
//
// Compile-time args: {EPS_BITS, SCALE_BITS, NG_EPS_BITS, NG_SCALE_BITS, LEVEL, RT_WORDS, SRC_TAG}.
// Runtime args: {half}.

// @@GDN_SEQ_BLOCK_BUILD@@

namespace {

constexpr uint32_t SB_IN_QK = 0, SB_IN_V = 1, SB_IN_Z = 2, SB_IN_GB = 3, SB_S0 = 4, SB_FB = 5;
constexpr uint32_t SB_ONES = 6;
constexpr uint32_t SB_SOUT = 7, SB_W = 8, SB_OUT = 9;
constexpr uint32_t SB_X = 10, SB_QKN = 11, SB_VF = 12, SB_ZF = 13, SB_GF = 14, SB_BF = 15, SB_GEXP = 16;
constexpr uint32_t SB_WF = 17, SB_NSQ = 18, SB_SUM = 19, SB_FAC_Q = 20;
constexpr uint32_t SB_SOUT2 = 21;  // the snapshot's K rows 2-3, compute -> reader
constexpr uint32_t SB_TOKA = 22, SB_TOKB = 23, SB_S = 24, SB_H = 25, SB_UD = 26, SB_DL = 27;
constexpr uint32_t SB_OT = 28, SB_O = 29, SB_RT = 30;
static_assert(SB_ONES == cb_ones, "rowsum_k and rowbcast_delta read the ones tile at cb_ones");

constexpr uint32_t SB_T = 16;                 // tokens (rows) per block
constexpr uint32_t SB_KT = 4;                 // K tiles
constexpr uint32_t SB_VT = 4;                 // V tiles of the whole head (the epilogue's width)
constexpr uint32_t SB_LVT = GDN_SPLIT_LVT;    // V tiles this core runs: 2 of 4
constexpr uint32_t SB_KVL = SB_KT * SB_LVT;   // state tiles this core holds
static_assert(SB_LVT == 2 && SB_VT == 2 * SB_LVT, "the value split is two ways");

// TOKA: [a = exp(g)[t,h] at [0,0], beta[t,h] at [0,0], v row t x LVt (row 0), k~ row t x4 (row 0)].
// Pages LVt..3 of the v range are never read (the layout is the K5-A one so the k~ pages do not move).
constexpr uint32_t TOKA_A = 0, TOKA_BETA = 1, TOKA_V = 2, TOKA_K = 6, TOKA_PAGES = 10;
// TOKB: [k~ row t as column 0 x4, q~ row t x4 (row 0)].
constexpr uint32_t TOKB_KCOL = 0, TOKB_Q = 4, TOKB_PAGES = 8;
static_assert(TOKA_A == 0, "the served bcast_scalar_mul reads its scalar at page 0");
static_assert(TOKB_KCOL == 0, "the T5 helpers read kcol_i at page i");

constexpr uint32_t SB_VARIANT_A = 0, SB_VARIANT_N5R = 1, SB_VARIANT_N5X = 2;
constexpr uint32_t SB_DIAG_NONE = 0, SB_DIAG_NOSNAP = 1, SB_DIAG_PASSTHROUGH = 2;
constexpr uint32_t SB_LEVEL = GDN_SEQ_BLOCK_LEVEL;
constexpr uint32_t SB_VARIANT = GDN_SEQ_BLOCK_VARIANT;
constexpr uint32_t SB_DIAG = GDN_SEQ_BLOCK_DIAG;
static_assert(SB_VARIANT <= SB_VARIANT_N5X && SB_DIAG <= SB_DIAG_PASSTHROUGH, "unknown build");
static_assert(SB_LEVEL == 0, "no A+ increment is implemented; level 0 only");
constexpr uint32_t SB_RT_USED = 1;  // runtime words read: half

// mm (compute.cpp:57-77) with an A page offset and a column-major B: o[j] = sum_ki A[a0+ki] @ B(ki, j),
// B(ki, j) at page 4j + ki, for this core's SB_LVT columns. One DST per column, the served K order.
void mm_cols(uint32_t a, uint32_t a0, uint32_t b, uint32_t o) {
    cb_reserve_back(o, SB_LVT);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);
    matmul_init(a, b, 0);
    for (uint32_t j = 0; j < SB_LVT; j++) {
        tile_regs_acquire();
        for (uint32_t ki = 0; ki < SB_KT; ki++) {
            matmul_tiles(a, b, a0 + ki, j * SB_KT + ki, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, j);
        tile_regs_release();
    }
    cb_push_back(o, SB_LVT);
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

// The served L2 norm (compute.cpp:434-446 for q, 448-460 for k) over the front SB_KT pages of `x`,
// all 16 token rows at once; identical to gdn_seq_block_compute.cpp (every op is row-independent and
// spans K, which both halves hold whole).
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

// T5, served [495] + [499] with the add in DST: per column j (this core's SB_LVT), windows of two
// tiles (gdn_outer_add.HELPER's arithmetic): DST[s] = D'_j * kcol_i[:,0], then DST[s] = H(i,j) + DST[s]
// (DEST_TO_SRCB), packed to the new state. Pushed per column.
void outer_add_cols(uint32_t dprime, uint32_t kcol, uint32_t state, uint32_t o) {
    pack_reconfig_data_format(o);
    for (uint32_t j = 0; j < SB_LVT; j++) {
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

// T6: the served copy_tiles(snew, sout) and copy_tiles(snew, feedback) as ONE unpack packed twice
// (gdn_dual_state_copy.py), per column, the snapshot split by K row over two rings (rows 0-1 to the
// writer's NoC, rows 2-3 to the reader's): tile (i, j) of this core's columns to `out` for i < 2 and
// `out2` for i >= 2, two pages per column each. Same packs, same values; only the width differs.
constexpr uint32_t SB_KT_HALF = SB_KT / 2;
void state_out(uint32_t s, uint32_t out, uint32_t out2, uint32_t feedback, bool more) {
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(s);
    copy_tile_to_dst_init_short(s);
    for (uint32_t j = 0; j < SB_LVT; j++) {
        cb_reserve_back(out, SB_KT_HALF);
        cb_reserve_back(out2, SB_KT_HALF);
        if (more) {
            cb_reserve_back(feedback, SB_KT);
        }
        for (uint32_t i = 0; i < SB_KT; i++) {
            tile_regs_acquire();
            copy_tile(s, j * SB_KT + i, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, i < SB_KT_HALF ? out : out2, i % SB_KT_HALF);
            if (more) {
                pack_tile(0, feedback, i);
            }
            tile_regs_release();
        }
        cb_push_back(out, SB_KT_HALF);
        cb_push_back(out2, SB_KT_HALF);
        if (more) {
            cb_push_back(feedback, SB_KT);
        }
    }
}

// Negative control N5r: the epilogue's rowsum_k with the value tiles added in the order 2, 3, 0, 1
// (the owner's own tiles last) - the order a partial-sum split would have produced. Known to change
// bits whenever the halves differ; the probe must see it.
void rowsum_k_rotated(uint32_t in, uint32_t o, uint32_t Kt) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(in, cb_ones);
    matmul_init(in, cb_ones, 0);
    tile_regs_acquire();
    for (uint32_t k = 0; k < Kt; k++) {
        matmul_tiles(in, cb_ones, (k + Kt / 2) % Kt, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(0);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(1);
    constexpr uint32_t NG_EPS_BITS = get_compile_time_arg_val(2);
    constexpr uint32_t NG_SCALE_BITS = get_compile_time_arg_val(3);
    constexpr uint32_t LEVEL = get_compile_time_arg_val(4);
    constexpr uint32_t RT_WORDS = get_compile_time_arg_val(5);
    constexpr uint32_t SRC_TAG = get_compile_time_arg_val(6);  // keys the JIT cache on content
    static_assert(LEVEL == SB_LEVEL, "LEVEL compile arg must equal the generated level");
    static_assert(SB_RT_USED <= RT_WORDS, "every runtime word the kernel reads is one the host sends");
    (void)SRC_TAG;

    const bool owner = get_arg_val<uint32_t>(0) == 0;

    compute_kernel_hw_startup(SB_IN_QK, SB_IN_V, SB_OUT);

    // ---- prologue, once per block, both halves (norm_w and z are the owner's) ----
    WAIT(SB_IN_GB, 2);
    copy_tiles(SB_IN_GB, SB_GF, 1);
    POP(SB_IN_GB, 1);
    copy_tiles(SB_IN_GB, SB_BF, 1);  // -> reader
    POP(SB_IN_GB, 1);
    WAIT(SB_GF, 1);
    expc(SB_GF, SB_GEXP, 1);  // [t,h] = exp(g[t,h]) [463]; -> reader
    POP(SB_GF, 1);
    WAIT(SB_IN_V, SB_LVT);
    copy_tiles(SB_IN_V, SB_VF, SB_LVT);  // [412]; -> reader (this half's value tiles)
    POP(SB_IN_V, SB_LVT);
    if (owner) {
        WAIT(SB_IN_Z, SB_VT);
        copy_tiles(SB_IN_Z, SB_ZF, SB_VT);  // [528], for the epilogue
        POP(SB_IN_Z, SB_VT);
    }
    WAIT(SB_IN_QK, 2 * SB_KT);
    copy_tiles(SB_IN_QK, SB_X, 2 * SB_KT);  // q | k [410-411]
    POP(SB_IN_QK, 2 * SB_KT);
    WAIT(SB_X, 2 * SB_KT);
    WAIT(SB_ONES, 1);
    l2_norm_rows(SB_X, SB_FAC_Q, SB_QKN, EPS_BITS, SCALE_BITS, true);  // q~ -> QKN pages 0-3 [434-446]
    POP(SB_X, SB_KT);
    l2_norm_rows(SB_X, SB_FAC_Q, SB_QKN, EPS_BITS, SCALE_BITS, false);  // k~ -> QKN pages 4-7 [448-460]
    POP(SB_X, SB_KT);

    // ---- the chain, token by token ----
    for (uint32_t t = 0; t < SB_T; ++t) {
        const uint32_t src = t == 0 ? SB_S0 : SB_FB;
        const bool more = t + 1 < SB_T;

        // T1: bf16 state -> fp32 [415].
        WAIT(src, SB_KVL);
        copy_tiles(src, SB_S, SB_KVL);
        POP(src, SB_KVL);
        WAIT(SB_S, SB_KVL);
        WAIT(SB_TOKA, TOKA_PAGES);

        if constexpr (SB_DIAG == SB_DIAG_PASSTHROUGH) {
            // Diagnostic only: the chain's math replaced by pass-through (bytes-bound timing).
            POP(SB_TOKA, TOKA_PAGES);
            WAIT(SB_TOKB, TOKB_PAGES);
            state_out(SB_S, SB_SOUT, SB_SOUT2, SB_FB, more);
            copy_tiles(SB_TOKB, SB_OT, SB_LVT);
            POP(SB_S, SB_KVL);
            POP(SB_TOKB, TOKB_PAGES);
            continue;
        }

        // T2: h = S * exp(g) [466] (TOKA page 0 holds the scalar at [0,0]).
        bcast_scalar_mul(SB_S, SB_TOKA, SB_H, SB_KVL);
        WAIT(SB_H, SB_KVL);
        POP(SB_S, SB_KVL);
        // T3a: v_read = k~ @ h [472].
        mm_cols(SB_TOKA, TOKA_K, SB_H, SB_UD);
        WAIT(SB_UD, SB_LVT);
        // T3b: delta = v - v_read [474] (DL pages 0..LVt-1).
        ew_off(SB_TOKA, TOKA_V, SB_UD, 0, SB_DL, SB_LVT, 1);
        WAIT(SB_DL, SB_LVT);
        POP(SB_UD, SB_LVT);
        // T3c: delta *= beta [478] (the served in-place ring).
        bcast_scalar_mul_at(SB_DL, SB_TOKA, TOKA_BETA, SB_DL, SB_LVT);
        POP(SB_TOKA, TOKA_PAGES);
        POP(SB_DL, SB_LVT);  // drop the pre-beta pages
        WAIT(SB_DL, SB_LVT);
        // T4: D'_j = delta row 0 broadcast down every row [492].
        rowbcast_delta(SB_DL, SB_UD, SB_LVT);
        WAIT(SB_UD, SB_LVT);
        POP(SB_DL, SB_LVT);
        // T5: h_new = h + k~^T (x) D' [495, 499].
        WAIT(SB_TOKB, TOKB_PAGES);
        outer_add_cols(SB_UD, SB_TOKB, SB_H, SB_S);
        POP(SB_UD, SB_LVT);
        POP(SB_H, SB_KVL);
        WAIT(SB_S, SB_KVL);
        // T6: the bf16 snapshot for the writer and reader and, before the last token, the bf16 feedback.
        state_out(SB_S, SB_SOUT, SB_SOUT2, SB_FB, more);
        // T7: o = q~ @ h_new [506] (fp32, row 0) -> the writer assembles row t of O.
        mm_cols(SB_TOKB, TOKB_Q, SB_S, SB_OT);
        POP(SB_S, SB_KVL);
        POP(SB_TOKB, TOKB_PAGES);
    }

    // ---- epilogue, once per block, the OWNER only: the served fused norm/gate [507-553] on 16 rows,
    // over all four O tiles (its own two and the helper's two, assembled by its writer) ----
    if (owner) {
        // norm_w -> fp32 [393-396], while the writer still waits for the helper's O rows.
        WAIT(SB_W, SB_VT);
        copy_tiles(SB_W, SB_WF, SB_VT);
        POP(SB_W, SB_VT);
        WAIT(SB_WF, SB_VT);
        WAIT(SB_O, SB_VT);
        ew(SB_O, SB_O, SB_NSQ, SB_VT, 2);  // o^2
        WAIT(SB_NSQ, SB_VT);
        if constexpr (SB_VARIANT == SB_VARIANT_N5R) {
            rowsum_k_rotated(SB_NSQ, SB_SUM, SB_VT);
        } else {
            rowsum_k(SB_NSQ, SB_SUM, SB_VT);
        }
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
}
