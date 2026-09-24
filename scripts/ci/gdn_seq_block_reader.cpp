// gdn_seq_block reader (RISCV_1, K5-A). See gdn_seq_block.py and the K5 plan, section 3.5.
//
// One core, one (user, head), one T=16 block. Every DRAM read is a FULL 2 KiB page into the CB
// that consumes it (the sub-page rule, reader_decode_gated_delta_rule.cpp:12-17): 38 pages per
// core per block, none of them per token, all issued back to back behind ONE read barrier (the
// ONES tile is built while they are in flight). The per-token row operands are then staged from
// the fp32 words the compute prologue packed (QKN, VF, GEXP, BF): pure bit copies, no arithmetic.
//
//   ONES   fp32 all-ones tile, as served (reader.cpp:219-228)
//   IN_GB  g page h/32 and beta page h/32, whole pages (element (t, h) is token t's scalar)
//   IN_V   v pages VOT + 4h;  IN_QK: q pages QOT + 4(h/RF), then k pages KOT + 4(h/RF)
//   IN_Z   z pages ZOT + 4h
//   S0     carry page h*16 + 4i + j -> CB page 4j + i (column-major)
//   per token t, once the prologue has packed QKN (q~ | k~), VF, GEXP and BF:
//     TOKA [0][0,0] = exp(g)[t,h]; [1][0,0] = beta[t,h]; tiles 2-5 row 0 = v row t;
//          tiles 6-9 row 0 = k~ row t
//     TOKB tiles 0-3 column 0 = k~ row t (element c of K tile i -> row c); tiles 4-7 row 0 = q~ row t
//   W      norm_w pages 0-3, read with the inputs but finished and pushed only after the last
//          token is staged (the compute reads it in its epilogue, so this is off the chain's
//          critical path): row 0 copied into rows 1-15 so that row t of xn * w is the served row-0
//          product; rows 16-31 zeroed as served (reader.cpp:230-247).
//
//   SOUT2  the snapshot's K rows 2, 3 (tile (i, j) at CB page 2(j) + i - 2, two per column), which
//          this RISC writes on NoC 1 while the writer writes rows 0, 1 on NoC 0: tile (i, j) of token
//          t to states page (h + H*t)*16 + 4i + j, the served layout. Token t's half is issued once
//          TOKA(t + 1) is staged, in page order q = 4(i - 2) + j rotated by a per-core start (the
//          writer's bank walk, started four banks away), and flushed and popped after TOKB(t + 1):
//          the compute needs TOKA and TOKB before it packs SOUT2 again, so neither waits on the other.
//
// TOKA and TOKB are NOT zeroed: only what is staged each token is ever read. The scalars are read
// at [0,0] only (scalar broadcast); the v and k~ rows and the q~ row feed ops whose output row 0
// depends on input row 0 alone (T3b elementwise, T3a/T7 matmul rows), and rows 1-31 of those
// outputs are dropped (T4 broadcasts DL's row 0; the writer copies OT's row 0); the k~ column is
// read at column 0 only (column broadcast). The served reader dropped its own zeroing for the same
// reason (reader.cpp:182-185, 200-201).
//
// Element (r, c) of a tile is word ((r/16)*2 + c/16)*256 + (r%16)*16 + c%16 (reader.cpp:7-10).
//
// Compile args: {Kt, Vt, H, RF, Ct, QOT, KOT, VOT, WTZ, ZOT, SRC_TAG} + accessor args for
// (qkv, beta, g, carry, z, norm_w, states). Runtime args: {h, qkv, beta, g, carry, z, norm_w, states}.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// @@GDN_SEQ_BLOCK_BUILD@@

namespace {

constexpr uint32_t SB_IN_QK = 0, SB_IN_V = 1, SB_IN_Z = 2, SB_IN_GB = 3, SB_S0 = 4;
constexpr uint32_t SB_ONES = 6, SB_W = 8;
constexpr uint32_t SB_QKN = 11, SB_VF = 12, SB_BF = 15, SB_GEXP = 16;
constexpr uint32_t SB_TOKA = 22, SB_TOKB = 23;
constexpr uint32_t SB_SOUT2 = 21;
constexpr uint32_t SB_T = 16;
constexpr uint32_t SB_DIAG_NOSNAP = 1;
constexpr uint32_t SB_DIAG = GDN_SEQ_BLOCK_DIAG;
constexpr uint32_t TOKA_A = 0, TOKA_BETA = 1, TOKA_V = 2, TOKA_K = 6, TOKA_PAGES = 10;
constexpr uint32_t TOKB_KCOL = 0, TOKB_Q = 4, TOKB_PAGES = 8;
constexpr uint32_t FP32_WORDS = 1024;  // words per fp32 tile

// Word offset of element (r, c) in a 32x32 tile of 4-byte elements.
constexpr uint32_t at(uint32_t r, uint32_t c) { return ((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + c % 16; }

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t RF = get_compile_time_arg_val(3);
    constexpr uint32_t Ct = get_compile_time_arg_val(4);
    constexpr uint32_t QOT = get_compile_time_arg_val(5);
    constexpr uint32_t KOT = get_compile_time_arg_val(6);
    constexpr uint32_t VOT = get_compile_time_arg_val(7);
    constexpr uint32_t WTZ = get_compile_time_arg_val(8);
    constexpr uint32_t ZOT = get_compile_time_arg_val(9);
    constexpr uint32_t SRC_TAG = get_compile_time_arg_val(10);  // keys the JIT cache on content
    static_assert(Kt == 4 && Vt == 4, "the staging layout is written for 128-wide K and V heads");
    static_assert(H <= 32, "all heads' g/beta sit in one tile column page");
    // One 16-row block is tile-row 0 of the packed [1, T, C] tensors (reader.cpp:256): page = column.
    static_assert(QOT + (H / RF) * Kt <= KOT && KOT + (H / RF) * Kt <= VOT && VOT + H * Vt <= Ct,
                  "q | k | v channel tiles");
    static_assert(ZOT + H * Vt <= WTZ, "z channel tiles");
    (void)SRC_TAG;

    constexpr auto qkv_a = TensorAccessorArgs<11>();
    constexpr auto beta_a = TensorAccessorArgs<qkv_a.next_compile_time_args_offset()>();
    constexpr auto g_a = TensorAccessorArgs<beta_a.next_compile_time_args_offset()>();
    constexpr auto s0_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();
    constexpr auto z_a = TensorAccessorArgs<s0_a.next_compile_time_args_offset()>();
    constexpr auto w_a = TensorAccessorArgs<z_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t qkv_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t g_addr = get_arg_val<uint32_t>(3);
    const uint32_t s0_addr = get_arg_val<uint32_t>(4);
    const uint32_t z_addr = get_arg_val<uint32_t>(5);
    const uint32_t w_addr = get_arg_val<uint32_t>(6);
    const uint32_t s_addr = get_arg_val<uint32_t>(7);

    const uint32_t tb = get_tile_size(SB_IN_QK);  // bf16 page
    const uint32_t tf = get_tile_size(SB_TOKA);   // fp32 page
    const auto qkv_acc = TensorAccessor(qkv_a, qkv_addr, tb);
    const auto beta_acc = TensorAccessor(beta_a, beta_addr, tb);
    const auto g_acc = TensorAccessor(g_a, g_addr, tb);
    const auto s0_acc = TensorAccessor(s0_a, s0_addr, tb);
    const auto z_acc = TensorAccessor(z_a, z_addr, tb);
    const auto w_acc = TensorAccessor(w_a, w_addr, tb);
    const auto s_acc = TensorAccessor(s_a, s_addr, tb);

    Noc noc;

    // SOUT2: token t's K rows 2, 3 -> states pages (h + H*t)*16 + 8 + q, q = 4(i - 2) + j, from CB
    // page 2j + i - 2. Issue walks the banks from a per-core start four banks from the writer's.
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t rows = Kt / 2;
    constexpr uint32_t half = rows * Vt;
    const uint32_t start = (h + half / 2) % half;
    CircularBuffer sout2(SB_SOUT2);
    const uint32_t sout2_limit = get_local_cb_interface(SB_SOUT2).fifo_limit;
    const uint32_t sout2_size = get_local_cb_interface(SB_SOUT2).fifo_size;
    auto issue_snapshot = [&](uint32_t t) {
        sout2.wait_front(half);
        if constexpr (SB_DIAG != SB_DIAG_NOSNAP) {
            const uint32_t base_page = (h + H * t) * kv + half;
            const uint32_t src = sout2.get_read_ptr();
            for (uint32_t k = 0; k < half; k++) {
                const uint32_t q = (start + k) % half;
                const uint32_t slot = (q % Vt) * rows + q / Vt;
                uint32_t addr = src + slot * tb;
                if (addr >= sout2_limit) {
                    addr -= sout2_size;
                }
                noc.async_write(CoreLocalMem<uint32_t>(addr), s_acc, tb, {}, {.page_id = base_page + q});
            }
        }
    };
    auto retire_snapshot = [&]() {
        if constexpr (SB_DIAG != SB_DIAG_NOSNAP) {
            noc.async_writes_flushed();
        }
        for (uint32_t j = 0; j < Vt; j++) {
            sout2.pop_front(rows);  // a column at a time: the ring (14) is a multiple of 2, not of 8
        }
    };

    // n full pages first_page.. into `cb` at page offset `slot` of the caller's reservation.
    auto read_pages = [&](const auto& acc, CircularBuffer& cb, uint32_t first_page, uint32_t n, uint32_t slot) {
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = first_page + t}, {.offset_bytes = (slot + t) * tb});
        }
    };

    // Row t of an fp32 tile -> row 0 of another (faces 0 and 1: 16 words each).
    auto copy_row = [&](uint32_t src_tile, uint32_t t, uint32_t dst_tile) {
        auto s = CoreLocalMem<volatile uint32_t>(src_tile);
        auto d = CoreLocalMem<volatile uint32_t>(dst_tile);
        for (uint32_t c = 0; c < 16; c++) {
            d[c] = s[at(t, c)];
            d[256 + c] = s[at(t, 16 + c)];
        }
    };

    // Row t of an fp32 tile -> column 0 of another (element c -> row c).
    auto copy_row_to_column = [&](uint32_t src_tile, uint32_t t, uint32_t dst_tile) {
        auto s = CoreLocalMem<volatile uint32_t>(src_tile);
        auto d = CoreLocalMem<volatile uint32_t>(dst_tile);
        for (uint32_t c = 0; c < 32; c++) {
            d[at(c, 0)] = s[at(t, c)];
        }
    };

    // Every input DRAM read of the block, back to back, in the order the compute consumes them.
    // Each CB is fresh and exactly as large as its reservation, so no reserve here can wait.
    CircularBuffer in_gb(SB_IN_GB), in_v(SB_IN_V), in_z(SB_IN_Z), in_qk(SB_IN_QK), s0(SB_S0), w(SB_W);
    in_gb.reserve_back(2);
    read_pages(g_acc, in_gb, h / 32, 1, 0);  // g then beta, the head's tile-column page of each
    read_pages(beta_acc, in_gb, h / 32, 1, 1);
    in_v.reserve_back(Vt);
    read_pages(qkv_acc, in_v, VOT + h * Vt, Vt, 0);
    in_z.reserve_back(Vt);
    read_pages(z_acc, in_z, ZOT + h * Vt, Vt, 0);
    in_qk.reserve_back(2 * Kt);
    read_pages(qkv_acc, in_qk, QOT + (h / RF) * Kt, Kt, 0);  // q then k of GQA source head h / RF
    read_pages(qkv_acc, in_qk, KOT + (h / RF) * Kt, Kt, Kt);
    s0.reserve_back(Kt * Vt);
    for (uint32_t j = 0; j < Vt; j++) {
        for (uint32_t i = 0; i < Kt; i++) {
            noc.async_read(s0_acc, s0, tb, {.page_id = h * Kt * Vt + i * Vt + j}, {.offset_bytes = (j * Kt + i) * tb});
        }
    }
    w.reserve_back(Vt);
    const uint32_t w_base = w.get_write_ptr();
    read_pages(w_acc, w, 0, Vt, 0);

    // ONES while the reads are in flight: the fp32 all-ones tile for rowsum_k and rowbcast_delta.
    {
        CircularBuffer cb(SB_ONES);
        cb.reserve_back(1);
        auto ptr = CoreLocalMem<uint32_t>(cb.get_write_ptr());
        for (uint32_t word = 0; word < FP32_WORDS; word++) {
            ptr[word] = 0x3F800000u;  // fp32 1.0
        }
        cb.push_back(1);
    }

    noc.async_read_barrier();
    in_gb.push_back(2);
    in_v.push_back(Vt);
    in_z.push_back(Vt);
    in_qk.push_back(2 * Kt);
    s0.push_back(Kt * Vt);

    // Per-token staging from the prologue's packed fp32 words.
    CircularBuffer gexp(SB_GEXP), bf(SB_BF), vf(SB_VF), qkn(SB_QKN);
    gexp.wait_front(1);
    bf.wait_front(1);
    vf.wait_front(Vt);
    qkn.wait_front(2 * Kt);
    const uint32_t a_base = gexp.get_read_ptr();
    const uint32_t b_base = bf.get_read_ptr();
    const uint32_t v_base = vf.get_read_ptr();
    const uint32_t qk_base = qkn.get_read_ptr();  // q~ tiles 0..Kt-1, k~ tiles Kt..2Kt-1

    CircularBuffer toka(SB_TOKA), tokb(SB_TOKB);
    for (uint32_t t = 0; t < SB_T; t++) {
        toka.reserve_back(TOKA_PAGES);
        const uint32_t pa = toka.get_write_ptr();
        asm volatile("" ::: "memory");
        {
            auto a = CoreLocalMem<volatile uint32_t>(a_base);
            auto b = CoreLocalMem<volatile uint32_t>(b_base);
            auto da = CoreLocalMem<volatile uint32_t>(pa + TOKA_A * tf);
            auto db = CoreLocalMem<volatile uint32_t>(pa + TOKA_BETA * tf);
            da[0] = a[at(t, h)];
            db[0] = b[at(t, h)];
        }
        for (uint32_t j = 0; j < Vt; j++) {
            copy_row(v_base + j * tf, t, pa + (TOKA_V + j) * tf);
        }
        for (uint32_t i = 0; i < Kt; i++) {
            copy_row(qk_base + (Kt + i) * tf, t, pa + (TOKA_K + i) * tf);
        }
        asm volatile("" ::: "memory");
        toka.push_back(TOKA_PAGES);

        if (t > 0) {
            issue_snapshot(t - 1);  // packed at T6(t - 1), before T7(t - 1) frees TOKB below
        }

        tokb.reserve_back(TOKB_PAGES);
        const uint32_t pb = tokb.get_write_ptr();
        asm volatile("" ::: "memory");
        for (uint32_t i = 0; i < Kt; i++) {
            copy_row_to_column(qk_base + (Kt + i) * tf, t, pb + (TOKB_KCOL + i) * tf);
        }
        for (uint32_t i = 0; i < Kt; i++) {
            copy_row(qk_base + i * tf, t, pb + (TOKB_Q + i) * tf);
        }
        asm volatile("" ::: "memory");
        tokb.push_back(TOKB_PAGES);

        if (t > 0) {
            retire_snapshot();
        }
    }

    gexp.pop_front(1);
    bf.pop_front(1);
    vf.pop_front(Vt);
    qkn.pop_front(2 * Kt);

    // W, now that the chain no longer waits on this RISC: norm_w [1,1,V] row 0 -> rows 0-15 of Vt
    // bf16 tiles (a face row is 8 words; face 0 row 0 at word 0, face 1 row 0 at word 128, faces
    // 2-3 at 256-511). Its pages landed behind the barrier above.
    asm volatile("" ::: "memory");
    for (uint32_t tile = 0; tile < Vt; tile++) {
        auto words = CoreLocalMem<volatile uint32_t>(w_base + tile * tb);
        for (uint32_t r = 1; r < 16; r++) {
            for (uint32_t word = 0; word < 8; word++) {
                words[r * 8 + word] = words[word];
                words[128 + r * 8 + word] = words[128 + word];
            }
        }
        for (uint32_t word = 256; word < 512; word++) {
            words[word] = 0u;
        }
    }
    asm volatile("" ::: "memory");
    w.push_back(Vt);

    // The last token's half, then every snapshot write acknowledged before the kernel ends.
    issue_snapshot(SB_T - 1);
    retire_snapshot();
    noc.async_write_barrier();
}
