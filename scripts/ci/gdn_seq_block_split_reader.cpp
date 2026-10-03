// gdn_seq_block_split reader (RISCV_1, V5, QWEN_FAST_GDN_SPLIT_V=2). See gdn_seq_block_split.py.
//
// gdn_seq_block_reader.cpp (K5-A) for ONE HALF of a (user, head): half 0 (the owner) holds value tiles
// 0..LVt-1, half 1 (the helper) tiles LVt..2LVt-1 (c0 = half * LVt), with LVt = GDN_SPLIT_LVT = 2.
// Every DRAM read is still a FULL 2 KiB page into the CB that consumes it, behind ONE read barrier:
//   both halves  g and beta pages, v pages VOT + 4h + c0 + jl (jl < LVt), the head's q and k pages
//                (the prologue is redundant: both halves norm the same q and k), the carry tiles
//                h*16 + 4i + c0 + jl -> S0 page jl*4 + i, and the ONES tile;
//   owner only   z pages ZOT + 4h (all four) and norm_w, which only the owner's epilogue reads.
// The per-token staging is K5-A's (TOKA, TOKB from the fp32 words the prologue packed) with LVt v rows.
// TOKA and TOKB are two tokens deep (gdn_seq_block_split.CB_PLAN), so this RISC stages token t's
// operands BEFORE issuing token t-1's snapshot half: the compute is never made to wait for the copy.
//
// SOUT2 is the snapshot's K rows 2, 3 of this half's columns: tile (i, j) of token t to states page
// (h + H*t)*16 + 4i + c0 + jl, written on NoC 1 while the writer writes rows 0, 1 on NoC 0. Token t's
// half is 4 pages in page order q = LVt*(i-2) + jl rotated by a per-core start, and the bank walk
// (page mod the DRAM bank count) puts owners on banks {0,1,4,5} and helpers on {2,3,6,7}.
//
// Compile args: {Kt, Vt, H, RF, Ct, QOT, KOT, VOT, WTZ, ZOT, RT_WORDS, SRC_TAG} + accessor args for
// (qkv, beta, g, carry, z, norm_w, states). Runtime args (RT_WORDS = 9):
// {h, half, qkv, beta, g, carry, z, norm_w, states}.

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
constexpr uint32_t SB_LVT = GDN_SPLIT_LVT;  // value tiles this half holds
constexpr uint32_t SB_RT_USED = 9;          // runtime words read: the highest get_arg_val index + 1

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
    constexpr uint32_t RT_WORDS = get_compile_time_arg_val(10);
    constexpr uint32_t SRC_TAG = get_compile_time_arg_val(11);  // keys the JIT cache on content
    static_assert(SB_RT_USED <= RT_WORDS, "every runtime word the kernel reads is one the host sends");
    static_assert(SB_LVT == 2 && SB_LVT * 2 == Vt, "the value split is two ways");
    static_assert(Kt == 4 && Vt == 4, "the staging layout is written for 128-wide K and V heads");
    static_assert(H <= 32, "all heads' g/beta sit in one tile column page");
    // One 16-row block is tile-row 0 of the packed [1, T, C] tensors (reader.cpp:256): page = column.
    static_assert(QOT + (H / RF) * Kt <= KOT && KOT + (H / RF) * Kt <= VOT && VOT + H * Vt <= Ct,
                  "q | k | v channel tiles");
    static_assert(ZOT + H * Vt <= WTZ, "z channel tiles");
    (void)SRC_TAG;

    constexpr auto qkv_a = TensorAccessorArgs<12>();
    constexpr auto beta_a = TensorAccessorArgs<qkv_a.next_compile_time_args_offset()>();
    constexpr auto g_a = TensorAccessorArgs<beta_a.next_compile_time_args_offset()>();
    constexpr auto s0_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();
    constexpr auto z_a = TensorAccessorArgs<s0_a.next_compile_time_args_offset()>();
    constexpr auto w_a = TensorAccessorArgs<z_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t half_id = get_arg_val<uint32_t>(1);
    const uint32_t qkv_addr = get_arg_val<uint32_t>(2);
    const uint32_t beta_addr = get_arg_val<uint32_t>(3);
    const uint32_t g_addr = get_arg_val<uint32_t>(4);
    const uint32_t s0_addr = get_arg_val<uint32_t>(5);
    const uint32_t z_addr = get_arg_val<uint32_t>(6);
    const uint32_t w_addr = get_arg_val<uint32_t>(7);
    const uint32_t s_addr = get_arg_val<uint32_t>(8);
    const bool owner = half_id == 0;
    const uint32_t c0 = half_id * SB_LVT;  // first global value tile of this half

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

    // SOUT2: token t's K rows 2, 3 of this half's columns -> states pages (h + H*t)*16 + 8 + 4ii + c0 + jl
    // (ii = i - 2, q = LVt*ii + jl), from CB page jl*2 + ii. Issue walks the banks from a per-core start
    // two pages from the writer's.
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t rows = Kt / 2;
    constexpr uint32_t half = rows * SB_LVT;
    const uint32_t start = (h + half / 2) % half;
    CircularBuffer sout2(SB_SOUT2);
    const uint32_t sout2_limit = get_local_cb_interface(SB_SOUT2).fifo_limit;
    const uint32_t sout2_size = get_local_cb_interface(SB_SOUT2).fifo_size;
    auto issue_snapshot = [&](uint32_t t) {
        sout2.wait_front(half);
        if constexpr (SB_DIAG != SB_DIAG_NOSNAP) {
            const uint32_t base_page = (h + H * t) * kv + rows * Vt;
            const uint32_t src = sout2.get_read_ptr();
            for (uint32_t k = 0; k < half; k++) {
                const uint32_t q = (start + k) % half;
                const uint32_t ii = q / SB_LVT;
                const uint32_t jl = q % SB_LVT;
                const uint32_t slot = jl * rows + ii;
                uint32_t addr = src + slot * tb;
                if (addr >= sout2_limit) {
                    addr -= sout2_size;
                }
                noc.async_write(CoreLocalMem<uint32_t>(addr), s_acc, tb, {},
                                {.page_id = base_page + Vt * ii + c0 + jl});
            }
        }
    };
    auto retire_snapshot = [&]() {
        if constexpr (SB_DIAG != SB_DIAG_NOSNAP) {
            noc.async_writes_flushed();
        }
        for (uint32_t j = 0; j < SB_LVT; j++) {
            sout2.pop_front(rows);  // a column at a time: the ring (14) is a multiple of 2, not of 4
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
    in_v.reserve_back(SB_LVT);
    read_pages(qkv_acc, in_v, VOT + h * Vt + c0, SB_LVT, 0);
    if (owner) {
        in_z.reserve_back(Vt);
        read_pages(z_acc, in_z, ZOT + h * Vt, Vt, 0);
    }
    in_qk.reserve_back(2 * Kt);
    read_pages(qkv_acc, in_qk, QOT + (h / RF) * Kt, Kt, 0);  // q then k of GQA source head h / RF
    read_pages(qkv_acc, in_qk, KOT + (h / RF) * Kt, Kt, Kt);
    s0.reserve_back(Kt * SB_LVT);
    for (uint32_t jl = 0; jl < SB_LVT; jl++) {
        for (uint32_t i = 0; i < Kt; i++) {
            noc.async_read(s0_acc, s0, tb, {.page_id = h * Kt * Vt + i * Vt + c0 + jl},
                           {.offset_bytes = (jl * Kt + i) * tb});
        }
    }
    uint32_t w_base = 0;
    if (owner) {
        w.reserve_back(Vt);
        w_base = w.get_write_ptr();
        read_pages(w_acc, w, 0, Vt, 0);
    }

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
    in_v.push_back(SB_LVT);
    if (owner) {
        in_z.push_back(Vt);
    }
    in_qk.push_back(2 * Kt);
    s0.push_back(Kt * SB_LVT);

    // Per-token staging from the prologue's packed fp32 words.
    CircularBuffer gexp(SB_GEXP), bf(SB_BF), vf(SB_VF), qkn(SB_QKN);
    gexp.wait_front(1);
    bf.wait_front(1);
    vf.wait_front(SB_LVT);
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
        for (uint32_t j = 0; j < SB_LVT; j++) {
            copy_row(v_base + j * tf, t, pa + (TOKA_V + j) * tf);
        }
        for (uint32_t i = 0; i < Kt; i++) {
            copy_row(qk_base + (Kt + i) * tf, t, pa + (TOKA_K + i) * tf);
        }
        asm volatile("" ::: "memory");
        toka.push_back(TOKA_PAGES);

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
            issue_snapshot(t - 1);  // packed at T6(t - 1); token t's operands are already staged (depth 2)
            retire_snapshot();
        }
    }

    gexp.pop_front(1);
    bf.pop_front(1);
    vf.pop_front(SB_LVT);
    qkn.pop_front(2 * Kt);

    // W, now that the chain no longer waits on this RISC: norm_w [1,1,V] row 0 -> rows 0-15 of Vt
    // bf16 tiles (a face row is 8 words; face 0 row 0 at word 0, face 1 row 0 at word 128, faces
    // 2-3 at 256-511). Its pages landed behind the barrier above.
    asm volatile("" ::: "memory");
    for (uint32_t tile = 0; owner && tile < Vt; tile++) {
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
    if (owner) {
        w.push_back(Vt);
    }

    // The last token's half, then every snapshot write acknowledged before the kernel ends.
    issue_snapshot(SB_T - 1);
    retire_snapshot();
    noc.async_write_barrier();
}
