// gdn_seq_block_split writer (RISCV_0, V5, QWEN_FAST_GDN_SPLIT_V=2). See gdn_seq_block_split.py.
//
// gdn_seq_block_writer.cpp (K5-A) for ONE HALF of a (user, head): half 0 (the owner) holds value tiles
// 0..LVt-1, half 1 (the helper) tiles LVt..2LVt-1 (c0 = half * LVt), with LVt = GDN_SPLIT_LVT = 2.
//   per token t: HALF the snapshot of this half's columns - the 4 bf16 tiles (i, jl) with K row i = 0, 1,
//     which the compute packs into SOUT at CB page jl*2 + i - to states page (h + H*t)*16 + 4i + c0 + jl
//     (the served layout); the reader writes K rows 2, 3 on NoC 1. The half-token's pages go out in the
//     order q = LVt*i + jl rotated by a per-core start, q = (start + k) mod 4, start = h mod 4.
//     Then OT: row 0 of this half's LVt fp32 o tiles copied into row t of the fp32 O block.
//   end, helper: its O pages 0..LVt-1 (all 4 KiB each, faces 2-3 zeroed) are written to the owner's O
//     pages LVt..2LVt-1 at the same L1 address (the O ring is allocated once over the union of cores, so
//     the address is the same everywhere), the write is acknowledged, then the owner's semaphore is
//     incremented. The helper writes no output and pushes no O.
//   end, owner: waits for the helper's increment, resets the semaphore, pushes O (all four tiles now
//     in order 0..3 - exactly the fp32 bits K5-A would have assembled) for the compute's epilogue,
//     then writes the four bf16 gated tiles at page 4h + tile as K5-A does.
// Only each core's OWN pages are zeroed or assembled locally; the owner's pages LVt..2LVt-1 are written
// by the helper alone, so no two RISCs ever write the same L1 byte.
// Nothing here computes.
//
// Compile args: {Kt, Vt, H, RT_WORDS, SRC_TAG} + accessor args for (out, states).
// Runtime args (RT_WORDS = 6): {h, half, out, states, peer_x, peer_y}; the peer is the other half of
// the pair (NOC coordinates).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/page.h"

// @@GDN_SEQ_BLOCK_BUILD@@

namespace {

constexpr uint32_t SB_SOUT = 7, SB_OUT = 9, SB_OT = 28, SB_O = 29;
constexpr uint32_t SB_T = 16;
constexpr uint32_t SB_VARIANT_N5X = 2;
constexpr uint32_t SB_VARIANT = GDN_SEQ_BLOCK_VARIANT;
constexpr uint32_t SB_DIAG_NOSNAP = 1;
constexpr uint32_t SB_DIAG = GDN_SEQ_BLOCK_DIAG;
constexpr uint32_t SB_LVT = GDN_SPLIT_LVT;  // value tiles this half holds
constexpr uint32_t SB_RT_USED = 6;          // runtime words read: the highest get_arg_val index + 1

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t RT_WORDS = get_compile_time_arg_val(3);
    constexpr uint32_t SRC_TAG = get_compile_time_arg_val(4);  // keys the JIT cache on content
    static_assert(Kt == 4 && Vt == 4, "the assembly layout is written for 128-wide K and V heads");
    static_assert(SB_LVT == 2 && SB_LVT * 2 == Vt, "the value split is two ways");
    static_assert(SB_RT_USED <= RT_WORDS, "every runtime word the kernel reads is one the host sends");
    (void)SRC_TAG;

    constexpr auto out_a = TensorAccessorArgs<5>();
    constexpr auto s_a = TensorAccessorArgs<out_a.next_compile_time_args_offset()>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t half_id = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t s_addr = get_arg_val<uint32_t>(3);
    const uint32_t peer_x = get_arg_val<uint32_t>(4);
    const uint32_t peer_y = get_arg_val<uint32_t>(5);
    const bool owner = half_id == 0;
    const uint32_t c0 = half_id * SB_LVT;  // first global value tile of this half
    (void)peer_x;
    (void)peer_y;

    const uint32_t tb = get_tile_size(SB_OUT);  // bf16 page
    const uint32_t tf = get_tile_size(SB_O);    // fp32 page
    const auto out_acc = TensorAccessor(out_a, out_addr, tb);
    const auto s_acc = TensorAccessor(s_a, s_addr, tb);

    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t rows = Kt / 2;         // K rows per half-token: 0, 1 here (2, 3 on the reader)
    constexpr uint32_t half = rows * SB_LVT;  // pages per half-token of this half
    // The core's first page: heads 0..11 start on every bank walk position equally often.
    const uint32_t start = h % half;

    Noc noc;

    auto zero = [&](uint32_t base, uint32_t n_words) {
        auto ptr = CoreLocalMem<volatile uint32_t>(base);
        for (uint32_t w = 0; w < n_words; w++) {
            ptr[w] = 0u;
        }
        asm volatile("" ::: "memory");
    };

    auto copy_words = [&](uint32_t src_bytes, uint32_t dst_bytes, uint32_t n_words) {
        asm volatile("" ::: "memory");
        auto s = CoreLocalMem<volatile uint32_t>(src_bytes);
        auto d = CoreLocalMem<volatile uint32_t>(dst_bytes);
        for (uint32_t w = 0; w < n_words; w++) {
            d[w] = s[w];
        }
        asm volatile("" ::: "memory");
    };

    // O: the fp32 o block the owner's epilogue normalises. This core's LVt tiles (pages 0..LVt-1 of its
    // own ring) have rows 0-15 written below and rows 16-31 (faces 2-3, words 512-1023) zeroed so they
    // are defined. The owner's pages LVt..2LVt-1 are the helper's, written remotely: not touched here.
    CircularBuffer o(SB_O);
    o.reserve_back(Vt);
    const uint32_t o_base = o.get_write_ptr();
    for (uint32_t tile = 0; tile < SB_LVT; tile++) {
        zero(o_base + tile * tf + 512 * 4, 512);
    }

    CircularBuffer sout(SB_SOUT), ot(SB_OT);
    const uint32_t sout_limit = get_local_cb_interface(SB_SOUT).fifo_limit;
    const uint32_t sout_size = get_local_cb_interface(SB_SOUT).fifo_size;
    for (uint32_t t = 0; t < SB_T; t++) {
        const uint32_t base_page = (h + H * t) * kv;
        sout.wait_front(half);  // this half of the token
        if constexpr (SB_DIAG != SB_DIAG_NOSNAP) {
            const uint32_t src = sout.get_read_ptr();
            for (uint32_t k = 0; k < half; k++) {
                const uint32_t q = (start + k) % half;  // q = LVt*i + jl, i = 0, 1
                const uint32_t ii = q / SB_LVT;
                const uint32_t jl = q % SB_LVT;
                const uint32_t slot = jl * rows + ii;   // CB page jl*2 + i
                uint32_t addr = src + slot * tb;
                if (addr >= sout_limit) {
                    addr -= sout_size;
                }
                noc.async_write(CoreLocalMem<uint32_t>(addr), s_acc, tb, {},
                                {.page_id = base_page + Vt * ii + c0 + jl});
            }
        }

        ot.wait_front(SB_LVT);
        const uint32_t src = ot.get_read_ptr();
        for (uint32_t tile = 0; tile < SB_LVT; tile++) {
            copy_words(src + tile * tf, o_base + tile * tf + (t * 16) * 4, 16);                  // cols 0-15
            copy_words(src + tile * tf + 256 * 4, o_base + tile * tf + (256 + t * 16) * 4, 16);  // cols 16-31
        }
        ot.pop_front(SB_LVT);

        if constexpr (SB_DIAG != SB_DIAG_NOSNAP) {
            noc.async_writes_flushed();
        }
        for (uint32_t j = 0; j < SB_LVT; j++) {
            sout.pop_front(rows);
        }
    }

    if (!owner) {
        // The helper's O rows to the owner's pages LVt..2LVt-1 (same L1 address on every core), then the
        // semaphore, only after the write is acknowledged. N5x (negative control): no exchange.
        if constexpr (SB_VARIANT != SB_VARIANT_N5X) {
            noc_async_write(o_base, get_noc_addr(peer_x, peer_y, o_base + SB_LVT * tf), SB_LVT * tf);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(peer_x, peer_y, get_semaphore(0)), 1);
            noc_async_atomic_barrier();
        }
        noc.async_write_barrier();
        return;
    }

    if constexpr (SB_VARIANT != SB_VARIANT_N5X) {
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        noc_semaphore_wait(arrived, 1);
        noc_semaphore_set(arrived, 0);
    }
    o.push_back(Vt);

    CircularBuffer out(SB_OUT);
    out.wait_front(Vt);
    const uint32_t out_base = out.get_read_ptr();
    for (uint32_t tile = 0; tile < Vt; tile++) {
        zero(out_base + tile * tb + tb / 2, tb / 8);  // bf16 faces 2-3: the second half of the page
    }
    for (uint32_t tile = 0; tile < Vt; tile++) {
        noc.async_write(CoreLocalMem<uint32_t>(out_base + tile * tb), out_acc, tb, {}, {.page_id = h * Vt + tile});
    }
    noc.async_write_barrier();
    out.pop_front(Vt);
}
