// gdn_seq_block writer (RISCV_0, K5-A). See gdn_seq_block.py and the K5 plan, section 3.6.
//
// One core, one (user, head), one T=16 block:
//   per token t: the 16 bf16 snapshot tiles from SOUT, one column (4 pages) at a time, each page
//     written to states page (h + H*t)*16 + 4i + j - the served order (writer.cpp:161-164 under
//     gdn_multitoken's token transform) - then flushed and popped so the compute can refill it;
//     then OT: row 0 of the four fp32 o tiles (faces 0 and 1, 16 words each) copied into row t of
//     the fp32 O block, which is pushed to the compute's epilogue once all 16 rows are in.
//   end: the four bf16 gated tiles from OUT, faces 2-3 (rows 16-31) zeroed as the served zeroed
//     assembly leaves them (gdn_multitoken.py:47), written at page 4h + tile (gdn_multitoken.py:65-72).
// SOUT is drained before OT each token (the plan lists OT first): the compute pushes SOUT (T6)
// before OT (T7), so the snapshot writes overlap T7. Nothing here computes.
//
// Compile args: {Kt, Vt, H, SRC_TAG} + accessor args for (out, states).
// Runtime args: {h, out, states}.

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
constexpr uint32_t SB_DIAG_NOSNAP = 1;
constexpr uint32_t SB_DIAG = GDN_SEQ_BLOCK_DIAG;

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t SRC_TAG = get_compile_time_arg_val(3);  // keys the JIT cache on content
    static_assert(Kt == 4 && Vt == 4, "the assembly layout is written for 128-wide K and V heads");
    (void)SRC_TAG;

    constexpr auto out_a = TensorAccessorArgs<4>();
    constexpr auto s_a = TensorAccessorArgs<out_a.next_compile_time_args_offset()>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_arg_val<uint32_t>(2);

    const uint32_t tb = get_tile_size(SB_OUT);  // bf16 page
    const uint32_t tf = get_tile_size(SB_O);    // fp32 page
    const auto out_acc = TensorAccessor(out_a, out_addr, tb);
    const auto s_acc = TensorAccessor(s_a, s_addr, tb);

    constexpr uint32_t kv = Kt * Vt;

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

    // O: the fp32 o block the epilogue normalises. Rows 0-15 are all written below; rows 16-31
    // (faces 2-3, words 512-1023 of each tile) are zeroed so they are defined.
    CircularBuffer o(SB_O);
    o.reserve_back(Vt);
    const uint32_t o_base = o.get_write_ptr();
    for (uint32_t tile = 0; tile < Vt; tile++) {
        zero(o_base + tile * tf + 512 * 4, 512);
    }

    CircularBuffer sout(SB_SOUT), ot(SB_OT);
    for (uint32_t t = 0; t < SB_T; t++) {
        const uint32_t base_page = (h + H * t) * kv;
        for (uint32_t j = 0; j < Vt; j++) {
            sout.wait_front(Kt);
            if constexpr (SB_DIAG != SB_DIAG_NOSNAP) {
                auto src = use<CircularBuffer::AddrSelector::READ_PTR>(sout);
                for (uint32_t i = 0; i < Kt; i++) {
                    noc.async_write(src, s_acc, tb, {.offset_bytes = i * tb}, {.page_id = base_page + i * Vt + j});
                }
                noc.async_writes_flushed();
            }
            sout.pop_front(Kt);
        }

        ot.wait_front(Vt);
        const uint32_t src = ot.get_read_ptr();
        for (uint32_t tile = 0; tile < Vt; tile++) {
            copy_words(src + tile * tf, o_base + tile * tf + (t * 16) * 4, 16);                // cols 0-15
            copy_words(src + tile * tf + 256 * 4, o_base + tile * tf + (256 + t * 16) * 4, 16);  // cols 16-31
        }
        ot.pop_front(Vt);
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
