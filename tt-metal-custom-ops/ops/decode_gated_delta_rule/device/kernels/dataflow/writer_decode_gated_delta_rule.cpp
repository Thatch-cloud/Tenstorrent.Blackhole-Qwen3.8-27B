// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer for the fused T=1 decode gated delta rule. Device 2.0 API.
//
// o is [B,1,H,V] ROW_MAJOR: its flat 2D is [B*H, V], so page bh is head bh's
// own [V] stick — each head owns its DRAM page EXCLUSIVELY and the write is a
// single full-page accessor write (page_id form), the same proven write shape
// the new-state path uses. The stick is staged in L1 from the [1,Vt] o tiles
// (row 0 of each tile holds cols 32t..32t+31 as two 16-element face chunks),
// then written with one full-page noc write.
//
// The new state [B,H,K,V] is TILE and head-aligned (K%32==0): full-tile pages,
// also written exclusively per head.
//
// Compile args: {Kt, Vt} + accessor args for (o, new_state).
// Runtime args: {bh_start, n_inst, o addr, o page bytes, state addr}: this
// core loops over its contiguous instance chunk [bh_start, bh_start+n_inst).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/page.h"

constexpr uint32_t cb_out = 19, cb_sout = 18, cb_scratch = 27;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    // Fused norm + gate: each head's gated row is scattered by NOC into the L1 assembly slot
    // of its (batch-tile, head) assembler core and counted on that core's semaphore; the
    // assembler writes the Vt full output tiles once all rows are in.
    constexpr uint32_t FNG = get_compile_time_arg_val(2);
    constexpr uint32_t PER_CORE = get_compile_time_arg_val(3);
    constexpr uint32_t H = get_compile_time_arg_val(4);
    constexpr uint32_t B = get_compile_time_arg_val(5);

    constexpr auto o_a = TensorAccessorArgs<6>();
    constexpr auto s_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();

    const uint32_t bh_start = get_arg_val<uint32_t>(0);
    const uint32_t n_inst = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_arg_val<uint32_t>(2);
    const uint32_t o_page = get_arg_val<uint32_t>(3);  // RM stick page bytes
    const uint32_t s_addr = get_arg_val<uint32_t>(4);

    const uint32_t tb_io = get_tile_size(cb_out);
    const uint32_t elem = tb_io / 1024;
    const auto o_acc = TensorAccessor(o_a, o_addr, o_page);
    const auto s_acc = TensorAccessor(s_a, s_addr, tb_io);

    constexpr uint32_t kv = Kt * Vt;

    Noc noc;
    uint32_t stage = 0;  // staged o stick base (state write below is untouched)

    // Fused mode: the assembly buffer (cb_scratch: PER_CORE slots of Vt tiles) has the same L1
    // address on every core, so a remote slot is base + slot*Vt*tb on the assembler's L1.
    const uint32_t asm_base = FNG ? CircularBuffer(cb_scratch).get_write_ptr() : 0;

    // Zero an L1 region of n words (page padding stays exact zeros).
    // Volatile: these words are later read back after core-side copies.
    auto zero = [&](uint32_t base, uint32_t n_words) {
        auto ptr = CoreLocalMem<volatile uint32_t>(base);
        for (uint32_t w = 0; w < n_words; w++) {
            ptr[w] = 0u;
        }
        asm volatile("" ::: "memory");
    };

    // Core-side L1 copy of n words (volatile both sides + compiler barriers).
    auto copy_words = [&](uint32_t src_bytes, uint32_t dst_bytes, uint32_t n_words) {
        asm volatile("" ::: "memory");
        auto s = CoreLocalMem<volatile uint32_t>(src_bytes);
        auto d = CoreLocalMem<volatile uint32_t>(dst_bytes);
        for (uint32_t w = 0; w < n_words; w++) {
            d[w] = s[w];
        }
        asm volatile("" ::: "memory");
    };

    if constexpr (FNG) {
        // Rows at or beyond B of my own assembly slots stay zero: nobody writes them.
        for (uint32_t s = 0; s < n_inst; s++) {
            const uint32_t bh = bh_start + s;
            const uint32_t b = bh / H;
            if (b % 32 != 0) {
                continue;  // not an assembler slot
            }
            const uint32_t rows = (B - b < 32) ? (B - b) : 32;
            for (uint32_t r = rows; r < 32; r++) {
                const uint32_t e0 = ((r / 16) * 2) * 256 + (r % 16) * 16;
                for (uint32_t t = 0; t < Vt; t++) {
                    const uint32_t tile = asm_base + (s * Vt + t) * tb_io;
                    zero(tile + e0 * elem, 16 * elem / 4);
                    zero(tile + (e0 + 256) * elem, 16 * elem / 4);
                }
            }
        }
    }

    // Per-instance loop: this core owns [bh_start, bh_start + n_inst).
    for (uint32_t bh = bh_start; bh < bh_start + n_inst; ++bh) {
        if constexpr (FNG) {
            // gated row (row 0 of Vt tiles in cb_out) -> row r of the assembler's slot tiles.
            CircularBuffer cb(cb_out);
            cb.wait_front(Vt);
            const uint32_t src = cb.get_read_ptr();
            const uint32_t b = bh / H;
            const uint32_t h = bh % H;
            const uint32_t r = b % 32;
            const uint32_t bh0 = (b - r) * H + h;
            const uint32_t slot = bh0 % PER_CORE;
            const uint32_t i = bh - bh_start;
            const uint32_t ax = get_arg_val<uint32_t>(5 + 2 * i);
            const uint32_t ay = get_arg_val<uint32_t>(6 + 2 * i);
            const uint32_t e0 = ((r / 16) * 2) * 256 + (r % 16) * 16;
            const uint32_t chunk = 16 * elem;
            for (uint32_t t = 0; t < Vt; t++) {
                const uint32_t dst_tile = asm_base + (slot * Vt + t) * tb_io;
                noc_async_write(src + t * tb_io, get_noc_addr(ax, ay, dst_tile + e0 * elem), chunk);
                noc_async_write(src + t * tb_io + 256 * elem, get_noc_addr(ax, ay, dst_tile + (e0 + 256) * elem), chunk);
            }
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(ax, ay, get_semaphore(slot)), 1);
            cb.pop_front(Vt);
        } else {
        // o: stage head bh's [V] stick in scratch, then ONE full-page write to
        // o page bh via the o accessor's page_id form (the same write shape the
        // new-state write below uses and that ttsim/silicon both land). Tile t's
        // row 0 holds o cols 32t..32t+31: face 0 (cols 0-15) at element offset 0,
        // face 1 (cols 16-31) at element offset 256.
        {
            CircularBuffer cb(cb_out);
            cb.wait_front(Vt);
            const uint32_t src = cb.get_read_ptr();
            CircularBuffer scb(cb_scratch);
            scb.reserve_back(1);
            const uint32_t stage_ = scb.get_write_ptr();
            stage = stage_;
            zero(stage, (o_page + 3) / 4);
            const uint32_t cw = 16 * elem / 4;  // words per 16-element face chunk
            for (uint32_t t = 0; t < Vt; t++) {
                const uint32_t s = src + t * tb_io;
                copy_words(s, stage + (32 * t) * elem, cw);
                copy_words(s + 256 * elem, stage + (32 * t + 16) * elem, cw);
            }
            noc.async_write(CoreLocalMem<uint32_t>(stage), o_acc, o_page, {}, {.page_id = bh});
            noc.async_write_barrier();
            scb.pop_front(1);
            cb.pop_front(Vt);
        }
        }  // !FNG

        // new state: full tiles, head-aligned contiguous pages.
        {
            CircularBuffer cb(cb_sout);
            cb.wait_front(kv);
            const uint32_t base_page = bh * kv;
            auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
            for (uint32_t t = 0; t < kv; t++) {
                noc.async_write(src, s_acc, tb_io, {.offset_bytes = t * tb_io}, {.page_id = base_page + t});
            }
            noc.async_write_barrier();
            cb.pop_front(kv);
        }
    }  // per-instance loop

    if constexpr (FNG) {
        // Assembler duty: for each of my slots that is the row-0 head of a batch tile, wait for
        // that tile's rows, write the Vt full output tiles of gated [1,B,H*Vt*32], reset.
        for (uint32_t s = 0; s < n_inst; s++) {
            const uint32_t bh = bh_start + s;
            const uint32_t b = bh / H;
            if (b % 32 != 0) {
                continue;
            }
            const uint32_t h = bh % H;
            const uint32_t bt = b / 32;
            const uint32_t rows = (B - b < 32) ? (B - b) : 32;
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(s));
            noc_semaphore_wait(sem, rows);
            asm volatile("" ::: "memory");
            for (uint32_t t = 0; t < Vt; t++) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(asm_base + (s * Vt + t) * tb_io),
                    o_acc,
                    o_page,
                    {},
                    {.page_id = bt * (H * Vt) + h * Vt + t});
            }
            noc.async_write_barrier();
            noc_semaphore_set(sem, 0);
        }
    }
}
