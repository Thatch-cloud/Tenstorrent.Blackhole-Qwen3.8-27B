// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// F1 reader for the fused GDN decode conv + gates (QWEN_FAST_TP4_CONV_GATES_SPREAD, gdn_conv_gates_spread.py). Device 2.0 API.
//
// reader_gdn_conv_gates.cpp (the served reader, sha256-pinned in gdn_conv_gates_spread.py) with exactly two changes, both data movement:
//   F1a  a GATE START runtime word (index 16): gate instance gi of this core is global gate tile gate_start + gi, so a gate tile can sit
//        on a core of its own (the served plan puts every gate tile on ONE core, whose reader runs them one after another).
//   F1b  the gate tile's a and b are gathered by 32-bit words (16-bit only at a run's odd edge or at an odd column offset) instead of one
//        bfloat16 at a time, and the whole gate's reads (a, b, dt_bias, neg_exp_A) issue behind ONE barrier. The same bytes land in the
//        same positions of the same zeroed tile; nothing is computed.
// The conv instance loop is the served one verbatim.
//
// Compile args: {K, Ct, Nvt, B, xBt, Wt, GP, AWt, ACOL, BWt, BCOL, Nv} + accessor args (x, st0..st3, tap0..tap3, a, b, dt_bias,
// neg_exp_A) + RT_WORDS. Runtime args (RT_WORDS = 17): {inst_start, n_inst, g_n, x, st0..st3, tap0..tap3, a, b, dt_bias, neg_exp_A,
// gate_start}.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_win = 0, cb_tap = 1;
constexpr uint32_t cb_a = 6, cb_b = 7, cb_dtb = 8, cb_nega = 9, cb_gsrc = 13;

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t Ct = get_compile_time_arg_val(1);
    constexpr uint32_t Nvt = get_compile_time_arg_val(2);
    constexpr uint32_t B = get_compile_time_arg_val(3);
    constexpr uint32_t xBt = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);    // x's width in tiles (Ct when not wider)
    constexpr uint32_t GP = get_compile_time_arg_val(6);    // 1: gather a/b per element from windows
    constexpr uint32_t AWt = get_compile_time_arg_val(7);
    constexpr uint32_t ACOL = get_compile_time_arg_val(8);
    constexpr uint32_t BWt = get_compile_time_arg_val(9);
    constexpr uint32_t BCOL = get_compile_time_arg_val(10);
    constexpr uint32_t NV = get_compile_time_arg_val(11);   // gate width (columns >= NV are padding)
    static_assert(K == 4, "reader is written for K == 4 taps");

    constexpr auto x_a = TensorAccessorArgs<12>();
    constexpr auto s0_a = TensorAccessorArgs<x_a.next_compile_time_args_offset()>();
    constexpr auto s1_a = TensorAccessorArgs<s0_a.next_compile_time_args_offset()>();
    constexpr auto s2_a = TensorAccessorArgs<s1_a.next_compile_time_args_offset()>();
    constexpr auto s3_a = TensorAccessorArgs<s2_a.next_compile_time_args_offset()>();
    constexpr auto t0_a = TensorAccessorArgs<s3_a.next_compile_time_args_offset()>();
    constexpr auto t1_a = TensorAccessorArgs<t0_a.next_compile_time_args_offset()>();
    constexpr auto t2_a = TensorAccessorArgs<t1_a.next_compile_time_args_offset()>();
    constexpr auto t3_a = TensorAccessorArgs<t2_a.next_compile_time_args_offset()>();
    constexpr auto a_a = TensorAccessorArgs<t3_a.next_compile_time_args_offset()>();
    constexpr auto b_a = TensorAccessorArgs<a_a.next_compile_time_args_offset()>();
    constexpr auto dtb_a = TensorAccessorArgs<b_a.next_compile_time_args_offset()>();
    constexpr auto nega_a = TensorAccessorArgs<dtb_a.next_compile_time_args_offset()>();
    constexpr uint32_t RT_WORDS = get_compile_time_arg_val(nega_a.next_compile_time_args_offset());
    static_assert(16 < RT_WORDS, "the reader reads runtime argument 16 (gate_start)");

    const uint32_t inst_start = get_arg_val<uint32_t>(0);
    const uint32_t n_inst = get_arg_val<uint32_t>(1);
    const uint32_t g_n = get_arg_val<uint32_t>(2);
    const uint32_t x_addr = get_arg_val<uint32_t>(3);
    const uint32_t s0_addr = get_arg_val<uint32_t>(4);
    const uint32_t s1_addr = get_arg_val<uint32_t>(5);
    const uint32_t s2_addr = get_arg_val<uint32_t>(6);
    const uint32_t s3_addr = get_arg_val<uint32_t>(7);
    const uint32_t t0_addr = get_arg_val<uint32_t>(8);
    const uint32_t t1_addr = get_arg_val<uint32_t>(9);
    const uint32_t t2_addr = get_arg_val<uint32_t>(10);
    const uint32_t t3_addr = get_arg_val<uint32_t>(11);
    const uint32_t a_addr = get_arg_val<uint32_t>(12);
    const uint32_t b_addr = get_arg_val<uint32_t>(13);
    const uint32_t dtb_addr = get_arg_val<uint32_t>(14);
    const uint32_t nega_addr = get_arg_val<uint32_t>(15);
    const uint32_t gate_start = get_arg_val<uint32_t>(16);
    (void)s0_addr;  // the oldest state is dropped by the shift; never read

    const uint32_t tb = get_tile_size(cb_win);
    const uint32_t elem = tb / 1024;
    const auto x_acc = TensorAccessor(x_a, x_addr, tb);
    const auto s1_acc = TensorAccessor(s1_a, s1_addr, tb);
    const auto s2_acc = TensorAccessor(s2_a, s2_addr, tb);
    const auto s3_acc = TensorAccessor(s3_a, s3_addr, tb);
    const auto t0_acc = TensorAccessor(t0_a, t0_addr, tb);
    const auto t1_acc = TensorAccessor(t1_a, t1_addr, tb);
    const auto t2_acc = TensorAccessor(t2_a, t2_addr, tb);
    const auto t3_acc = TensorAccessor(t3_a, t3_addr, tb);
    const auto a_acc = TensorAccessor(a_a, a_addr, tb);
    const auto b_acc = TensorAccessor(b_a, b_addr, tb);
    const auto dtb_acc = TensorAccessor(dtb_a, dtb_addr, tb);
    const auto nega_acc = TensorAccessor(nega_a, nega_addr, tb);

    Noc noc;

    auto zero_words = [&](uint32_t base, uint32_t n_words) {
        auto ptr = CoreLocalMem<volatile uint32_t>(base);
        for (uint32_t w = 0; w < n_words; w++) {
            ptr[w] = 0u;
        }
        asm volatile("" ::: "memory");
    };
    // Zero row r of the tile at `base`: two 16-element face chunks (faces (r/16)*2 and +1).
    auto zero_row = [&](uint32_t base, uint32_t r) {
        const uint32_t e0 = ((r / 16) * 2) * 256 + (r % 16) * 16;
        zero_words(base + e0 * elem, 16 * elem / 4);
        zero_words(base + (e0 + 256) * elem, 16 * elem / 4);
    };

    for (uint32_t inst = inst_start; inst < inst_start + n_inst; ++inst) {
        const uint32_t bt = inst / Ct;
        const uint32_t cc = inst % Ct;

        // window = [st1, st2, st3, x]  (page `inst` in every [1,Bmax,C] tensor)
        {
            CircularBuffer cb(cb_win);
            cb.reserve_back(K);
            const uint32_t base = cb.get_write_ptr();
            noc.async_read(s1_acc, cb, tb, {.page_id = inst}, {.offset_bytes = 0 * tb});
            noc.async_read(s2_acc, cb, tb, {.page_id = inst}, {.offset_bytes = 1 * tb});
            noc.async_read(s3_acc, cb, tb, {.page_id = inst}, {.offset_bytes = 2 * tb});
            const uint32_t xbase = base + 3 * tb;
            const bool have_x = bt < xBt;
            if (have_x) {
                noc.async_read(x_acc, cb, tb, {.page_id = bt * Wt + cc}, {.offset_bytes = 3 * tb});
            }
            noc.async_read_barrier();
            if (!have_x) {
                zero_words(xbase, tb / 4);
            } else {
                // rows at or beyond the active batch enter the shift register as zeros
                const uint32_t row0 = bt * 32;
                for (uint32_t r = 0; r < 32; r++) {
                    if (row0 + r >= B) {
                        zero_row(xbase, r);
                    }
                }
            }
            cb.push_back(K);
        }
        // taps: page cc of each [1,1,C] tap tensor
        {
            CircularBuffer cb(cb_tap);
            cb.reserve_back(K);
            noc.async_read(t0_acc, cb, tb, {.page_id = cc}, {.offset_bytes = 0 * tb});
            noc.async_read(t1_acc, cb, tb, {.page_id = cc}, {.offset_bytes = 1 * tb});
            noc.async_read(t2_acc, cb, tb, {.page_id = cc}, {.offset_bytes = 2 * tb});
            noc.async_read(t3_acc, cb, tb, {.page_id = cc}, {.offset_bytes = 3 * tb});
            noc.async_read_barrier();
            cb.push_back(K);
        }
    }

    // Copy `n` elements of every row of a tile: source element column sc of the source tile at `sp`, destination column dc of the tile at
    // `dst`. A run never crosses a 16-column face of either tile (a 32-column tile boundary is a face boundary too). bfloat16 runs that
    // start at an even column in both tiles move as 32-bit words, an odd leftover element as one 16-bit store; everything else
    // (an odd column offset, fp32) moves element by element, which is what the served reader does for every element.
    auto copy_run = [&](uint32_t sp, uint32_t sc, uint32_t dst, uint32_t dc, uint32_t run) {
        for (uint32_t r = 0; r < 32; r++) {
            const uint32_t se = ((r / 16) * 2 + (sc / 16)) * 256 + (r % 16) * 16 + (sc % 16);
            const uint32_t de = ((r / 16) * 2 + (dc / 16)) * 256 + (r % 16) * 16 + (dc % 16);
            if (elem == 2 && ((sc | dc) & 1u) == 0) {
                const uint32_t words = run / 2;
                auto sv = CoreLocalMem<volatile uint32_t>(sp + se * 2);
                auto dv = CoreLocalMem<volatile uint32_t>(dst + de * 2);
                for (uint32_t w = 0; w < words; w++) {
                    dv[w] = sv[w];
                }
                if (run & 1u) {
                    auto sh = CoreLocalMem<volatile uint16_t>(sp + (se + run - 1) * 2);
                    auto dh = CoreLocalMem<volatile uint16_t>(dst + (de + run - 1) * 2);
                    dh[0] = sh[0];
                }
            } else if (elem == 2) {
                for (uint32_t e = 0; e < run; e++) {
                    auto sv = CoreLocalMem<volatile uint16_t>(sp + (se + e) * 2);
                    auto dv = CoreLocalMem<volatile uint16_t>(dst + (de + e) * 2);
                    dv[0] = sv[0];
                }
            } else {
                for (uint32_t e = 0; e < run; e++) {
                    auto sv = CoreLocalMem<volatile uint32_t>(sp + (se + e) * 4);
                    auto dv = CoreLocalMem<volatile uint32_t>(dst + (de + e) * 4);
                    dv[0] = sv[0];
                }
            }
        }
    };

    // Gate tile t of tile row bt_g: a and b gathered (columns col0 + h, h in [32t, 32t+32) intersect [0, NV)) from sources `wt` tiles
    // wide, dt_bias and neg_exp_A page t. All four reads issue first and share one barrier; the two source windows of a and b land in the
    // four pages of the scratch ring (a: slots 0-1, b: slots 2-3), then the words are copied into the zeroed a and b tiles.
    auto gate = [&](uint32_t bt_g, uint32_t t) {
        CircularBuffer cba(cb_a);
        CircularBuffer cbb(cb_b);
        CircularBuffer cbd(cb_dtb);
        CircularBuffer cbn(cb_nega);
        CircularBuffer scb(cb_gsrc);
        cba.reserve_back(1);
        cbb.reserve_back(1);
        cbd.reserve_back(1);
        cbn.reserve_back(1);
        scb.reserve_back(4);
        const uint32_t dst_a = cba.get_write_ptr();
        const uint32_t dst_b = cbb.get_write_ptr();
        const uint32_t sbase = scb.get_write_ptr();
        const uint32_t h0 = t * 32;
        const uint32_t h1 = (h0 + 32 < NV) ? h0 + 32 : NV;  // exclusive: only real heads are gathered
        const uint32_t ap0 = (ACOL + h0) / 32;
        const uint32_t ap1 = (ACOL + h1 - 1) / 32;
        const uint32_t bp0 = (BCOL + h0) / 32;
        const uint32_t bp1 = (BCOL + h1 - 1) / 32;
        for (uint32_t p = ap0; p <= ap1; p++) {
            noc.async_read(a_acc, scb, tb, {.page_id = bt_g * AWt + p}, {.offset_bytes = (p - ap0) * tb});
        }
        for (uint32_t p = bp0; p <= bp1; p++) {
            noc.async_read(b_acc, scb, tb, {.page_id = bt_g * BWt + p}, {.offset_bytes = (2 + p - bp0) * tb});
        }
        noc.async_read(dtb_acc, cbd, tb, {.page_id = t}, {.offset_bytes = 0});
        noc.async_read(nega_acc, cbn, tb, {.page_id = t}, {.offset_bytes = 0});
        zero_words(dst_a, tb / 4);
        zero_words(dst_b, tb / 4);
        noc.async_read_barrier();
        asm volatile("" ::: "memory");
        for (uint32_t h = h0; h < h1;) {
            const uint32_t c = ACOL + h;
            const uint32_t sc = c % 32;
            const uint32_t dc = h - h0;
            uint32_t run = h1 - h;
            run = (16 - sc % 16 < run) ? 16 - sc % 16 : run;
            run = (16 - dc % 16 < run) ? 16 - dc % 16 : run;
            copy_run(sbase + (c / 32 - ap0) * tb, sc, dst_a, dc, run);
            h += run;
        }
        for (uint32_t h = h0; h < h1;) {
            const uint32_t c = BCOL + h;
            const uint32_t sc = c % 32;
            const uint32_t dc = h - h0;
            uint32_t run = h1 - h;
            run = (16 - sc % 16 < run) ? 16 - sc % 16 : run;
            run = (16 - dc % 16 < run) ? 16 - dc % 16 : run;
            copy_run(sbase + (2 + c / 32 - bp0) * tb, sc, dst_b, dc, run);
            h += run;
        }
        asm volatile("" ::: "memory");
        scb.push_back(4);
        scb.pop_front(4);
        cba.push_back(1);
        cbb.push_back(1);
        cbd.push_back(1);
        cbn.push_back(1);
    };

    // gates: one tile per instance gi = bt_g*Nvt + t, numbered from gate_start (this core's first global gate tile)
    for (uint32_t gi = gate_start; gi < gate_start + g_n; ++gi) {
        const uint32_t t = gi % Nvt;
        const uint32_t bt_g = gi / Nvt;
        if constexpr (GP) {
            gate(bt_g, t);
        } else {
            auto one = [&](const auto& acc, uint32_t cb_id, uint32_t page) {
                CircularBuffer cb(cb_id);
                cb.reserve_back(1);
                noc.async_read(acc, cb, tb, {.page_id = page}, {.offset_bytes = 0});
                noc.async_read_barrier();
                cb.push_back(1);
            };
            one(a_acc, cb_a, gi);
            one(b_acc, cb_b, gi);
            one(dtb_acc, cb_dtb, t);
            one(nega_acc, cb_nega, t);
        }
    }
}
