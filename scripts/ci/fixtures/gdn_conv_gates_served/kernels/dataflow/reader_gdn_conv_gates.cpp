// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the fused GDN decode conv + gates. Device 2.0 API.
//
// Per conv instance (page p = bt*Ct + cc of every [1,Bmax,C] tensor): DMA conv_states[1..K-1]
// page p and x page p (or zeros when x has fewer tile-rows) into the window CB, zero the x
// rows at or beyond the active batch B (they enter the shift register as zeros -- bucketed
// decode), then DMA the K tap pages cc (taps are [1,1,C]: one tile-row, row 0 holds the tap).
// All reads are full tile pages; the row masking is a core-side L1 word write.
//
// Gate instances (on the gate core only): one tile each -- a and b page gi, dt_bias and
// neg_exp_A page (gi % Nvt).
//
// Compile args: {K, Ct, Nvt, B, xBt, Wt, GP, AWt, ACOL, BWt, BCOL, Nv} + accessor args (x, st0..st3, tap0..tap3, a, b,
// dt_bias, neg_exp_A). Runtime args: {inst_start, n_inst, g_n, x, st0..st3, tap0..tap3,
// a, b, dt_bias, neg_exp_A}.

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

    // Gather gate tile t (output cols h in [32t, 32t+32) ∩ [0,Nv)) from a window that starts
    // at element column `col0` of a tensor `wt` tiles wide: the source columns col0+h span one
    // or two source tiles. Full-page reads into the scratch CB, then per-element L1 copies
    // (16-bit at bf16, 32-bit at fp32) into the zeroed target tile.
    auto gather_gate = [&](const auto& acc, uint32_t cb_id, uint32_t wt, uint32_t col0, uint32_t bt_g, uint32_t t) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        const uint32_t dst = cb.get_write_ptr();
        zero_words(dst, tb / 4);
        const uint32_t h0 = t * 32;
        const uint32_t h1 = (h0 + 32 < NV) ? h0 + 32 : NV;  // exclusive: only real heads are gathered
        const uint32_t c0 = col0 + h0;
        const uint32_t c1 = col0 + h1 - 1;
        const uint32_t p0 = c0 / 32;
        const uint32_t p1 = c1 / 32;
        CircularBuffer scb(cb_gsrc);
        scb.reserve_back(2);
        const uint32_t sbase = scb.get_write_ptr();
        for (uint32_t p = p0; p <= p1; p++) {
            noc.async_read(acc, scb, tb, {.page_id = bt_g * wt + p}, {.offset_bytes = (p - p0) * tb});
        }
        noc.async_read_barrier();
        asm volatile("" ::: "memory");
        for (uint32_t h = h0; h < h1; h++) {
            const uint32_t c = col0 + h;
            const uint32_t sp = sbase + (c / 32 - p0) * tb;
            const uint32_t sc = c % 32;
            const uint32_t dc = h - h0;
            for (uint32_t r = 0; r < 32; r++) {
                const uint32_t se = ((r / 16) * 2 + (sc / 16)) * 256 + (r % 16) * 16 + (sc % 16);
                const uint32_t de = ((r / 16) * 2 + (dc / 16)) * 256 + (r % 16) * 16 + (dc % 16);
                if (elem == 2) {
                    auto sv = CoreLocalMem<volatile uint16_t>(sp + se * 2);
                    auto dv = CoreLocalMem<volatile uint16_t>(dst + de * 2);
                    dv[0] = sv[0];
                } else {
                    auto sv = CoreLocalMem<volatile uint32_t>(sp + se * 4);
                    auto dv = CoreLocalMem<volatile uint32_t>(dst + de * 4);
                    dv[0] = sv[0];
                }
            }
        }
        asm volatile("" ::: "memory");
        scb.push_back(2);
        scb.pop_front(2);
        cb.push_back(1);
    };

    // gates: one tile per instance gi = bt_g*Nvt + t
    for (uint32_t gi = 0; gi < g_n; ++gi) {
        const uint32_t t = gi % Nvt;
        const uint32_t bt_g = gi / Nvt;
        auto one = [&](const auto& acc, uint32_t cb_id, uint32_t page) {
            CircularBuffer cb(cb_id);
            cb.reserve_back(1);
            noc.async_read(acc, cb, tb, {.page_id = page}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb.push_back(1);
        };
        if constexpr (GP) {
            gather_gate(a_acc, cb_a, AWt, ACOL, bt_g, t);
            gather_gate(b_acc, cb_b, BWt, BCOL, bt_g, t);
        } else {
            one(a_acc, cb_a, gi);
            one(b_acc, cb_b, gi);
        }
        one(dtb_acc, cb_dtb, t);
        one(nega_acc, cb_nega, t);
    }
}
