// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// WC reader for the fused GDN windows + conv + gates launch (QWEN_FAST_TP4_GDN_WC, gdn_wc.py). Runs on RISCV_1 beside the SERVED conv-gates compute
// kernel (unchanged) and gdn_wc_writer.cpp. Old-style dataflow API (the calls gdn_conv_windows_packed.cpp and gdn_rows_dma_tp.cpp already run in
// production); nothing here is computed: tiles are copied as bytes, and the one value rule is canonical_half, the rule gdn_rows_dma_tp.cpp applies.
//
// A TILE page is 2048 bytes: four 512-byte faces (0: rows 0-15 x cols 0-15, 1: rows 0-15 x cols 16-31, 2: rows 16-31 x cols 0-15, 3: rows 16-31 x
// cols 16-31), a face row 32 bytes. In the packed 64-row block a tile holds user 2*bt in faces 0-1 (bytes 0-1023) and user 2*bt+1 in faces 2-3.
//
// Per core (the roles are exclusive; the runtime words say which):
//   conv instance inst = bt * Ct + cc (n_inst of them, from inst_start):
//     the conv CB gets [st1, st2, st3, x]:
//       x    = the block's projection tile (bt, cc) with faces 2-3 canonicalised (the canon block's tile),
//       st_s = per face F (user 2*bt + F/2, face column F%2): rows r < 4 - s from row 0 of that face of history r + s of that user, rows 4 - s..15 from rows
//              0..11 + s of x's same face (one block of (12 + s) * 32 bytes) - the T2 window map (gdn_conv_windows_packed.window_copies) on the V1 stack's tile.
//     the four taps (page cc of each).
//   gate tile gi = bt_g * Nvt + t (g_n of them, from gate_start): a and b gathered by words from the projection tile (bt_g, page of column ACOL / BCOL),
//     faces 2-3 canonicalised, dt_bias and neg_exp_A page t. (F1's gather, with the canonical rule the canon block would have applied.)
//   z tile zi = bt * ZCOLS + column (n_z of them, from z_start): the projection tile (bt, ZFIRST + column) with faces 2-3 canonicalised, queued for the writer.
//
// Compile args: {K, Ct, Nvt, USERS, Wt, ACOL, BCOL, NV, ZFIRST, ZCOLS} + accessor args (projection, history, small) + RT_WORDS. Per-core runtime words
// (RT_WORDS = 8): {inst_start, n_inst, g_n, gate_start, z_start, n_z, 0, 0}. Common runtime words (19): x, then H[u][1..3] for u = 0..3 (index 1 + 3u + h - 1),
// then tap0..3 (13..16), dt_bias (17), neg_exp_A (18).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#ifndef CANON_DENORM
#define CANON_DENORM 1
#endif

namespace wc {
constexpr uint32_t cb_win = 0, cb_tap = 1;
constexpr uint32_t cb_a = 6, cb_b = 7, cb_dtb = 8, cb_nega = 9;
constexpr uint32_t cb_gsrc = 13, cb_hist = 14, cb_z = 15;
constexpr uint32_t page = 2048;
constexpr uint32_t half_page = 1024;
constexpr uint32_t face = 512;
constexpr uint32_t row = 32;
constexpr uint32_t span = 64;
constexpr uint32_t history = 4;
constexpr uint32_t history_used = 3;
constexpr uint32_t K = 4;
// common word indices
constexpr uint32_t X_WORD = 0;
constexpr uint32_t H_WORD = 1;                       // + history_used * user + (h - 1)
constexpr uint32_t TAP_WORD = 1 + 4 * history_used;  // 13
constexpr uint32_t DTB_WORD = TAP_WORD + K;          // 17
constexpr uint32_t NEGA_WORD = DTB_WORD + 1;         // 18

FORCE_INLINE uint32_t canonical_half(uint32_t value) {
#if CANON_DENORM
    return (value & 0x7F80u) == 0 ? 0u : value;
#else
    return value == 0x8000u ? 0u : value;
#endif
}

// Canonicalise `bytes` bytes of bfloat16 at an L1 address (both halves of every 32-bit word), in place.
FORCE_INLINE void canonicalise(uint32_t address, uint32_t bytes) {
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    for (uint32_t word = 0; word < bytes / 4; ++word) {
        const uint32_t value = words[word];
        words[word] = canonical_half(value & 0xFFFFu) | (canonical_half(value >> 16) << 16);
    }
}

FORCE_INLINE void fill_words(uint32_t address, uint32_t bytes, uint32_t value) {
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    for (uint32_t word = 0; word < bytes / 4; ++word) {
        words[word] = value;
    }
}

// L1 -> L1 copy of `bytes` (a multiple of 32, both ends 32-byte aligned) through this RISC's NoC: the gdn_conv_windows_packed copy_noc primitive.
FORCE_INLINE void local_copy(uint32_t source, uint32_t destination, uint32_t bytes) {
    noc_async_read(get_noc_addr(source), destination, bytes);
}
}  // namespace wc

void kernel_main() {
    using namespace wc;
    constexpr uint32_t K_ARG = get_compile_time_arg_val(0);
    constexpr uint32_t Ct = get_compile_time_arg_val(1);
    constexpr uint32_t Nvt = get_compile_time_arg_val(2);
    constexpr uint32_t USERS = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t ACOL = get_compile_time_arg_val(5);
    constexpr uint32_t BCOL = get_compile_time_arg_val(6);
    constexpr uint32_t NV = get_compile_time_arg_val(7);
    constexpr uint32_t ZFIRST = get_compile_time_arg_val(8);
    constexpr uint32_t ZCOLS = get_compile_time_arg_val(9);
    static_assert(K_ARG == K, "the reader is written for K == 4 taps");
    static_assert(USERS == 2 || USERS == 4, "whole tiles of two users only");

    constexpr auto x_args = TensorAccessorArgs<10>();
    constexpr auto hist_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto small_args = TensorAccessorArgs<hist_args.next_compile_time_args_offset()>();
    constexpr uint32_t RT_WORDS = get_compile_time_arg_val(small_args.next_compile_time_args_offset());
    static_assert(5 < RT_WORDS, "the reader reads runtime argument 5 (n_z)");

    const uint32_t inst_start = get_arg_val<uint32_t>(0);
    const uint32_t n_inst = get_arg_val<uint32_t>(1);
    const uint32_t g_n = get_arg_val<uint32_t>(2);
    const uint32_t gate_start = get_arg_val<uint32_t>(3);
    const uint32_t z_start = get_arg_val<uint32_t>(4);
    const uint32_t n_z = get_arg_val<uint32_t>(5);

    const auto x_acc = TensorAccessor(x_args, get_common_arg_val<uint32_t>(X_WORD), page);

    // ---- conv instances ----
    for (uint32_t inst = inst_start; inst < inst_start + n_inst; ++inst) {
        const uint32_t bt = inst / Ct;
        const uint32_t cc = inst % Ct;

        cb_reserve_back(cb_win, K);
        const uint32_t win = get_write_ptr(cb_win);
        const uint32_t x_tile = win + 3 * page;
        const uint32_t stage = get_write_ptr(cb_hist);

        noc_async_read_tile(bt * Wt + cc, x_acc, x_tile);
        // the 64-byte spans of row 0 of faces 0 and 1 of histories 1..3 of the tile's two users: stage[((ul * 3 + h - 1) * 2 + f) * 64]
        for (uint32_t ul = 0; ul < 2; ++ul) {
            const uint32_t user = 2 * bt + ul;
            for (uint32_t h = 1; h < history; ++h) {
                const auto hist = TensorAccessor(hist_args, get_common_arg_val<uint32_t>(H_WORD + history_used * user + (h - 1)), page);
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_read(hist.get_noc_addr(cc, f * face), stage + ((ul * history_used + (h - 1)) * 2 + f) * span, span);
                }
            }
        }
        noc_async_read_barrier();
        // The NoC wrote this L1: drop any line of it this RISC cached from the instance that used the CB slot before.
        invalidate_l1_cache();
        // x: the second user's half canonical (the canon block's tile). The first user's half is raw.
        canonicalise(x_tile + half_page, half_page);
        asm volatile("" ::: "memory");
        // st1..st3 into slots 0..2, per face
        for (uint32_t s = 1; s < history; ++s) {
            const uint32_t slot = win + (s - 1) * page;
            for (uint32_t face_index = 0; face_index < 4; ++face_index) {
                const uint32_t ul = face_index >> 1;
                const uint32_t f = face_index & 1;
                const uint32_t destination = slot + face_index * face;
                for (uint32_t r = 0; r + s < history; ++r) {
                    local_copy(stage + ((ul * history_used + (r + s - 1)) * 2 + f) * span, destination + r * row, row);
                }
                local_copy(x_tile + face_index * face, destination + (history - s) * row, (16 - history + s) * row);
            }
        }
        noc_async_read_barrier();
        asm volatile("" ::: "memory");
        cb_push_back(cb_win, K);

        cb_reserve_back(cb_tap, K);
        const uint32_t taps = get_write_ptr(cb_tap);
        for (uint32_t j = 0; j < K; ++j) {
            const auto tap = TensorAccessor(small_args, get_common_arg_val<uint32_t>(TAP_WORD + j), page);
            noc_async_read_tile(cc, tap, taps + j * page);
        }
        noc_async_read_barrier();
        cb_push_back(cb_tap, K);
    }

    // ---- gate tiles ----
    // Copy `run` elements of every row of a tile: source column sc of the tile at `sp`, destination column dc of the tile at `dst`. A run never crosses a
    // 16-column face of either tile. Runs starting at an even column in both tiles move as 32-bit words (an odd leftover element as one 16-bit store),
    // everything else element by element: F1's copy_run for bfloat16.
    auto copy_run = [&](uint32_t sp, uint32_t sc, uint32_t dst, uint32_t dc, uint32_t run) {
        for (uint32_t r = 0; r < 32; r++) {
            const uint32_t se = ((r / 16) * 2 + (sc / 16)) * 256 + (r % 16) * 16 + (sc % 16);
            const uint32_t de = ((r / 16) * 2 + (dc / 16)) * 256 + (r % 16) * 16 + (dc % 16);
            if (((sc | dc) & 1u) == 0) {
                auto sv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sp + se * 2);
                auto dv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + de * 2);
                for (uint32_t w = 0; w < run / 2; w++) {
                    dv[w] = sv[w];
                }
                if (run & 1u) {
                    auto sh = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sp + (se + run - 1) * 2);
                    auto dh = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst + (de + run - 1) * 2);
                    dh[0] = sh[0];
                }
            } else {
                for (uint32_t e = 0; e < run; e++) {
                    auto sv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sp + (se + e) * 2);
                    auto dv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst + (de + e) * 2);
                    dv[0] = sv[0];
                }
            }
        }
    };
    for (uint32_t gi = gate_start; gi < gate_start + g_n; ++gi) {
        const uint32_t t = gi % Nvt;
        const uint32_t bt_g = gi / Nvt;
        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);
        cb_reserve_back(cb_dtb, 1);
        cb_reserve_back(cb_nega, 1);
        const uint32_t dst_a = get_write_ptr(cb_a);
        const uint32_t dst_b = get_write_ptr(cb_b);
        const uint32_t sbase = get_write_ptr(cb_gsrc);
        const uint32_t h0 = t * 32;
        const uint32_t h1 = (h0 + 32 < NV) ? h0 + 32 : NV;  // exclusive: only real heads are gathered
        const uint32_t ap0 = (ACOL + h0) / 32;
        const uint32_t ap1 = (ACOL + h1 - 1) / 32;
        const uint32_t bp0 = (BCOL + h0) / 32;
        const uint32_t bp1 = (BCOL + h1 - 1) / 32;
        // all four reads issue behind ONE barrier: a's source pages into ring slots 0-1, b's into 2-3
        for (uint32_t p = ap0; p <= ap1; p++) {
            noc_async_read_tile(bt_g * Wt + p, x_acc, sbase + (p - ap0) * page);
        }
        for (uint32_t p = bp0; p <= bp1; p++) {
            noc_async_read_tile(bt_g * Wt + p, x_acc, sbase + (2 + p - bp0) * page);
        }
        const auto dtb = TensorAccessor(small_args, get_common_arg_val<uint32_t>(DTB_WORD), page);
        const auto nega = TensorAccessor(small_args, get_common_arg_val<uint32_t>(NEGA_WORD), page);
        noc_async_read_tile(t, dtb, get_write_ptr(cb_dtb));
        noc_async_read_tile(t, nega, get_write_ptr(cb_nega));
        fill_words(dst_a, page, 0);
        fill_words(dst_b, page, 0);
        noc_async_read_barrier();
        invalidate_l1_cache();
        asm volatile("" ::: "memory");
        for (uint32_t h = h0; h < h1;) {
            const uint32_t c = ACOL + h;
            const uint32_t sc = c % 32;
            const uint32_t dc = h - h0;
            uint32_t run = h1 - h;
            run = (16 - sc % 16 < run) ? 16 - sc % 16 : run;
            run = (16 - dc % 16 < run) ? 16 - dc % 16 : run;
            copy_run(sbase + (c / 32 - ap0) * page, sc, dst_a, dc, run);
            h += run;
        }
        for (uint32_t h = h0; h < h1;) {
            const uint32_t c = BCOL + h;
            const uint32_t sc = c % 32;
            const uint32_t dc = h - h0;
            uint32_t run = h1 - h;
            run = (16 - sc % 16 < run) ? 16 - sc % 16 : run;
            run = (16 - dc % 16 < run) ? 16 - dc % 16 : run;
            copy_run(sbase + (2 + c / 32 - bp0) * page, sc, dst_b, dc, run);
            h += run;
        }
        // the canon block's rows 16-31 (the second user) are canonical: the rule is element-wise, so it holds after the gather as before it
        canonicalise(dst_a + half_page, half_page);
        canonicalise(dst_b + half_page, half_page);
        asm volatile("" ::: "memory");
        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
        cb_push_back(cb_dtb, 1);
        cb_push_back(cb_nega, 1);
    }

    // ---- z tiles ----
    for (uint32_t zi = z_start; zi < z_start + n_z; ++zi) {
        const uint32_t bt = zi / ZCOLS;
        const uint32_t column = zi % ZCOLS;
        cb_reserve_back(cb_z, 1);
        const uint32_t tile = get_write_ptr(cb_z);
        noc_async_read_tile(bt * Wt + ZFIRST + column, x_acc, tile);
        noc_async_read_barrier();
        invalidate_l1_cache();
        canonicalise(tile + half_page, half_page);
        asm volatile("" ::: "memory");
        cb_push_back(cb_z, 1);
    }
}
