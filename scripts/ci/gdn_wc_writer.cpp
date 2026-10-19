// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// WC writer for the fused GDN windows + conv + gates launch (QWEN_FAST_TP4_GDN_WC, gdn_wc.py). Runs on RISCV_0 beside the SERVED conv-gates compute
// kernel (unchanged) and gdn_wc_reader.cpp. Old-style dataflow API (the calls gdn_conv_windows_packed.cpp and gdn_rows_dma_tp.cpp run in production).
//
// The compute's output tiles hold two users each (bytes 0-1023 user 2*bt, bytes 1024-2047 user 2*bt+1). Every per-user tensor page this launch writes is a
// WHOLE 2 KiB page: the user's half in bytes 0-1023 and zeros in bytes 1024-2047, composed in a staging ring (CB 16: ten user pages and one zero source
// page) by 1 KiB local L1 -> L1 copies through the NoC. The upper halves of the ten ring pages are zeroed once at start (a zeroed source page copied ten
// times) and are never written again; only the lower halves are re-copied for each instance. No DRAM access is smaller than a page.
//
//   conv instance inst = bt * Ct + cc: cb_out tile (conv) -> conv[u] page cc, and cb_shift tile j (the compute's pass-through [st1, st2, st3, x], the
//     advanced windows new st0..st3) -> window[u][j] page cc, for u = 2*bt and 2*bt + 1: ten pages.
//   gate tile gi = bt_g * Nvt + t: cb_beta -> beta[u] page t, cb_g -> g[u] page t, for u = 2*bt_g, 2*bt_g + 1: four pages.
//   z tile zi = bt * ZCOLS + column: cb_z -> z[u] page column, u = 2*bt, 2*bt + 1: two pages.
//
// Compile args: {K, Ct, Nvt, USERS, ZCOLS} + accessor args (output: every destination tensor shares one layout) + RT_WORDS. Per-core runtime words (RT_WORDS =
// 8): {inst_start, n_inst, g_n, gate_start, z_start, n_z, 0, 0}. Common runtime words (32): conv[0..3] at 0, beta at 4, g at 8, z at 12, window[u][s] at 16 + 4u + s.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace wc {
constexpr uint32_t cb_out = 4, cb_shift = 5, cb_beta = 11, cb_g = 12, cb_z = 15, cb_ustage = 16;
constexpr uint32_t page = 2048;
constexpr uint32_t half_page = 1024;
constexpr uint32_t K = 4;
constexpr uint32_t user_pages = 10;
constexpr uint32_t CONV_WORD = 0, BETA_WORD = 4, G_WORD = 8, Z_WORD = 12, WINDOW_WORD = 16;

FORCE_INLINE void fill_words(uint32_t address, uint32_t bytes, uint32_t value) {
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    for (uint32_t word = 0; word < bytes / 4; ++word) {
        words[word] = value;
    }
}

// L1 -> L1 copy of `bytes` (a multiple of 32, both ends 32-byte aligned) through this RISC's NoC.
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
    constexpr uint32_t ZCOLS = get_compile_time_arg_val(4);
    static_assert(K_ARG == K, "the writer is written for K == 4 taps");
    static_assert(USERS == 2 || USERS == 4, "whole tiles of two users only");

    constexpr auto out_args = TensorAccessorArgs<5>();
    constexpr uint32_t RT_WORDS = get_compile_time_arg_val(out_args.next_compile_time_args_offset());
    static_assert(5 < RT_WORDS, "the writer reads runtime argument 5 (n_z)");

    const uint32_t inst_start = get_arg_val<uint32_t>(0);
    const uint32_t n_inst = get_arg_val<uint32_t>(1);
    const uint32_t g_n = get_arg_val<uint32_t>(2);
    const uint32_t gate_start = get_arg_val<uint32_t>(3);
    const uint32_t z_start = get_arg_val<uint32_t>(4);
    const uint32_t n_z = get_arg_val<uint32_t>(5);

    // The staging ring: upper halves of the ten user pages zero, once.
    const uint32_t ustage = get_write_ptr(cb_ustage);
    const uint32_t zero = ustage + user_pages * page;
    fill_words(zero, half_page, 0);
    asm volatile("" ::: "memory");
    for (uint32_t k = 0; k < user_pages; ++k) {
        local_copy(zero, ustage + k * page + half_page, half_page);
    }
    noc_async_read_barrier();

    // ---- conv instances ----
    for (uint32_t inst = inst_start; inst < inst_start + n_inst; ++inst) {
        const uint32_t bt = inst / Ct;
        const uint32_t cc = inst % Ct;
        cb_wait_front(cb_out, 1);
        cb_wait_front(cb_shift, K);
        const uint32_t out_tile = get_read_ptr(cb_out);
        const uint32_t shift = get_read_ptr(cb_shift);
        for (uint32_t ul = 0; ul < 2; ++ul) {
            local_copy(out_tile + ul * half_page, ustage + ul * page, half_page);
            for (uint32_t j = 0; j < K; ++j) {
                local_copy(shift + j * page + ul * half_page, ustage + (2 + 2 * j + ul) * page, half_page);
            }
        }
        noc_async_read_barrier();
        asm volatile("" ::: "memory");
        cb_pop_front(cb_out, 1);
        cb_pop_front(cb_shift, K);
        for (uint32_t ul = 0; ul < 2; ++ul) {
            const uint32_t user = 2 * bt + ul;
            const auto conv = TensorAccessor(out_args, get_common_arg_val<uint32_t>(CONV_WORD + user), page);
            noc_async_write_tile(cc, conv, ustage + ul * page);
            for (uint32_t j = 0; j < K; ++j) {
                const auto window = TensorAccessor(out_args, get_common_arg_val<uint32_t>(WINDOW_WORD + 4 * user + j), page);
                noc_async_write_tile(cc, window, ustage + (2 + 2 * j + ul) * page);
            }
        }
        noc_async_write_barrier();
    }

    // ---- gate tiles ----
    for (uint32_t gi = gate_start; gi < gate_start + g_n; ++gi) {
        const uint32_t bt_g = gi / Nvt;
        const uint32_t t = gi % Nvt;
        cb_wait_front(cb_beta, 1);
        cb_wait_front(cb_g, 1);
        const uint32_t beta_tile = get_read_ptr(cb_beta);
        const uint32_t g_tile = get_read_ptr(cb_g);
        for (uint32_t ul = 0; ul < 2; ++ul) {
            local_copy(beta_tile + ul * half_page, ustage + ul * page, half_page);
            local_copy(g_tile + ul * half_page, ustage + (2 + ul) * page, half_page);
        }
        noc_async_read_barrier();
        asm volatile("" ::: "memory");
        cb_pop_front(cb_beta, 1);
        cb_pop_front(cb_g, 1);
        for (uint32_t ul = 0; ul < 2; ++ul) {
            const uint32_t user = 2 * bt_g + ul;
            const auto beta = TensorAccessor(out_args, get_common_arg_val<uint32_t>(BETA_WORD + user), page);
            noc_async_write_tile(t, beta, ustage + ul * page);
            const auto gate = TensorAccessor(out_args, get_common_arg_val<uint32_t>(G_WORD + user), page);
            noc_async_write_tile(t, gate, ustage + (2 + ul) * page);
        }
        noc_async_write_barrier();
    }

    // ---- z tiles ----
    for (uint32_t zi = z_start; zi < z_start + n_z; ++zi) {
        const uint32_t bt = zi / ZCOLS;
        const uint32_t column = zi % ZCOLS;
        cb_wait_front(cb_z, 1);
        const uint32_t tile = get_read_ptr(cb_z);
        for (uint32_t ul = 0; ul < 2; ++ul) {
            local_copy(tile + ul * half_page, ustage + ul * page, half_page);
        }
        noc_async_read_barrier();
        asm volatile("" ::: "memory");
        cb_pop_front(cb_z, 1);
        for (uint32_t ul = 0; ul < 2; ++ul) {
            const auto z = TensorAccessor(out_args, get_common_arg_val<uint32_t>(Z_WORD + 2 * bt + ul), page);
            noc_async_write_tile(column, z, ustage + ul * page);
        }
        noc_async_write_barrier();
    }
}
