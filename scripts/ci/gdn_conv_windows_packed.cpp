// gdn_conv_windows_packed: every packed user's four causal-conv windows in one launch.
// See gdn_conv_windows_packed.py (verify-trace T2 cut #1). One source, three roles:
//
//   VTW_ROLE_READER (RISCV_1)  per (user, page) task: the piece tile and the four history tiles
//                              (or, under VTW_HIST_ROW, only the two 64-byte spans of each
//                              history page the windows use) into CB_IN, one read barrier.
//   VTW_ROLE_WRITER (RISCV_0)  faces 2-3 of every scratch page zeroed ONCE (rows == 16 only);
//                              per task the four windows built from CB_IN with 32-byte row
//                              copies (gdn_conv_windows_packed.window_copies) and written.
//   VTW_PORT        (RISCV_0)  gdn_conv_windows.cpp transcribed literally onto (user, page)
//                              tasks: zero the scratch, copy, write and barrier per window.
//
// Every value is moved as bits: RISC-V uint32 word copies (the served primitive) or, under
// VTW_COPY_NOC, local L1-to-L1 NOC copies. Nothing is computed, packed or unpacked.
//
// Compile-time args: [USERS, PAGES, ROWS, NBUF, TASKS] then the TensorAccessorArgs of the piece,
// history and output classes. Common runtime args (user-major): P[u] at u, H[u][h] at
// USERS + 4u + h, O[u][s] at 5 USERS + 4u + s. Per-core runtime args: [task_start, task_count].

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace vtw {
constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_scratch = 1;
constexpr uint32_t page = 2048;
constexpr uint32_t face = 512;
constexpr uint32_t row = 32;
constexpr uint32_t tiles_in = 5;
constexpr uint32_t slots = 4;
constexpr uint32_t history = 4;
// A history page's rows 0 of faces 0 and 1 are read as two 64-byte spans (DRAM reads are
// 64-byte granular on Blackhole); only the first 32 bytes of each span are ever copied.
constexpr uint32_t history_span = 64;

#ifdef VTW_NEG_PAD
constexpr uint32_t pad_word = 0xFFFFFFFFu;  // negative control: the compare must see padding
#else
constexpr uint32_t pad_word = 0;
#endif

// The window a slot is built as (VTW_NEG_SLOT: slot s gets slot (s + 1) % 4's content).
FORCE_INLINE uint32_t source_slot(uint32_t slot) {
#ifdef VTW_NEG_SLOT
    return (slot + 1) % slots;
#else
    return slot;
#endif
}

// The history a window row reads (VTW_NEG_HIST: one later, clamped to the last).
FORCE_INLINE uint32_t history_index(uint32_t index) {
#ifdef VTW_NEG_HIST
    return index + 1 > history - 1 ? history - 1 : index + 1;
#else
    return index;
#endif
}

// The user whose piece a task reads (VTW_NEG_USER: the next user's).
template <uint32_t users>
FORCE_INLINE uint32_t piece_user(uint32_t user) {
#ifdef VTW_NEG_USER
    return (user + 1) % users;
#else
    return user;
#endif
}

FORCE_INLINE void fill_words(uint32_t address, uint32_t bytes, uint32_t value) {
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    for (uint32_t i = 0; i < bytes / 4; ++i) {
        words[i] = value;
    }
}

// L1 -> L1 copy of n bytes (a multiple of 32, both ends 32-byte aligned).
FORCE_INLINE void copy_bytes(uint32_t source, uint32_t destination, uint32_t bytes) {
#ifdef VTW_COPY_NOC
    noc_async_read(get_noc_addr(source), destination, bytes);
#else
    auto from = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(source);
    auto to = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination);
    for (uint32_t i = 0; i < bytes / 4; ++i) {
        to[i] = from[i];
    }
#endif
}

// Window `slot` into `destination` (faces 2-3 already zero): per face, rows 0..3-s from
// history s..3 (row 0 of each history page), rows 4-s..15 from piece rows 0..11+s in one block.
FORCE_INLINE void build_window(uint32_t staged, uint32_t destination, uint32_t slot) {
    for (uint32_t f = 0; f < 2; ++f) {
        for (uint32_t r = 0; r + slot < history; ++r) {
            copy_bytes(staged + (1 + history_index(r + slot)) * page + f * face, destination + f * face + r * row, row);
        }
        copy_bytes(staged + f * face, destination + f * face + (history - slot) * row, (16 - history + slot) * row);
    }
}
}  // namespace vtw

void kernel_main() {
    using namespace vtw;
    constexpr uint32_t USERS = get_compile_time_arg_val(0);
    constexpr uint32_t PAGES = get_compile_time_arg_val(1);
    constexpr uint32_t ROWS = get_compile_time_arg_val(2);
    constexpr uint32_t NBUF = get_compile_time_arg_val(3);
    constexpr auto piece_args = TensorAccessorArgs<5>();
    constexpr auto history_args = TensorAccessorArgs<piece_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<history_args.next_compile_time_args_offset()>();
    constexpr uint32_t H_BASE = USERS;
    constexpr uint32_t O_BASE = USERS + history * USERS;
    const uint32_t task_start = get_arg_val<uint32_t>(0);
    const uint32_t task_count = get_arg_val<uint32_t>(1);

#if defined(VTW_ROLE_READER)
    for (uint32_t t = task_start; t < task_start + task_count; ++t) {
        const uint32_t user = t / PAGES;
        const uint32_t p = t % PAGES;
        cb_reserve_back(cb_in, tiles_in);
        const uint32_t staged = get_write_ptr(cb_in);
        const auto piece = TensorAccessor(piece_args, get_common_arg_val<uint32_t>(piece_user<USERS>(user)), page);
        noc_async_read_tile(p, piece, staged);
        for (uint32_t h = 0; h < history; ++h) {
            const auto source = TensorAccessor(history_args, get_common_arg_val<uint32_t>(H_BASE + history * user + h), page);
            const uint32_t destination = staged + (1 + h) * page;
#ifdef VTW_HIST_ROW
            noc_async_read(source.get_noc_addr(p, 0), destination, history_span);
            noc_async_read(source.get_noc_addr(p, face), destination + face, history_span);
#else
            noc_async_read_tile(p, source, destination);
#endif
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, tiles_in);
    }

#elif defined(VTW_ROLE_WRITER)
    // Zeroing faces 2-3 once is exact only because no token >= 16 exists: every window row
    // lives in faces 0-1, each of whose 16 rows is rewritten for every window.
    static_assert(ROWS == 16, "the trimmed windows kernel zeroes faces 2-3 once: rows must be 16");
    static_assert(NBUF == 1 || NBUF == 2, "one or two scratch sets");
    const uint32_t scratch_base = get_write_ptr(cb_scratch);
    for (uint32_t index = 0; index < slots * NBUF; ++index) {
        fill_words(scratch_base + index * page + 2 * face, 2 * face, pad_word);
    }
    asm volatile("" ::: "memory");
    uint32_t local = 0;
    for (uint32_t t = task_start; t < task_start + task_count; ++t, ++local) {
        const uint32_t user = t / PAGES;
        const uint32_t p = t % PAGES;
        const uint32_t set = NBUF == 2 ? (local & 1) : 0;
        const uint32_t scratch = scratch_base + set * slots * page;
        if constexpr (NBUF == 2) {
            if (local >= 2) {
                noc_async_writes_flushed();  // this set's writes of two tasks ago have left L1
            }
        }
        cb_wait_front(cb_in, tiles_in);
#ifndef VTW_COPY_NOC
        // The word copies below read CB_IN through this RISC's L1 data cache, and on Blackhole
        // cb_wait_front does not invalidate it: drop any line of this half cached from the
        // task that used it two tasks ago, whatever barrier policy runs between (one fence).
        invalidate_l1_cache();
#endif
        const uint32_t staged = get_read_ptr(cb_in);
        for (uint32_t s = 0; s < slots; ++s) {
            build_window(staged, scratch + s * page, source_slot(s));
        }
#ifdef VTW_COPY_NOC
        noc_async_read_barrier();
#endif
        asm volatile("" ::: "memory");
        cb_pop_front(cb_in, tiles_in);
        for (uint32_t s = 0; s < slots; ++s) {
            const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(O_BASE + slots * user + s), page);
            noc_async_write_tile(p, output, scratch + s * page);
        }
        if constexpr (NBUF == 1) {
            noc_async_write_barrier();
        }
    }
    noc_async_write_barrier();

#elif defined(VTW_PORT)
    // gdn_conv_windows.cpp:4-58, one (user, page) task at a time: 5 staging tiles, one scratch.
    const uint32_t staged = get_write_ptr(cb_in);
    const uint32_t scratch = staged + tiles_in * page;
    for (uint32_t t = task_start; t < task_start + task_count; ++t) {
        const uint32_t user = t / PAGES;
        const uint32_t p = t % PAGES;
        const auto piece = TensorAccessor(piece_args, get_common_arg_val<uint32_t>(piece_user<USERS>(user)), page);
        noc_async_read_tile(p, piece, staged);
        for (uint32_t h = 0; h < history; ++h) {
            const auto source = TensorAccessor(history_args, get_common_arg_val<uint32_t>(H_BASE + history * user + h), page);
            noc_async_read_tile(p, source, staged + (1 + h) * page);
        }
        noc_async_read_barrier();
        for (uint32_t slot = 0; slot < slots; ++slot) {
            const uint32_t built = source_slot(slot);
            fill_words(scratch, page, 0);
#ifdef VTW_NEG_PAD
            fill_words(scratch + 2 * face, 2 * face, pad_word);
#endif
            asm volatile("" ::: "memory");
            for (uint32_t token = 0; token < ROWS; token++) {
                const uint32_t index = token + built;
                const uint32_t source_row = index < history ? 0 : index - history;
                const uint32_t source_offset = ((source_row / 16) * 512 + (source_row % 16) * 16) * 2;
                const uint32_t destination_row = ((token / 16) * 512 + (token % 16) * 16) * 2;
                for (uint32_t f = 0; f < 2; f++) {
                    const uint32_t tile = index < history ? history_index(index) + 1 : 0;
                    const auto source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        staged + tile * page + source_offset + f * face);
                    auto target = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + destination_row + f * face);
                    for (uint32_t word = 0; word < 8; word++) {
                        target[word] = source[word];
                    }
                }
            }
            asm volatile("" ::: "memory");
            const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(O_BASE + slots * user + slot), page);
            noc_async_write_tile(p, output, scratch);
            noc_async_write_barrier();
        }
    }

#else
#error "gdn_conv_windows_packed.cpp: define VTW_ROLE_READER, VTW_ROLE_WRITER or VTW_PORT"
#endif
}
