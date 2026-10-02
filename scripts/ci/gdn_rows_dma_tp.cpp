// Half-tile row mover for the packed 64-row GDN block (QWEN_FAST_TP4_GDN_GLUE / _GDN_BLOCK_CONV, tp4/vglue V2 and V1).
//
// One task composes ONE destination tile from two half-tile sources: rows 0-15 (faces 0 and 1, bytes 0-1023 of the tile)
// come from source A and rows 16-31 (faces 2 and 3, bytes 1024-2047) from source B. Each source is a page of any tensor
// the launch's source accessor describes, one of its two halves, copied raw or canonicalised, or nothing (zeros). The
// four-user block keeps user u in tile row u / 2, half u % 2, and a per-user tensor keeps its 16 rows in half 0 with
// half 1 zero, so splitting the block, merging the users' outputs and re-stacking windows are all this one move.
//
// Canonicalisation is the value rule of the served untilize / slice / tilize round trip (gdn_prefill_conv_exact.py
// "canonical on both state paths", card M pcx-20260923T063828): a bf16 with a zero exponent (-0 and every denormal) comes
// out +0. CANON_DENORM (default on) selects it; off, only 0x8000 becomes 0. Everything else is copied bit for bit.
//
// Runtime args: [tasks, then per task: destination address, destination page, then per half (A, B): source address,
// source page, mode]. Mode 0 = zeros, 1 / 2 = raw half 0 / half 1 of the source page, 3 / 4 = canonical half 0 / half 1.
// Compile-time args: the source accessor, then the destination accessor (every source of a launch shares one layout, and
// so does every destination), then CAPACITY, the most tasks any core carries: every core's runtime args are padded to
// 1 + 8 * CAPACITY words, and CAPACITY being a compile-time arg keeps launches with different list lengths in different
// program cache entries. Eight tasks are in flight per barrier.
#include "api/dataflow/dataflow_api.h"

#ifndef CANON_DENORM
#define CANON_DENORM 1
#endif

constexpr uint32_t LANES = 8;
constexpr uint32_t TASK_WORDS = 8;
constexpr uint32_t HALF_BYTES = 1024;

FORCE_INLINE uint32_t canonical_half(uint32_t value) {
#if CANON_DENORM
    return (value & 0x7F80u) == 0 ? 0u : value;
#else
    return value == 0x8000u ? 0u : value;
#endif
}

void kernel_main() {
    constexpr auto source_args = TensorAccessorArgs<0>();
    constexpr auto destination_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    constexpr uint32_t CAPACITY = get_compile_time_arg_val(destination_args.next_compile_time_args_offset());
    const uint32_t given = get_arg_val<uint32_t>(0);
    const uint32_t tasks = given < CAPACITY ? given : CAPACITY;  // never reads past 1 + TASK_WORDS * CAPACITY
    const uint32_t scratch = get_write_ptr(0);
    for (uint32_t first = 0; first < tasks; first += LANES) {
        const uint32_t lanes = tasks - first < LANES ? tasks - first : LANES;
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            const uint32_t base = 1 + (first + lane) * TASK_WORDS;
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t mode = get_arg_val<uint32_t>(base + 2 + half * 3 + 2);
                if (mode == 0) { continue; }
                const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(base + 2 + half * 3), 2048);
                const uint32_t page = get_arg_val<uint32_t>(base + 2 + half * 3 + 1);
                const uint32_t from = ((mode - 1) & 1) * HALF_BYTES;
                noc_async_read(source.get_noc_addr(page, from), scratch + lane * 2048 + half * HALF_BYTES, HALF_BYTES);
            }
        }
        noc_async_read_barrier();
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            const uint32_t base = 1 + (first + lane) * TASK_WORDS;
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t mode = get_arg_val<uint32_t>(base + 2 + half * 3 + 2);
                auto words = reinterpret_cast<volatile uint32_t*>(scratch + lane * 2048 + half * HALF_BYTES);
                if (mode == 0) {
                    for (uint32_t word = 0; word < HALF_BYTES / 4; ++word) { words[word] = 0; }
                } else if (mode >= 3) {
                    for (uint32_t word = 0; word < HALF_BYTES / 4; ++word) {
                        const uint32_t value = words[word];
                        words[word] = canonical_half(value & 0xFFFFu) | (canonical_half(value >> 16) << 16);
                    }
                }
            }
        }
        asm volatile("" ::: "memory");
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            const uint32_t base = 1 + (first + lane) * TASK_WORDS;
            const auto destination = TensorAccessor(destination_args, get_arg_val<uint32_t>(base), 2048);
            noc_async_write_tile(get_arg_val<uint32_t>(base + 1), destination, scratch + lane * 2048);
        }
        noc_async_write_barrier();
    }
}
