// Tile-native per-shard argmax, the TWO-LEVEL FOLD (fusion programme WP1, QWEN_FAST_TP4_SHARD_ARGMAX_FOLD2, default off; S1).
//
// Why. The one-core fold (tp4_shard_argmax_fold.cpp) walks rows x tasks records serially: 64 rows x 110 tasks (or 32 x 220) is 7,040
// strict-greater steps with the order-key branches, an estimated 25 cycles each, so an estimated 0.13 ms on one RISC-V while 109 workers idle.
// The measured S1 net saving (0.74 ms of trace_ms against the 1.03 ms it replaces) leaves 0.29 ms for scan + fold, and the fold is the part that
// serialises. This kernel is the same fold split over GROUPS cores, then one core folds the group winners (an estimated 25 us in all).
//
// One source, two launches, chosen by the FOLD2_LEVEL define:
//
// LEVEL 1, one core per group g (0 <= g < groups). Task t of a tile row belongs to group g when
// g * per_tile_row / groups <= t < (g + 1) * per_tile_row / groups: contiguous, ascending, every group non-empty (the host guarantees
// groups <= per_tile_row). The core reads its tasks' partials pages of EVERY tile row, and for each live row folds the row's tasks of
// its range in ascending order with the same strict greater-than on the same total order key as the one-core fold, then writes one
// record per row (the winning task's own word, column << 16 | bits) to page g of the group buffer (64 words; rows at and past `rows` 0).
// Runtime args (seven words): partials address, group buffer address, rows, tasks per tile row, tile rows, groups, group.
//
// LEVEL 2, one core. It reads the `groups` group pages and, per row, folds the group records in ascending group order with the same
// strict greater-than, then writes ids, values and words exactly as the one-core fold does (see its header for the three pages).
// Runtime args (six words): group buffer address, ids address, values address, words address, rows, groups.
//
// Exactness. The one-core fold is: best = first record, then replace on a STRICTLY greater key, tasks in ascending order. A strict-greater
// scan over a sequence equals the strict-greater scan over its contiguous blocks' winners in block order (the winner of a block is its first
// maximum-key record; the next block replaces it only on a strictly greater key), so the tree returns the same record as the one-core fold
// for every row: the first column holding the maximum, NaN above everything, -0 and +0 one key.
#include "api/dataflow/dataflow_api.h"

#ifndef FOLD2_LEVEL
#error "FOLD2_LEVEL must be defined to 1 or 2"
#endif
#ifndef FOLD2_GROUP_PAGES
#define FOLD2_GROUP_PAGES 29
#endif

namespace {

constexpr uint32_t PAGE_BYTES = 128;        // a scan partials page: 32 words
constexpr uint32_t GROUP_BYTES = 256;       // a group page: 64 words
constexpr uint32_t IDS_BYTES = 256;
constexpr uint32_t VALUES_BYTES = 128;
constexpr uint32_t WORDS_BYTES = 256;
constexpr uint32_t GROUP_PAGES = FOLD2_GROUP_PAGES;  // the most partials pages one level-1 core reads

inline uint32_t order_key(uint32_t bits) {
    const uint32_t magnitude = bits & 0x7fffu;
    if (magnitude > 0x7f80u) {
        return 0x10000u;
    }
    if (magnitude == 0) {
        return 0x8000u;
    }
    return (bits & 0x8000u) ? 0x8000u - magnitude : 0x8000u + magnitude;
}

}  // namespace

#if FOLD2_LEVEL == 1

void kernel_main() {
    constexpr auto partial_args = TensorAccessorArgs<0>();
    constexpr auto group_args = TensorAccessorArgs<partial_args.next_compile_time_args_offset()>();
    const auto partials = TensorAccessor(partial_args, get_arg_val<uint32_t>(0), PAGE_BYTES);
    const auto group_pages = TensorAccessor(group_args, get_arg_val<uint32_t>(1), GROUP_BYTES);
    const uint32_t rows = get_arg_val<uint32_t>(2);
    const uint32_t per_tile_row = get_arg_val<uint32_t>(3);
    const uint32_t tile_rows = get_arg_val<uint32_t>(4);
    const uint32_t groups = get_arg_val<uint32_t>(5);
    const uint32_t group = get_arg_val<uint32_t>(6);
    const uint32_t first = group * per_tile_row / groups;
    const uint32_t last = (group + 1) * per_tile_row / groups;
    const uint32_t count = last - first;  // tasks of this group per tile row
    const uint32_t scratch = get_write_ptr(0);

    // Page (tile row t, task first + i) lands at slot t * count + i; a slot past the scratch is never read (the host sizes the scratch).
    for (uint32_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t slot = tile_row * count + i;
            if (slot < GROUP_PAGES) {
                noc_async_read(partials.get_noc_addr(tile_row * per_tile_row + first + i), scratch + slot * PAGE_BYTES, PAGE_BYTES);
            }
        }
    }
    noc_async_read_barrier();

    const auto records = reinterpret_cast<volatile const uint32_t*>(scratch);
    auto out = reinterpret_cast<volatile uint32_t*>(scratch + GROUP_PAGES * PAGE_BYTES);
    for (uint32_t row = 0; row < 64; ++row) {
        uint32_t winner = 0;
        if (row < rows) {
            const uint32_t tile_row = row >> 5;
            const uint32_t word = row & 31;
            const volatile uint32_t* run = records + tile_row * count * 32 + word;
            uint32_t best_key = 0;  // every key is at least 0x80, so the first task always wins
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t record = run[i * 32];
                const uint32_t key = order_key(record & 0xffffu);
                if (key > best_key) {
                    best_key = key;
                    winner = record;
                }
            }
        }
        out[row] = winner;
    }
    asm volatile("" ::: "memory");
    noc_async_write(scratch + GROUP_PAGES * PAGE_BYTES, group_pages.get_noc_addr(group), GROUP_BYTES);
    noc_async_write_barrier();
}

#else

void kernel_main() {
    constexpr auto group_args = TensorAccessorArgs<0>();
    constexpr auto ids_args = TensorAccessorArgs<group_args.next_compile_time_args_offset()>();
    constexpr auto values_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto words_args = TensorAccessorArgs<values_args.next_compile_time_args_offset()>();
    const auto group_pages = TensorAccessor(group_args, get_arg_val<uint32_t>(0), GROUP_BYTES);
    const auto ids = TensorAccessor(ids_args, get_arg_val<uint32_t>(1), IDS_BYTES);
    const auto values = TensorAccessor(values_args, get_arg_val<uint32_t>(2), VALUES_BYTES);
    const auto words = TensorAccessor(words_args, get_arg_val<uint32_t>(3), WORDS_BYTES);
    const uint32_t rows = get_arg_val<uint32_t>(4);
    const uint32_t groups = get_arg_val<uint32_t>(5);
    const uint32_t scratch = get_write_ptr(0);

    for (uint32_t group = 0; group < groups; ++group) {
        noc_async_read(group_pages.get_noc_addr(group), scratch + group * GROUP_BYTES, GROUP_BYTES);
    }
    noc_async_read_barrier();

    const auto records = reinterpret_cast<volatile const uint32_t*>(scratch);
    const uint32_t out_base = scratch + groups * GROUP_BYTES;  // the three output pages follow the group pages
    auto out_ids = reinterpret_cast<volatile uint32_t*>(out_base);
    auto out_values = reinterpret_cast<volatile uint16_t*>(out_base + IDS_BYTES);
    auto out_words = reinterpret_cast<volatile uint32_t*>(out_base + IDS_BYTES + VALUES_BYTES);
    for (uint32_t row = 0; row < 64; ++row) {
        uint32_t column = 0;
        uint32_t bits = 0;
        if (row < rows) {
            uint32_t best_key = 0;  // every key is at least 0x80, so the first group always wins
            for (uint32_t group = 0; group < groups; ++group) {
                const uint32_t record = records[group * 64 + row];
                const uint32_t key = order_key(record & 0xffffu);
                if (key > best_key) {
                    best_key = key;
                    column = record >> 16;
                    bits = record & 0xffffu;
                }
            }
        }
        out_ids[row] = column;
        out_values[row] = static_cast<uint16_t>(bits);
        out_words[row] = (column << 16) | bits;
    }
    asm volatile("" ::: "memory");
    noc_async_write(out_base, ids.get_noc_addr(0), IDS_BYTES);
    noc_async_write(out_base + IDS_BYTES, values.get_noc_addr(0), VALUES_BYTES);
    noc_async_write(out_base + IDS_BYTES + VALUES_BYTES, words.get_noc_addr(0), WORDS_BYTES);
    noc_async_write_barrier();
}

#endif
