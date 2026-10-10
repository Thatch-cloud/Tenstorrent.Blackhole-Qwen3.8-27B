// Tile-native per-shard argmax, the FOLD kernel (tp4/samp-draft, QWEN_FAST_TP4_SHARD_ARGMAX, default off; S1).
//
// One core. It reads the scan tasks' partials pages (32 words each, word r = (column << 16) | bf16 bits of row r's best element in
// that task's run of columns), and for each row folds the row's tasks IN ASCENDING COLUMN ORDER (the tasks of one tile row are
// stored in that order) with a strict greater-than on the total order key, so a tie keeps the lowest column. NaN is the greatest
// key (the first NaN wins), -0 and +0 share a key: torch.argmax's rule, which combine_shards then applies across chips.
//
// Output (the readback layout, shared with the W3 MR work): three pages, each one page of 64 words, rows at and past `rows` zero.
//   ids    256 bytes, 64 uint32: row r = the shard-local column of the row's maximum;
//   values 128 bytes, 64 bf16:   row r = the maximum element's own bits;
//   words  256 bytes, 64 uint32: row r = (column << 16) | bits, the scan's own record for the winner (ids and values joined, so one mesh
//                                read can carry both: ids = word >> 16, bits = word & 0xffff).
// combine_shards reads ids and values with to_torch(...).reshape(-1)[:rows].
//
// Runtime args (seven words, always): partials address, ids address, values address, words address, rows (1..64), tasks per tile row,
// tile rows. The scratch CB holds FOLD_MAX_PAGES partials pages, then the three output pages (a define; 220 = 2 x the 110-worker grid).
// Single-core is the S1 baseline; tp4_shard_argmax_fold2.cpp is the same fold as a two-level tree.
#include "api/dataflow/dataflow_api.h"

#ifndef FOLD_MAX_PAGES
#define FOLD_MAX_PAGES 220
#endif

namespace {

constexpr uint32_t PAGE_BYTES = 128;
constexpr uint32_t IDS_BYTES = 256;
constexpr uint32_t VALUES_BYTES = 128;
constexpr uint32_t WORDS_BYTES = 256;
constexpr uint32_t MAX_PAGES = FOLD_MAX_PAGES;

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

void kernel_main() {
    constexpr auto partial_args = TensorAccessorArgs<0>();
    constexpr auto ids_args = TensorAccessorArgs<partial_args.next_compile_time_args_offset()>();
    constexpr auto values_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto words_args = TensorAccessorArgs<values_args.next_compile_time_args_offset()>();
    const auto partials = TensorAccessor(partial_args, get_arg_val<uint32_t>(0), PAGE_BYTES);
    const auto ids = TensorAccessor(ids_args, get_arg_val<uint32_t>(1), IDS_BYTES);
    const auto values = TensorAccessor(values_args, get_arg_val<uint32_t>(2), VALUES_BYTES);
    const auto words = TensorAccessor(words_args, get_arg_val<uint32_t>(3), WORDS_BYTES);
    const uint32_t rows = get_arg_val<uint32_t>(4);
    const uint32_t per_tile_row = get_arg_val<uint32_t>(5);
    const uint32_t tile_rows = get_arg_val<uint32_t>(6);
    const uint32_t pages = per_tile_row * tile_rows < MAX_PAGES ? per_tile_row * tile_rows : MAX_PAGES;
    const uint32_t scratch = get_write_ptr(0);
    for (uint32_t page = 0; page < pages; ++page) {
        noc_async_read(partials.get_noc_addr(page), scratch + page * PAGE_BYTES, PAGE_BYTES);
    }
    noc_async_read_barrier();

    const auto records = reinterpret_cast<volatile const uint32_t*>(scratch);
    constexpr uint32_t OUT = MAX_PAGES * PAGE_BYTES;
    auto out_ids = reinterpret_cast<volatile uint32_t*>(scratch + OUT);
    auto out_values = reinterpret_cast<volatile uint16_t*>(scratch + OUT + IDS_BYTES);
    auto out_words = reinterpret_cast<volatile uint32_t*>(scratch + OUT + IDS_BYTES + VALUES_BYTES);
    for (uint32_t row = 0; row < 64; ++row) {
        uint32_t column = 0;
        uint32_t bits = 0;
        if (row < rows) {
            const uint32_t tile_row = row >> 5;
            const uint32_t word = row & 31;
            const volatile uint32_t* run = records + tile_row * per_tile_row * 32 + word;
            uint32_t best_key = 0;  // every key is at least 0x80, so the first task always wins
            for (uint32_t task = 0; task < per_tile_row; ++task) {
                const uint32_t record = run[task * 32];
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
    noc_async_write(scratch + OUT, ids.get_noc_addr(0), IDS_BYTES);
    noc_async_write(scratch + OUT + IDS_BYTES, values.get_noc_addr(0), VALUES_BYTES);
    noc_async_write(scratch + OUT + IDS_BYTES + VALUES_BYTES, words.get_noc_addr(0), WORDS_BYTES);
    noc_async_write_barrier();
}
