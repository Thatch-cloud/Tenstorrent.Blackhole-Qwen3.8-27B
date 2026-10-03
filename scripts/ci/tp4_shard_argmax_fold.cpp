// Tile-native per-shard argmax, the FOLD kernel (tp4/samp-draft, QWEN_FAST_TP4_SHARD_ARGMAX, default off; S1).
//
// One core. It reads the scan tasks' partials pages (32 words each, word r = (column << 16) | bf16 bits of row r's best element in
// that task's run of columns), and for each row folds the row's tasks IN ASCENDING COLUMN ORDER (the tasks of one tile row are
// stored in that order) with a strict greater-than on the total order key, so a tie keeps the lowest column. NaN is the greatest
// key (the first NaN wins), -0 and +0 share a key: torch.argmax's rule, which combine_shards then applies across chips.
//
// Output: ids, one 256-byte page of 64 uint32 (row r = the shard-local column), and values, one 128-byte page of 64 bf16 (row r =
// the maximum's bits); rows at and past `rows` are zero. combine_shards reads each with to_torch(...).reshape(-1)[:rows].
//
// Runtime args (six words, always): partials address, ids address, values address, rows (1..64), tasks per tile row, tile rows.
#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t PAGE_BYTES = 128;
constexpr uint32_t IDS_BYTES = 256;
constexpr uint32_t VALUES_BYTES = 128;
constexpr uint32_t MAX_PAGES = 220;

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
    const auto partials = TensorAccessor(partial_args, get_arg_val<uint32_t>(0), PAGE_BYTES);
    const auto ids = TensorAccessor(ids_args, get_arg_val<uint32_t>(1), IDS_BYTES);
    const auto values = TensorAccessor(values_args, get_arg_val<uint32_t>(2), VALUES_BYTES);
    const uint32_t rows = get_arg_val<uint32_t>(3);
    const uint32_t per_tile_row = get_arg_val<uint32_t>(4);
    const uint32_t tile_rows = get_arg_val<uint32_t>(5);
    const uint32_t pages = per_tile_row * tile_rows < MAX_PAGES ? per_tile_row * tile_rows : MAX_PAGES;
    const uint32_t scratch = get_write_ptr(0);
    for (uint32_t page = 0; page < pages; ++page) {
        noc_async_read(partials.get_noc_addr(page), scratch + page * PAGE_BYTES, PAGE_BYTES);
    }
    noc_async_read_barrier();

    const auto records = reinterpret_cast<volatile const uint32_t*>(scratch);
    auto out_ids = reinterpret_cast<volatile uint32_t*>(scratch + MAX_PAGES * PAGE_BYTES);
    auto out_values = reinterpret_cast<volatile uint16_t*>(scratch + MAX_PAGES * PAGE_BYTES + IDS_BYTES);
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
    }
    asm volatile("" ::: "memory");
    noc_async_write(scratch + MAX_PAGES * PAGE_BYTES, ids.get_noc_addr(0), IDS_BYTES);
    noc_async_write(scratch + MAX_PAGES * PAGE_BYTES + IDS_BYTES, values.get_noc_addr(0), VALUES_BYTES);
    noc_async_write_barrier();
}
