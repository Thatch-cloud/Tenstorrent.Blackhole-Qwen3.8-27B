// Q4 probe-local copy of scripts/ci/draft_convolution_fused_io.cpp, extended to a 64-row block (E1 / E1b,
// quad-draft-plan.md section 3.5). NOT SHIPPED: run_card_b.sh mounts it beside the harness. The per-page
// arithmetic (the served draft_convolution_fused_compute.cpp) is untouched; only which tiles a page reads and
// which rows carry change. Against the served kernel:
//   - pages run over 160 per tile row (320 at 64 rows) with the worker count as the stride (runtime arg 8), so
//     E1 is 80 workers x 4 pages and E1b 110 workers x 3 (workers 100-109: 2);
//   - tile_row = page / 160, col = page % 160: the hidden and output tile is `page`, the base tiles are read at
//     `col` (the bases are one tile row), the dynamic tile is tile_row * 10 + col / 16;
//   - `rows` and the seam word are per tile row: the live rows of tile row t are min(32, rows - 32 t) and its
//     seams are runtime arg 9 (tile row 0) or 10 (tile row 1). Rows 32 and 48 begin users 2 and 3, so no causal
//     carry crosses a tile row; the harness refuses a 64-row block without a seam at row 32.
// At rows <= 32 with 80 workers this reads and writes exactly what the served kernel does.
#include "api/dataflow/dataflow_api.h"

uint32_t lane(uint32_t row, uint32_t column) {
    return (row / 16) * 512 + (column / 16) * 256 + (row % 16) * 16 + column % 16;
}

void kernel_main() {
    constexpr auto hidden_args = TensorAccessorArgs<0>();
    constexpr auto dynamic0_args = TensorAccessorArgs<hidden_args.next_compile_time_args_offset()>();
    constexpr auto dynamic1_args = TensorAccessorArgs<dynamic0_args.next_compile_time_args_offset()>();
    constexpr auto base0_args = TensorAccessorArgs<dynamic1_args.next_compile_time_args_offset()>();
    constexpr auto base1_args = TensorAccessorArgs<base0_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<base1_args.next_compile_time_args_offset()>();
    const auto hidden = TensorAccessor(hidden_args, get_arg_val<uint32_t>(0), 2048);
    const auto dynamic0 = TensorAccessor(dynamic0_args, get_arg_val<uint32_t>(1), 2048);
    const auto dynamic1 = TensorAccessor(dynamic1_args, get_arg_val<uint32_t>(2), 2048);
    const auto base0 = TensorAccessor(base0_args, get_arg_val<uint32_t>(3), 2048);
    const auto base1 = TensorAccessor(base1_args, get_arg_val<uint32_t>(4), 2048);
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(5), 2048);
    const uint32_t rows = get_arg_val<uint32_t>(6);
    const uint32_t worker = get_arg_val<uint32_t>(7);
    const uint32_t workers = get_arg_val<uint32_t>(8);
    // Bit r of a tile row's word set means row r of that tile row begins a packed user's segment, so its causal
    // shift reads zero instead of the row above. Bit 0 is always set in both words.
    const uint32_t seams_low = get_arg_val<uint32_t>(9);
    const uint32_t seams_high = get_arg_val<uint32_t>(10);
    const uint32_t tile_rows = (rows + 31) / 32;
    const uint32_t scratch = get_write_ptr(1);
    for (uint32_t page = worker; page < 160 * tile_rows; page += workers) {
        const uint32_t tile_row = page / 160;
        const uint32_t col = page % 160;
        const uint32_t live = rows - 32 * tile_row < 32 ? rows - 32 * tile_row : 32;
        const uint32_t seams = tile_row == 0 ? seams_low : seams_high;
        const uint32_t dynamic_tile = tile_row * 10 + col / 16;
        cb_reserve_back(0, 7);
        const uint32_t destination = get_write_ptr(0);
        noc_async_read_tile(page, hidden, destination);
        noc_async_read_tile(col, base0, destination + 2 * 2048);
        noc_async_read_tile(col, base1, destination + 4 * 2048);
        noc_async_read_tile(dynamic_tile, dynamic0, scratch);
        noc_async_read_tile(dynamic_tile, dynamic1, scratch + 2048);
        noc_async_read_barrier();
        auto tiles = reinterpret_cast<volatile uint16_t*>(destination);
        const auto coefficients = reinterpret_cast<volatile const uint16_t*>(scratch);
        for (uint32_t row = 0; row < 32; ++row) {
            for (uint32_t column = 0; column < 32; ++column) {
                const uint32_t element = lane(row, column);
                const uint32_t group = lane(row, 2 * (col % 16) + column / 16);
                const bool carries = row < live && !((seams >> row) & 1u);
                tiles[1024 + element] = carries ? tiles[lane(row - 1, column)] : 0;
                tiles[2 * 1024 + element] = tiles[2 * 1024 + lane(0, column)];
                tiles[3 * 1024 + element] = row < live ? coefficients[group] : 0;
                tiles[4 * 1024 + element] = tiles[4 * 1024 + lane(0, column)];
                tiles[5 * 1024 + element] = row < live ? coefficients[1024 + group] : 0;
                tiles[6 * 1024 + element] = 0;
                if (row >= live) {
                    tiles[element] = 0;
                }
            }
        }
        asm volatile("" ::: "memory");
        cb_push_back(0, 7);
        cb_wait_front(16, 1);
        noc_async_write_tile(page, output, get_read_ptr(16));
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
}
