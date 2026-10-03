// The drafter's fused-convolution INPUT kernel at four cards (tp4/samp-draft, QWEN_FAST_TP4_DRAFT_CONV, default off).
//
// It builds, for each page, exactly the seven-tile input image the served kernels build (draft_convolution_fused_io.cpp for one
// or two users, quad_conv_io.cpp for the 64-row quad block): tile 0 hidden (rows at and above `live` zeroed), tile 1 the causal
// shift (row r holds row r-1 of tile 0, or zero for a row at or past `live` and for a row that begins a packed segment), tiles 2
// and 4 the two base kernels broadcast from row 0, tiles 3 and 5 the two dynamic coefficients (the value for row r and 16-column
// half h is one bf16 replicated across the half), tile 6 the constant zero tile. Byte for byte the same image, so the served
// compute kernel (draft_convolution_fused_compute.cpp) produces the same output and the drafter proposes the same tokens.
//
// What differs is the cost. The served kernels fill the image one bf16 element at a time (five lane() evaluations, six 16-bit
// stores per element, 2,048 elements a page: about 150 us of a 177 us call) and wait for each page's output before they read the
// next page. Here every row of every face is 8 aligned 32-bit words, so each step is a word copy or fill, and CB 0 holds two
// seven-tile slots so the next page is read and prepared while the compute kernel works on this one (the output is written by
// draft_conv_out.cpp on the other data-movement core).
//
// Runtime args: five buffer addresses (hidden, dynamic0, dynamic1, base0, base1), rows, worker, workers, seams_low, seams_high.
// Page p of tile row p / 160 is column p % 160; workers take pages worker, worker + workers, ... (the quad's page plan; at rows <=
// 32 and 80 workers it is the pair's). Bit r of a tile row's seam word means row r begins a packed user's segment; bit 0 is always
// set, so row 0 never shifts in a row of the previous tile (the served kernels never read it either).
#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t TILE_BYTES = 2048;
constexpr uint32_t TILE_WORDS = 512;

// Word offset, inside a tile, of the 16 columns `half` of row `row` (8 words). A tile is four 16 x 16 faces in the order
// (rows 0-15, cols 0-15), (rows 0-15, cols 16-31), (rows 16-31, cols 0-15), (rows 16-31, cols 16-31); each face is 16 rows of
// 16 bf16, so a row of a face is 32 bytes.
inline uint32_t row_words(uint32_t row, uint32_t half) { return (row >> 4) * 256 + half * 128 + (row & 15) * 8; }

inline void copy_row(volatile uint32_t* to, const volatile uint32_t* from) {
    for (uint32_t word = 0; word < 8; ++word) {
        to[word] = from[word];
    }
}

inline void fill_row(volatile uint32_t* to, uint32_t value) {
    for (uint32_t word = 0; word < 8; ++word) {
        to[word] = value;
    }
}

// The element index, in a 1,024-element tile, of row `row`, column `column` (the served lane()).
inline uint32_t lane(uint32_t row, uint32_t column) {
    return (row / 16) * 512 + (column / 16) * 256 + (row % 16) * 16 + column % 16;
}

// Build tiles 1-5 of the image at `destination` (tile 0, tile 2 and tile 4 already hold the NoC-read hidden and base tiles, the
// dynamic coefficient tiles sit in `scratch`) and zero tile 0's rows at and above `live`.
inline void prepare(uint32_t destination, uint32_t scratch, uint32_t col, uint32_t live, uint32_t seams) {
    auto hidden = reinterpret_cast<volatile uint32_t*>(destination);
    auto shift = hidden + TILE_WORDS;
    auto base0 = hidden + 2 * TILE_WORDS;
    auto dynamic0 = hidden + 3 * TILE_WORDS;
    auto base1 = hidden + 4 * TILE_WORDS;
    auto dynamic1 = hidden + 5 * TILE_WORDS;
    const auto coefficients0 = reinterpret_cast<volatile const uint16_t*>(scratch);
    const auto coefficients1 = coefficients0 + 1024;
    const uint32_t first_column = 2 * (col % 16);
    seams |= 1u;
    for (uint32_t row = 0; row < 32; ++row) {
        const bool carries = row < live && !((seams >> row) & 1u);
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t at = row_words(row, half);
            if (carries) {
                copy_row(shift + at, hidden + row_words(row - 1, half));
            } else {
                fill_row(shift + at, 0);
            }
            if (row != 0) {
                copy_row(base0 + at, base0 + row_words(0, half));
                copy_row(base1 + at, base1 + row_words(0, half));
            }
            if (row < live) {
                const uint32_t index = lane(row, first_column + half);
                const uint32_t low = coefficients0[index];
                const uint32_t high = coefficients1[index];
                fill_row(dynamic0 + at, (low << 16) | low);
                fill_row(dynamic1 + at, (high << 16) | high);
            } else {
                fill_row(dynamic0 + at, 0);
                fill_row(dynamic1 + at, 0);
                fill_row(hidden + at, 0);
            }
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr auto hidden_args = TensorAccessorArgs<0>();
    constexpr auto dynamic0_args = TensorAccessorArgs<hidden_args.next_compile_time_args_offset()>();
    constexpr auto dynamic1_args = TensorAccessorArgs<dynamic0_args.next_compile_time_args_offset()>();
    constexpr auto base0_args = TensorAccessorArgs<dynamic1_args.next_compile_time_args_offset()>();
    constexpr auto base1_args = TensorAccessorArgs<base0_args.next_compile_time_args_offset()>();
    const auto hidden = TensorAccessor(hidden_args, get_arg_val<uint32_t>(0), TILE_BYTES);
    const auto dynamic0 = TensorAccessor(dynamic0_args, get_arg_val<uint32_t>(1), TILE_BYTES);
    const auto dynamic1 = TensorAccessor(dynamic1_args, get_arg_val<uint32_t>(2), TILE_BYTES);
    const auto base0 = TensorAccessor(base0_args, get_arg_val<uint32_t>(3), TILE_BYTES);
    const auto base1 = TensorAccessor(base1_args, get_arg_val<uint32_t>(4), TILE_BYTES);
    const uint32_t rows = get_arg_val<uint32_t>(5);
    const uint32_t worker = get_arg_val<uint32_t>(6);
    const uint32_t workers = get_arg_val<uint32_t>(7);
    const uint32_t seams_low = get_arg_val<uint32_t>(8);
    const uint32_t seams_high = get_arg_val<uint32_t>(9);
    const uint32_t tile_rows = (rows + 31) / 32;
    const uint32_t scratch = get_write_ptr(1);

    // CB 0 is two seven-tile slots and nothing else ever writes tile 6 of a slot (the NoC reads fill tiles 0, 2 and 4, prepare()
    // tiles 1, 3, 5 and the rows of tile 0), so the constant zero tile is written once here, before the first reserve.
    const uint32_t slots = get_write_ptr(0);
    for (uint32_t slot = 0; slot < 2; ++slot) {
        auto zero = reinterpret_cast<volatile uint32_t*>(slots + (slot * 7 + 6) * TILE_BYTES);
        for (uint32_t word = 0; word < TILE_WORDS; ++word) {
            zero[word] = 0;
        }
    }

    for (uint32_t page = worker; page < 160 * tile_rows; page += workers) {
        const uint32_t tile_row = page / 160;
        const uint32_t col = page % 160;
        const uint32_t live = rows - 32 * tile_row < 32 ? rows - 32 * tile_row : 32;
        const uint32_t seams = tile_row == 0 ? seams_low : seams_high;
        const uint32_t dynamic_tile = tile_row * 10 + col / 16;
        cb_reserve_back(0, 7);
        const uint32_t destination = get_write_ptr(0);
        noc_async_read_tile(page, hidden, destination);
        noc_async_read_tile(col, base0, destination + 2 * TILE_BYTES);
        noc_async_read_tile(col, base1, destination + 4 * TILE_BYTES);
        noc_async_read_tile(dynamic_tile, dynamic0, scratch);
        noc_async_read_tile(dynamic_tile, dynamic1, scratch + TILE_BYTES);
        noc_async_read_barrier();
        prepare(destination, scratch, col, live, seams);
        asm volatile("" ::: "memory");
        cb_push_back(0, 7);
    }
}
