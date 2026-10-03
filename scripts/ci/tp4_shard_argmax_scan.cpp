// Tile-native per-shard argmax, the SCAN kernel (tp4/samp-draft, QWEN_FAST_TP4_SHARD_ARGMAX, default off; S1).
//
// One task scans a run of tile columns [first, last) of ONE tile row of a chip's bf16 TILE logits shard and writes, for each of
// its `live` rows, one word (column << 16) | bits: the first column holding the row's maximum inside the run, and that element's
// bf16 bits. The columns are shard-local (< 62,080 < 65,536). Data-movement kernels only: the tiles are read straight from DRAM
// (no untilize) and the scan is word-wide on the data-movement RISC-V, two tasks per core (one per RISC, role 0 and 1).
//
// Order. The scan walks tiles in ascending order, per tile the 16-column half 0 then half 1, per half the 8 words in order, per
// word the low element then the high (the lower address is the lower column), and keeps a candidate only on a STRICT improvement,
// so a tie keeps the lowest column: the rule of torch.argmax, of the pinned sampler and of combine_shards.
//
// Fast path. A bf16's raw bits read as a signed 16-bit integer order every non-negative value numerically and put every
// non-negative value above every negative one. So for a row whose best raw value is > 0, the first column holding the best raw
// value is the first column holding the numeric maximum (a positive maximum has one bit pattern). A word's NaN test is one add:
// ((w & 0x7fff7fff) + 0x007f007f) has bit 15 (bit 31) set iff the low (high) half is a NaN, and no carry crosses halves.
// Slow path. A row with a NaN, or whose best raw value is <= 0 (a maximum of +0, or all values negative), is rescanned over the run
// with the full total order: NaN above everything (the first NaN wins), -0 equal to +0, negative values ordered by flipped
// magnitude. Real logits take the fast path.
//
// Runtime args (nine words, always): logits address, partials address, tile row, first tile column, one past the last tile column,
// live rows (1..32), task (the partials page this task writes), role (0 or 1: which scratch CB), tile columns per tile row.
// The partials page is 32 words (128 bytes): word r is row r, unused words are zero.
#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t TILE_BYTES = 2048;
constexpr uint32_t TILE_WORDS = 512;
constexpr uint32_t PAGE_BYTES = 128;
constexpr uint32_t MAX_TILES = 18;

// Total order on bf16 bits: larger is greater, every NaN is the same (greatest) key, -0 and +0 share a key.
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
    constexpr auto logits_args = TensorAccessorArgs<0>();
    constexpr auto partial_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    const auto logits = TensorAccessor(logits_args, get_arg_val<uint32_t>(0), TILE_BYTES);
    const auto partials = TensorAccessor(partial_args, get_arg_val<uint32_t>(1), PAGE_BYTES);
    const uint32_t tile_row = get_arg_val<uint32_t>(2);
    const uint32_t first = get_arg_val<uint32_t>(3);
    const uint32_t last = get_arg_val<uint32_t>(4);
    const uint32_t live = get_arg_val<uint32_t>(5);
    const uint32_t task = get_arg_val<uint32_t>(6);
    const uint32_t role = get_arg_val<uint32_t>(7);
    const uint32_t tile_columns = get_arg_val<uint32_t>(8);
    const uint32_t tiles = last - first < MAX_TILES ? last - first : MAX_TILES;
    const uint32_t scratch = get_write_ptr(role);
    // The result page sits after the tiles (the CB is MAX_TILES + 1 pages).
    auto result = reinterpret_cast<volatile uint32_t*>(scratch + MAX_TILES * TILE_BYTES);

    for (uint32_t tile = 0; tile < tiles; ++tile) {
        noc_async_read_tile(tile_row * tile_columns + first + tile, logits, scratch + tile * TILE_BYTES);
    }
    noc_async_read_barrier();

    const auto words = reinterpret_cast<volatile const uint32_t*>(scratch);
    for (uint32_t row = 0; row < live; ++row) {
        const uint32_t row_offset = (row >> 4) * 256 + (row & 15) * 8;  // words of this row's half 0 in a tile; half 1 is +128
        int32_t best = -65536;                                            // below every int16
        uint32_t best_at = 0;                                             // column inside the run
        uint32_t nan_flags = 0;
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            for (uint32_t half = 0; half < 2; ++half) {
                const volatile uint32_t* source = words + tile * TILE_WORDS + half * 128 + row_offset;
                const uint32_t at = tile * 32 + half * 16;
                for (uint32_t k = 0; k < 8; ++k) {
                    const uint32_t word = source[k];
                    nan_flags |= (word & 0x7fff7fffu) + 0x007f007fu;
                    const int32_t low = static_cast<int16_t>(word & 0xffffu);
                    const int32_t high = static_cast<int16_t>(word >> 16);
                    if (low > best) {
                        best = low;
                        best_at = at + 2 * k;
                    }
                    if (high > best) {
                        best = high;
                        best_at = at + 2 * k + 1;
                    }
                }
            }
        }
        uint32_t bits = static_cast<uint32_t>(best) & 0xffffu;
        if ((nan_flags & 0x80008000u) != 0 || best <= 0) {
            uint32_t best_key = 0;  // every key is at least 0x80, so the first element always wins
            for (uint32_t tile = 0; tile < tiles; ++tile) {
                for (uint32_t half = 0; half < 2; ++half) {
                    const volatile uint32_t* source = words + tile * TILE_WORDS + half * 128 + row_offset;
                    const uint32_t at = tile * 32 + half * 16;
                    for (uint32_t k = 0; k < 8; ++k) {
                        const uint32_t word = source[k];
                        const uint32_t low_key = order_key(word & 0xffffu);
                        if (low_key > best_key) {
                            best_key = low_key;
                            best_at = at + 2 * k;
                            bits = word & 0xffffu;
                        }
                        const uint32_t high_key = order_key(word >> 16);
                        if (high_key > best_key) {
                            best_key = high_key;
                            best_at = at + 2 * k + 1;
                            bits = word >> 16;
                        }
                    }
                }
            }
        }
        result[row] = ((first * 32 + best_at) << 16) | bits;
    }
    for (uint32_t row = live; row < 32; ++row) {
        result[row] = 0;
    }
    asm volatile("" ::: "memory");
    noc_async_write(scratch + MAX_TILES * TILE_BYTES, partials.get_noc_addr(task), PAGE_BYTES);
    noc_async_write_barrier();
}
