// Quarter-tile row mover for the octo block's packed 64-row GDN block (QWEN_FAST_OCTO_GLUE8, tp4/octo-2 lever 2).
//
// The twin of gdn_rows_dma_tp.cpp for EIGHT-row users: the octo block keeps user u in tile row u / 4, quarter u % 4 (rows 8q .. 8q + 7 of a 32 x 32 tile), and a per-user
// tensor keeps its 8 rows in quarter 0 with quarters 1-3 zero. Splitting the block, merging the users' outputs, canonicalising the block and stacking / unstacking windows are
// all this one move. gdn_rows_dma_tp.cpp (half tiles, sixteen-row M3 users) is unchanged and is not read by the octo block.
//
// A tile is four 16 x 16 faces of 512 bytes (face f at byte 512 f, row r of a face at 32 r). Quarter q (rows 8q .. 8q + 7) is therefore two 256-byte chunks: rows 8 (q % 2) ..
// of faces 2 (q / 2) and 2 (q / 2) + 1, at byte offsets quarter_offset(q) = 1024 (q / 2) + 256 (q % 2) and quarter_offset(q) + 512. One task composes ONE destination tile from
// four quarter sources: destination quarter j (bytes quarter_offset(j) and + 512) comes from source quarter q of a page of any tensor the launch's source accessor describes,
// copied raw or canonicalised, or from nothing (zeros).
//
// Canonicalisation is the value rule of the served untilize / slice / tilize round trip (gdn_prefill_conv_exact.py "canonical on both state paths", card M): a bf16 with a zero
// exponent (-0 and every denormal) comes out +0. CANON_DENORM (default on) selects it; off, only 0x8000 becomes 0. Everything else is copied bit for bit.
//
// Runtime args (padded to 1 + NSRC + NDST + TASK_WORDS * CAPACITY words): [tasks, NSRC source addresses, NDST destination addresses, then per task five words:
//   word 0     destination index (bits 16..23) and destination page (bits 0..15),
//   words 1-4  destination quarter j = 0..3: source index (bits 24..31), mode (bits 16..23), source page (bits 0..15).
// Mode 0 = zeros (index and page ignored), 1..4 = raw quarter mode - 1 of the source page, 5..8 = canonical quarter mode - 5.]
// Compile-time args: the source accessor, then the destination accessor (every source of a launch shares one layout, and so does every destination), then NSRC, NDST and
// CAPACITY (the most tasks any core carries): all three are program-cache key material, because generic_op caches on compile-time args and not on runtime-arg lengths.
// Eight tasks are in flight per barrier.
#include "api/dataflow/dataflow_api.h"

#ifndef CANON_DENORM
#define CANON_DENORM 1
#endif

constexpr uint32_t LANES = 8;
constexpr uint32_t TASK_WORDS = 5;
constexpr uint32_t TILE_BYTES = 2048;
constexpr uint32_t FACE_BYTES = 512;
constexpr uint32_t CHUNK_BYTES = 256;

FORCE_INLINE uint32_t canonical_half(uint32_t value) {
#if CANON_DENORM
    return (value & 0x7F80u) == 0 ? 0u : value;
#else
    return value == 0x8000u ? 0u : value;
#endif
}

// Byte offset of quarter q (rows 8q .. 8q + 7) inside the left face of its row group; the right face is FACE_BYTES further on.
FORCE_INLINE uint32_t quarter_offset(uint32_t quarter) {
    return (quarter >> 1) * (2 * FACE_BYTES) + (quarter & 1) * CHUNK_BYTES;
}

void kernel_main() {
    constexpr auto source_args = TensorAccessorArgs<0>();
    constexpr auto destination_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    constexpr uint32_t NSRC = get_compile_time_arg_val(destination_args.next_compile_time_args_offset());
    constexpr uint32_t NDST = get_compile_time_arg_val(destination_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t CAPACITY = get_compile_time_arg_val(destination_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t SOURCE_TABLE = 1;
    constexpr uint32_t DESTINATION_TABLE = SOURCE_TABLE + NSRC;
    constexpr uint32_t TASKS = DESTINATION_TABLE + NDST;
    const uint32_t given = get_arg_val<uint32_t>(0);
    const uint32_t tasks = given < CAPACITY ? given : CAPACITY;  // never reads past 1 + NSRC + NDST + TASK_WORDS * CAPACITY
    const uint32_t scratch = get_write_ptr(0);
    for (uint32_t first = 0; first < tasks; first += LANES) {
        const uint32_t lanes = tasks - first < LANES ? tasks - first : LANES;
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            const uint32_t base = TASKS + (first + lane) * TASK_WORDS;
            for (uint32_t quarter = 0; quarter < 4; ++quarter) {
                const uint32_t word = get_arg_val<uint32_t>(base + 1 + quarter);
                const uint32_t mode = (word >> 16) & 0xFFu;
                if (mode == 0) { continue; }
                const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(SOURCE_TABLE + (word >> 24)), TILE_BYTES);
                const uint32_t from = quarter_offset((mode - 1) & 3);
                const uint32_t to = scratch + lane * TILE_BYTES + quarter_offset(quarter);
                noc_async_read(source.get_noc_addr(word & 0xFFFFu, from), to, CHUNK_BYTES);
                noc_async_read(source.get_noc_addr(word & 0xFFFFu, from + FACE_BYTES), to + FACE_BYTES, CHUNK_BYTES);
            }
        }
        noc_async_read_barrier();
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            const uint32_t base = TASKS + (first + lane) * TASK_WORDS;
            for (uint32_t quarter = 0; quarter < 4; ++quarter) {
                const uint32_t mode = (get_arg_val<uint32_t>(base + 1 + quarter) >> 16) & 0xFFu;
                for (uint32_t face = 0; face < 2; ++face) {
                    auto words = reinterpret_cast<volatile uint32_t*>(scratch + lane * TILE_BYTES + quarter_offset(quarter) + face * FACE_BYTES);
                    if (mode == 0) {
                        for (uint32_t word = 0; word < CHUNK_BYTES / 4; ++word) { words[word] = 0; }
                    } else if (mode >= 5) {
                        for (uint32_t word = 0; word < CHUNK_BYTES / 4; ++word) {
                            const uint32_t value = words[word];
                            words[word] = canonical_half(value & 0xFFFFu) | (canonical_half(value >> 16) << 16);
                        }
                    }
                }
            }
        }
        asm volatile("" ::: "memory");
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            const uint32_t base = TASKS + (first + lane) * TASK_WORDS;
            const uint32_t head = get_arg_val<uint32_t>(base);
            const auto destination = TensorAccessor(destination_args, get_arg_val<uint32_t>(DESTINATION_TABLE + ((head >> 16) & 0xFFu)), TILE_BYTES);
            noc_async_write_tile(head & 0xFFFFu, destination, scratch + lane * TILE_BYTES);
        }
        noc_async_write_barrier();
    }
}
