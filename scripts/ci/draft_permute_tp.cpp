// Tile / quarter-tile permutation behind the drafter's K/V assembly, query fold and output unfold (tp4/fx-wp7, QWEN_FAST_DRAFT_PERMUTE, default off; F-F2).
//
// The drafter quad (and the pair and the octo pass) assemble each layer's K and V by cutting every user's rows out of the shared 64-row block with slices that start inside a
// tile, round-tripping them through untilize / tilize, and joining them with the users' cached banks in a row-major concat that is tilized again: about 285 us a tensor and layer.
// The query fold and the output unfold move eight- or sixteen-row blocks between heads the same way. Every one of those moves is a permutation of the bytes of 32 x 32 tiles
// (head_dim 128 is four whole tiles), plus one value rule: the served untilize / tilize round trip maps a bf16 with a zero exponent (-0 and every denormal) to +0. This kernel
// does the moves in one launch and applies the rule to exactly the tiles / quarters the host planner marks canonical (draft_permute_tp.py derives the set from the served
// composition op by op; the planner marks nothing it cannot account for).
//
// A tile is four 16 x 16 faces of 512 bytes (face f at byte 512 f, row r of a face at 32 r). Quarter q (rows 8q .. 8q + 7) is two 256-byte chunks: rows 8 (q % 2) .. of faces
// 2 (q / 2) and 2 (q / 2) + 1, at byte offsets quarter_offset(q) = 1024 (q / 2) + 256 (q % 2) and quarter_offset(q) + 512 (gdn_rows_dma8_tp.cpp's mover; the same arithmetic).
//
// Two record types, in a core's runtime args after the address tables:
//   RUN  (4 words)  copies COUNT consecutive tiles: [TYPE_RUN << 28 | canonical << 24 | destination index << 8 | source index, COUNT, destination page, source page].
//                   The tiles are read eight at a time into the scratch CB, canonicalised in L1 when the record says so, and written.
//   MIX  (6 words)  composes ONE destination tile from four quarter sources: [TYPE_MIX << 28 | destination index << 8, destination page, then per destination quarter
//                   0..3: source index << 24 | mode << 16 | source page]. Mode 0 = zeros, 1..4 = raw quarter mode - 1 of the source page, 5..8 = canonical quarter mode - 5.
// Runtime args: [words used after the tables, NSRC source addresses, NDST destination addresses, records..., zero padding to CAPACITY words]. Compile-time args: the source
// accessor, the destination accessor (every source of a launch shares one layout, and so does every destination), then CAPACITY (the padded length of every core's list),
// NSRC, NDST and SCRATCH_CB (the CB index of this processor's eight-tile scratch). CAPACITY is program-cache key material: generic_op caches on compile-time args and not on
// runtime-arg lengths. A malformed or overlong list stops the loop; the kernel never reads past CAPACITY words.
//
// Canonicalisation is the value rule of gdn_rows_dma_tp.cpp (+0 where the exponent field is zero; CANON_DENORM, default on, selects it; off, only 0x8000 becomes 0). The
// default form works on two bf16 lanes of a 32-bit word without a branch: a lane's exponent field is nonzero iff adding 0x7F80 to the masked field sets bit 15 (the sum
// cannot carry across lanes), and the lane's mask is that bit smeared down.
#include "api/dataflow/dataflow_api.h"

#ifndef CANON_DENORM
#define CANON_DENORM 1
#endif

constexpr uint32_t LANES = 8;
constexpr uint32_t TILE_BYTES = 2048;
constexpr uint32_t FACE_BYTES = 512;
constexpr uint32_t CHUNK_BYTES = 256;
constexpr uint32_t TYPE_RUN = 1;
constexpr uint32_t TYPE_MIX = 2;
constexpr uint32_t RUN_WORDS = 4;
constexpr uint32_t MIX_WORDS = 6;

#if CANON_DENORM
FORCE_INLINE uint32_t canonical_pair(uint32_t word) {
    const uint32_t present = ((word & 0x7F807F80u) + 0x7F807F80u) & 0x80008000u;
    return word & (present | (present - (present >> 15)));
}
#else
FORCE_INLINE uint32_t canonical_half(uint32_t value) {
    return value == 0x8000u ? 0u : value;
}
FORCE_INLINE uint32_t canonical_pair(uint32_t word) {
    return canonical_half(word & 0xFFFFu) | (canonical_half(word >> 16) << 16);
}
#endif

// Canonicalise `words` 32-bit words (a multiple of four) at L1 address `at`.
FORCE_INLINE void canonicalise(uint32_t at, uint32_t words) {
    auto data = reinterpret_cast<volatile uint32_t*>(at);
    for (uint32_t word = 0; word < words; word += 4) {
        const uint32_t a = data[word];
        const uint32_t b = data[word + 1];
        const uint32_t c = data[word + 2];
        const uint32_t d = data[word + 3];
        data[word] = canonical_pair(a);
        data[word + 1] = canonical_pair(b);
        data[word + 2] = canonical_pair(c);
        data[word + 3] = canonical_pair(d);
    }
}

// Byte offset of quarter q inside the left face of its row group; the right face is FACE_BYTES further on.
FORCE_INLINE uint32_t quarter_offset(uint32_t quarter) {
    return (quarter >> 1) * (2 * FACE_BYTES) + (quarter & 1) * CHUNK_BYTES;
}

void kernel_main() {
    constexpr auto source_args = TensorAccessorArgs<0>();
    constexpr auto destination_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    constexpr uint32_t TAIL = destination_args.next_compile_time_args_offset();
    constexpr uint32_t CAPACITY = get_compile_time_arg_val(TAIL);
    constexpr uint32_t NSRC = get_compile_time_arg_val(TAIL + 1);
    constexpr uint32_t NDST = get_compile_time_arg_val(TAIL + 2);
    constexpr uint32_t SCRATCH_CB = get_compile_time_arg_val(TAIL + 3);
    constexpr uint32_t SOURCE_TABLE = 1;
    constexpr uint32_t DESTINATION_TABLE = SOURCE_TABLE + NSRC;
    constexpr uint32_t RECORDS = DESTINATION_TABLE + NDST;
    const uint32_t scratch = get_write_ptr(SCRATCH_CB);
    const uint32_t given = RECORDS + get_arg_val<uint32_t>(0);
    const uint32_t end = given < CAPACITY ? given : CAPACITY;  // never reads past CAPACITY words
    uint32_t at = RECORDS;
    while (at < end) {
        const uint32_t head = get_arg_val<uint32_t>(at);
        const uint32_t type = head >> 28;
        if (type == TYPE_RUN && at + RUN_WORDS <= end) {
            const bool canonical = ((head >> 24) & 1u) != 0;
            const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(SOURCE_TABLE + (head & 0xFFu)), TILE_BYTES);
            const auto destination =
                TensorAccessor(destination_args, get_arg_val<uint32_t>(DESTINATION_TABLE + ((head >> 8) & 0xFFu)), TILE_BYTES);
            const uint32_t count = get_arg_val<uint32_t>(at + 1);
            const uint32_t destination_page = get_arg_val<uint32_t>(at + 2);
            const uint32_t source_page = get_arg_val<uint32_t>(at + 3);
            for (uint32_t first = 0; first < count; first += LANES) {
                const uint32_t lanes = count - first < LANES ? count - first : LANES;
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    noc_async_read_tile(source_page + first + lane, source, scratch + lane * TILE_BYTES);
                }
                noc_async_read_barrier();
                if (canonical) {
                    for (uint32_t lane = 0; lane < lanes; ++lane) { canonicalise(scratch + lane * TILE_BYTES, TILE_BYTES / 4); }
                }
                asm volatile("" ::: "memory");
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    noc_async_write_tile(destination_page + first + lane, destination, scratch + lane * TILE_BYTES);
                }
                noc_async_write_barrier();
            }
            at += RUN_WORDS;
        } else if (type == TYPE_MIX && at + MIX_WORDS <= end) {
            const auto destination =
                TensorAccessor(destination_args, get_arg_val<uint32_t>(DESTINATION_TABLE + ((head >> 8) & 0xFFu)), TILE_BYTES);
            const uint32_t destination_page = get_arg_val<uint32_t>(at + 1);
            for (uint32_t quarter = 0; quarter < 4; ++quarter) {
                const uint32_t word = get_arg_val<uint32_t>(at + 2 + quarter);
                const uint32_t mode = (word >> 16) & 0xFFu;
                if (mode == 0) { continue; }
                const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(SOURCE_TABLE + (word >> 24)), TILE_BYTES);
                const uint32_t from = quarter_offset((mode - 1) & 3);
                const uint32_t to = scratch + quarter_offset(quarter);
                noc_async_read(source.get_noc_addr(word & 0xFFFFu, from), to, CHUNK_BYTES);
                noc_async_read(source.get_noc_addr(word & 0xFFFFu, from + FACE_BYTES), to + FACE_BYTES, CHUNK_BYTES);
            }
            noc_async_read_barrier();
            for (uint32_t quarter = 0; quarter < 4; ++quarter) {
                const uint32_t mode = (get_arg_val<uint32_t>(at + 2 + quarter) >> 16) & 0xFFu;
                for (uint32_t face = 0; face < 2; ++face) {
                    const uint32_t chunk = scratch + quarter_offset(quarter) + face * FACE_BYTES;
                    if (mode == 0) {
                        auto words = reinterpret_cast<volatile uint32_t*>(chunk);
                        for (uint32_t word = 0; word < CHUNK_BYTES / 4; ++word) { words[word] = 0; }
                    } else if (mode >= 5) {
                        canonicalise(chunk, CHUNK_BYTES / 4);
                    }
                }
            }
            asm volatile("" ::: "memory");
            noc_async_write_tile(destination_page, destination, scratch);
            noc_async_write_barrier();
            at += MIX_WORDS;
        } else {
            break;
        }
    }
}
