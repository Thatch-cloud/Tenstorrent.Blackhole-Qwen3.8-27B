// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// F-B1: the page-parallel K/V cache writer (QWEN_FAST_KV_PAGE_WRITER, kv_page_writer_tp4.py). One source, four roles chosen by define:
//
//   KVPW_ROLE_READER  (RISCV_1)  per unit: the group's positions, the page-table window of the unit's tile row, the cache tiles of the unit,
//                                the update rows; the update rows are assembled into WT synthetic bfloat16 "update tiles".
//   KVPW_ROLE_WRITER  (RISCV_0)  per unit: copies the update rows into the cache's untilized rows (L1), hands the block back to the compute
//                                kernel and writes the tilized result.
//   KVPW_ROLE_AUDIT_PREP  / KVPW_ROLE_AUDIT_CHECK  (RISCV_1, one kernel each, the audit launches only; see below)
//
// The COMPUTE kernel is not in this file: it is the served ordered K/V writer's own compute kernel (ordered_cache.load_kernels(...)['compute'],
// hash-pinned, run unchanged), with the compile arguments the served launch passes except Wt (= WT here) and the head count (1). It
// untilizes one input block, untilizes one cache block, waits for the writer's in-place edit of the untilized cache block (CB 25 is CB 24's
// memory) and tilizes it into CB 16 - the served read-modify-write, once per unit instead of once per row.
//
// WHAT A UNIT IS. The packed block is 4 groups (one packed user each) of 16 consecutive rows; a group's positions are consecutive, so its rows
// touch at most TWO cache tile rows (a 32-row tile row of a 64-token page): "slot" 0 is the tile row of the group's first row, slot 1 the first
// row whose tile row differs. One unit = (cache K or V, group, slot, WT column tiles): the read-modify-write of those WT tiles of that one tile
// row with EVERY row of the group that targets it inserted, in row order. A unit whose slot does not exist (valid = 0) runs the same
// dataflow on garbage and writes nothing. There is no semaphore chain: two units never touch one tile (two groups on one (page, tile row) is
// the kv_conflict rule, checked on the host; the warm forward's placeholders, which break it on purpose, run the ORDERED mode below).
//
// EXACTNESS. The served chain applies read-modify-write once PER ROW; a unit applies it once per tile. They are the same bytes exactly when
// pack(unpack(x)) is the identity on every bfloat8_b block the packer writes (the bfloat8_b read-modify-write is idempotent): a card-M proof
// (ordered_writer_tp4_card_test.py, the page64 arm) records it, and the lever refuses to engage without that record (kv_page_writer_tp4.py).
// The update rows go through the same unpack and pack-untilize as the served path's input (a synthetic tile, row o of it holding the row for
// cache row o), so the bits the tilize sees are the served ones.
//
// ORDERED (runtime word, the warm forward only): group g waits for group g-1's unit (same cache, slot, column tiles) to finish writing before
// it reads its cache tiles, and signals group g+1 after its write: the served single chain's order for placeholders that all sit on one tile
// row. Programs of the two modes are identical; only the runtime words differ.
//
// SOURCE: the update rows are read from the prepared K/V tensor: interleaved DRAM (page r * 8 + c is row r, column tile c) or, under
// KVPW_SRC_L1, straight from the height-sharded L1 shards of the AttnPrep output (row r on the core whose NOC coordinates the reader is
// given, tile c at the shard base + 2048 c). Only row 0 of each tile (the one KV head of a four-card chip) is read: faces 0 and 1 (at bytes 0
// and 512 of the tile), bytes 0-31 of each, read as 64-byte spans.
//
// Compile-time args, reader:  [GROUP_ROWS, WT, PAGE_BYTES, BLOCK_ROWS, RT_WORDS] + TensorAccessorArgs(cache, positions, pages[, packed]).
//                     writer: [WT, RT_WORDS] + TensorAccessorArgs(cache).
//                     audit prep:  [GROUP_ROWS, WT, PAGE_BYTES, BLOCK_ROWS, RT_WORDS] + accessors(cache, shadow, positions, pages, shadow positions).
//                     audit check: [GROUP_ROWS, WT, PAGE_BYTES, BLOCK_ROWS, RT_WORDS] + accessors(cache, shadow, positions, pages, counters).
// Runtime words (fixed length per role; generic_op does not hash them):
//   reader (8 + 2 GROUP_ROWS): cache, positions, pages, packed base, first row, slot, first column tile, wait flag, then per row of the group
//                              its source core's NOC x, y (zeros under DRAM source).
//   writer (8):               cache, first column tile, signal flag, target NOC x, y, 0, 0, 0.
//   audit prep (10):          cache, shadow, positions, pages, shadow positions, first row, slot, first column tile, writes-positions flag, group.
//   audit check (9):          cache, shadow, positions, pages, counters, first row, slot, first column tile, unit.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace kvpw {

constexpr uint32_t cb_cache = 0;           // bfloat8_b cache tiles in  (reader -> compute)
constexpr uint32_t cb_in = 1;              // bfloat16 update tiles in  (reader -> compute)
constexpr uint32_t cb_scratch = 2;         // raw scratch, never pushed
constexpr uint32_t cb_out = 16;            // bfloat8_b tilized result  (compute -> writer)
constexpr uint32_t cb_untilized = 24;      // untilized cache block     (compute -> writer)
constexpr uint32_t cb_untilized2 = 25;     // the same memory, edited   (writer -> compute)
constexpr uint32_t cb_untilized_in = 26;   // untilized update block    (compute -> writer)
constexpr uint32_t cb_meta = 27;           // one 64-byte unit record   (reader -> writer)

constexpr uint32_t tile_bytes = 2048;        // a bfloat16 tile
constexpr uint32_t cache_tile_bytes = 1088;  // a bfloat8_b tile: 1024 mantissa bytes and 64 exponent bytes
constexpr uint32_t face_bytes = 512;
constexpr uint32_t face_row_bytes = 32;
constexpr uint32_t span_bytes = 64;          // DRAM reads are 64-byte granular on Blackhole
constexpr uint32_t row_tiles = 8;            // 256 columns of K or V
constexpr uint32_t tiles_per_tile_row = row_tiles;
constexpr uint32_t semaphore_id = 0;

// Scratch layout (cb_scratch).
constexpr uint32_t positions_offset = 0;     // BLOCK_ROWS words
constexpr uint32_t window_offset = 256;      // one 64-byte window of the page-table row
constexpr uint32_t stage_offset = 320;       // reader: [row][tile][half] 64-byte spans
constexpr uint32_t audit_tiles_offset = 512; // audit check: the cache's tiles, then the shadow's

// A position's tile-row key: (page-table entry, tile row inside the 64-token page).
FORCE_INLINE uint32_t key_of(uint32_t position) {
    return ((position >> 6) << 1) | ((position >> 5) & 1u);
}

FORCE_INLINE uint32_t word_at(uint32_t base, uint32_t index) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base)[index];
}

FORCE_INLINE void set_word(uint32_t base, uint32_t index, uint32_t value) {
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base)[index] = value;
}

FORCE_INLINE void fill_words(uint32_t address, uint32_t bytes, uint32_t value) {
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    for (uint32_t i = 0; i < bytes / 4; ++i) {
        words[i] = value;
    }
}

FORCE_INLINE void copy_words(uint32_t source, uint32_t destination, uint32_t bytes) {
    auto from = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(source);
    auto to = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination);
    for (uint32_t i = 0; i < bytes / 4; ++i) {
        to[i] = from[i];
    }
}

// The target key of slot `slot` of the group whose positions are words [first, first + rows) at `positions`: slot 0 is the first row's key,
// slot 1 the key of the first row that differs from it (valid = false when every row shares the first key).
FORCE_INLINE uint32_t target_key(uint32_t positions, uint32_t first, uint32_t rows, uint32_t slot, bool& valid) {
    const uint32_t first_key = key_of(word_at(positions, first));
#ifdef KVPW_NEG_SLOT
    slot = 0;  // negative control: slot 1 writes slot 0's tile row
#endif
    valid = true;
    if (slot == 0) {
        return first_key;
    }
    for (uint32_t j = 1; j < rows; ++j) {
        const uint32_t key = key_of(word_at(positions, first + j));
        if (key != first_key) {
            return key;
        }
    }
    valid = false;
    return first_key;
}

// Bit o of the mask is set when a row of the group with the target key writes cache row o of its tile row.
FORCE_INLINE uint32_t row_mask(uint32_t positions, uint32_t first, uint32_t rows, uint32_t target) {
    uint32_t mask = 0;
    for (uint32_t j = 0; j < rows; ++j) {
        const uint32_t position = word_at(positions, first + j);
        if (key_of(position) == target) {
            mask |= 1u << (position & 31u);
        }
    }
#ifdef KVPW_NEG_DROP
    mask &= mask - 1u;  // negative control: the lowest row of the unit is not written
#endif
    return mask;
}

// Cache tile id of (block, tile row, column tile) for a one-head (N, 1, 64, 256) bfloat8_b cache.
FORCE_INLINE uint32_t cache_tile(uint32_t block, uint32_t tile_row, uint32_t column) {
    return (block * 2u + tile_row) * tiles_per_tile_row + column;
}

// Byte offset of row o, half h (columns 16 h .. 16 h + 15) of a 32 x 32 face-ordered bfloat16 tile.
FORCE_INLINE uint32_t face_row_offset(uint32_t row, uint32_t half) {
    return ((row >> 4) * 2u + half) * face_bytes + (row & 15u) * face_row_bytes;
}

}  // namespace kvpw

#if defined(KVPW_ROLE_READER)

void kernel_main() {
    using namespace kvpw;
    constexpr uint32_t GROUP_ROWS = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
    constexpr uint32_t PAGE_BYTES = get_compile_time_arg_val(2);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(3);
    constexpr uint32_t RT_WORDS = get_compile_time_arg_val(4);
    constexpr auto cache_args = TensorAccessorArgs<5>();
    constexpr auto positions_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto pages_args = TensorAccessorArgs<positions_args.next_compile_time_args_offset()>();
#ifndef KVPW_SRC_L1
    constexpr auto packed_args = TensorAccessorArgs<pages_args.next_compile_time_args_offset()>();
#endif
    static_assert(RT_WORDS == 8 + 2 * GROUP_ROWS, "the reader reads 8 words and one NOC pair per group row");
    static_assert(WT == 1 || WT == 2 || WT == 4 || WT == 8, "a unit is 1, 2, 4 or 8 column tiles");
    static_assert(GROUP_ROWS <= 32, "a group's rows must fit in one tile row's offsets");

    const uint32_t cache_address = get_arg_val<uint32_t>(0);
    const uint32_t positions_address = get_arg_val<uint32_t>(1);
    const uint32_t pages_address = get_arg_val<uint32_t>(2);
    const uint32_t packed_address = get_arg_val<uint32_t>(3);
    const uint32_t first_row = get_arg_val<uint32_t>(4);
    const uint32_t slot = get_arg_val<uint32_t>(5);
    const uint32_t first_column = get_arg_val<uint32_t>(6);
    const uint32_t wait_first = get_arg_val<uint32_t>(7);

    const auto cache = TensorAccessor(cache_args, cache_address, cache_tile_bytes);
    const auto positions_tensor = TensorAccessor(positions_args, positions_address, BLOCK_ROWS * 4);
    const auto pages = TensorAccessor(pages_args, pages_address, PAGE_BYTES);
#ifndef KVPW_SRC_L1
    const auto packed = TensorAccessor(packed_args, packed_address, tile_bytes);
#endif

    cb_reserve_back(cb_in, WT);
    cb_reserve_back(cb_cache, WT);
    cb_reserve_back(cb_meta, 1);
    const uint32_t in_pointer = get_write_ptr(cb_in);
    const uint32_t cache_pointer = get_write_ptr(cb_cache);
    const uint32_t meta_pointer = get_write_ptr(cb_meta);
    const uint32_t scratch = get_write_ptr(cb_scratch);
    const uint32_t positions = scratch + positions_offset;

    noc_async_read(positions_tensor.get_noc_addr(0), positions, BLOCK_ROWS * 4);
    noc_async_read_barrier();
    invalidate_l1_cache();

    bool valid = false;
    const uint32_t target = target_key(positions, first_row, GROUP_ROWS, slot, valid);
    const uint32_t mask = valid ? row_mask(positions, first_row, GROUP_ROWS, target) : 0u;
    uint32_t block = 0;

    if (wait_first != 0) {
        // ORDERED: the predecessor group's unit (same cache, slot and columns) has written its tiles and signalled.
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(semaphore_id));
        noc_semaphore_wait(arrived, 1);
        noc_semaphore_set(arrived, 0);
    }

    if (valid) {
        // The page-table window holding the unit's entry (the 64-byte aligned window of the first row's table: every row of a group carries
        // the packed user's one table), and the update rows' source spans, behind one barrier.
        const uint32_t entry_byte = (target >> 1) * 4u;
        const uint32_t window = entry_byte & ~63u;
        noc_async_read(pages.get_noc_addr(first_row, window), scratch + window_offset, span_bytes);
        for (uint32_t j = 0; j < GROUP_ROWS; ++j) {
            if (key_of(word_at(positions, first_row + j)) != target) {
                continue;
            }
            for (uint32_t u = 0; u < WT; ++u) {
                const uint32_t column = first_column + u;
                const uint32_t span = scratch + stage_offset + ((j * WT + u) * 2u) * span_bytes;
#ifdef KVPW_SRC_L1
                const uint32_t source_x = get_arg_val<uint32_t>(8 + 2 * j);
                const uint32_t source_y = get_arg_val<uint32_t>(9 + 2 * j);
                const uint32_t tile_address = packed_address + column * tile_bytes;
                noc_async_read(get_noc_addr(source_x, source_y, tile_address), span, span_bytes);
                noc_async_read(get_noc_addr(source_x, source_y, tile_address + face_bytes), span + span_bytes, span_bytes);
#else
                const uint32_t page = (first_row + j) * row_tiles + column;
                noc_async_read(packed.get_noc_addr(page, 0), span, span_bytes);
                noc_async_read(packed.get_noc_addr(page, face_bytes), span + span_bytes, span_bytes);
#endif
            }
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        block = word_at(scratch + window_offset, (entry_byte - window) >> 2);

        // The cache tiles of the unit, one read barrier.
        const uint32_t tile_row = target & 1u;
        for (uint32_t u = 0; u < WT; ++u) {
            noc_async_read(cache.get_noc_addr(cache_tile(block, tile_row, first_column + u)), cache_pointer + u * cache_tile_bytes,
                           cache_tile_bytes);
        }

        // The synthetic update tiles: zero, then each selected row's two 32-byte pieces at cache row o of its tile (rows in order, so a
        // repeated offset ends as the last row's, as the served chain ends).
        fill_words(in_pointer, WT * tile_bytes, 0);
        for (uint32_t j = 0; j < GROUP_ROWS; ++j) {
            const uint32_t position = word_at(positions, first_row + j);
            if (key_of(position) != target) {
                continue;
            }
            const uint32_t offset = position & 31u;
            for (uint32_t u = 0; u < WT; ++u) {
                const uint32_t span = scratch + stage_offset + ((j * WT + u) * 2u) * span_bytes;
                copy_words(span, in_pointer + u * tile_bytes + face_row_offset(offset, 0), face_row_bytes);
                copy_words(span + span_bytes, in_pointer + u * tile_bytes + face_row_offset(offset, 1), face_row_bytes);
            }
        }
        asm volatile("" ::: "memory");
        noc_async_read_barrier();
    }

    set_word(meta_pointer, 0, valid ? 1u : 0u);
    set_word(meta_pointer, 1, block);
    set_word(meta_pointer, 2, target & 1u);
    set_word(meta_pointer, 3, mask);
    asm volatile("" ::: "memory");
    cb_push_back(cb_in, WT);
    cb_push_back(cb_cache, WT);
    cb_push_back(cb_meta, 1);
}

#elif defined(KVPW_ROLE_WRITER)

void kernel_main() {
    using namespace kvpw;
    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t RT_WORDS = get_compile_time_arg_val(1);
    constexpr auto cache_args = TensorAccessorArgs<2>();
    static_assert(RT_WORDS == 8, "the writer reads 8 runtime words");

    const uint32_t cache_address = get_arg_val<uint32_t>(0);
    const uint32_t first_column = get_arg_val<uint32_t>(1);
    const uint32_t signal = get_arg_val<uint32_t>(2);
    const uint32_t target_x = get_arg_val<uint32_t>(3);
    const uint32_t target_y = get_arg_val<uint32_t>(4);
    const auto cache = TensorAccessor(cache_args, cache_address, cache_tile_bytes);

    cb_wait_front(cb_meta, 1);
    invalidate_l1_cache();
    const uint32_t meta_pointer = get_read_ptr(cb_meta);
    const bool valid = word_at(meta_pointer, 0) != 0;
    const uint32_t block = word_at(meta_pointer, 1);
    const uint32_t tile_row = word_at(meta_pointer, 2);
    const uint32_t mask = word_at(meta_pointer, 3);

    // The compute kernel has untilized the cache block (CB 24) and the update block (CB 26): edit the cache block in place, hand it back
    // through CB 25 (the same memory), and wait for the tilized block in CB 16.
    cb_wait_front(cb_untilized, WT);
    cb_wait_front(cb_untilized_in, WT);
    invalidate_l1_cache();
    const uint32_t cache_rows = get_read_ptr(cb_untilized);
    const uint32_t update_rows = get_read_ptr(cb_untilized_in);
    constexpr uint32_t row_bytes = WT * 64;  // WT tiles of 32 bfloat16 columns
    if (valid) {
        for (uint32_t row = 0; row < 32; ++row) {
            if (((mask >> row) & 1u) != 0) {
                copy_words(update_rows + row * row_bytes, cache_rows + row * row_bytes, row_bytes);
            }
        }
    }
    asm volatile("" ::: "memory");
    cb_reserve_back(cb_untilized2, WT);
    cb_push_back(cb_untilized2, WT);
    cb_pop_front(cb_untilized, WT);
    cb_pop_front(cb_untilized_in, WT);

    cb_wait_front(cb_out, WT);
    if (valid) {
        const uint32_t out_pointer = get_read_ptr(cb_out);
        for (uint32_t u = 0; u < WT; ++u) {
            noc_async_write(out_pointer + u * cache_tile_bytes, cache.get_noc_addr(cache_tile(block, tile_row, first_column + u)),
                            cache_tile_bytes);
        }
        noc_async_write_barrier();
    }
    cb_pop_front(cb_out, WT);
    cb_pop_front(cb_meta, 1);
    if (signal != 0) {
        noc_semaphore_inc(get_noc_addr(target_x, target_y, get_semaphore(semaphore_id)), 1);
        noc_async_atomic_barrier();
    }
}

#elif defined(KVPW_ROLE_AUDIT_PREP)

// Audit launch 1 (before the page write): for the unit's tile row, copy its WT tiles of the real cache into the shadow cache's block
// (group * 2 + slot) at the same tile row; and (the unit of slot 0, column 0 of cache K only) write the group's shadow positions
// ((group * 2 + slot of the row) * 64 + tile row * 32 + offset) so the SERVED chained writer, run next on the shadow, makes the same writes.
void kernel_main() {
    using namespace kvpw;
    constexpr uint32_t GROUP_ROWS = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
    constexpr uint32_t PAGE_BYTES = get_compile_time_arg_val(2);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(3);
    constexpr uint32_t RT_WORDS = get_compile_time_arg_val(4);
    constexpr auto cache_args = TensorAccessorArgs<5>();
    constexpr auto shadow_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto positions_args = TensorAccessorArgs<shadow_args.next_compile_time_args_offset()>();
    constexpr auto pages_args = TensorAccessorArgs<positions_args.next_compile_time_args_offset()>();
    constexpr auto shadow_positions_args = TensorAccessorArgs<pages_args.next_compile_time_args_offset()>();
    static_assert(RT_WORDS == 10, "audit prep reads 10 runtime words");

    const auto cache = TensorAccessor(cache_args, get_arg_val<uint32_t>(0), cache_tile_bytes);
    const auto shadow = TensorAccessor(shadow_args, get_arg_val<uint32_t>(1), cache_tile_bytes);
    const auto positions_tensor = TensorAccessor(positions_args, get_arg_val<uint32_t>(2), BLOCK_ROWS * 4);
    const auto pages = TensorAccessor(pages_args, get_arg_val<uint32_t>(3), PAGE_BYTES);
    const auto shadow_positions = TensorAccessor(shadow_positions_args, get_arg_val<uint32_t>(4), BLOCK_ROWS * 4);
    const uint32_t first_row = get_arg_val<uint32_t>(5);
    const uint32_t slot = get_arg_val<uint32_t>(6);
    const uint32_t first_column = get_arg_val<uint32_t>(7);
    const uint32_t writes_positions = get_arg_val<uint32_t>(8);
    const uint32_t group = get_arg_val<uint32_t>(9);

    const uint32_t scratch = get_write_ptr(cb_scratch);
    const uint32_t positions = scratch + positions_offset;
    noc_async_read(positions_tensor.get_noc_addr(0), positions, BLOCK_ROWS * 4);
    noc_async_read_barrier();
    invalidate_l1_cache();
    bool valid = false;
    const uint32_t target = target_key(positions, first_row, GROUP_ROWS, slot, valid);

    if (writes_positions != 0) {
        // The group's 16 shadow positions: 64 bytes of the shadow positions page at byte offset 4 * first_row (first_row is a multiple of 16).
        const uint32_t first_key = key_of(word_at(positions, first_row));
        const uint32_t output = scratch + audit_tiles_offset;
        for (uint32_t j = 0; j < GROUP_ROWS; ++j) {
            const uint32_t position = word_at(positions, first_row + j);
            const uint32_t row_slot = key_of(position) == first_key ? 0u : 1u;
            set_word(output, j, (group * 2u + row_slot) * 64u + ((position >> 5) & 1u) * 32u + (position & 31u));
        }
        asm volatile("" ::: "memory");
        noc_async_write(output, shadow_positions.get_noc_addr(0, 4u * first_row), GROUP_ROWS * 4u);
        noc_async_write_barrier();
    }
    if (valid) {
        const uint32_t entry_byte = (target >> 1) * 4u;
        const uint32_t window = entry_byte & ~63u;
        noc_async_read(pages.get_noc_addr(first_row, window), scratch + window_offset, span_bytes);
        noc_async_read_barrier();
        invalidate_l1_cache();
        const uint32_t block = word_at(scratch + window_offset, (entry_byte - window) >> 2);
        const uint32_t tile_row = target & 1u;
        const uint32_t copy = scratch + audit_tiles_offset;
        for (uint32_t u = 0; u < WT; ++u) {
            noc_async_read(cache.get_noc_addr(cache_tile(block, tile_row, first_column + u)), copy + u * cache_tile_bytes, cache_tile_bytes);
        }
        noc_async_read_barrier();
        const uint32_t shadow_block = group * 2u + slot;
        for (uint32_t u = 0; u < WT; ++u) {
            noc_async_write(copy + u * cache_tile_bytes, shadow.get_noc_addr(cache_tile(shadow_block, tile_row, first_column + u)),
                            cache_tile_bytes);
        }
        noc_async_write_barrier();
    }
}

#elif defined(KVPW_ROLE_AUDIT_CHECK)

// Audit launch 3 (after the page write and the served write on the shadow): count the 32-bit words of the unit's WT cache tiles that differ
// between the real cache and the shadow; write a 64-byte record [mismatched words, valid, block, tile row] to the unit's page of the counters.
void kernel_main() {
    using namespace kvpw;
    constexpr uint32_t GROUP_ROWS = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
    constexpr uint32_t PAGE_BYTES = get_compile_time_arg_val(2);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(3);
    constexpr uint32_t RT_WORDS = get_compile_time_arg_val(4);
    constexpr auto cache_args = TensorAccessorArgs<5>();
    constexpr auto shadow_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto positions_args = TensorAccessorArgs<shadow_args.next_compile_time_args_offset()>();
    constexpr auto pages_args = TensorAccessorArgs<positions_args.next_compile_time_args_offset()>();
    constexpr auto counters_args = TensorAccessorArgs<pages_args.next_compile_time_args_offset()>();
    static_assert(RT_WORDS == 9, "audit check reads 9 runtime words");

    const auto cache = TensorAccessor(cache_args, get_arg_val<uint32_t>(0), cache_tile_bytes);
    const auto shadow = TensorAccessor(shadow_args, get_arg_val<uint32_t>(1), cache_tile_bytes);
    const auto positions_tensor = TensorAccessor(positions_args, get_arg_val<uint32_t>(2), BLOCK_ROWS * 4);
    const auto pages = TensorAccessor(pages_args, get_arg_val<uint32_t>(3), PAGE_BYTES);
    const auto counters = TensorAccessor(counters_args, get_arg_val<uint32_t>(4), span_bytes);
    const uint32_t first_row = get_arg_val<uint32_t>(5);
    const uint32_t slot = get_arg_val<uint32_t>(6);
    const uint32_t first_column = get_arg_val<uint32_t>(7);
    const uint32_t unit = get_arg_val<uint32_t>(8);

    const uint32_t scratch = get_write_ptr(cb_scratch);
    const uint32_t positions = scratch + positions_offset;
    noc_async_read(positions_tensor.get_noc_addr(0), positions, BLOCK_ROWS * 4);
    noc_async_read_barrier();
    invalidate_l1_cache();
    bool valid = false;
    const uint32_t target = target_key(positions, first_row, GROUP_ROWS, slot, valid);
    uint32_t mismatched = 0;
    uint32_t block = 0;
    if (valid) {
        const uint32_t entry_byte = (target >> 1) * 4u;
        const uint32_t window = entry_byte & ~63u;
        noc_async_read(pages.get_noc_addr(first_row, window), scratch + window_offset, span_bytes);
        noc_async_read_barrier();
        invalidate_l1_cache();
        block = word_at(scratch + window_offset, (entry_byte - window) >> 2);
        const uint32_t tile_row = target & 1u;
        const uint32_t real = scratch + audit_tiles_offset;
        const uint32_t served = real + WT * cache_tile_bytes;
        const uint32_t shadow_block = (first_row / GROUP_ROWS) * 2u + slot;
        for (uint32_t u = 0; u < WT; ++u) {
            noc_async_read(cache.get_noc_addr(cache_tile(block, tile_row, first_column + u)), real + u * cache_tile_bytes, cache_tile_bytes);
            noc_async_read(shadow.get_noc_addr(cache_tile(shadow_block, tile_row, first_column + u)), served + u * cache_tile_bytes,
                           cache_tile_bytes);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t word = 0; word < WT * cache_tile_bytes / 4; ++word) {
            mismatched += word_at(real, word) != word_at(served, word) ? 1u : 0u;
        }
    }
    const uint32_t record = scratch + audit_tiles_offset + 2 * WT * cache_tile_bytes;
    fill_words(record, span_bytes, 0);
    set_word(record, 0, mismatched);
    set_word(record, 1, valid ? 1u : 0u);
    set_word(record, 2, block);
    set_word(record, 3, target & 1u);
    asm volatile("" ::: "memory");
    noc_async_write(record, counters.get_noc_addr(unit, 0), span_bytes);
    noc_async_write_barrier();
}

#else
#error "define one of KVPW_ROLE_READER, KVPW_ROLE_WRITER, KVPW_ROLE_AUDIT_PREP, KVPW_ROLE_AUDIT_CHECK"
#endif
