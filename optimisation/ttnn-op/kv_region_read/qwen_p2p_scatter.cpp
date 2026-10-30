// qwen_p2p_scatter: the device kernel behind ttnn.qwen_write_blocks_from_staging (docs/tp4-p2p-restore.md, section 3).
//
// Copies whole tile pages from ONE contiguous byte range in ONE DRAM bank (the P2P staging buffer, optimisation/ttnn-op/kv_region_read/
// qwen_kv_read.cpp: P2PStaging) into the pages of dimension-0 slices (paged KV cache blocks) of an interleaved DRAM tensor:
//
//   block blocks[i], page j (0 <= j < pages_per_block)  <-  staging bank bytes  src_offset + (i * pages_per_block + j) * page_bytes
//
// Every DRAM access is one WHOLE page (page_bytes, a multiple of 64): a sub-page DRAM read or write silently does nothing on this stack.
// The staging bytes are read straight from the bank (TensorAccessor page id = the bank id, so the address is base + offset in that bank),
// so no page of the staging buffer's own interleaving is ever assumed. The same bytes come out as went in: no pack, no unpack.
//
// Compile-time args: [page_bytes, pages_per_block, chunk_pages, max_blocks] + TensorAccessorArgs(destination) + TensorAccessorArgs(staging).
// Per-core runtime args (a CONSTANT count, the program is cached and only its values change between calls):
//   [0] destination buffer address   [1] staging buffer address   [2] staging bank   [3] staging byte offset of this core's first block
//   [4] number of blocks of this core (<= max_blocks)             [5 .. 5 + max_blocks) the block ids (unused entries are not read)
// The scratch is the L1 of circular buffer 0 (never pushed): two halves of chunk_pages pages, so the reads of chunk c + 1 overlap the
// writes of chunk c.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t pages_per_block = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_pages = get_compile_time_arg_val(2);
    constexpr uint32_t max_blocks = get_compile_time_arg_val(3);
    constexpr auto destination_args = TensorAccessorArgs<4>();
    constexpr auto staging_args = TensorAccessorArgs<destination_args.next_compile_time_args_offset()>();

    const uint32_t destination_address = get_arg_val<uint32_t>(0);
    const uint32_t staging_address = get_arg_val<uint32_t>(1);
    const uint32_t bank = get_arg_val<uint32_t>(2);
    const uint32_t source_offset = get_arg_val<uint32_t>(3);
    const uint32_t blocks = get_arg_val<uint32_t>(4);
    const auto destination = TensorAccessor(destination_args, destination_address, page_bytes);
    const auto staging = TensorAccessor(staging_args, staging_address, page_bytes);

    const uint32_t total = blocks * pages_per_block;
    if (blocks == 0 || blocks > max_blocks) {
        return;
    }
    const uint32_t scratch = (get_write_ptr(0) + 63) & ~63u;

    // First chunk: read it into half 0.
    uint32_t first = 0;
    uint32_t count = total < chunk_pages ? total : chunk_pages;
    uint32_t half = 0;
    for (uint32_t k = 0; k < count; ++k) {
        noc_async_read(staging.get_noc_addr(bank, source_offset + k * page_bytes), scratch + k * page_bytes, page_bytes);
    }
    noc_async_read_barrier();

    while (true) {
        // Write the chunk that has landed in this half to its destination pages.
        const uint32_t base = scratch + half * chunk_pages * page_bytes;
        for (uint32_t k = 0; k < count; ++k) {
            const uint32_t page = first + k;
            const uint32_t block = get_arg_val<uint32_t>(5 + page / pages_per_block);
            const uint32_t destination_page = block * pages_per_block + page % pages_per_block;
            noc_async_write(base + k * page_bytes, destination.get_noc_addr(destination_page), page_bytes);
        }
        // Meanwhile read the next chunk into the other half (its previous writes were awaited at the end of the last turn).
        const uint32_t next = first + count;
        uint32_t next_count = 0;
        if (next < total) {
            next_count = total - next < chunk_pages ? total - next : chunk_pages;
            const uint32_t other = scratch + (half ^ 1u) * chunk_pages * page_bytes;
            for (uint32_t k = 0; k < next_count; ++k) {
                noc_async_read(
                    staging.get_noc_addr(bank, source_offset + (next + k) * page_bytes), other + k * page_bytes, page_bytes);
            }
        }
        noc_async_read_barrier();
        noc_async_write_barrier();
        if (next >= total) {
            break;
        }
        first = next;
        count = next_count;
        half ^= 1u;
    }
}
