// qwen_p2p_gather: the device kernel behind ttnn.qwen_blocks_to_staging (the spill's first step; the exact mirror of qwen_p2p_scatter.cpp).
//
// Copies whole tile pages from the pages of dimension-0 slices (paged KV cache blocks) of an interleaved DRAM tensor into ONE byte range in ONE DRAM bank (the P2P
// staging buffer, optimisation/ttnn-op/kv_region_read/qwen_kv_read.cpp: P2PStaging):
//
//   staging bank bytes  dst_offset + i * row_stride + j * page_bytes   <-   block blocks[i], page j (0 <= j < pages_per_block) of the tensor
//
// Row i (the bytes of block blocks[i], pages_per_block pages) is written at dst_offset + i * row_stride. row_stride == pages_per_block * page_bytes is the
// contiguous layout; a larger stride (a multiple of 64) puts the rows apart, for example one cache tensor's row of every block of a slot laid out object-major
// (Gemma: 69,632-byte rows, 696,320-byte shards: row_stride 696,320, dst_offset = slot base + tensor * 69,632). The bytes between two rows are never written.
//
// Every DRAM access is one WHOLE page (page_bytes, a multiple of 64): a sub-page DRAM read or write silently does nothing on this stack. The staging bytes are
// addressed straight in the bank (TensorAccessor page id = the bank id, so the address is base + offset in that bank), so no page of the staging buffer's own
// interleaving is ever assumed. The same bytes come out as went in: no pack, no unpack. The kernel ends only after every write has been acknowledged, so when the
// program completes the bytes are in DRAM.
//
// Compile-time args: [page_bytes, pages_per_block, chunk_pages, max_blocks] + TensorAccessorArgs(source) + TensorAccessorArgs(staging).
// Per-core runtime args (a CONSTANT count, the program is cached and only its values change between calls; the stride is an argument, so one program serves
// every stride):
//   [0] source tensor address   [1] staging buffer address   [2] staging bank   [3] staging byte offset of this core's first block
//   [4] number of blocks of this core (<= max_blocks)       [5] row stride in bytes
//   [6 .. 6 + max_blocks) the block ids (unused entries are not read)
// The scratch is the L1 of circular buffer 0 (never pushed): two halves of chunk_pages pages, so the reads of chunk c + 1 overlap the writes of chunk c.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Staging byte offset of page `page` of this core's run (pages count from the core's first block): the block's row, then the page inside it.
inline uint32_t qwen_p2p_destination(
    uint32_t destination_offset, uint32_t row_stride, uint32_t page, uint32_t pages_per_block, uint32_t page_bytes) {
    return destination_offset + (page / pages_per_block) * row_stride + (page % pages_per_block) * page_bytes;
}

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t pages_per_block = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_pages = get_compile_time_arg_val(2);
    constexpr uint32_t max_blocks = get_compile_time_arg_val(3);
    constexpr auto source_args = TensorAccessorArgs<4>();
    constexpr auto staging_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();

    const uint32_t source_address = get_arg_val<uint32_t>(0);
    const uint32_t staging_address = get_arg_val<uint32_t>(1);
    const uint32_t bank = get_arg_val<uint32_t>(2);
    const uint32_t destination_offset = get_arg_val<uint32_t>(3);
    const uint32_t blocks = get_arg_val<uint32_t>(4);
    const uint32_t row_stride = get_arg_val<uint32_t>(5);
    constexpr uint32_t block_ids = 6;
    const auto source = TensorAccessor(source_args, source_address, page_bytes);
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
        const uint32_t block = get_arg_val<uint32_t>(block_ids + k / pages_per_block);
        noc_async_read(source.get_noc_addr(block * pages_per_block + k % pages_per_block), scratch + k * page_bytes, page_bytes);
    }
    noc_async_read_barrier();

    while (true) {
        // Write the chunk that has landed in this half to its rows of the staging range.
        const uint32_t base = scratch + half * chunk_pages * page_bytes;
        for (uint32_t k = 0; k < count; ++k) {
            const uint32_t page = first + k;
            noc_async_write(
                base + k * page_bytes,
                staging.get_noc_addr(bank, qwen_p2p_destination(destination_offset, row_stride, page, pages_per_block, page_bytes)),
                page_bytes);
        }
        // Meanwhile read the next chunk into the other half (its previous writes were awaited at the end of the last turn).
        const uint32_t next = first + count;
        uint32_t next_count = 0;
        if (next < total) {
            next_count = total - next < chunk_pages ? total - next : chunk_pages;
            const uint32_t other = scratch + (half ^ 1u) * chunk_pages * page_bytes;
            for (uint32_t k = 0; k < next_count; ++k) {
                const uint32_t page = next + k;
                const uint32_t block = get_arg_val<uint32_t>(block_ids + page / pages_per_block);
                noc_async_read(source.get_noc_addr(block * pages_per_block + page % pages_per_block), other + k * page_bytes, page_bytes);
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
