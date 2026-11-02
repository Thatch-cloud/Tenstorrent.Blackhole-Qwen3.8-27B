// qwen_p2p_digest_rows: the device kernel behind ttnn.qwen_blocks_digest_async (the integrity check of the P2P KV offload: the digest of KV ROWS, at spill over the
// source rows and at restore over the destination rows).
//
// The digest of block b is the checksum of qwen_p2p_checksum.cpp over the block's SHARD, the concatenation in canonical order of its row in each of the T tensors:
//
//   shard(b) = row(tensor 0, b) | row(tensor 1, b) | ... | row(tensor T-1, b)        row(t, b) = pages b * pages_per_block .. + pages_per_block - 1 of tensor t
//   x_k = the little-endian uint32 at byte offset 4 k of shard(b),   k = 0 .. T * row_bytes / 4 - 1        (k counts from the SHARD start)
//   w_k = (k * 2654435761 + 1) mod 2^32
//   S(b) = sum_k (w_k * x_k)  mod 2^64
//
// so it equals qwen_staging_checksum over the same bytes laid out as a staged object shard (Gemma: 10 tensors x 69,632 bytes = 696,320 bytes). The weight is carried
// from one tensor to the next (k does not restart between tensors), which is what makes the digest of a gathered shard in staging and the digest of the rows it came
// from the same number, and what makes one flipped byte, or two rows swapped, change it.
//
// The work is divided BY BLOCK: a core digests whole rows, all T tensors, and writes ONE 64-byte partial page per block (the 64-bit sum: low word, high word, the
// rest zero) at partials page first_slot + i for its i-th block; the host reads the pages back and needs no further addition. A row is read in chunks of chunk_pages
// whole pages into two halves of L1, so a read overlaps the arithmetic on the other half. Every DRAM access is one whole page (a sub-page access silently does
// nothing on this stack), and the kernel only READS the tensors.
//
// The arithmetic between the markers is the checksum kernel's, character for character (a CPU test compares the two blocks); it is also compiled on its own there.
//
// Compile-time args: [page_bytes, pages_per_block, chunk_pages, max_blocks_per_core, max_tensors] + TensorAccessorArgs(tensors) + TensorAccessorArgs(partials).
// All the tensors of one call share their accessor arguments (the same layout and page size); each has its own address.
// Per-core runtime args (a CONSTANT count, the program is cached and only its values change between calls):
//   [0] partials address  [1] number of tensors T (1 .. max_tensors)  [2] number of blocks of this core (0 .. max_blocks_per_core)  [3] this core's first partials page
//   [4 .. 4 + max_tensors) the tensor addresses  [4 + max_tensors .. + max_blocks_per_core) the block ids

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// BEGIN-CHECKSUM-ARITHMETIC
constexpr uint32_t kQwenP2PStep = 2654435761u;

// Accumulates words[0 .. count) with weights w, w + step, w + 2 step, ... (mod 2^32); returns the updated sum and leaves w advanced.
inline uint64_t qwen_p2p_accumulate(volatile tt_l1_ptr uint32_t* words, uint32_t count, uint32_t& w, uint64_t sum) {
    uint32_t weight = w;
    uint32_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const uint32_t x0 = words[i];
        const uint32_t x1 = words[i + 1];
        const uint32_t x2 = words[i + 2];
        const uint32_t x3 = words[i + 3];
        sum += static_cast<uint64_t>(weight) * x0;
        weight += kQwenP2PStep;
        sum += static_cast<uint64_t>(weight) * x1;
        weight += kQwenP2PStep;
        sum += static_cast<uint64_t>(weight) * x2;
        weight += kQwenP2PStep;
        sum += static_cast<uint64_t>(weight) * x3;
        weight += kQwenP2PStep;
    }
    for (; i < count; ++i) {
        sum += static_cast<uint64_t>(weight) * words[i];
        weight += kQwenP2PStep;
    }
    w = weight;
    return sum;
}
// END-CHECKSUM-ARITHMETIC

// Starts the read of chunk `chunk` of block `block`'s shard into `destination` (the chunks of the shard, in order: tensor 0's row in chunks_per_row chunks, then tensor
// 1's, ...); returns the number of pages it reads. The pages of a chunk are consecutive pages of ONE tensor's row.
template <typename Accessor>
inline uint32_t qwen_p2p_read_row_chunk(
    const Accessor& tensor, uint32_t block, uint32_t within, uint32_t pages_per_block, uint32_t chunk_pages, uint32_t page_bytes, uint32_t destination) {
    const uint32_t first_page = within * chunk_pages;
    const uint32_t count = pages_per_block - first_page < chunk_pages ? pages_per_block - first_page : chunk_pages;
    for (uint32_t p = 0; p < count; ++p) {
        noc_async_read(tensor.get_noc_addr(block * pages_per_block + first_page + p), destination + p * page_bytes, page_bytes);
    }
    return count;
}

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t pages_per_block = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_pages = get_compile_time_arg_val(2);
    constexpr uint32_t max_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t max_tensors = get_compile_time_arg_val(4);
    constexpr auto tensor_args = TensorAccessorArgs<5>();
    constexpr auto partials_args = TensorAccessorArgs<tensor_args.next_compile_time_args_offset()>();
    constexpr uint32_t partial_bytes = 64;

    const uint32_t partials_address = get_arg_val<uint32_t>(0);
    const uint32_t tensors = get_arg_val<uint32_t>(1);
    const uint32_t blocks = get_arg_val<uint32_t>(2);
    const uint32_t first_slot = get_arg_val<uint32_t>(3);
    constexpr uint32_t tensor_addresses = 4;
    constexpr uint32_t block_ids = 4 + max_tensors;
    const auto partials = TensorAccessor(partials_args, partials_address, partial_bytes);

    // A core with no blocks (or arguments outside the limits) writes nothing: every block of the call has its own page and a core writes only its own.
    if (blocks == 0 || blocks > max_blocks || tensors == 0 || tensors > max_tensors) {
        return;
    }

    // L1 of circular buffer 0 (never pushed): two read halves of chunk_pages pages, then the output page.
    constexpr uint32_t half_bytes = chunk_pages * page_bytes;
    const uint32_t scratch = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t out_page = scratch + 2 * half_bytes;
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_page);
    for (uint32_t i = 2; i < partial_bytes / 4; ++i) {
        out[i] = 0;
    }

    constexpr uint32_t chunks_per_row = (pages_per_block + chunk_pages - 1) / chunk_pages;
    const uint32_t chunks = tensors * chunks_per_row;

    for (uint32_t b = 0; b < blocks; ++b) {
        const uint32_t block = get_arg_val<uint32_t>(block_ids + b);
        uint32_t weight = 1;  // w_0: the weight of word 0 of the shard
        uint64_t sum = 0;
        // The chunks of the shard, in order, are one stream: chunk i + 1 is read once chunk i has landed, so it overlaps the arithmetic on chunk i.
        uint32_t pages = qwen_p2p_read_row_chunk(
            TensorAccessor(tensor_args, get_arg_val<uint32_t>(tensor_addresses), page_bytes), block, 0, pages_per_block, chunk_pages, page_bytes, scratch);
        for (uint32_t i = 0; i < chunks; ++i) {
            const uint32_t half = i & 1u;
            noc_async_read_barrier();
            uint32_t next_pages = 0;
            if (i + 1 < chunks) {
                const uint32_t next = i + 1;
                next_pages = qwen_p2p_read_row_chunk(
                    TensorAccessor(tensor_args, get_arg_val<uint32_t>(tensor_addresses + next / chunks_per_row), page_bytes),
                    block,
                    next % chunks_per_row,
                    pages_per_block,
                    chunk_pages,
                    page_bytes,
                    scratch + (half ^ 1u) * half_bytes);
            }
            sum = qwen_p2p_accumulate(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + half * half_bytes), (pages * page_bytes) >> 2, weight, sum);
            pages = next_pages;
        }
        out[0] = static_cast<uint32_t>(sum);
        out[1] = static_cast<uint32_t>(sum >> 32);
        noc_async_write(out_page, partials.get_noc_addr(first_slot + b), partial_bytes);
        noc_async_write_barrier();  // the page is reused by the next block
    }
}
