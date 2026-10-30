// qwen_p2p_checksum: the device kernel behind ttnn.qwen_staging_checksum (docs/tp4-p2p-restore.md, section 4).
//
// The checksum of a byte range of the P2P staging buffer (one contiguous range in one DRAM bank), computed in parallel by many cores:
//
//   x_k = the little-endian uint32 at byte offset (start + 4 k),   k = 0 .. length / 4 - 1        (k counts from the OBJECT start)
//   w_k = (k * 2654435761 + 1) mod 2^32
//   S   = sum_k (w_k * x_k)  mod 2^64
//
// Each core takes a run of 4096-byte units, starts its weight at w_(k0) (a runtime arg: the host computes it with 64-bit arithmetic),
// walks it with w += 2654435761 (mod 2^32, a plain add) and accumulates the 32 x 32 -> 64 bit products in a uint64 (a Blackhole data
// movement core is rv32im: the product is one mul and one mulhu). It writes its 64-bit partial sum, as a whole 64-byte page, to its own
// slot of the partials buffer; the host adds the partials mod 2^64. The range is read from the bank in chunks into two L1 halves, so a
// read overlaps the arithmetic on the other half. Every DRAM access is a whole aligned chunk (a multiple of 4096 bytes); the output is
// one whole 64-byte page per core.
//
// The arithmetic between the markers is compiled on a CPU by the unit test (test_p2p_staging.py) and compared with the numpy reference.
//
// Compile-time args: [chunk_bytes, out_page_bytes] + TensorAccessorArgs(staging) + TensorAccessorArgs(partials).
// Per-core runtime args: [0] staging address  [1] partials address  [2] staging bank  [3] staging byte offset of this core's first byte
//   [4] bytes for this core (0 for an idle core; a multiple of 4096)  [5] w_(k0) (uint32)  [6] this core's slot (a page id of partials)

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

void kernel_main() {
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(1);
    constexpr auto staging_args = TensorAccessorArgs<2>();
    constexpr auto partials_args = TensorAccessorArgs<staging_args.next_compile_time_args_offset()>();

    const uint32_t staging_address = get_arg_val<uint32_t>(0);
    const uint32_t partials_address = get_arg_val<uint32_t>(1);
    const uint32_t bank = get_arg_val<uint32_t>(2);
    const uint32_t source_offset = get_arg_val<uint32_t>(3);
    const uint32_t bytes = get_arg_val<uint32_t>(4);
    uint32_t weight = get_arg_val<uint32_t>(5);
    const uint32_t slot = get_arg_val<uint32_t>(6);
    const auto staging = TensorAccessor(staging_args, staging_address, chunk_bytes);
    const auto partials = TensorAccessor(partials_args, partials_address, out_page_bytes);

    // L1 of circular buffer 0 (never pushed): two read halves of chunk_bytes, then the output page.
    const uint32_t scratch = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t out_page = scratch + 2 * chunk_bytes;

    uint64_t sum = 0;
    if (bytes != 0) {
        uint32_t done = 0;
        uint32_t half = 0;
        uint32_t count = bytes < chunk_bytes ? bytes : chunk_bytes;
        noc_async_read(staging.get_noc_addr(bank, source_offset), scratch, count);
        while (true) {
            noc_async_read_barrier();
            const uint32_t next = done + count;
            uint32_t next_count = 0;
            if (next < bytes) {
                next_count = bytes - next < chunk_bytes ? bytes - next : chunk_bytes;
                noc_async_read(
                    staging.get_noc_addr(bank, source_offset + next),
                    scratch + (half ^ 1u) * chunk_bytes,
                    next_count);
            }
            sum = qwen_p2p_accumulate(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + half * chunk_bytes), count >> 2, weight, sum);
            if (next >= bytes) {
                break;
            }
            done = next;
            count = next_count;
            half ^= 1u;
        }
    }

    // The partial sum as one whole page: low word, high word, then zeros.
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_page);
    out[0] = static_cast<uint32_t>(sum);
    out[1] = static_cast<uint32_t>(sum >> 32);
    for (uint32_t i = 2; i < out_page_bytes / 4; ++i) {
        out[i] = 0;
    }
    noc_async_write(out_page, partials.get_noc_addr(slot), out_page_bytes);
    noc_async_write_barrier();
}
