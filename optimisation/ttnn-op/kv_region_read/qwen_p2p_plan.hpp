// qwen_p2p_plan: the pure arithmetic of the P2P device ops (qwen_kv_read.cpp, version 4: scatter, gather, block-row digest, the checksum oracle), with no
// tt-metal dependency, so a CPU test compiles it and compares every function with its Python mirror (p2p_staging_ref.py). Nothing here touches a device.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace qwen_p2p {

// Every DRAM access is a whole page, and a Blackhole DRAM address is aligned to 64 bytes.
constexpr uint32_t kDramAlignment = 64;

// qwen_write_blocks_from_staging: 16 cores (8 columns x 2 rows) each take up to 32 blocks of one launch, so one launch moves up to 512
// blocks; more blocks are several launches. A page chunk is 16 pages (one 1,088-byte bf8 block of one head), two halves of scratch.
constexpr uint32_t kScatterColumns = 8;
constexpr uint32_t kScatterRows = 2;
constexpr uint32_t kScatterCores = kScatterColumns * kScatterRows;
constexpr uint32_t kScatterMaxBlocksPerCore = 32;
constexpr uint32_t kScatterChunkPages = 16;
// The scatter's L1 scratch is two halves of kScatterChunkPages pages: a page above this would not leave room next to a served model's L1 buffers.
constexpr uint32_t kScatterMaxPageBytes = 8192;
constexpr uint32_t kScatterLaunchBlocks = kScatterCores * kScatterMaxBlocksPerCore;
// Runtime args of one scatter core: staging/destination addresses, bank, source offset, block count, row stride, then the block ids (padded).
constexpr uint32_t kScatterFixedArgs = 6;
constexpr uint32_t kScatterArgs = kScatterFixedArgs + kScatterMaxBlocksPerCore;

// qwen_staging_checksum: the cores work on 4,096-byte units, read the range in 32 KiB chunks, and write a 64-byte partial-sum page each.
constexpr uint32_t kChecksumUnit = 4096;
constexpr uint32_t kChecksumChunkBytes = 32768;
constexpr uint32_t kChecksumPartialBytes = 64;
constexpr uint32_t kChecksumColumns = 8;
constexpr uint32_t kChecksumDefaultCores = 64;
constexpr uint32_t kChecksumMaxCores = 80;  // 8 columns x 10 rows: the Blackhole compute grid is 13 x 10
// A launch checksums up to this many consecutive ranges (qwen_staging_checksum(..., count=N) with more is several launches). Every core writes one 64-byte
// partial page per range, so the partials buffer is kChecksumMaxRanges x kChecksumMaxCores pages.
constexpr uint32_t kChecksumMaxRanges = 512;
constexpr uint32_t kChecksumArgs = 10;
constexpr uint64_t kChecksumStep = 2654435761ull;

// qwen_blocks_to_staging (the gather, qwen_p2p_gather.cpp) is the exact mirror of the scatter and uses ITS constants: 16 cores (8 columns x 2 rows), up to 32
// blocks per core, 16-page chunks, two halves of scratch. One call is one launch, so a call carries at most kP2PMaxBlocksPerCall blocks (= 16 x 32), and the
// runtime args of a gather core are the scatter's (the source is the tensor, the destination the staging).
constexpr uint32_t kP2PMaxBlocksPerCall = 512;

// qwen_blocks_digest_async (the block-row digest, qwen_p2p_digest_rows.cpp): the work is divided BY BLOCK, so a core digests whole rows. One call takes up to
// kDigestMaxTensors tensors (their addresses are runtime args: one program serves any number of tensors up to the limit) and kDigestMaxBlocks blocks, on
// kDigestMaxCores cores (10 columns x 8 rows, which fits the 10 x 9 worker grid the Gemma serving image pins with TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE as well as the
// 13 x 10 grid of a bare Blackhole; a core past the end of the blocks has none). A core takes at most kDigestMaxBlocksPerCore = ceil(128 / 80) blocks.
// A block's digest is ONE 64-byte partial page: the 64-bit sum in words 0 (low) and 1 (high), the rest zero. A row is read in chunks of kDigestChunkPages
// pages into two halves of scratch. The partials buffer of one call holds kDigestMaxBlocks pages; kDigestRingBuffers of them take turns.
constexpr uint32_t kDigestColumns = 10;
constexpr uint32_t kDigestMaxCores = 80;
constexpr uint32_t kDigestMaxTensors = 16;
constexpr uint32_t kDigestMaxBlocks = 128;
constexpr uint32_t kDigestMaxBlocksPerCore = 2;
constexpr uint32_t kDigestChunkPages = 8;
constexpr uint32_t kDigestMaxPageBytes = 8192;
constexpr uint32_t kDigestPartialBytes = 64;
constexpr uint32_t kDigestRingBuffers = 2;
// Runtime args of one digest core: [0] partials address [1] tensor count [2] block count of this core (<= kDigestMaxBlocksPerCore) [3] the core's first partials
// page (the index of its first block in the call), then kDigestMaxTensors tensor addresses (unused ones zero), then kDigestMaxBlocksPerCore block ids.
constexpr uint32_t kDigestFixedArgs = 4;
constexpr uint32_t kDigestArgs = 22;
static_assert(kDigestArgs == kDigestFixedArgs + kDigestMaxTensors + kDigestMaxBlocksPerCore, "digest runtime args");
static_assert(kDigestMaxBlocksPerCore * kDigestMaxCores >= kDigestMaxBlocks, "every block of a call has a core");
static_assert(kDigestMaxCores % kDigestColumns == 0, "the digest cores are whole rows");
static_assert(kP2PMaxBlocksPerCall == kScatterLaunchBlocks, "a gather or scatter call is one launch");

// The staging buffer is an interleaved DRAM buffer of 8 banks; its page size is a parameter (the default keeps a 128 MiB range at 128 pages).
constexpr uint64_t kDefaultStagingPageBytes = 1ull << 20;

inline uint64_t ceil_div(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

// The scatter launches of n blocks: (first block, block count) each, every count <= kScatterLaunchBlocks.
inline std::vector<std::pair<uint32_t, uint32_t>> scatter_launches(uint32_t n) {
    std::vector<std::pair<uint32_t, uint32_t>> launches;
    for (uint32_t first = 0; first < n; first += kScatterLaunchBlocks) {
        launches.emplace_back(first, n - first < kScatterLaunchBlocks ? n - first : kScatterLaunchBlocks);
    }
    return launches;
}

// The checksum launches of n ranges: (first range, range count) each, every count <= kChecksumMaxRanges. The launches share one partials buffer, so the host
// reads each back before the next.
inline std::vector<std::pair<uint32_t, uint32_t>> checksum_launches(uint32_t n) {
    std::vector<std::pair<uint32_t, uint32_t>> launches;
    for (uint32_t first = 0; first < n; first += kChecksumMaxRanges) {
        launches.emplace_back(first, n - first < kChecksumMaxRanges ? n - first : kChecksumMaxRanges);
    }
    return launches;
}

// How a launch of n (<= kScatterLaunchBlocks) blocks is shared by `cores` cores: core c takes a contiguous run (first, count) of the launch's
// blocks, the first n % cores cores one block longer. Always `cores` entries; a core past the end gets (n, 0).
inline std::vector<std::pair<uint32_t, uint32_t>> split_blocks(uint32_t n, uint32_t cores) {
    std::vector<std::pair<uint32_t, uint32_t>> runs;
    const uint32_t base = n / cores;
    const uint32_t extra = n % cores;
    uint32_t first = 0;
    for (uint32_t core = 0; core < cores; ++core) {
        const uint32_t count = base + (core < extra ? 1u : 0u);
        runs.emplace_back(first, count);
        first += count;
    }
    return runs;
}

// How `units` 4 KiB units are shared by `cores` cores: core c takes units [floor(c * units / cores), floor((c + 1) * units / cores)).
inline std::vector<std::pair<uint64_t, uint64_t>> split_units(uint64_t units, uint32_t cores) {
    std::vector<std::pair<uint64_t, uint64_t>> runs;
    for (uint32_t core = 0; core < cores; ++core) {
        runs.emplace_back(units * core / cores, units * (core + 1) / cores);
    }
    return runs;
}

// w_k = (k * 2654435761 + 1) mod 2^32: the weight of word k (uint64 arithmetic wraps mod 2^64, which is a multiple of 2^32).
inline uint32_t start_weight(uint64_t k) { return static_cast<uint32_t>(k * kChecksumStep + 1ull); }

// L1 scratch (one circular buffer, never pushed) of each kernel: the kernel aligns its start up to 64 bytes, so 64 bytes of slack lead.
inline uint32_t scatter_cb_bytes(uint32_t page_bytes) { return 2 * kScatterChunkPages * page_bytes + 64; }
inline uint32_t checksum_cb_bytes() { return 64 + 2 * kChecksumChunkBytes + kChecksumPartialBytes; }
// The gather's scratch is the scatter's. The digest's is the aligned start, two halves of kDigestChunkPages pages, then the 64-byte output page.
inline uint32_t gather_cb_bytes(uint32_t page_bytes) { return scatter_cb_bytes(page_bytes); }
inline uint32_t digest_cb_bytes(uint32_t page_bytes) { return 64 + 2 * kDigestChunkPages * page_bytes + kDigestPartialBytes; }

// The runtime args of one scatter core (kScatterArgs words, the same count on every call): `run` is the core's (first, count) inside the launch
// that starts at block launch_first of `blocks`; the staging byte offset is where that core's first row begins. Row i of the call (block blocks[i])
// is read from byte_offset + i * row_stride; the kernel finds the later rows of the core from the stride.
inline std::vector<uint32_t> scatter_core_args(
    uint32_t destination_address,
    uint32_t staging_address,
    uint32_t bank,
    uint64_t byte_offset,
    uint64_t row_stride,
    const std::vector<uint32_t>& blocks,
    uint32_t launch_first,
    std::pair<uint32_t, uint32_t> run) {
    std::vector<uint32_t> values(kScatterArgs, 0);
    values[0] = destination_address;
    values[1] = staging_address;
    values[2] = bank;
    values[3] = static_cast<uint32_t>(byte_offset + uint64_t(launch_first + run.first) * row_stride);
    values[4] = run.second;
    values[5] = static_cast<uint32_t>(row_stride);
    for (uint32_t i = 0; i < run.second; ++i) {
        values[kScatterFixedArgs + i] = blocks[launch_first + run.first + i];
    }
    return values;
}

// The runtime args of one checksum core (kChecksumArgs words). The launch checksums `ranges` consecutive ranges of range_bytes each, the first at byte_offset; every
// range is split among the cores the same way, so a core takes its units [first, last) of EACH range (units are relative to the range start) and starts its
// weight at w_(k0) of its first word counted from the start of that range. Its partial for range r goes to partials page slot + r * slot_stride.
inline std::vector<uint32_t> checksum_core_args(
    uint32_t staging_address,
    uint32_t partials_address,
    uint32_t bank,
    uint64_t byte_offset,
    std::pair<uint64_t, uint64_t> units,
    uint32_t slot,
    uint32_t ranges,
    uint64_t range_bytes,
    uint32_t slot_stride) {
    std::vector<uint32_t> values(kChecksumArgs, 0);
    values[0] = staging_address;
    values[1] = partials_address;
    values[2] = bank;
    values[3] = static_cast<uint32_t>(byte_offset + units.first * kChecksumUnit);
    values[4] = static_cast<uint32_t>((units.second - units.first) * kChecksumUnit);
    values[5] = start_weight(units.first * kChecksumUnit / 4);
    values[6] = slot;
    values[7] = ranges;
    values[8] = static_cast<uint32_t>(range_bytes);
    values[9] = slot_stride;
    return values;
}

// The runtime args of one gather core: the scatter's layout (a constant count), with the SOURCE tensor address first and the staging address second:
// row i of the call (block blocks[i]) is written to staging[byte_offset + i * row_stride : + slice bytes].
inline std::vector<uint32_t> gather_core_args(
    uint32_t source_address,
    uint32_t staging_address,
    uint32_t bank,
    uint64_t byte_offset,
    uint64_t row_stride,
    const std::vector<uint32_t>& blocks,
    uint32_t launch_first,
    std::pair<uint32_t, uint32_t> run) {
    return scatter_core_args(source_address, staging_address, bank, byte_offset, row_stride, blocks, launch_first, run);
}

// The runtime args of one digest core (kDigestArgs words, the same count on every call). `run` is the core's (first, count) of the call's blocks; its
// partials pages are first .. first + count - 1.
inline std::vector<uint32_t> digest_core_args(
    uint32_t partials_address,
    const std::vector<uint32_t>& tensor_addresses,
    const std::vector<uint32_t>& blocks,
    std::pair<uint32_t, uint32_t> run) {
    std::vector<uint32_t> values(kDigestArgs, 0);
    values[0] = partials_address;
    values[1] = static_cast<uint32_t>(tensor_addresses.size());
    values[2] = run.second;
    values[3] = run.first;
    for (size_t t = 0; t < tensor_addresses.size() && t < kDigestMaxTensors; ++t) {
        values[kDigestFixedArgs + t] = tensor_addresses[t];
    }
    for (uint32_t i = 0; i < run.second && i < kDigestMaxBlocksPerCore; ++i) {
        values[kDigestFixedArgs + kDigestMaxTensors + i] = blocks[run.first + i];
    }
    return values;
}

// Argument checks shared by the ops; each returns nullptr when the arguments are fine, else the reason.
// Rows are row_stride apart (row_stride == slice_bytes: contiguous). The last row ends at byte_offset + (blocks - 1) * row_stride + slice_bytes, which must lie inside the staging range.
inline const char* check_scatter(
    uint64_t byte_offset, uint64_t slice_bytes, uint64_t page_bytes, uint64_t blocks, uint64_t staging_bytes, uint64_t row_stride) {
    if (page_bytes == 0 || page_bytes % kDramAlignment != 0) {
        return "the tensor's page size must be a multiple of 64 bytes (a whole DRAM-aligned page)";
    }
    if (page_bytes > kScatterMaxPageBytes) {
        return "the tensor's page size is above 8192 bytes (the scatter's L1 scratch is two halves of 16 pages)";
    }
    if (slice_bytes == 0 || slice_bytes % page_bytes != 0) {
        return "a block must be a whole number of tensor pages";
    }
    if (byte_offset % kDramAlignment != 0) {
        return "byte_offset must be a multiple of 64";
    }
    if (row_stride % kDramAlignment != 0 || row_stride < slice_bytes) {
        return "row_stride must be a multiple of 64 and at least the block size (the slice bytes)";
    }
    if (blocks == 0) {
        return "no blocks";
    }
    if (byte_offset > staging_bytes || slice_bytes > staging_bytes - byte_offset ||
        blocks - 1 > (staging_bytes - byte_offset - slice_bytes) / row_stride) {
        return "the byte range lies outside the staging range";
    }
    return nullptr;
}

// `ranges` consecutive ranges of `length` bytes from byte_offset (ranges == 1: one range). `ranges` has no limit of its own: the op makes several launches.
inline const char* check_checksum(
    uint64_t byte_offset, uint64_t length, uint64_t staging_bytes, uint32_t cores, uint64_t ranges = 1) {
    if (length == 0 || length % kChecksumUnit != 0) {
        return "length must be a positive multiple of 4096";
    }
    if (byte_offset % kDramAlignment != 0) {
        return "byte_offset must be a multiple of 64";
    }
    if (ranges == 0) {
        return "count must be at least 1";
    }
    if (byte_offset > staging_bytes || length > staging_bytes - byte_offset || ranges - 1 > (staging_bytes - byte_offset - length) / length) {
        return "the byte range lies outside the staging range";
    }
    if (cores == 0 || cores > kChecksumMaxCores || cores % kChecksumColumns != 0) {
        return "cores must be a multiple of 8, at most 80";
    }
    return nullptr;
}

// The staging geometry: the usable range of every bank is size_bytes (bytes per bank) of an interleaved buffer of num_banks * size_bytes bytes. bank_bytes is what the
// allocator can give in one bank; the range must fit it, and its offsets are 32-bit in the kernels' runtime arguments. The size is a multiple of the staging page
// size and of 4096 (the drive's I/O unit and the canary page).
inline const char* check_staging(uint64_t size_bytes, uint64_t page_bytes, uint32_t num_banks, uint64_t bank_bytes) {
    if (page_bytes == 0 || page_bytes % kDramAlignment != 0) {
        return "the staging page size must be a multiple of 64";
    }
    if (size_bytes == 0 || size_bytes % page_bytes != 0) {
        return "the staging size must be a positive multiple of the staging page size";
    }
    if (size_bytes % 4096 != 0) {
        return "the staging size per bank must be a multiple of 4096";
    }
    if (num_banks == 0) {
        return "the device has no DRAM banks";
    }
    if (size_bytes > bank_bytes || size_bytes >= (1ull << 32)) {
        return "the staging size is larger than the allocatable bytes of one DRAM bank";
    }
    return nullptr;
}

// A bank argument of an op: it must name a DRAM bank of the device.
inline const char* check_bank(uint64_t bank, uint32_t num_banks) {
    if (bank >= num_banks) {
        return "bank is outside the device's DRAM banks";
    }
    return nullptr;
}

// The shared arguments of the v4 gather (qwen_blocks_to_staging) and scatter (qwen_write_blocks_from_staging): the v3 checks of check_scatter, plus pages_per_block
// (the caller's own figure, which must be the tensor's block: slice_bytes / page_bytes) and the limit of one launch per call (kP2PMaxBlocksPerCall blocks).
inline const char* check_block_rows(
    uint64_t byte_offset,
    uint64_t slice_bytes,
    uint64_t page_bytes,
    uint64_t pages_per_block,
    uint64_t blocks,
    uint64_t staging_bytes,
    uint64_t row_stride) {
    if (blocks > kP2PMaxBlocksPerCall) {
        return "at most 512 blocks per call (one launch)";
    }
    if (pages_per_block == 0 || page_bytes == 0 || slice_bytes % page_bytes != 0 || slice_bytes / page_bytes != pages_per_block) {
        return "pages_per_block must be the tensor's block size divided by its page size";
    }
    return check_scatter(byte_offset, slice_bytes, page_bytes, blocks, staging_bytes, row_stride);
}

inline const char* check_gather(
    uint64_t byte_offset,
    uint64_t slice_bytes,
    uint64_t page_bytes,
    uint64_t pages_per_block,
    uint64_t blocks,
    uint64_t staging_bytes,
    uint64_t row_stride) {
    return check_block_rows(byte_offset, slice_bytes, page_bytes, pages_per_block, blocks, staging_bytes, row_stride);
}

// The block ids of a call against the tensor's `slices` blocks. A scatter needs distinct ids (two writes to one block have no order); a gather and a digest only
// read, so a repeated id is fine. On a refusal *bad_index (when given) is the position of the first offender.
inline const char* check_block_ids(const std::vector<uint32_t>& blocks, uint64_t slices, bool require_distinct, size_t* bad_index = nullptr) {
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (blocks[i] >= slices) {
            if (bad_index != nullptr) {
                *bad_index = i;
            }
            return "a block id is outside the tensor";
        }
    }
    if (require_distinct) {
        std::vector<uint32_t> sorted(blocks);
        std::sort(sorted.begin(), sorted.end());
        for (size_t i = 1; i < sorted.size(); ++i) {
            if (sorted[i] == sorted[i - 1]) {
                if (bad_index != nullptr) {
                    for (size_t j = 0; j < blocks.size(); ++j) {
                        if (blocks[j] == sorted[i]) {
                            *bad_index = j;
                            break;
                        }
                    }
                }
                return "the block ids must be distinct (two writes to one block have no order)";
            }
        }
    }
    return nullptr;
}

// The arguments of qwen_blocks_digest_async that do not depend on one tensor: how many tensors and blocks, and the geometry of a block. Every tensor must have the
// same page size and block size (checked by the caller against pages_per_block * page_bytes), and the page ids block * pages_per_block + j must fit 32 bits.
inline const char* check_digest_rows(uint64_t tensors, uint64_t blocks, uint64_t pages_per_block, uint64_t page_bytes, uint64_t slices) {
    if (tensors == 0 || tensors > kDigestMaxTensors) {
        return "tensors must be 1 to 16";
    }
    if (blocks == 0 || blocks > kDigestMaxBlocks) {
        return "blocks must be 1 to 128";
    }
    if (page_bytes == 0 || page_bytes % kDramAlignment != 0) {
        return "the tensor's page size must be a multiple of 64 bytes (a whole DRAM-aligned page)";
    }
    if (page_bytes > kDigestMaxPageBytes) {
        return "the tensor's page size is above 8192 bytes (the digest's L1 scratch is two halves of 8 pages)";
    }
    if (pages_per_block == 0) {
        return "pages_per_block must be positive";
    }
    if (slices == 0 || slices * pages_per_block >= (1ull << 32)) {
        return "the tensor's page ids do not fit 32 bits";
    }
    return nullptr;
}

}  // namespace qwen_p2p
