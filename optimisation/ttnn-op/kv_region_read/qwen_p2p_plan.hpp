// qwen_p2p_plan: the pure arithmetic of the P2P restore's device ops (qwen_kv_read.cpp, version 3), with no tt-metal dependency, so a CPU test
// compiles it and compares every function with its Python mirror (p2p_staging_ref.py). Nothing here touches a device.
#pragma once

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
// Runtime args of one scatter core: staging/destination addresses, bank, source offset, block count, then the block ids (padded).
constexpr uint32_t kScatterFixedArgs = 5;
constexpr uint32_t kScatterArgs = kScatterFixedArgs + kScatterMaxBlocksPerCore;

// qwen_staging_checksum: the cores work on 4,096-byte units, read the range in 32 KiB chunks, and write a 64-byte partial-sum page each.
constexpr uint32_t kChecksumUnit = 4096;
constexpr uint32_t kChecksumChunkBytes = 32768;
constexpr uint32_t kChecksumPartialBytes = 64;
constexpr uint32_t kChecksumColumns = 8;
constexpr uint32_t kChecksumDefaultCores = 64;
constexpr uint32_t kChecksumMaxCores = 80;  // 8 columns x 10 rows: the Blackhole compute grid is 13 x 10
constexpr uint32_t kChecksumArgs = 7;
constexpr uint64_t kChecksumStep = 2654435761ull;

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

// The runtime args of one scatter core (kScatterArgs words, the same count on every call): `run` is the core's (first, count) inside the launch
// that starts at block launch_first of `blocks`; the staging byte offset is where that core's first block begins.
inline std::vector<uint32_t> scatter_core_args(
    uint32_t destination_address,
    uint32_t staging_address,
    uint32_t bank,
    uint64_t byte_offset,
    uint64_t slice_bytes,
    const std::vector<uint32_t>& blocks,
    uint32_t launch_first,
    std::pair<uint32_t, uint32_t> run) {
    std::vector<uint32_t> values(kScatterArgs, 0);
    values[0] = destination_address;
    values[1] = staging_address;
    values[2] = bank;
    values[3] = static_cast<uint32_t>(byte_offset + uint64_t(launch_first + run.first) * slice_bytes);
    values[4] = run.second;
    for (uint32_t i = 0; i < run.second; ++i) {
        values[kScatterFixedArgs + i] = blocks[launch_first + run.first + i];
    }
    return values;
}

// The runtime args of one checksum core (kChecksumArgs words): its units [first, last) of the range that starts at byte_offset, and its partial slot.
inline std::vector<uint32_t> checksum_core_args(
    uint32_t staging_address,
    uint32_t partials_address,
    uint32_t bank,
    uint64_t byte_offset,
    std::pair<uint64_t, uint64_t> units,
    uint32_t slot) {
    std::vector<uint32_t> values(kChecksumArgs, 0);
    values[0] = staging_address;
    values[1] = partials_address;
    values[2] = bank;
    values[3] = static_cast<uint32_t>(byte_offset + units.first * kChecksumUnit);
    values[4] = static_cast<uint32_t>((units.second - units.first) * kChecksumUnit);
    values[5] = start_weight(units.first * kChecksumUnit / 4);
    values[6] = slot;
    return values;
}

// Argument checks shared by the ops; each returns nullptr when the arguments are fine, else the reason.
inline const char* check_scatter(
    uint64_t byte_offset, uint64_t slice_bytes, uint64_t page_bytes, uint64_t blocks, uint64_t staging_bytes) {
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
    if (blocks == 0) {
        return "no blocks";
    }
    if (byte_offset > staging_bytes || blocks > (staging_bytes - byte_offset) / slice_bytes) {
        return "the byte range lies outside the staging range";
    }
    return nullptr;
}

inline const char* check_checksum(uint64_t byte_offset, uint64_t length, uint64_t staging_bytes, uint32_t cores) {
    if (length == 0 || length % kChecksumUnit != 0) {
        return "length must be a positive multiple of 4096";
    }
    if (byte_offset % kDramAlignment != 0) {
        return "byte_offset must be a multiple of 64";
    }
    if (byte_offset > staging_bytes || length > staging_bytes - byte_offset) {
        return "the byte range lies outside the staging range";
    }
    if (cores == 0 || cores > kChecksumMaxCores || cores % kChecksumColumns != 0) {
        return "cores must be a multiple of 8, at most 80";
    }
    return nullptr;
}

// The staging geometry: a range of size_bytes in one bank is the usable part of an interleaved buffer of num_banks * size_bytes bytes. bank_bytes is what the
// allocator can give in one bank; the range must fit it, and its offsets are 32-bit in the kernels' runtime arguments.
inline const char* check_staging(uint64_t size_bytes, uint64_t page_bytes, uint32_t bank, uint32_t num_banks, uint64_t bank_bytes) {
    if (page_bytes == 0 || page_bytes % kDramAlignment != 0) {
        return "the staging page size must be a multiple of 64";
    }
    if (size_bytes == 0 || size_bytes % page_bytes != 0) {
        return "the staging size must be a positive multiple of the staging page size";
    }
    if (num_banks == 0 || bank >= num_banks) {
        return "the staging bank is outside the device's DRAM banks";
    }
    if (size_bytes > bank_bytes || size_bytes >= (1ull << 32)) {
        return "the staging size is larger than the allocatable bytes of one DRAM bank";
    }
    return nullptr;
}

}  // namespace qwen_p2p
