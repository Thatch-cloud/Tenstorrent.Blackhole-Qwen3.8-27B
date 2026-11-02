// Runs qwen_p2p_gather.cpp on the CPU (against the stand-in dataflow API, emu/api/dataflow/dataflow_api.h) the way the host drives it: the launches, the per-core
// argument vectors and the circular buffer size all come from qwen_p2p_plan.hpp, the same header the extension uses.
//   emu_gather <scenario> <tensor.bin> <staging_in.bin> <staging_out.bin>
// scenario: page_bytes pages_per_block bank staging_address byte_offset tensor_address tensor_pages bank_bytes blocks... and, optionally, row_stride (default: the
// block size, contiguous rows)
// tensor.bin = tensor_pages pages in page order (the source, interleaved over the 8 banks from tensor_address); staging_in/out = the bytes of the staging range
// [staging_address, + its length) in the staging bank before and after.
// The harness lets the kernel READ exactly the pages of the blocks it was asked for and WRITE exactly the pages of the rows (the bytes between two rows are not its to
// touch). Prints "OK reads=<n> writes=<n> stray=<bytes changed outside the rows>".
#include "api/dataflow/dataflow_api.h"
#include "emu_common.hpp"
#include "../qwen_p2p_plan.hpp"
#ifdef EMU_KERNEL_FILE
#include EMU_KERNEL_FILE  // a deliberately broken copy, for the "has teeth" tests
#else
#include "../qwen_p2p_gather.cpp"
#endif

int main(int argc, char** argv) {
    if (argc != 5) return 2;
    const auto config = emu_run::read_config(argv[1]);
    const uint32_t page = static_cast<uint32_t>(emu_run::one(config, "page_bytes"));
    const uint32_t pages_per_block = static_cast<uint32_t>(emu_run::one(config, "pages_per_block"));
    const uint32_t bank = static_cast<uint32_t>(emu_run::one(config, "bank"));
    const uint32_t staging_address = static_cast<uint32_t>(emu_run::one(config, "staging_address"));
    const uint64_t byte_offset = emu_run::one(config, "byte_offset");
    const uint32_t tensor_address = static_cast<uint32_t>(emu_run::one(config, "tensor_address"));
    const uint32_t tensor_pages = static_cast<uint32_t>(emu_run::one(config, "tensor_pages"));
    const uint64_t bank_bytes = emu_run::one(config, "bank_bytes");
    std::vector<uint32_t> blocks;
    for (uint64_t block : config.at("blocks")) blocks.push_back(static_cast<uint32_t>(block));
    const uint64_t slice_bytes = uint64_t(page) * pages_per_block;
    const uint64_t row_stride = emu_run::one_or(config, "row_stride", slice_bytes);
    const uint32_t aligned = (page + 63u) & ~63u;

    auto& state = emu::state;
    state.banks.assign(emu::kBanks, std::vector<uint8_t>(bank_bytes, 0xEE));
    const std::vector<uint8_t> tensor = emu_run::read_file(argv[2]);
    if (tensor.size() != uint64_t(tensor_pages) * page) return 2;
    for (uint32_t p = 0; p < tensor_pages; ++p) {
        const uint64_t at = tensor_address + uint64_t(p / emu::kBanks) * aligned;
        if (at + page > bank_bytes) return 2;
        std::copy_n(tensor.begin() + uint64_t(p) * page, page, state.banks[p % emu::kBanks].begin() + at);
    }
    const std::vector<uint8_t> staging_in = emu_run::read_file(argv[3]);
    if (staging_address + staging_in.size() > bank_bytes) return 2;
    std::copy(staging_in.begin(), staging_in.end(), state.banks[bank].begin() + staging_address);
    const auto before = state.banks;

    const uint32_t l1 = emu_run::l1_region();
    state.cb_base = l1 + 16;
    state.cb_bytes = qwen_p2p::gather_cb_bytes(page);
    // reads: the pages of the blocks asked for (any bank), exactly; writes: the rows of the staging range, exactly their pages
    state.read_regions = {emu::Region{-1, tensor_address, bank_bytes, 0, 0}};
    state.exact_read_size = page;
    for (uint32_t block : blocks) {
        for (uint32_t j = 0; j < pages_per_block; ++j) {
            const uint64_t id = uint64_t(block) * pages_per_block + j;
            state.exact_reads.insert((uint64_t(id % emu::kBanks) << 48) | (tensor_address + (id / emu::kBanks) * aligned));
        }
    }
    const uint64_t rows_lo = uint64_t(staging_address) + byte_offset;
    state.write_regions = {emu::Region{int(bank), rows_lo, rows_lo + uint64_t(blocks.size() - 1) * row_stride + slice_bytes, row_stride, slice_bytes}};
    state.write_page = page;
    state.exact_write_size = page;
    for (size_t i = 0; i < blocks.size(); ++i) {
        for (uint32_t j = 0; j < pages_per_block; ++j) {
            state.exact_writes.insert((uint64_t(bank) << 48) | (rows_lo + i * row_stride + uint64_t(j) * page));
        }
    }

    for (const auto& [launch_first, launch_count] : qwen_p2p::scatter_launches(static_cast<uint32_t>(blocks.size()))) {
        const auto runs = qwen_p2p::split_blocks(launch_count, qwen_p2p::kScatterCores);
        for (uint32_t core = 0; core < qwen_p2p::kScatterCores; ++core) {
            state.args = qwen_p2p::gather_core_args(tensor_address, staging_address, bank, byte_offset, row_stride, blocks, launch_first, runs[core]);
            kernel_main();
            if (!state.reads.empty() || !state.writes.empty()) emu::violation("the kernel ended with a transfer pending");
        }
    }

    std::vector<uint8_t> staging_out(staging_in.size());
    std::copy_n(state.banks[bank].begin() + staging_address, staging_out.size(), staging_out.begin());
    emu_run::write_file(argv[4], staging_out);
    uint64_t stray = 0;
    for (uint32_t b = 0; b < emu::kBanks; ++b) {
        for (uint64_t at = 0; at < bank_bytes; ++at) {
            bool in_row = false;
            if (b == bank && at >= rows_lo) {
                const uint64_t rel = at - rows_lo;
                in_row = rel / row_stride < blocks.size() && rel % row_stride < slice_bytes;
            }
            if (!in_row && state.banks[b][at] != before[b][at]) ++stray;
        }
    }
    std::printf("OK reads=%llu writes=%llu stray=%llu\n", static_cast<unsigned long long>(state.reads_issued), static_cast<unsigned long long>(state.writes_issued),
                static_cast<unsigned long long>(stray));
    return 0;
}
