// Runs qwen_p2p_scatter.cpp on the CPU (against the stand-in dataflow API, emu/api/dataflow/dataflow_api.h) the way the host drives it: the
// launches, the per-core argument vectors and the circular buffer size all come from qwen_p2p_plan.hpp, the same header the extension uses.
//   emu_scatter <scenario> <staging.bin> <dst_in.bin> <dst_out.bin>
// scenario: page_bytes pages_per_block bank staging_address byte_offset dst_address dst_pages bank_bytes blocks...
// staging.bin = the bytes of the staging range [staging_address, + n * slice_bytes + byte_offset); dst_in/out = dst_pages pages in page order.
// Prints "OK reads=<n> writes=<n> stray=<bytes changed outside the destination>".
#include "api/dataflow/dataflow_api.h"
#include "emu_common.hpp"
#include "../qwen_p2p_plan.hpp"
#ifdef EMU_KERNEL_FILE
#include EMU_KERNEL_FILE  // a deliberately broken copy, for the "has teeth" tests
#else
#include "../qwen_p2p_scatter.cpp"
#endif

int main(int argc, char** argv) {
    if (argc != 5) return 2;
    const auto config = emu_run::read_config(argv[1]);
    const uint32_t page = static_cast<uint32_t>(emu_run::one(config, "page_bytes"));
    const uint32_t pages_per_block = static_cast<uint32_t>(emu_run::one(config, "pages_per_block"));
    const uint32_t bank = static_cast<uint32_t>(emu_run::one(config, "bank"));
    const uint32_t staging_address = static_cast<uint32_t>(emu_run::one(config, "staging_address"));
    const uint64_t byte_offset = emu_run::one(config, "byte_offset");
    const uint32_t dst_address = static_cast<uint32_t>(emu_run::one(config, "dst_address"));
    const uint32_t dst_pages = static_cast<uint32_t>(emu_run::one(config, "dst_pages"));
    const uint64_t bank_bytes = emu_run::one(config, "bank_bytes");
    std::vector<uint32_t> blocks;
    for (uint64_t block : config.at("blocks")) blocks.push_back(static_cast<uint32_t>(block));
    const uint64_t slice_bytes = uint64_t(page) * pages_per_block;
    const uint32_t aligned = (page + 63u) & ~63u;

    auto& state = emu::state;
    state.banks.assign(emu::kBanks, std::vector<uint8_t>(bank_bytes, 0xEE));
    const std::vector<uint8_t> staging = emu_run::read_file(argv[2]);
    if (staging_address + staging.size() > bank_bytes) return 2;
    std::copy(staging.begin(), staging.end(), state.banks[bank].begin() + staging_address);
    const std::vector<uint8_t> dst_in = emu_run::read_file(argv[3]);
    if (dst_in.size() != uint64_t(dst_pages) * page) return 2;
    for (uint32_t p = 0; p < dst_pages; ++p) {
        std::copy_n(dst_in.begin() + uint64_t(p) * page, page, state.banks[p % emu::kBanks].begin() + dst_address + uint64_t(p / emu::kBanks) * aligned);
    }
    const auto before = state.banks;

    const uint32_t l1 = emu_run::l1_region();
    state.cb_base = l1 + 16;
    state.cb_bytes = qwen_p2p::scatter_cb_bytes(page);
    state.read_bank = bank;
    state.read_lo = staging_address + byte_offset;
    state.read_hi = state.read_lo + uint64_t(blocks.size()) * slice_bytes;
    state.write_lo = dst_address;
    state.write_hi = bank_bytes;
    state.write_page = page;

    for (const auto& [launch_first, launch_count] : qwen_p2p::scatter_launches(static_cast<uint32_t>(blocks.size()))) {
        const auto runs = qwen_p2p::split_blocks(launch_count, qwen_p2p::kScatterCores);
        for (uint32_t core = 0; core < qwen_p2p::kScatterCores; ++core) {
            state.args = qwen_p2p::scatter_core_args(dst_address, staging_address, bank, byte_offset, slice_bytes, blocks, launch_first, runs[core]);
            kernel_main();
            if (!state.reads.empty() || !state.writes.empty()) emu::violation("the kernel ended with a transfer pending");
        }
    }

    std::vector<uint8_t> dst_out(uint64_t(dst_pages) * page);
    for (uint32_t p = 0; p < dst_pages; ++p) {
        std::copy_n(state.banks[p % emu::kBanks].begin() + dst_address + uint64_t(p / emu::kBanks) * aligned, page, dst_out.begin() + uint64_t(p) * page);
    }
    emu_run::write_file(argv[4], dst_out);
    uint64_t stray = 0;
    const uint64_t dst_lo = dst_address;
    const uint64_t dst_hi = dst_address + uint64_t((dst_pages + emu::kBanks - 1) / emu::kBanks) * aligned;
    for (uint32_t b = 0; b < emu::kBanks; ++b) {
        for (uint64_t at = 0; at < bank_bytes; ++at) {
            if ((at < dst_lo || at >= dst_hi) && state.banks[b][at] != before[b][at]) ++stray;
        }
    }
    std::printf("OK reads=%llu writes=%llu stray=%llu\n", static_cast<unsigned long long>(state.reads_issued),
                static_cast<unsigned long long>(state.writes_issued), static_cast<unsigned long long>(stray));
    return 0;
}
