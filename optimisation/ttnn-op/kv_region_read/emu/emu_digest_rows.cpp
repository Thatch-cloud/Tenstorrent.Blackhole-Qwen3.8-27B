// Runs qwen_p2p_digest_rows.cpp on the CPU (against the stand-in dataflow API, emu/api/dataflow/dataflow_api.h) the way the host drives it: the per-core argument
// vectors (the split of the blocks over the cores, the tensor addresses) and the circular buffer size come from qwen_p2p_plan.hpp, the same header the extension uses.
//   emu_digest_rows <scenario> <tensors.bin>
// scenario: page_bytes pages_per_block tensors tensor_pages bank_bytes partials_address tensor_addresses (one per tensor) blocks...
// tensors.bin = the T tensors back to back, each tensor_pages pages in page order (interleaved over the 8 banks from its own address). The partials buffer is
// interleaved 64-byte pages from partials_address: block i's partial is page i.
// The harness lets the kernel READ exactly the pages of the blocks it was asked for (in every tensor) and WRITE exactly the partial pages of those blocks.
// Prints "DIGESTS <uint64>..." (one per block, from the partial pages) and "OK reads=<n> writes=<n> stray=<bytes changed outside the partial pages>".
#include "api/dataflow/dataflow_api.h"
#include "emu_common.hpp"
#include "../qwen_p2p_plan.hpp"
#ifdef EMU_KERNEL_FILE
#include EMU_KERNEL_FILE  // a deliberately broken copy, for the "has teeth" tests
#else
#include "../qwen_p2p_digest_rows.cpp"
#endif

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const auto config = emu_run::read_config(argv[1]);
    const uint32_t page = static_cast<uint32_t>(emu_run::one(config, "page_bytes"));
    const uint32_t pages_per_block = static_cast<uint32_t>(emu_run::one(config, "pages_per_block"));
    const uint32_t tensors = static_cast<uint32_t>(emu_run::one(config, "tensors"));
    const uint32_t tensor_pages = static_cast<uint32_t>(emu_run::one(config, "tensor_pages"));
    const uint64_t bank_bytes = emu_run::one(config, "bank_bytes");
    const uint32_t partials_address = static_cast<uint32_t>(emu_run::one(config, "partials_address"));
    std::vector<uint32_t> addresses;
    for (uint64_t address : config.at("tensor_addresses")) addresses.push_back(static_cast<uint32_t>(address));
    std::vector<uint32_t> blocks;
    for (uint64_t block : config.at("blocks")) blocks.push_back(static_cast<uint32_t>(block));
    if (addresses.size() != tensors) return 2;
    const uint32_t aligned = (page + 63u) & ~63u;

    auto& state = emu::state;
    state.banks.assign(emu::kBanks, std::vector<uint8_t>(bank_bytes, 0xEE));
    const std::vector<uint8_t> data = emu_run::read_file(argv[2]);
    if (data.size() != uint64_t(tensors) * tensor_pages * page) return 2;
    for (uint32_t t = 0; t < tensors; ++t) {
        for (uint32_t p = 0; p < tensor_pages; ++p) {
            const uint64_t at = addresses[t] + uint64_t(p / emu::kBanks) * aligned;
            if (at + page > bank_bytes) return 2;
            std::copy_n(data.begin() + (uint64_t(t) * tensor_pages + p) * page, page, state.banks[p % emu::kBanks].begin() + at);
        }
    }
    const auto before = state.banks;

    const uint32_t l1 = emu_run::l1_region();
    state.cb_base = l1 + 16;
    state.cb_bytes = qwen_p2p::digest_cb_bytes(page);
    state.read_regions = {emu::Region{-1, 0, bank_bytes, 0, 0}};
    state.exact_read_size = page;
    for (uint32_t t = 0; t < tensors; ++t) {
        for (uint32_t block : blocks) {
            for (uint32_t j = 0; j < pages_per_block; ++j) {
                const uint64_t id = uint64_t(block) * pages_per_block + j;
                state.exact_reads.insert((uint64_t(id % emu::kBanks) << 48) | (addresses[t] + (id / emu::kBanks) * aligned));
            }
        }
    }
    state.write_regions = {emu::Region{-1, partials_address, bank_bytes, 0, 0}};
    state.write_page = qwen_p2p::kDigestPartialBytes;
    state.exact_write_size = qwen_p2p::kDigestPartialBytes;
    auto partial_address = [&](uint64_t i) { return (uint64_t(i % emu::kBanks) << 48) | (partials_address + (i / emu::kBanks) * qwen_p2p::kDigestPartialBytes); };
    for (size_t i = 0; i < blocks.size(); ++i) state.exact_writes.insert(partial_address(i));

    const auto runs = qwen_p2p::split_blocks(static_cast<uint32_t>(blocks.size()), qwen_p2p::kDigestMaxCores);
    for (uint32_t core = 0; core < qwen_p2p::kDigestMaxCores; ++core) {
        state.args = qwen_p2p::digest_core_args(partials_address, addresses, blocks, runs[core]);
        kernel_main();
        if (!state.reads.empty() || !state.writes.empty()) emu::violation("the kernel ended with a transfer pending");
    }

    std::printf("DIGESTS");
    for (size_t i = 0; i < blocks.size(); ++i) {
        const uint64_t id = i;
        uint64_t digest = 0;
        std::memcpy(&digest, state.banks[id % emu::kBanks].data() + partials_address + (id / emu::kBanks) * qwen_p2p::kDigestPartialBytes, 8);
        std::printf(" %llu", static_cast<unsigned long long>(digest));
    }
    uint64_t stray = 0;
    for (uint32_t b = 0; b < emu::kBanks; ++b) {
        for (uint64_t at = 0; at < bank_bytes; ++at) {
            bool in_partial = false;
            if (at >= partials_address) {
                const uint64_t page_index = (at - partials_address) / qwen_p2p::kDigestPartialBytes * emu::kBanks + b;
                in_partial = page_index < blocks.size();
            }
            if (!in_partial && state.banks[b][at] != before[b][at]) ++stray;
        }
    }
    std::printf("\nOK reads=%llu writes=%llu stray=%llu\n", static_cast<unsigned long long>(state.reads_issued), static_cast<unsigned long long>(state.writes_issued),
                static_cast<unsigned long long>(stray));
    return 0;
}
