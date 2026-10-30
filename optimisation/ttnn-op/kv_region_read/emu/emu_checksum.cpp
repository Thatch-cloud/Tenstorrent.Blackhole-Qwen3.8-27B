// Runs qwen_p2p_checksum.cpp on the CPU the way the host drives it (units, weights, slots, circular buffer from qwen_p2p_plan.hpp).
//   emu_checksum <scenario> <staging.bin>
// scenario: staging_address partials_address bank byte_offset length cores bank_bytes; staging.bin = the bytes of [staging_address + byte_offset, + length).
// Prints "SUM <uint64>" (the partial sums added mod 2^64) and "OK reads=<n> writes=<n>".
#include "api/dataflow/dataflow_api.h"
#include "emu_common.hpp"
#include "../qwen_p2p_plan.hpp"
#ifdef EMU_KERNEL_FILE
#include EMU_KERNEL_FILE  // a deliberately broken copy, for the "has teeth" tests
#else
#include "../qwen_p2p_checksum.cpp"
#endif

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const auto config = emu_run::read_config(argv[1]);
    const uint32_t staging_address = static_cast<uint32_t>(emu_run::one(config, "staging_address"));
    const uint32_t partials_address = static_cast<uint32_t>(emu_run::one(config, "partials_address"));
    const uint32_t bank = static_cast<uint32_t>(emu_run::one(config, "bank"));
    const uint64_t byte_offset = emu_run::one(config, "byte_offset");
    const uint64_t length = emu_run::one(config, "length");
    const uint32_t cores = static_cast<uint32_t>(emu_run::one(config, "cores"));
    const uint64_t bank_bytes = emu_run::one(config, "bank_bytes");

    auto& state = emu::state;
    state.banks.assign(emu::kBanks, std::vector<uint8_t>(bank_bytes, 0xEE));
    const std::vector<uint8_t> staging = emu_run::read_file(argv[2]);
    if (staging.size() != length) return 2;
    std::copy(staging.begin(), staging.end(), state.banks[bank].begin() + staging_address + byte_offset);

    const uint32_t l1 = emu_run::l1_region();
    state.cb_base = l1 + 16;
    state.cb_bytes = qwen_p2p::checksum_cb_bytes();
    state.read_bank = bank;
    state.read_lo = staging_address + byte_offset;
    state.read_hi = state.read_lo + length;
    state.read_unit = qwen_p2p::kChecksumUnit;
    state.write_lo = partials_address;
    state.write_hi = bank_bytes;
    state.write_page = qwen_p2p::kChecksumPartialBytes;

    const auto units = qwen_p2p::split_units(length / qwen_p2p::kChecksumUnit, cores);
    for (uint32_t core = 0; core < cores; ++core) {
        state.args = qwen_p2p::checksum_core_args(staging_address, partials_address, bank, byte_offset, units[core], core);
        kernel_main();
        if (!state.reads.empty() || !state.writes.empty()) emu::violation("the kernel ended with a transfer pending");
    }
    uint64_t sum = 0;
    for (uint32_t core = 0; core < cores; ++core) {
        uint64_t partial = 0;
        const uint8_t* page = state.banks[core % emu::kBanks].data() + partials_address + uint64_t(core / emu::kBanks) * 64;
        std::memcpy(&partial, page, 8);
        sum += partial;
    }
    std::printf("SUM %llu\nOK reads=%llu writes=%llu\n", static_cast<unsigned long long>(sum),
                static_cast<unsigned long long>(state.reads_issued), static_cast<unsigned long long>(state.writes_issued));
    return 0;
}
