// A host stand-in for the device dataflow API, just enough for qwen_p2p_scatter.cpp and qwen_p2p_checksum.cpp to be compiled by an ordinary C++
// compiler and RUN on the CPU by test_p2p_staging.py (emu_scatter.cpp, emu_checksum.cpp). It is a test double, not tt-metal: it models
//   * DRAM as 8 banks of host memory and a NOC address as (bank << 48 | byte address in the bank);
//   * an interleaved TensorAccessor (page id -> bank id % 8, address base + (id / 8) * page rounded up to 64, plus the offset);
//   * L1 as one host region at a 32-bit address (so a uint32 L1 address IS a pointer), with the kernel's circular buffer inside it;
//   * ASYNCHRONOUS NOC transfers, at the two extremes of "lands any time between the call and the barrier": by default a read is performed at
//     noc_async_read_barrier() and a write at noc_async_write_barrier() (LATE: a kernel that uses data before the barrier, or reuses an L1 buffer
//     a pending write still needs, computes a wrong answer), and with -DEMU_EAGER=1 both are performed when issued (EAGER: a kernel that
//     refills a buffer it is still reading, or changes a buffer after issuing the write that sends it, computes a wrong answer). A correct
//     kernel is right under both;
//   * the rules that bite on this stack: every DRAM access starts 64-byte aligned, L1 destinations of reads are 64-byte aligned, an access
//     is a WHOLE page (the sub-page DRAM no-op), nothing is read or written outside the ranges the harness allows, all L1 traffic stays
//     inside the circular buffer, runtime args are read only inside the argument count, and a kernel ends with no transfer pending.
// Any violation prints "EMU-VIOLATION: ..." and exits 3.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// The device headers define this as an address-space attribute of the RISC-V compiler; on the host it is nothing.
#define tt_l1_ptr

namespace emu {

constexpr uint32_t kBanks = 8;

struct Pending {
    uint64_t noc_address;
    uint32_t l1;
    uint32_t size;
};

struct State {
    std::vector<std::vector<uint8_t>> banks;
    std::vector<Pending> reads, writes;
    uint32_t cb_base = 0, cb_bytes = 0;
    std::vector<uint32_t> args;
    // The ranges the harness allows: reads (bank, [lo, hi)) and writes ([lo, hi) in every bank), and the page size writes must use (0: any).
    uint32_t read_bank = 0;
    uint64_t read_lo = 0, read_hi = 0;
    uint64_t write_lo = 0, write_hi = 0;
    uint32_t write_page = 0;
    uint32_t read_unit = 0;  // reads must be a multiple of this many bytes
    uint64_t reads_issued = 0, writes_issued = 0;
};
inline State state;

[[noreturn]] inline void violation(const char* what, uint64_t a = 0, uint64_t b = 0) {
    std::printf("EMU-VIOLATION: %s (%llu, %llu)\n", what, static_cast<unsigned long long>(a), static_cast<unsigned long long>(b));
    std::fflush(stdout);
    std::exit(3);
}

inline bool in_cb(uint64_t l1, uint64_t size) { return l1 >= state.cb_base && l1 + size <= uint64_t(state.cb_base) + state.cb_bytes; }

inline void perform_read(const Pending& read) {
    const uint32_t bank = static_cast<uint32_t>(read.noc_address >> 48);
    const uint64_t address = read.noc_address & ((uint64_t(1) << 48) - 1);
    std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(read.l1)), state.banks[bank].data() + address, read.size);
}

inline void perform_write(const Pending& write) {
    const uint32_t bank = static_cast<uint32_t>(write.noc_address >> 48);
    const uint64_t address = write.noc_address & ((uint64_t(1) << 48) - 1);
    std::memcpy(state.banks[bank].data() + address, reinterpret_cast<const void*>(static_cast<uintptr_t>(write.l1)), write.size);
}

#ifdef EMU_EAGER
constexpr bool kEager = true;
#else
constexpr bool kEager = false;
#endif

}  // namespace emu

// ---- compile-time and runtime args -----------------------------------------------------------------------------------------------------
namespace emu {
constexpr uint32_t kCompileArgs[] = {EMU_CT_ARGS};
}
#define get_compile_time_arg_val(i) (::emu::kCompileArgs[i])

template <typename T>
inline T get_arg_val(uint32_t index) {
    if (index >= emu::state.args.size()) {
        emu::violation("runtime arg index outside the argument count", index, emu::state.args.size());
    }
    return static_cast<T>(emu::state.args[index]);
}

inline uint32_t get_write_ptr(uint32_t cb) {
    if (cb != 0) {
        emu::violation("only circular buffer 0 exists", cb);
    }
    return emu::state.cb_base;
}

// ---- accessors -------------------------------------------------------------------------------------------------------------------------
template <uint32_t Offset>
struct TensorAccessorArgs {
    constexpr TensorAccessorArgs() = default;
    static constexpr uint32_t next_compile_time_args_offset() { return Offset + 1; }
};

class TensorAccessor {
public:
    template <uint32_t Offset>
    TensorAccessor(const TensorAccessorArgs<Offset>&, uint32_t base, uint32_t page_size) : base_(base), page_(page_size) {}
    uint64_t get_noc_addr(uint32_t id, uint32_t offset = 0) const {
        const uint32_t aligned = (page_ + 63u) & ~63u;
        const uint64_t address = uint64_t(base_) + uint64_t(id / emu::kBanks) * aligned + offset;
        return (uint64_t(id % emu::kBanks) << 48) | address;
    }

private:
    uint32_t base_;
    uint32_t page_;
};

// ---- transfers -------------------------------------------------------------------------------------------------------------------------
inline void noc_async_read(uint64_t source, uint32_t destination_l1, uint32_t size) {
    using namespace emu;
    const uint32_t bank = static_cast<uint32_t>(source >> 48);
    const uint64_t address = source & ((uint64_t(1) << 48) - 1);
    if (size == 0) violation("zero-size read");
    if (bank >= kBanks) violation("read from a bank that does not exist", bank);
    if (address % 64 != 0) violation("read from a DRAM address that is not 64-byte aligned", address);
    if (destination_l1 % 64 != 0) violation("read into an L1 address that is not 64-byte aligned", destination_l1);
    if (!in_cb(destination_l1, size)) violation("read lands outside the circular buffer", destination_l1, size);
    if (bank != state.read_bank || address < state.read_lo || address + size > state.read_hi) {
        violation("read outside the allowed staging range", bank, address);
    }
    if (state.read_unit != 0 && size % state.read_unit != 0) violation("read is not a whole unit", size, state.read_unit);
    if (kEager) {
        perform_read({source, destination_l1, size});
    } else {
        state.reads.push_back({source, destination_l1, size});
    }
    ++state.reads_issued;
}

inline void noc_async_write(uint32_t source_l1, uint64_t destination, uint32_t size) {
    using namespace emu;
    const uint32_t bank = static_cast<uint32_t>(destination >> 48);
    const uint64_t address = destination & ((uint64_t(1) << 48) - 1);
    if (size == 0) violation("zero-size write");
    if (bank >= kBanks) violation("write to a bank that does not exist", bank);
    if (address % 64 != 0) violation("write to a DRAM address that is not 64-byte aligned", address);
    if (source_l1 % 16 != 0) violation("write from an L1 address that is not 16-byte aligned", source_l1);
    if (!in_cb(source_l1, size)) violation("write reads from outside the circular buffer", source_l1, size);
    if (address < state.write_lo || address + size > state.write_hi) violation("write outside the allowed range", bank, address);
    if (state.write_page != 0 && size != state.write_page) violation("a write that is not one whole page", size, state.write_page);
    if (kEager) {
        perform_write({destination, source_l1, size});
    } else {
        state.writes.push_back({destination, source_l1, size});
    }
    ++state.writes_issued;
}

inline void noc_async_read_barrier() {
    for (const emu::Pending& read : emu::state.reads) emu::perform_read(read);
    emu::state.reads.clear();
}

inline void noc_async_write_barrier() {
    for (const emu::Pending& write : emu::state.writes) emu::perform_write(write);
    emu::state.writes.clear();
}
