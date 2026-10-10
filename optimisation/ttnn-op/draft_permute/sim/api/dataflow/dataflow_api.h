// A host simulation of the dataflow API surface draft_permute_tp.cpp uses, so the REAL kernel source runs on the CPU in test_draft_permute_kernel_sim.py (no tt-metal tree needed).
//
// L1 is a fixed low mapping (the kernel casts a 32-bit L1 address to a pointer); DRAM is a host buffer addressed by the buffer base addresses the runtime args carry; interleaved paging is
// ignored (page p of a buffer is at base + 2048 p). NoC reads and writes are QUEUED and executed at their barrier: a kernel that touches L1 before the read barrier sees junk, one that
// changes a written tile after issuing the write changes what lands, and the transfers obey the alignment the hardware asks (DRAM side 32 B, L1 side 16 B, sizes a multiple of 16 B).
// Evidence, not proof: the stubs are declarations written from the call sites; the card-M probe is the proof.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define FORCE_INLINE inline __attribute__((always_inline))

namespace sim {
constexpr uint32_t L1_BASE = 0x20000000u;
constexpr uint32_t CB_STRIDE = 0x20000u;
constexpr uint32_t L1_BYTES = 0x100000u;
extern uint8_t* dram;
extern uint64_t dram_bytes;
extern uint32_t* runtime_args;
extern uint32_t runtime_count;
extern int errors;

struct Transfer {
    bool read;
    uint64_t noc;
    uint32_t l1;
    uint32_t size;
};
extern Transfer queue[4096];
extern int queued;

inline void fail(const char* what) {
    std::fprintf(stderr, "SIM ERROR: %s\n", what);
    ++errors;
}

inline void enqueue(bool read, uint64_t noc, uint32_t l1, uint32_t size) {
    if (queued >= 4096) { fail("transfer queue overflow"); return; }
    if ((noc & 31u) != 0 || (l1 & 15u) != 0 || (size & 15u) != 0) { fail("misaligned NoC transfer"); }
    if (l1 < L1_BASE || uint64_t(l1) + size > uint64_t(L1_BASE) + L1_BYTES) { fail("L1 address out of range"); return; }
    if (noc + size > dram_bytes) { fail("DRAM address out of range"); return; }
    queue[queued++] = Transfer{read, noc, l1, size};
}

inline void drain(bool read) {
    int kept = 0;
    for (int i = 0; i < queued; ++i) {
        const Transfer& t = queue[i];
        if (t.read != read) { queue[kept++] = t; continue; }
        if (read) { std::memcpy(reinterpret_cast<void*>(uintptr_t(t.l1)), dram + t.noc, t.size); }
        else { std::memcpy(dram + t.noc, reinterpret_cast<void*>(uintptr_t(t.l1)), t.size); }
    }
    queued = kept;
}
}  // namespace sim

template <uint32_t Base>
struct TensorAccessorArgs {
    static constexpr uint32_t next_compile_time_args_offset() { return Base + 1; }
};

struct TensorAccessor {
    uint32_t base;
    uint32_t page_size;
    template <typename Args>
    TensorAccessor(const Args&, uint32_t address, uint32_t size) : base(address), page_size(size) {}
    uint64_t get_noc_addr(uint32_t page, uint32_t offset = 0) const { return uint64_t(base) + uint64_t(page) * page_size + offset; }
};

template <typename T>
inline T get_arg_val(uint32_t index) {
    if (index >= sim::runtime_count) { sim::fail("runtime arg read past the list"); return T(0); }
    return static_cast<T>(sim::runtime_args[index]);
}

constexpr uint32_t sim_compile_time_args[] = {SIM_CT_ARGS};
constexpr uint32_t get_compile_time_arg_val(uint32_t index) { return sim_compile_time_args[index]; }

inline uint32_t get_write_ptr(uint32_t cb) { return sim::L1_BASE + cb * sim::CB_STRIDE; }

inline void noc_async_read(uint64_t noc, uint32_t l1, uint32_t size) { sim::enqueue(true, noc, l1, size); }
inline void noc_async_read_tile(uint32_t page, const TensorAccessor& accessor, uint32_t l1) { sim::enqueue(true, accessor.get_noc_addr(page), l1, accessor.page_size); }
inline void noc_async_write_tile(uint32_t page, const TensorAccessor& accessor, uint32_t l1) { sim::enqueue(false, accessor.get_noc_addr(page), l1, accessor.page_size); }
inline void noc_async_read_barrier() { sim::drain(true); }
inline void noc_async_write_barrier() { sim::drain(false); }
