// The simulation's translation unit: the stub's globals, the kernel under test (included as a whole, with the compile-time args the test passes as -DSIM_CT_ARGS=...) and a C entry point.
#include "api/dataflow/dataflow_api.h"

namespace sim {
uint8_t* dram = nullptr;
uint64_t dram_bytes = 0;
uint32_t* runtime_args = nullptr;
uint32_t runtime_count = 0;
int errors = 0;
Transfer queue[4096];
int queued = 0;
}  // namespace sim

#include SIM_KERNEL

#include <cerrno>
#include <sys/mman.h>

extern "C" int sim_run(uint8_t* dram, uint64_t dram_bytes, uint32_t* args, uint32_t count, uint32_t junk) {
    // L1 lives at one fixed low address for the whole process: every compiled kernel library maps it, and EEXIST means another library (or an earlier run) already did.
    static bool mapped = false;
    if (!mapped) {
        void* l1 = mmap(reinterpret_cast<void*>(uintptr_t(sim::L1_BASE)), sim::L1_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (l1 != reinterpret_cast<void*>(uintptr_t(sim::L1_BASE)) && errno != EEXIST) { std::fprintf(stderr, "SIM ERROR: cannot map L1\n"); return -1; }
        mapped = true;
    }
    void* l1 = reinterpret_cast<void*>(uintptr_t(sim::L1_BASE));
    uint32_t* words = static_cast<uint32_t*>(l1);
    for (uint32_t i = 0; i < sim::L1_BYTES / 4; ++i) { words[i] = junk * 2654435761u + i * 40503u; }
    sim::dram = dram;
    sim::dram_bytes = dram_bytes;
    sim::runtime_args = args;
    sim::runtime_count = count;
    sim::errors = 0;
    sim::queued = 0;
    kernel_main();
    if (sim::queued != 0) { std::fprintf(stderr, "SIM ERROR: %d transfers never reached a barrier\n", sim::queued); ++sim::errors; }
    return sim::errors;
}
