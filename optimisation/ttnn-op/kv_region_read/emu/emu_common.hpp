// Shared by emu_scatter.cpp and emu_checksum.cpp: reading the scenario, the host L1 region, and the files the test exchanges.
#pragma once

#include <sys/mman.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace emu_run {

// "key v1 v2 ..." per line.
inline std::map<std::string, std::vector<uint64_t>> read_config(const char* path) {
    std::map<std::string, std::vector<uint64_t>> config;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream words(line);
        std::string key;
        if (!(words >> key)) continue;
        uint64_t value = 0;
        while (words >> value) config[key].push_back(value);
        config[key];
    }
    return config;
}

inline uint64_t one(const std::map<std::string, std::vector<uint64_t>>& config, const char* key) {
    auto it = config.find(key);
    if (it == config.end() || it->second.size() != 1) {
        std::printf("EMU-VIOLATION: scenario key %s missing\n", key);
        std::exit(2);
    }
    return it->second[0];
}

inline std::vector<uint8_t> read_file(const char* path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

inline void write_file(const char* path, const std::vector<uint8_t>& data) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

// One host region in the low 4 GiB, so a 32-bit "L1 address" is a real pointer (MAP_32BIT exists on x86-64 only: elsewhere a hint address is tried).
inline uint32_t l1_region() {
    const size_t size = 2u << 20;
    for (uintptr_t hint : {uintptr_t(0), uintptr_t(0x20000000), uintptr_t(0x40000000), uintptr_t(0x60000000), uintptr_t(0x80000000)}) {
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_32BIT
        if (hint == 0) flags |= MAP_32BIT;
#endif
        void* region = mmap(reinterpret_cast<void*>(hint), size, PROT_READ | PROT_WRITE, flags, -1, 0);
        if (region == MAP_FAILED) continue;
        if (reinterpret_cast<uintptr_t>(region) + size <= (uintptr_t(1) << 32)) {
            return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(region));
        }
        munmap(region, size);
    }
    std::printf("EMU-VIOLATION: no mapping in the low 4 GiB for L1\n");
    std::exit(2);
}

}  // namespace emu_run
