#include "api/dataflow/dataflow_api.h"
#include "markov_cache_control.hpp"

void kernel_main() {
    constexpr auto state_args = TensorAccessorArgs<0>();
    constexpr auto request_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    constexpr auto decision_args = TensorAccessorArgs<request_args.next_compile_time_args_offset()>();
    constexpr auto status_args = TensorAccessorArgs<decision_args.next_compile_time_args_offset()>();
    const auto state_access = TensorAccessor(state_args, get_arg_val<uint32_t>(0), 32);
    const auto request_access = TensorAccessor(request_args, get_arg_val<uint32_t>(1), 32);
    const auto decision_access = TensorAccessor(decision_args, get_arg_val<uint32_t>(2), 32);
    const auto status_access = TensorAccessor(status_args, get_arg_val<uint32_t>(3), 32);
    const uint32_t scratch = get_write_ptr(0);
    const uint32_t request_address = scratch + sizeof(markov_cache::State);
    const uint32_t decision_address = request_address + 32;
    const uint32_t status_address = decision_address + 32;
    const uint32_t staging_base = (scratch + 3072 + 127) & ~127u;
    auto read_record = [&](uint64_t source, uint32_t destination) {
        const uint32_t staging = staging_base + static_cast<uint32_t>(source & 127);
        noc_async_read(source, staging, 32);
        noc_async_read_barrier();
        auto input = reinterpret_cast<volatile uint32_t*>(staging);
        auto output = reinterpret_cast<uint32_t*>(destination);
        for (uint32_t word = 0; word < 8; ++word) {
            output[word] = input[word];
        }
    };
    auto write_record = [&](uint32_t source, uint64_t destination) {
        const uint32_t staging = staging_base + static_cast<uint32_t>(destination & 127);
        auto input = reinterpret_cast<const uint32_t*>(source);
        auto output = reinterpret_cast<volatile uint32_t*>(staging);
        for (uint32_t word = 0; word < 8; ++word) {
            output[word] = input[word];
        }
        asm volatile("" ::: "memory");
        noc_async_write(staging, destination, 32);
        noc_async_write_barrier();
    };
    for (uint32_t page = 0; page <= markov_cache::slots; ++page) {
        read_record(get_noc_addr(page, state_access), scratch + page * 32);
    }
    read_record(get_noc_addr(0, request_access), request_address);
    read_record(get_noc_addr(0, decision_access), decision_address);
    asm volatile("" ::: "memory");
    auto& state = *reinterpret_cast<markov_cache::State*>(scratch);
    auto& decision = *reinterpret_cast<markov_cache::Decision*>(decision_address);
    auto request = reinterpret_cast<uint32_t*>(request_address);
#ifdef QWEN_CACHE_LIVE_ANCHOR
    if (request[0] == 0) {
        constexpr auto anchor_args = TensorAccessorArgs<status_args.next_compile_time_args_offset()>();
        const auto anchor = TensorAccessor(anchor_args, get_arg_val<uint32_t>(4), 4);
        const uint32_t anchor_address = scratch + 2304;
        read_record(get_noc_addr(0, anchor), anchor_address);
        request[1] = *reinterpret_cast<const uint32_t*>(anchor_address);
    }
#endif
    auto status = reinterpret_cast<uint32_t*>(status_address);
    for (uint32_t word = 0; word < 8; ++word) {
        status[word] = 0;
    }
    status[1] = request[0];
    if (request[0] == 0) {
        decision = markov_cache::lookup(state, request[1], request[2]);
        status[0] = decision.ok;
    } else if (request[0] == 1) {
        status[0] = markov_cache::commit(state, decision);
    }
    asm volatile("" ::: "memory");
    for (uint32_t page = 0; page <= markov_cache::slots; ++page) {
        write_record(scratch + page * 32, get_noc_addr(page, state_access));
    }
    write_record(decision_address, get_noc_addr(0, decision_access));
    write_record(status_address, get_noc_addr(0, status_access));
}
