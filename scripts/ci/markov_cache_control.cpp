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
    for (uint32_t page = 0; page <= markov_cache::slots; ++page) {
        noc_async_read(get_noc_addr(page, state_access), scratch + page * 32, 32);
    }
    noc_async_read(get_noc_addr(0, request_access), request_address, 32);
    noc_async_read(get_noc_addr(0, decision_access), decision_address, 32);
    noc_async_read_barrier();
    asm volatile("" ::: "memory");
    auto& state = *reinterpret_cast<markov_cache::State*>(scratch);
    auto& decision = *reinterpret_cast<markov_cache::Decision*>(decision_address);
    const auto request = reinterpret_cast<const uint32_t*>(request_address);
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
        noc_async_write(scratch + page * 32, get_noc_addr(page, state_access), 32);
    }
    noc_async_write(decision_address, get_noc_addr(0, decision_access), 32);
    noc_async_write(status_address, get_noc_addr(0, status_access), 32);
    noc_async_write_barrier();
}
