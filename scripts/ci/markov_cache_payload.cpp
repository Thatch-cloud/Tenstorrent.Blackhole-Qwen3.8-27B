#include "api/dataflow/dataflow_api.h"
#include "markov_cache_control.hpp"

void kernel_main() {
    constexpr auto state_args = TensorAccessorArgs<0>();
    constexpr auto decision_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<decision_args.next_compile_time_args_offset()>();
    constexpr auto cache_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    const uint32_t width = get_arg_val<uint32_t>(5);
    const auto state = TensorAccessor(state_args, get_arg_val<uint32_t>(0), 32);
    const auto decision = TensorAccessor(decision_args, get_arg_val<uint32_t>(1), 32);
    const auto bias = TensorAccessor(bias_args, get_arg_val<uint32_t>(2), 4096);
    const auto cache = TensorAccessor(cache_args, get_arg_val<uint32_t>(3), width * 4);
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(4), 4096);
    const uint32_t scratch = (get_write_ptr(0) + 127) & ~127u;
    auto read_record = [&](uint64_t source, uint32_t destination) {
        const uint32_t staging = scratch + 2048 + static_cast<uint32_t>(source & 127);
        noc_async_read(source, staging, 32);
        noc_async_read_barrier();
        auto input = reinterpret_cast<volatile uint32_t*>(staging);
        auto result = reinterpret_cast<uint32_t*>(destination);
        for (uint32_t word = 0; word < 8; ++word) {
            result[word] = input[word];
        }
    };
    read_record(get_noc_addr(0, decision), scratch);
    read_record(get_noc_addr(0, state), scratch + 128);
    const auto& ticket = *reinterpret_cast<const markov_cache::Decision*>(scratch);
    const auto header = reinterpret_cast<const uint32_t*>(scratch + 128);
    bool valid = ticket.ok == 1 && ticket.hit <= 1 && ticket.slot < 64 && ticket.token < 248320 &&
        ticket.epoch != 0 && ticket.ticket != 0 && ticket.epoch == header[0] && ticket.ticket == header[1];
    if (valid) {
        read_record(get_noc_addr(ticket.slot + 1, state), scratch + 256);
        const auto& entry = *reinterpret_cast<const markov_cache::Slot*>(scratch + 256);
        valid = entry.valid == ticket.hit && entry.token == ticket.token &&
            entry.epoch == ticket.epoch && entry.ticket == ticket.ticket;
    }
    const uint32_t payload = scratch + 512;
    for (uint32_t tile = get_arg_val<uint32_t>(6); tile < width / 32; tile += get_arg_val<uint32_t>(7)) {
        if (valid) {
            if (ticket.hit) {
                noc_async_read(cache.get_noc_addr(ticket.slot, tile * 128), payload, 128);
            } else {
                noc_async_read(bias.get_noc_addr(tile, 0), payload, 64);
                noc_async_read(bias.get_noc_addr(tile, 1024), payload + 128, 64);
            }
            noc_async_read_barrier();
            if (!ticket.hit) {
                auto packed = reinterpret_cast<volatile uint32_t*>(payload + 64);
                auto face = reinterpret_cast<volatile uint32_t*>(payload + 128);
                for (uint32_t word = 0; word < 16; ++word) {
                    packed[word] = face[word];
                }
                noc_async_write(payload, cache.get_noc_addr(ticket.slot, tile * 128), 128);
            }
        } else {
            auto poison = reinterpret_cast<volatile uint32_t*>(payload);
            for (uint32_t word = 0; word < 32; ++word) {
                poison[word] = 0x7fc00000;
            }
        }
        auto packed = reinterpret_cast<volatile uint32_t*>(payload + 64);
        auto face = reinterpret_cast<volatile uint32_t*>(payload + 128);
        for (uint32_t word = 0; word < 16; ++word) {
            face[word] = packed[word];
        }
        noc_async_write(payload, output.get_noc_addr(tile, 0), 64);
        noc_async_write(payload + 128, output.get_noc_addr(tile, 1024), 64);
        noc_async_write_barrier();
    }
}
