#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto source_args = TensorAccessorArgs<0>();
    constexpr auto destination_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    const uint32_t token = get_arg_val<uint32_t>(8) - 1;
    const uint32_t worker = get_arg_val<uint32_t>(9);
    const uint32_t staged = get_write_ptr(0);
    const uint32_t output = staged + 2048;
    const uint32_t offset = ((token / 16) * 512 + (token % 16) * 16) * 2;
    for (uint32_t task = worker; task < 640; task += 48) {
        const uint32_t slot = task / 160;
        const uint32_t page = task % 160;
        const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(slot), 2048);
        const auto destination = TensorAccessor(destination_args, get_arg_val<uint32_t>(4 + slot), 2048);
        noc_async_read_tile(page, source, staged);
        noc_async_read_barrier();
        auto words = reinterpret_cast<volatile uint32_t*>(output);
        for (uint32_t word = 0; word < 512; word++) { words[word] = 0; }
        for (uint32_t face = 0; face < 2; face++) {
            const auto input = reinterpret_cast<volatile const uint32_t*>(staged + offset + face * 512);
            auto target = reinterpret_cast<volatile uint32_t*>(output + face * 512);
            for (uint32_t word = 0; word < 8; word++) { target[word] = input[word]; }
        }
        asm volatile("" ::: "memory");
        noc_async_write_tile(page, destination, output);
        noc_async_write_barrier();
    }
}
