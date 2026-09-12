#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto source_args = TensorAccessorArgs<0>();
    constexpr auto destination_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(2));
    const auto destination = TensorAccessor(destination_args, get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(3));
    const uint32_t pages = get_arg_val<uint32_t>(4);
    const uint32_t worker = get_arg_val<uint32_t>(5);
    const uint32_t staged = get_write_ptr(0);
    auto words = reinterpret_cast<volatile uint32_t*>(staged);
    for (uint32_t word = 0; word < 512; word++) {
        words[word] = 0;
    }
    for (uint32_t page = worker; page < pages; page += 44) {
        noc_async_read(get_noc_addr(page, source), staged, 1024);
        noc_async_read_barrier();
        for (uint32_t face = 0; face < 2; face++) {
            for (uint32_t word = 64; word < 128; word++) {
                words[face * 128 + word] = 0;
            }
        }
        asm volatile("" ::: "memory");
        noc_async_write_tile(page, destination, staged);
        noc_async_write_barrier();
    }
}
