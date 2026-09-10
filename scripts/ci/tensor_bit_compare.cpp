#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto left_args = TensorAccessorArgs<0>();
    constexpr auto right_args = TensorAccessorArgs<left_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<right_args.next_compile_time_args_offset()>();
    const auto left = TensorAccessor(left_args, get_arg_val<uint32_t>(0), 2048);
    const auto right = TensorAccessor(right_args, get_arg_val<uint32_t>(1), 2048);
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(2), 128);
    const uint32_t tiles = get_arg_val<uint32_t>(3);
    const uint32_t worker = get_arg_val<uint32_t>(4);
    const uint32_t scratch = get_write_ptr(0);
    uint32_t mismatches = 0;
    for (uint32_t page = worker; page < tiles; page += 32) {
        noc_async_read_tile(page, left, scratch);
        noc_async_read_tile(page, right, scratch + 2048);
        noc_async_read_barrier();
        const auto left_words = reinterpret_cast<volatile const uint32_t*>(scratch);
        const auto right_words = reinterpret_cast<volatile const uint32_t*>(scratch + 2048);
        for (uint32_t word = 0; word < 512; ++word) {
            mismatches += left_words[word] != right_words[word];
        }
    }
    auto result = reinterpret_cast<volatile uint32_t*>(scratch);
    result[0] = mismatches;
    asm volatile("" ::: "memory");
    noc_async_write(scratch, output.get_noc_addr(0, worker * 4), 4);
    noc_async_write_barrier();
}
