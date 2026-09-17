#include "api/dataflow/dataflow_api.h"

template <uint32_t destination_offset>
void window(uint32_t slot, uint32_t page, uint32_t rows, uint32_t source_tiles) {
    constexpr auto destination_args = TensorAccessorArgs<destination_offset>();
    const auto destination = TensorAccessor(destination_args, get_arg_val<uint32_t>(5 + slot), 2048);
    const uint32_t scratch = source_tiles + 5 * 2048;
    {
        auto words = reinterpret_cast<volatile uint32_t*>(scratch);
        for (uint32_t word = 0; word < 512; word++) { words[word] = 0; }
        asm volatile("" ::: "memory");
        for (uint32_t token = 0; token < rows; token++) {
            const uint32_t history = token + slot;
            const uint32_t source_row = history < 4 ? 0 : history - 4;
            const uint32_t source_offset = ((source_row / 16) * 512 + (source_row % 16) * 16) * 2;
            const uint32_t destination_row = ((token / 16) * 512 + (token % 16) * 16) * 2;
            for (uint32_t face = 0; face < 2; face++) {
                const uint32_t tile = history < 4 ? history + 1 : 0;
                const auto source = reinterpret_cast<volatile const uint32_t*>(source_tiles + tile * 2048 + source_offset + face * 512);
                auto target = reinterpret_cast<volatile uint32_t*>(scratch + destination_row + face * 512);
                for (uint32_t word = 0; word < 8; word++) { target[word] = source[word]; }
            }
        }
        asm volatile("" ::: "memory");
        noc_async_write_tile(page, destination, scratch);
        noc_async_write_barrier();
    }
}

void kernel_main() {
    constexpr auto projected_args = TensorAccessorArgs<0>();
    constexpr auto first_args = TensorAccessorArgs<projected_args.next_compile_time_args_offset()>();
    constexpr auto second_args = TensorAccessorArgs<first_args.next_compile_time_args_offset()>();
    constexpr auto third_args = TensorAccessorArgs<second_args.next_compile_time_args_offset()>();
    constexpr auto fourth_args = TensorAccessorArgs<third_args.next_compile_time_args_offset()>();
    constexpr auto output0_args = TensorAccessorArgs<fourth_args.next_compile_time_args_offset()>();
    constexpr auto output1_args = TensorAccessorArgs<output0_args.next_compile_time_args_offset()>();
    constexpr auto output2_args = TensorAccessorArgs<output1_args.next_compile_time_args_offset()>();
    const auto projected = TensorAccessor(projected_args, get_arg_val<uint32_t>(0), 2048);
    const auto first = TensorAccessor(first_args, get_arg_val<uint32_t>(1), 2048);
    const auto second = TensorAccessor(second_args, get_arg_val<uint32_t>(2), 2048);
    const auto third = TensorAccessor(third_args, get_arg_val<uint32_t>(3), 2048);
    const auto fourth = TensorAccessor(fourth_args, get_arg_val<uint32_t>(4), 2048);
    const uint32_t rows = get_arg_val<uint32_t>(9);
    const uint32_t worker = get_arg_val<uint32_t>(10);
    const uint32_t scratch = get_write_ptr(0);
    for (uint32_t page = worker; page < 160; page += 48) {
        noc_async_read_tile(page, projected, scratch);
        noc_async_read_tile(page, first, scratch + 2048);
        noc_async_read_tile(page, second, scratch + 2 * 2048);
        noc_async_read_tile(page, third, scratch + 3 * 2048);
        noc_async_read_tile(page, fourth, scratch + 4 * 2048);
        noc_async_read_barrier();
        window<fourth_args.next_compile_time_args_offset()>(0, page, rows, scratch);
        window<output0_args.next_compile_time_args_offset()>(1, page, rows, scratch);
        window<output1_args.next_compile_time_args_offset()>(2, page, rows, scratch);
        window<output2_args.next_compile_time_args_offset()>(3, page, rows, scratch);
    }
}
