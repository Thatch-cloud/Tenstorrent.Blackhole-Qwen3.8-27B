#include "api/dataflow/dataflow_api.h"

uint32_t lane(uint32_t row, uint32_t column) {
    return (row / 16) * 512 + (column / 16) * 256 + (row % 16) * 16 + column % 16;
}

void kernel_main() {
    constexpr auto hidden_args = TensorAccessorArgs<0>();
    constexpr auto dynamic0_args = TensorAccessorArgs<hidden_args.next_compile_time_args_offset()>();
    constexpr auto dynamic1_args = TensorAccessorArgs<dynamic0_args.next_compile_time_args_offset()>();
    constexpr auto base0_args = TensorAccessorArgs<dynamic1_args.next_compile_time_args_offset()>();
    constexpr auto base1_args = TensorAccessorArgs<base0_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<base1_args.next_compile_time_args_offset()>();
    const auto hidden = TensorAccessor(hidden_args, get_arg_val<uint32_t>(0), 2048);
    const auto dynamic0 = TensorAccessor(dynamic0_args, get_arg_val<uint32_t>(1), 2048);
    const auto dynamic1 = TensorAccessor(dynamic1_args, get_arg_val<uint32_t>(2), 2048);
    const auto base0 = TensorAccessor(base0_args, get_arg_val<uint32_t>(3), 2048);
    const auto base1 = TensorAccessor(base1_args, get_arg_val<uint32_t>(4), 2048);
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(5), 2048);
    const uint32_t rows = get_arg_val<uint32_t>(6);
    const uint32_t worker = get_arg_val<uint32_t>(7);
    const uint32_t scratch = get_write_ptr(1);
    for (uint32_t page = worker; page < 160; page += 80) {
        cb_reserve_back(0, 7);
        const uint32_t destination = get_write_ptr(0);
        noc_async_read_tile(page, hidden, destination);
        noc_async_read_tile(page, base0, destination + 2 * 2048);
        noc_async_read_tile(page, base1, destination + 4 * 2048);
        noc_async_read_tile(page / 16, dynamic0, scratch);
        noc_async_read_tile(page / 16, dynamic1, scratch + 2048);
        noc_async_read_barrier();
        auto tiles = reinterpret_cast<volatile uint16_t*>(destination);
        const auto coefficients = reinterpret_cast<volatile const uint16_t*>(scratch);
        for (uint32_t row = 0; row < 32; ++row) {
            for (uint32_t column = 0; column < 32; ++column) {
                const uint32_t element = lane(row, column);
                const uint32_t group = lane(row, 2 * (page % 16) + column / 16);
                tiles[1024 + element] = row && row < rows ? tiles[lane(row - 1, column)] : 0;
                tiles[2 * 1024 + element] = tiles[2 * 1024 + lane(0, column)];
                tiles[3 * 1024 + element] = row < rows ? coefficients[group] : 0;
                tiles[4 * 1024 + element] = tiles[4 * 1024 + lane(0, column)];
                tiles[5 * 1024 + element] = row < rows ? coefficients[1024 + group] : 0;
                tiles[6 * 1024 + element] = 0;
                if (row >= rows) {
                    tiles[element] = 0;
                }
            }
        }
        asm volatile("" ::: "memory");
        cb_push_back(0, 7);
        cb_wait_front(16, 1);
        noc_async_write_tile(page, output, get_read_ptr(16));
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
}
