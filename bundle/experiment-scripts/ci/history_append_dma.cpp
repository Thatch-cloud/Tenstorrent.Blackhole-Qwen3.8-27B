#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto active_args = TensorAccessorArgs<0>();
    constexpr auto delta_args = TensorAccessorArgs<active_args.next_compile_time_args_offset()>();
    constexpr auto spare_args = TensorAccessorArgs<delta_args.next_compile_time_args_offset()>();
    const auto active = TensorAccessor(active_args, get_arg_val<uint32_t>(0), 2048);
    const auto delta = TensorAccessor(delta_args, get_arg_val<uint32_t>(1), 2048);
    const auto spare = TensorAccessor(spare_args, get_arg_val<uint32_t>(2), 2048);
    const uint32_t capacity_tiles = get_arg_val<uint32_t>(3);
    const uint32_t position = get_arg_val<uint32_t>(4);
    const uint32_t prefix = get_arg_val<uint32_t>(5);
    const uint32_t first_tile = get_arg_val<uint32_t>(6);
    const uint32_t end_tile = get_arg_val<uint32_t>(7);
    const uint32_t worker = get_arg_val<uint32_t>(8);
    const uint32_t head = worker / 4;
    const uint32_t column = worker % 4;
    const uint32_t staged = get_write_ptr(0);
    const uint32_t added = staged + 2048;
    const uint32_t output = staged + 4096;
    noc_async_read_tile(head * 4 + column, delta, added);
    noc_async_read_barrier();
    const auto old_words = reinterpret_cast<volatile const uint32_t*>(staged);
    const auto new_words = reinterpret_cast<volatile const uint32_t*>(added);
    auto output_words = reinterpret_cast<volatile uint32_t*>(output);
    for (uint32_t tile = first_tile; tile < end_tile; ++tile) {
        const uint32_t page = (head * capacity_tiles + tile) * 4 + column;
        noc_async_read_tile(page, active, staged);
        noc_async_read_barrier();
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t absolute_row = tile * 32 + row;
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t destination = (row / 16) * 256 + face * 128 + (row % 16) * 8;
                for (uint32_t word = 0; word < 8; ++word) {
                    uint32_t bits = 0;
                    if (absolute_row < position) {
                        bits = old_words[destination + word];
                    } else if (absolute_row < position + prefix) {
                        const uint32_t added_row = absolute_row - position;
                        const uint32_t source = (added_row / 16) * 256 + face * 128 + (added_row % 16) * 8;
                        bits = new_words[source + word];
                    }
                    output_words[destination + word] = bits;
                }
            }
        }
        asm volatile("" ::: "memory");
        noc_async_write_tile(page, spare, output);
        noc_async_write_barrier();
    }
}
