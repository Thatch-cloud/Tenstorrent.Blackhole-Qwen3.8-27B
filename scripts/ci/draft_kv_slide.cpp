#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto active_args = TensorAccessorArgs<0>();
    constexpr auto delta_args = TensorAccessorArgs<active_args.next_compile_time_args_offset()>();
    constexpr auto spare_args = TensorAccessorArgs<delta_args.next_compile_time_args_offset()>();
    const auto active = TensorAccessor(active_args, get_arg_val<uint32_t>(0), 2048);
    const auto delta = TensorAccessor(delta_args, get_arg_val<uint32_t>(1), 2048);
    const auto spare = TensorAccessor(spare_args, get_arg_val<uint32_t>(2), 2048);
    const uint32_t history_rows = get_arg_val<uint32_t>(3);
    const uint32_t prefix = get_arg_val<uint32_t>(4);
    const uint32_t drop = get_arg_val<uint32_t>(5);
    const uint32_t rows = get_arg_val<uint32_t>(6);
    const uint32_t worker = get_arg_val<uint32_t>(7);
    const uint32_t head = worker / 4;
    const uint32_t column = worker % 4;
    const uint32_t scratch = get_write_ptr(0);
    const uint32_t added = scratch + 4096;
    const uint32_t output = scratch + 6144;
    noc_async_read_tile(head * 4 + column, delta, added);
    noc_async_read_barrier();
    for (uint32_t tile = 0; tile < 64; ++tile) {
        const uint32_t source_start = tile * 32 + drop;
        const uint32_t source_tile = source_start / 32;
        if (source_start < history_rows) {
            noc_async_read_tile((head * 64 + source_tile) * 4 + column, active, scratch);
            if (source_start % 32 != 0 && (source_tile + 1) * 32 < history_rows) {
                noc_async_read_tile((head * 64 + source_tile + 1) * 4 + column, active, scratch + 2048);
            }
        }
        noc_async_read_barrier();
        auto output_words = reinterpret_cast<volatile uint32_t*>(output);
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t destination_row = tile * 32 + row;
            const uint32_t source_row = destination_row + drop;
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t destination = (row / 16) * 256 + face * 128 + (row % 16) * 8;
                uint32_t source_address = 0;
                uint32_t local_row = 0;
                if (destination_row < rows && source_row < history_rows) {
                    source_address = scratch + (source_row / 32 - source_tile) * 2048;
                    local_row = source_row % 32;
                } else if (destination_row < rows && source_row - history_rows < prefix) {
                    source_address = added;
                    local_row = source_row - history_rows;
                }
                const uint32_t source_offset = (local_row / 16) * 256 + face * 128 + (local_row % 16) * 8;
                for (uint32_t word = 0; word < 8; ++word) {
                    output_words[destination + word] = source_address == 0 ? 0 :
                        reinterpret_cast<volatile const uint32_t*>(source_address)[source_offset + word];
                }
            }
        }
        asm volatile("" ::: "memory");
        noc_async_write_tile((head * 64 + tile) * 4 + column, spare, output);
        noc_async_write_barrier();
    }
}
