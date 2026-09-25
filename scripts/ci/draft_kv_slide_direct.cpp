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
    const uint32_t scratch = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t output = get_write_ptr(0) + 6144;
    for (uint32_t tile = 0; tile < 64; ++tile) {
        uint32_t row = 0;
        while (row < 32) {
            const uint32_t destination_row = tile * 32 + row;
            const uint32_t logical_source = destination_row + drop;
            if (destination_row >= rows) {
                auto words = reinterpret_cast<volatile uint32_t*>(output);
                for (; row < 32; ++row) {
                    for (uint32_t face = 0; face < 2; ++face) {
                        const uint32_t offset = (row / 16) * 256 + face * 128 + (row % 16) * 8;
                        for (uint32_t word = 0; word < 8; ++word) {
                            words[offset + word] = 0;
                        }
                    }
                }
                break;
            }
            const bool historical = logical_source < history_rows;
            const uint32_t source_row = historical ? logical_source : logical_source - history_rows;
            const uint32_t remaining = historical ? history_rows - source_row : prefix - source_row;
            uint32_t count = 16 - row % 16;
            if (count > 16 - source_row % 16) { count = 16 - source_row % 16; }
            if (count > remaining) { count = remaining; }
            if (count > rows - destination_row) { count = rows - destination_row; }
            const uint32_t page = historical ? (head * 64 + source_row / 32) * 4 + column : head * 4 + column;
            const uint64_t source = historical ? get_noc_addr(page, active) : get_noc_addr(page, delta);
            const uint32_t source_start = ((source_row % 32) / 16) * 1024 + (source_row % 16) * 32;
            const uint32_t destination_start = (row / 16) * 1024 + (row % 16) * 32;
            const uint32_t source_alignment = static_cast<uint32_t>(source + source_start) & 63;
            const bool staged = source_alignment != ((output + destination_start) & 63);
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t destination = staged ? scratch + face * 1024 + source_alignment
                    : output + destination_start + face * 512;
                noc_async_read(source + source_start + face * 512, destination, count * 32);
            }
            if (staged) {
                noc_async_read_barrier();
                for (uint32_t face = 0; face < 2; ++face) {
                    noc_async_read(get_noc_addr(scratch + face * 1024 + source_alignment),
                        output + destination_start + face * 512, count * 32);
                }
                noc_async_read_barrier();
            }
            row += count;
        }
        noc_async_read_barrier();
        asm volatile("" ::: "memory");
        noc_async_write_tile((head * 64 + tile) * 4 + column, spare, output);
        noc_async_write_barrier();
    }
}
