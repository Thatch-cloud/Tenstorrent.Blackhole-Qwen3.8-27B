#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto source_args = TensorAccessorArgs<0>();
    constexpr auto destination_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(0), 2048);
    const auto destination = TensorAccessor(destination_args, get_arg_val<uint32_t>(1), 2048);
    const uint32_t rows = get_arg_val<uint32_t>(2);
    const bool inverse = get_arg_val<uint32_t>(3) != 0;
    const uint32_t offset = get_arg_val<uint32_t>(4);
    const uint32_t task = get_arg_val<uint32_t>(5);
    const uint32_t column = task % 8;
    const uint32_t staged = get_write_ptr(0);
    const uint32_t output = staged + 8192;
    const uint32_t source_tiles = inverse ? (rows * 12 + 31) / 32 : rows;
    for (uint32_t tile = 0; tile < source_tiles; tile++) {
        const uint32_t page = (tile + (inverse ? 0 : offset)) * 8 + column;
        noc_async_read_tile(page, source, staged + tile * 2048);
    }
    noc_async_read_barrier();
    auto words = reinterpret_cast<volatile uint32_t*>(output);
    for (uint32_t word = 0; word < 512; word++) { words[word] = 0; }
    for (uint32_t target_row = 0; target_row < 32; target_row++) {
        uint32_t source_tile;
        uint32_t source_head;
        if (inverse) {
            if (target_row >= 12) { continue; }
            const uint32_t token = task / 8;
            const uint32_t head = (target_row / 6) * rows * 6 + token * 6 + target_row % 6;
            source_tile = head / 32;
            source_head = head % 32;
        } else {
            const uint32_t head = (task / 8) * 32 + target_row;
            if (head >= rows * 12) { continue; }
            const uint32_t remainder = head % (rows * 6);
            source_tile = remainder / 6;
            source_head = (head / (rows * 6)) * 6 + remainder % 6;
        }
        const uint32_t source_offset = ((source_head / 16) * 512 + (source_head % 16) * 16) * 2;
        const uint32_t target_offset = ((target_row / 16) * 512 + (target_row % 16) * 16) * 2;
        for (uint32_t face = 0; face < 2; face++) {
            const auto input = reinterpret_cast<volatile const uint32_t*>(staged + source_tile * 2048 + source_offset + face * 512);
            auto target = reinterpret_cast<volatile uint32_t*>(output + target_offset + face * 512);
            for (uint32_t word = 0; word < 8; word++) { target[word] = input[word]; }
        }
    }
    asm volatile("" ::: "memory");
    noc_async_write_tile(task, destination, output);
    noc_async_write_barrier();
}
