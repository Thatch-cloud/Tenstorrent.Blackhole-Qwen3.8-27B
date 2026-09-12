#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto position_args = TensorAccessorArgs<0>();
    constexpr auto mask_args = TensorAccessorArgs<position_args.next_compile_time_args_offset()>();
    const auto positions = TensorAccessor(position_args, get_arg_val<uint32_t>(0), 32);
    const auto mask = TensorAccessor(mask_args, get_arg_val<uint32_t>(1), 2048);
    const uint32_t rows = get_arg_val<uint32_t>(2);
    const uint32_t capacity = get_arg_val<uint32_t>(3);
    const uint32_t offset = get_arg_val<uint32_t>(4);
    const uint32_t task = get_arg_val<uint32_t>(5);
    const uint32_t staged = get_write_ptr(0);
    noc_async_read(get_noc_addr(0, positions), staged, 32);
    noc_async_read_barrier();
    const uint32_t start = *reinterpret_cast<volatile uint32_t*>(staged);
    const uint32_t output = staged + 2048;
    auto words = reinterpret_cast<volatile uint16_t*>(output);
    const uint32_t head_tiles = (rows * 12 + 31) / 32;
    const uint32_t batch = task / (head_tiles * 8);
    const uint32_t head_tile = (task / 8) % head_tiles;
    const uint32_t column_tile = task % 8;
    for (uint32_t row = 0; row < 32; row++) {
        const uint32_t head = head_tile * 32 + row;
        const uint32_t position = start + offset + batch * rows + (head % (rows * 6)) / 6;
        for (uint32_t column = 0; column < 32; column++) {
            const uint32_t index = (row / 16) * 512 + (column / 16) * 256 + (row % 16) * 16 + column % 16;
            const uint32_t cache_position = capacity - 256 + column_tile * 32 + column;
            words[index] = head >= rows * 12 || cache_position > position ? 0xff80 : 0;
        }
    }
    asm volatile("" ::: "memory");
    const uint32_t page = (batch * head_tiles + head_tile) * (capacity / 32) + capacity / 32 - 8 + column_tile;
    noc_async_write_tile(page, mask, output);
    noc_async_write_barrier();
}

