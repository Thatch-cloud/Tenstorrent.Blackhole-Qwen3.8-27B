#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t block_rows = get_compile_time_arg_val(1);
    constexpr uint32_t receiver_columns = get_compile_time_arg_val(2);
    constexpr uint32_t total_columns = get_compile_time_arg_val(3);
    constexpr auto source_args = TensorAccessorArgs<4>();
    const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(0), tile_bytes);
    const uint32_t blocks = get_arg_val<uint32_t>(1);
    const uint32_t receivers = get_arg_val<uint32_t>(2);
    const uint32_t block_tiles = block_rows * receivers * receiver_columns;
    for (uint32_t block = 0; block < blocks; block++) {
        cb_reserve_back(0, block_tiles);
        uint32_t destination = get_write_ptr(0);
        for (uint32_t row = 0; row < block_rows; row++) {
            for (uint32_t receiver = 0; receiver < receivers; receiver++) {
                const uint32_t receiver_index = get_arg_val<uint32_t>(3 + receiver);
                for (uint32_t column = 0; column < receiver_columns; column++) {
                    const uint32_t page = (block * block_rows + row) * total_columns + receiver_index * receiver_columns + column;
                    noc_async_read_tile(page, source, destination);
                    destination += tile_bytes;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(0, block_tiles);
    }
    cb_wait_front(3, 1);
    cb_pop_front(3, 1);
}
