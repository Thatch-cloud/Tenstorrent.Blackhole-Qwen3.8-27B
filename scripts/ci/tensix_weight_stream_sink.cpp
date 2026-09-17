#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"

void kernel_main() {
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t block_rows = get_compile_time_arg_val(1);
    constexpr uint32_t receiver_columns = get_compile_time_arg_val(2);
    constexpr uint32_t total_columns = get_compile_time_arg_val(3);
    constexpr auto destination_args = TensorAccessorArgs<4>();
    const auto destination = TensorAccessor(destination_args, get_arg_val<uint32_t>(0), tile_bytes);
    const uint32_t blocks = get_arg_val<uint32_t>(1);
    const uint32_t receiver_index = get_arg_val<uint32_t>(2);
    experimental::resize_remote_receiver_cb_interface<true>(31, block_rows * receiver_columns * tile_bytes, noc_index);
    for (uint32_t block = 0; block < blocks; block++) {
        experimental::remote_cb_wait_front(31, 1);
        uint32_t source = get_remote_receiver_cb_interface(31).fifo_rd_ptr;
        for (uint32_t row = 0; row < block_rows; row++) {
            for (uint32_t column = 0; column < receiver_columns; column++) {
                const uint32_t page = (block * block_rows + row) * total_columns + receiver_index * receiver_columns + column;
                noc_async_write_tile(page, destination, source);
                source += tile_bytes;
            }
        }
        noc_async_write_barrier();
        experimental::remote_cb_pop_front(31, 1);
    }
    experimental::update_remote_cb_config_in_l1(31);
    noc_async_atomic_barrier();
}
