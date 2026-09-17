#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"

void kernel_main() {
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t block_rows = get_compile_time_arg_val(1);
    constexpr uint32_t receiver_columns = get_compile_time_arg_val(2);
    const uint32_t blocks = get_arg_val<uint32_t>(0);
    const uint32_t receivers = get_arg_val<uint32_t>(1);
    const uint32_t block_tiles = block_rows * receivers * receiver_columns;
    experimental::resize_remote_sender_cb_interface<true>(31, block_rows * receiver_columns * tile_bytes, noc_index);
    for (uint32_t block = 0; block < blocks; block++) {
        cb_wait_front(0, block_tiles);
        experimental::remote_cb_reserve_back(31, 1);
        experimental::remote_cb_push_back_and_write_pages<false>(
            31, get_read_ptr(0), 1, block_rows, receiver_columns, tile_bytes, noc_index);
        noc_async_writes_flushed();
        cb_pop_front(0, block_tiles);
    }
    experimental::remote_cb_sender_barrier(31);
    experimental::update_remote_cb_config_in_l1(31);
    noc_async_atomic_barrier();
    cb_reserve_back(3, 1);
    cb_push_back(3, 1);
}
