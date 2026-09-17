#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"

void kernel_main() {
    constexpr uint32_t block_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t receiver_columns = get_compile_time_arg_val(1);
    constexpr auto output_args = TensorAccessorArgs<2>();
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(0), 2048);
    const uint32_t blocks = get_arg_val<uint32_t>(1);
    const uint32_t receiver = get_arg_val<uint32_t>(2);
    const uint32_t fifo_tiles = get_local_cb_interface(1).fifo_num_pages;
    for (uint32_t block = 0; block < blocks; block++) {
        cb_reserve_back(1, block_tiles);
        experimental::remote_cb_wait_front(31, block == 0 ? 1u : 2u);
        cb_push_back(1, block_tiles);
        if (block >= 1) {
            while (!cb_pages_reservable_at_back(1, fifo_tiles - block_tiles)) {
                invalidate_l1_cache();
            }
            experimental::remote_cb_pop_front(31, 1);
        }
    }
    while (!cb_pages_reservable_at_back(1, fifo_tiles)) {
        invalidate_l1_cache();
    }
    experimental::remote_cb_pop_front(31, 1);
    cb_wait_front(4, receiver_columns);
    uint32_t source = get_read_ptr(4);
    for (uint32_t column = 0; column < receiver_columns; column++) {
        noc_async_write_tile(receiver * receiver_columns + column, output, source);
        source += 2048;
    }
    noc_async_write_barrier();
    cb_pop_front(4, receiver_columns);
    experimental::update_remote_cb_config_in_l1(31);
    noc_async_atomic_barrier();
}
