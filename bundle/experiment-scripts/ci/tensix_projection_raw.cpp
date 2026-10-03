#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr auto source_args = TensorAccessorArgs<1>();
    constexpr auto destination_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(0), tile_bytes);
    const auto destination = TensorAccessor(destination_args, get_arg_val<uint32_t>(1), tile_bytes);
    const uint32_t pages = get_arg_val<uint32_t>(2);
    const uint32_t worker = get_arg_val<uint32_t>(3);
    for (uint32_t first_page = worker * 8; first_page < pages; first_page += 64) {
        cb_reserve_back(0, 8);
        const uint32_t scratch = get_write_ptr(0);
        for (uint32_t offset = 0; offset < 8 && first_page + offset < pages; offset++) {
            noc_async_read_tile(first_page + offset, source, scratch + offset * tile_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t offset = 0; offset < 8 && first_page + offset < pages; offset++) {
            noc_async_write_tile(first_page + offset, destination, scratch + offset * tile_bytes);
        }
        noc_async_write_barrier();
        cb_push_back(0, 8);
        cb_wait_front(0, 8);
        cb_pop_front(0, 8);
    }
}
