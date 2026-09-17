#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto gate_args = TensorAccessorArgs<0>();
    constexpr auto up_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<up_args.next_compile_time_args_offset()>();
    const auto gate = TensorAccessor(gate_args, get_arg_val<uint32_t>(0), 1024);
    const auto up = TensorAccessor(up_args, get_arg_val<uint32_t>(1), 1024);
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(2), 1024);
    const uint32_t pages = get_arg_val<uint32_t>(3);
    const uint32_t worker = get_arg_val<uint32_t>(4);
    for (uint32_t page = worker; page < pages; page += 44) {
        cb_reserve_back(0, 1);
        cb_reserve_back(1, 1);
        noc_async_read_tile(page, gate, get_write_ptr(0));
        noc_async_read_tile(page, up, get_write_ptr(1));
        noc_async_read_barrier();
        cb_push_back(0, 1);
        cb_push_back(1, 1);
        cb_wait_front(16, 1);
        noc_async_write_tile(page, output, get_read_ptr(16));
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
}
