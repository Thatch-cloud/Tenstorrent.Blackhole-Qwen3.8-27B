#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(0), 4096);
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(1), 4096);
    const uint32_t task = get_arg_val<uint32_t>(2);
    const uint32_t columns = get_arg_val<uint32_t>(3);
    for (uint32_t column = 0; column < columns; column++) {
        cb_reserve_back(0, 1);
        noc_async_read_tile(task * columns + column, input, get_write_ptr(0));
        noc_async_read_barrier();
        cb_push_back(0, 1);
    }
    cb_wait_front(16, 1);
    noc_async_write_tile(task, output, get_read_ptr(16));
    noc_async_write_barrier();
    cb_pop_front(16, 1);
}
