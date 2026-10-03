#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto base_args = TensorAccessorArgs<0>();
    constexpr auto bias_args = TensorAccessorArgs<base_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    const uint32_t worker = get_arg_val<uint32_t>(3);
    const uint32_t workers = get_arg_val<uint32_t>(4);
    const uint32_t tiles = get_arg_val<uint32_t>(5);
    const uint32_t step = get_arg_val<uint32_t>(6);
    const auto base = TensorAccessor(base_args, get_arg_val<uint32_t>(0), 4096);
    const auto bias = TensorAccessor(bias_args, get_arg_val<uint32_t>(1), 4096);
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(2), tiles * 128);
    auto base_words = reinterpret_cast<volatile uint32_t*>(get_write_ptr(0));
    auto bias_words = reinterpret_cast<volatile uint32_t*>(get_write_ptr(1));
    for (uint32_t offset = 0; offset < 1024; offset++) {
        base_words[offset] = 0;
        bias_words[offset] = 0;
    }
    const uint32_t row_offset = ((step / 16) * 512 + (step % 16) * 16) * 4;
    for (uint32_t task = worker; task < tiles; task += workers) {
        cb_reserve_back(0, 1);
        cb_reserve_back(1, 1);
        noc_async_read(base.get_noc_addr(task, row_offset), get_write_ptr(0), 64);
        noc_async_read(base.get_noc_addr(task, row_offset + 1024), get_write_ptr(0) + 1024, 64);
        noc_async_read(bias.get_noc_addr(task, 0), get_write_ptr(1), 64);
        noc_async_read(bias.get_noc_addr(task, 1024), get_write_ptr(1) + 1024, 64);
        noc_async_read_barrier();
        cb_push_back(0, 1);
        cb_push_back(1, 1);
        cb_wait_front(16, 1);
        noc_async_write(get_read_ptr(16), output.get_noc_addr(0, task * 128), 64);
        noc_async_write(get_read_ptr(16) + 1024, output.get_noc_addr(0, task * 128 + 64), 64);
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
}
