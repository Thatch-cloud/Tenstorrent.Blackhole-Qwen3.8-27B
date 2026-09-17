#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto latent_args = TensorAccessorArgs<0>();
    constexpr auto weight_args = TensorAccessorArgs<latent_args.next_compile_time_args_offset()>();
    constexpr auto base_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<base_args.next_compile_time_args_offset()>();
    const uint32_t worker = get_arg_val<uint32_t>(4);
    const uint32_t workers = get_arg_val<uint32_t>(5);
    const uint32_t tiles = get_arg_val<uint32_t>(6);
    const uint32_t step = get_arg_val<uint32_t>(7);
    const auto latent = TensorAccessor(latent_args, get_arg_val<uint32_t>(0), 2048);
    const auto weight = TensorAccessor(weight_args, get_arg_val<uint32_t>(1), 2048);
    const auto base = TensorAccessor(base_args, get_arg_val<uint32_t>(2), 4096);
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(3), tiles * 128);
    const uint32_t scratch = get_write_ptr(3);
    cb_reserve_back(1, 8);
    for (uint32_t column = 0; column < 8; column++) {
        noc_async_read_tile(column, latent, get_write_ptr(1) + column * 2048);
    }
    noc_async_read_barrier();
    cb_push_back(1, 8);
    for (uint32_t task = worker; task < tiles; task += workers) {
        for (uint32_t column = 0; column < 8; column++) {
            cb_reserve_back(0, 1);
            noc_async_read_tile(task * 8 + column, weight, get_write_ptr(0));
            noc_async_read_barrier();
            cb_push_back(0, 1);
        }
        const uint32_t row_offset = ((step / 16) * 512 + (step % 16) * 16) * 4;
        noc_async_read(base.get_noc_addr(task, row_offset), scratch, 64);
        noc_async_read(base.get_noc_addr(task, row_offset + 1024), scratch + 64, 64);
        noc_async_read_barrier();
        cb_reserve_back(2, 1);
        auto unary = reinterpret_cast<volatile uint32_t*>(get_write_ptr(2));
        auto packed_base = reinterpret_cast<volatile const uint32_t*>(scratch);
        for (uint32_t offset = 0; offset < 1024; offset++) { unary[offset] = 0; }
        for (uint32_t row = 0; row < 32; row++) {
            unary[(row / 16) * 512 + (row % 16) * 16] = packed_base[row];
        }
        cb_push_back(2, 1);
        cb_wait_front(16, 1);
        auto scores = reinterpret_cast<volatile const uint32_t*>(get_read_ptr(16));
        auto packed_scores = reinterpret_cast<volatile uint32_t*>(scratch);
        for (uint32_t row = 0; row < 32; row++) {
            packed_scores[row] = scores[(row / 16) * 512 + (row % 16) * 16];
        }
        noc_async_write(scratch, output.get_noc_addr(0, task * 128), 128);
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
}
