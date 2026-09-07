#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto left_args = TensorAccessorArgs<0>();
    constexpr auto right_args = TensorAccessorArgs<left_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<right_args.next_compile_time_args_offset()>();
    const auto left = TensorAccessor(left_args, get_arg_val<uint32_t>(0), 4096);
    const auto right = TensorAccessor(right_args, get_arg_val<uint32_t>(1), 4096);
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(2), 4096);
    const uint32_t worker = get_arg_val<uint32_t>(3);
    const uint32_t workers = get_arg_val<uint32_t>(4);
    const uint32_t key_tiles = get_arg_val<uint32_t>(5);
    const uint32_t width_tiles = get_arg_val<uint32_t>(6);
    const bool cache_tiles = get_arg_val<uint32_t>(7) != 0;
    const uint32_t scratch = get_write_ptr(2);
    const uint32_t output_address = scratch + (cache_tiles ? width_tiles : 1) * 4096;
    auto assembled = reinterpret_cast<volatile uint32_t*>(output_address);
    for (uint32_t task = worker; task < 16 * key_tiles; task += workers) {
        const uint32_t head = task / key_tiles;
        if (cache_tiles) {
            cb_reserve_back(0, width_tiles);
            for (uint32_t column = 0; column < width_tiles; column++) {
                noc_async_read_tile(head * width_tiles + column, left, get_write_ptr(0) + column * 4096);
                noc_async_read_tile(task * width_tiles + column, right, scratch + column * 4096);
            }
            noc_async_read_barrier();
            cb_push_back(0, width_tiles);
        }
        for (uint32_t key = 0; key < 32; key++) {
            for (uint32_t column = 0; column < width_tiles; column++) {
                cb_reserve_back(1, 1);
                if (!cache_tiles) {
                    cb_reserve_back(0, 1);
                    noc_async_read_tile(head * width_tiles + column, left, get_write_ptr(0));
                    noc_async_read_tile(task * width_tiles + column, right, scratch);
                    noc_async_read_barrier();
                }
                auto source = reinterpret_cast<volatile const uint32_t*>(scratch + (cache_tiles ? column * 4096 : 0));
                auto broadcast = reinterpret_cast<volatile uint32_t*>(get_write_ptr(1));
                for (uint32_t lane = 0; lane < 32; lane++) {
                    broadcast[(lane / 16) * 256 + lane % 16] =
                        source[(key / 16) * 512 + (lane / 16) * 256 + (key % 16) * 16 + lane % 16];
                }
                if (!cache_tiles) { cb_push_back(0, 1); }
                cb_push_back(1, 1);
            }
            cb_wait_front(16, 1);
            auto reduced = reinterpret_cast<volatile const uint32_t*>(get_read_ptr(16));
            for (uint32_t row = 0; row < 32; row++) {
                assembled[(row / 16) * 512 + (key / 16) * 256 + (row % 16) * 16 + key % 16] =
                    reduced[(row / 16) * 512 + (row % 16) * 16];
            }
            cb_pop_front(16, 1);
        }
        noc_async_write_tile(task, output, output_address);
        noc_async_write_barrier();
    }
}
