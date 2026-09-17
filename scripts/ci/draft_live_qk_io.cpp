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
    const uint32_t scratch = get_write_ptr(2);
    const uint32_t output_address = scratch + 4 * 4096;
    auto assembled = reinterpret_cast<volatile uint32_t*>(output_address);
    for (uint32_t element = 0; element < 1024; element++) { assembled[element] = 0; }
    for (uint32_t task = worker; task < 16 * key_tiles; task += workers) {
        const uint32_t head = task / key_tiles;
        cb_reserve_back(0, 4);
        for (uint32_t column = 0; column < 4; column++) {
            noc_async_read_tile(head * 4 + column, left, get_write_ptr(0) + column * 4096);
            noc_async_read_tile(task * 4 + column, right, scratch + column * 4096);
        }
        noc_async_read_barrier();
        cb_push_back(0, 4);
        for (uint32_t key = 0; key < 32; key++) {
            for (uint32_t column = 0; column < 4; column++) {
                cb_reserve_back(1, 1);
                auto source = reinterpret_cast<volatile const uint32_t*>(scratch + column * 4096);
                auto broadcast = reinterpret_cast<volatile uint32_t*>(get_write_ptr(1));
                for (uint32_t lane = 0; lane < 32; lane++) {
                    broadcast[(lane / 16) * 256 + lane % 16] =
                        source[(key / 16) * 512 + (lane / 16) * 256 + (key % 16) * 16 + lane % 16];
                }
                cb_push_back(1, 1);
            }
            cb_wait_front(16, 1);
            auto reduced = reinterpret_cast<volatile const uint32_t*>(get_read_ptr(16));
            for (uint32_t row = 0; row < 8; row++) {
                assembled[(key / 16) * 256 + row * 16 + key % 16] = reduced[row * 16];
            }
            cb_pop_front(16, 1);
        }
        noc_async_write_tile(task, output, output_address);
        noc_async_write_barrier();
    }
}
