#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t weight_address = get_arg_val<uint32_t>(0);
    const uint32_t output_address = get_arg_val<uint32_t>(1);
    const uint32_t first_pair = get_arg_val<uint32_t>(2);
    const uint32_t valid_pairs = get_arg_val<uint32_t>(3);
    const uint32_t output_tiles = get_arg_val<uint32_t>(4);
    const uint32_t pairs_per_worker = get_arg_val<uint32_t>(5);
    constexpr auto weight_args = TensorAccessorArgs<0>();
    constexpr auto output_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    const auto weights = TensorAccessor(weight_args, weight_address, 576);
    const auto output = TensorAccessor(output_args, output_address, 2048);
    for (uint32_t block = 0; block < 20; ++block) {
        cb_reserve_back(1, 16 * pairs_per_worker);
        const uint32_t destination = get_write_ptr(1);
        for (uint32_t inner = 0; inner < 8; ++inner) {
            for (uint32_t column = 0; column < 2 * pairs_per_worker; ++column) {
                const uint32_t tile_address = destination + (inner * 2 * pairs_per_worker + column) * 576;
                if (column < valid_pairs * 2) {
                    const uint32_t page = (block * 8 + inner) * 544 + first_pair * 2 + column;
                    noc_async_read_tile(page, weights, tile_address);
                } else {
                    volatile tt_l1_ptr uint32_t* zeros = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_address);
                    for (uint32_t word = 0; word < 144; ++word) {
                        zeros[word] = 0;
                    }
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(1, 16 * pairs_per_worker);
    }
    for (uint32_t pair = 0; pair < pairs_per_worker; ++pair) {
        cb_wait_front(4, output_tiles);
        if (pair < valid_pairs) {
            for (uint32_t tile = 0; tile < output_tiles; ++tile) {
                noc_async_write_tile((first_pair + pair) * output_tiles + tile, output, get_read_ptr(4) + tile * 2048);
            }
            noc_async_write_barrier();
        }
        cb_pop_front(4, output_tiles);
    }
}
