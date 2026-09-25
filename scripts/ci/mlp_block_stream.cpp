#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t source_address = get_arg_val<uint32_t>(0);
    const uint32_t destination_address = get_arg_val<uint32_t>(1);
    const uint32_t worker = get_arg_val<uint32_t>(2);
    const uint32_t workers = get_arg_val<uint32_t>(3);
    const uint32_t pairs = get_arg_val<uint32_t>(4);
    const uint32_t blocks = get_arg_val<uint32_t>(5);
    constexpr auto source_args = TensorAccessorArgs<0>();
    constexpr auto destination_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    const auto source = TensorAccessor(source_args, source_address, 576);
    const auto destination = TensorAccessor(destination_args, destination_address, 27648);
    cb_reserve_back(0, 1);
    const uint32_t scratch = get_write_ptr(0);
    for (uint32_t block = 0; block < blocks; ++block) {
        for (uint32_t inner = 0; inner < 8; ++inner) {
            for (uint32_t column = 0; column < 6; ++column) {
                const uint32_t tile_address = scratch + (inner * 6 + column) * 576;
                if (worker * 6 + column < pairs * 2) {
                    const uint32_t page = (block * 8 + inner) * pairs * 2 + worker * 6 + column;
                    noc_async_read_tile(page, source, tile_address);
                } else {
                    volatile tt_l1_ptr uint32_t* zeros = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_address);
                    for (uint32_t word = 0; word < 144; ++word) {
                        zeros[word] = 0;
                    }
                }
            }
        }
        noc_async_read_barrier();
        noc_async_write(scratch, destination.get_noc_addr(block * workers + worker), 27648);
        noc_async_write_barrier();
    }
}
