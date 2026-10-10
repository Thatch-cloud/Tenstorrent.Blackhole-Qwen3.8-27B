// The WRITER shared by the drafter fusion launches of tp4/fx-wp6 (draft_reduce_tp.py, draft_tail_tp.py): on the second data-movement core,
// for each output tile this worker owns, in order, wait for the compute kernel's result tile in CB 16 and write it to the output tensor.
// (draft_conv_out.cpp's shape, with the page size and the tile order from the launch.)
//
// Runtime args: [output buffer address, first tile, tile count]. Compile-time args: the output tensor's accessor args, then the page
// size in bytes (4096 for an fp32 tile, 2048 for a bf16 one).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto output_args = TensorAccessorArgs<0>();
    constexpr uint32_t page_bytes = get_compile_time_arg_val(output_args.next_compile_time_args_offset());
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(0), page_bytes);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    for (uint32_t tile = first; tile < first + count; ++tile) {
        cb_wait_front(16, 1);
        noc_async_write_tile(tile, output, get_read_ptr(16));
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
}
