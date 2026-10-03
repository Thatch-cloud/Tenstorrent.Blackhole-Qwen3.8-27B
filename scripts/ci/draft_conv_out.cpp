// The drafter's fused-convolution OUTPUT kernel at four cards (tp4/samp-draft, QWEN_FAST_TP4_DRAFT_CONV, default off).
//
// The served I/O kernels write each page's output themselves, after waiting for it (so they cannot read the next page meanwhile).
// Here the writer is its own kernel on the second data-movement core: for each page this worker owns, in the order draft_conv_io_fast
// reads them, wait for the compute kernel's result tile in CB 16 and write it to the output tensor.
//
// Runtime args: the output buffer address, rows, worker, workers (the same page plan as draft_conv_io_fast.cpp).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(0), 2048);
    const uint32_t rows = get_arg_val<uint32_t>(1);
    const uint32_t worker = get_arg_val<uint32_t>(2);
    const uint32_t workers = get_arg_val<uint32_t>(3);
    const uint32_t tile_rows = (rows + 31) / 32;
    for (uint32_t page = worker; page < 160 * tile_rows; page += workers) {
        cb_wait_front(16, 1);
        noc_async_write_tile(page, output, get_read_ptr(16));
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
}
