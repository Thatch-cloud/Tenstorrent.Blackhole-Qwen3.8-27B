// The READER of the drafter's local reduce (QWEN_FAST_DRAFT_REDUCE, draft_reduce_tp.py; tp4/fx-wp6, F-F1).
//
// The served chain after the dim-0 all-gather is four slices (one per chip's partial), then three fp32 adds, ((p0 + p1) + p2) + p3 - eight
// launches, one of them a DRAM round trip per add. The gathered tensor is (chips, 1, rows, 5120) fp32 TILE in interleaved DRAM, so chip c's
// tile t is page c * stride + t (stride = tiles of one chip's block = ceil(rows / 32) * 160). This kernel hands the compute kernel the four
// partials of one output tile at a time, chip 0 first, in four consecutive 4,096-byte pages of CB 0; it reads each tile once and moves no
// element, so the adds see the very bytes the slices would have copied.
//
// Runtime args: [gathered buffer address, first tile, tile count, stride (tiles per chip block)]. Compile-time args: the gathered tensor's
// accessor args, then the chip count (4; at most 4 because the fp32 destination holds four tiles).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr uint32_t chips = get_compile_time_arg_val(input_args.next_compile_time_args_offset());
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(0), 4096);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t stride = get_arg_val<uint32_t>(3);
    for (uint32_t tile = first; tile < first + count; ++tile) {
        cb_reserve_back(0, chips);
        const uint32_t destination = get_write_ptr(0);
        for (uint32_t chip = 0; chip < chips; ++chip) {
            noc_async_read_tile(chip * stride + tile, input, destination + chip * 4096);
        }
        noc_async_read_barrier();
        cb_push_back(0, chips);
    }
}
