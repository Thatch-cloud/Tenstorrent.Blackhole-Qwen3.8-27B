// The READER of the drafter's tail launches (QWEN_FAST_DRAFT_TAIL, draft_tail_tp.py; tp4/fx-wp6, F-F3a): SwiGLU and the residual tail.
//
// Both launches are elementwise over two operand tensors, one output tile per pair of input tiles, in row-major tile order. A task is an output
// tile: task t sits at tile row t / columns, tile column t % columns, and its two operand tiles are
//     A: page  row * a_stride + column              (SwiGLU: the gate half; residual: the finished convolution)
//     B: page  row * b_stride + b_col + column      (SwiGLU: the up half; residual: the hidden block)
// so a fused gate|up matmul output (a_stride = b_stride = 2 * columns, b_col = columns) and two separate tensors (strides = columns, b_col = 0)
// are read by the same loop. Tile A goes to CB 0 and tile B to CB 1, one page each; nothing is moved or changed.
//
// Runtime args: [A address, B address, first task, task count, a_stride, b_stride, b_col, columns]. Compile-time args: A's accessor args, B's
// accessor args, then the page sizes of A and B in bytes (4096 for fp32 tiles, 2048 for bf16 ones).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr uint32_t a_bytes = get_compile_time_arg_val(b_args.next_compile_time_args_offset());
    constexpr uint32_t b_bytes = get_compile_time_arg_val(b_args.next_compile_time_args_offset() + 1);
    const auto a = TensorAccessor(a_args, get_arg_val<uint32_t>(0), a_bytes);
    const auto b = TensorAccessor(b_args, get_arg_val<uint32_t>(1), b_bytes);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t a_stride = get_arg_val<uint32_t>(4);
    const uint32_t b_stride = get_arg_val<uint32_t>(5);
    const uint32_t b_col = get_arg_val<uint32_t>(6);
    const uint32_t columns = get_arg_val<uint32_t>(7);
    uint32_t row = first / columns;
    uint32_t column = first % columns;
    for (uint32_t task = 0; task < count; ++task) {
        cb_reserve_back(0, 1);
        cb_reserve_back(1, 1);
        noc_async_read_tile(row * a_stride + column, a, get_write_ptr(0));
        noc_async_read_tile(row * b_stride + b_col + column, b, get_write_ptr(1));
        noc_async_read_barrier();
        cb_push_back(0, 1);
        cb_push_back(1, 1);
        if (++column == columns) {
            column = 0;
            ++row;
        }
    }
}
