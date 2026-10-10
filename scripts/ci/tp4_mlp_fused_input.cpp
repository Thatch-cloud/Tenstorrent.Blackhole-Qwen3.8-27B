#include "api/dataflow/dataflow_api.h"

// WP4 F-D1, the activation side of the fused gate|up launch at the TP4 per-chip shape (the twin of fused_1d_input.cpp, which is the TP2 single-tile-row kernel
// and is never edited): worker 0 reads one K block of the (row_tiles x K tiles) activation from L1 and multicasts it to every core of the rectangle, in the block
// layout the native matmul compute kernel reads, [row tile][K tile in block]. Cores that hold no output columns drain their copy.
void kernel_main() {
    const uint32_t address = get_arg_val<uint32_t>(0);
    const uint32_t worker = get_arg_val<uint32_t>(1);
    const uint32_t first_x = get_arg_val<uint32_t>(2);
    const uint32_t first_y = get_arg_val<uint32_t>(3);
    const uint32_t last_x = get_arg_val<uint32_t>(4);
    const uint32_t last_y = get_arg_val<uint32_t>(5);
    const uint32_t workers = get_arg_val<uint32_t>(6);
    const uint32_t receivers = get_arg_val<uint32_t>(7) - 1;
    constexpr uint32_t k_blocks = get_named_compile_time_arg_val("k_blocks");
    constexpr uint32_t block_tiles = get_named_compile_time_arg_val("block_tiles");
    constexpr uint32_t row_tiles = get_named_compile_time_arg_val("row_tiles");
    constexpr uint32_t row_stride = get_named_compile_time_arg_val("row_stride");
    constexpr uint32_t row_bytes = block_tiles * 2048;
    constexpr auto tensor_args = TensorAccessorArgs<0>();
    const auto input = TensorAccessor(tensor_args, address, 2048);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* received = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    for (uint32_t block = 0; block < k_blocks; ++block) {
        cb_reserve_back(0, block_tiles * row_tiles);
        const uint32_t destination = get_write_ptr(0);
        if (worker == 0) {
            for (uint32_t row = 0; row < row_tiles; ++row) {
                for (uint32_t tile = 0; tile < block_tiles; ++tile) {
                    noc_async_read_tile(row * row_stride + block * block_tiles + tile, input, destination + (row * block_tiles + tile) * 2048);
                }
            }
            noc_async_read_barrier();
            noc_semaphore_wait(ready, receivers);
            noc_semaphore_set(ready, 0);
            for (uint32_t row = 0; row < row_tiles; ++row) {
                const uint64_t target = get_noc_multicast_addr(last_x, last_y, first_x, first_y, destination + row * row_bytes);
                noc_async_write_multicast(destination + row * row_bytes, target, row_bytes, receivers);
            }
            noc_async_write_barrier();
            noc_semaphore_set(received, 1);
            const uint64_t signal = get_noc_multicast_addr(last_x, last_y, first_x, first_y, get_semaphore(1));
            noc_semaphore_set_multicast(get_semaphore(1), signal, receivers);
        } else {
            noc_semaphore_inc(get_noc_addr(first_x, first_y, get_semaphore(0)), 1);
            noc_semaphore_wait(received, 1);
            noc_semaphore_set(received, 0);
        }
        cb_push_back(0, block_tiles * row_tiles);
        if (worker >= workers) {
            cb_wait_front(0, block_tiles * row_tiles);
            cb_pop_front(0, block_tiles * row_tiles);
        }
    }
    // Drain before exit (as the TP2 kernel): worker 0's last act is a multicast of the received signal and the receivers' last acts are semaphore
    // increments, and a kernel that exits with either in flight hands the next program on the core a stale NoC counter. Each barrier is a no-op for
    // the other role, so one exit sequence serves both.
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
