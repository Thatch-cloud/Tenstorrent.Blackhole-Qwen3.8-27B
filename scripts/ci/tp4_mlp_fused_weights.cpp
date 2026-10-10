#include "api/dataflow/dataflow_api.h"

// WP4 F-D1, the weight and output side of the fused gate|up launch at the TP4 per-chip shape (the twin of fused_1d_weights.cpp, which reads one pair-packed
// tensor): each worker streams its pairs' gate tiles from the served w1 tensor and up tiles from the served w3 tensor, K block by K block, into the block layout
// the native matmul compute kernel reads ([K tile][gate n0, up n0, gate n1, up n1, ...], bfloat4_b tiles of 576 bytes), and writes the SwiGLU products the
// compute kernel leaves in the output circular buffer to the interleaved output tensor, row tile by row tile. No packed copy of the weights is read or built.
void kernel_main() {
    const uint32_t gate_address = get_arg_val<uint32_t>(0);
    const uint32_t up_address = get_arg_val<uint32_t>(1);
    const uint32_t output_address = get_arg_val<uint32_t>(2);
    const uint32_t first_pair = get_arg_val<uint32_t>(3);
    const uint32_t valid_pairs = get_arg_val<uint32_t>(4);
    constexpr uint32_t pairs_per_worker = get_named_compile_time_arg_val("pairs_per_worker");
    constexpr uint32_t k_blocks = get_named_compile_time_arg_val("k_blocks");
    constexpr uint32_t block_tiles = get_named_compile_time_arg_val("block_tiles");
    constexpr uint32_t row_tiles = get_named_compile_time_arg_val("row_tiles");
    constexpr uint32_t pair_columns = get_named_compile_time_arg_val("pair_columns");
    constexpr auto gate_args = TensorAccessorArgs<0>();
    constexpr auto up_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<up_args.next_compile_time_args_offset()>();
    const auto gate = TensorAccessor(gate_args, gate_address, 576);
    const auto up = TensorAccessor(up_args, up_address, 576);
    const auto output = TensorAccessor(output_args, output_address, 2048);
    // The padding slots of a short last worker (pair >= valid_pairs) are zeroed ONCE, before the K loop, in both halves of the two-block weight circular buffer: this kernel never
    // writes them again and the compute kernel never writes this buffer, so they stay zero. (Zeroing them per K block cost 20 blocks of 288-word stores per padding pair on the
    // reader RISC: the p3/p5/p7 pathology the first card run measured, +274 / +630 / +659 microseconds.)
    if (valid_pairs < pairs_per_worker) {
        const uint32_t base = get_write_ptr(1);          // nothing is reserved or pushed yet: the base of the buffer
        for (uint32_t slot_row = 0; slot_row < 2 * block_tiles; ++slot_row) {
            for (uint32_t pair = valid_pairs; pair < pairs_per_worker; ++pair) {
                const uint32_t gate_tile = base + (slot_row * 2 * pairs_per_worker + 2 * pair) * 576;
                volatile tt_l1_ptr uint32_t* gate_zeros = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gate_tile);
                volatile tt_l1_ptr uint32_t* up_zeros = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gate_tile + 576);
                for (uint32_t word = 0; word < 144; ++word) {
                    gate_zeros[word] = 0;
                    up_zeros[word] = 0;
                }
            }
        }
    }
    for (uint32_t block = 0; block < k_blocks; ++block) {
        cb_reserve_back(1, 2 * block_tiles * pairs_per_worker);
        const uint32_t destination = get_write_ptr(1);
        for (uint32_t inner = 0; inner < block_tiles; ++inner) {
            for (uint32_t pair = 0; pair < valid_pairs; ++pair) {
                const uint32_t gate_tile = destination + (inner * 2 * pairs_per_worker + 2 * pair) * 576;
                const uint32_t up_tile = gate_tile + 576;
                const uint32_t page = (block * block_tiles + inner) * pair_columns + first_pair + pair;
                noc_async_read_tile(page, gate, gate_tile);
                noc_async_read_tile(page, up, up_tile);
            }
        }
        noc_async_read_barrier();
        cb_push_back(1, 2 * block_tiles * pairs_per_worker);
    }
    for (uint32_t row = 0; row < row_tiles; ++row) {
        for (uint32_t pair = 0; pair < pairs_per_worker; ++pair) {
            cb_wait_front(4, 1);
            if (pair < valid_pairs) {
                noc_async_write_tile(row * pair_columns + first_pair + pair, output, get_read_ptr(4));
                noc_async_write_barrier();
            }
            cb_pop_front(4, 1);
        }
    }
}
