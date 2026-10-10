#include "api/dataflow/dataflow_api.h"

// WP4 F-D3 (QWEN_FAST_MLP_GATEUP_BANK), the weight and output side of the bank-strided fused gate|up launch at the TP4 per-chip shape.
//
// THE LAYOUT FACT. An interleaved DRAM tensor puts page p in bank p % banks at bank offset (p / banks) * page_bytes. The gate and up weights have 136 tile columns,
// a multiple of the 8 banks, so the tiles (k, c), (k, c + banks), (k, c + 2 banks) ... of one tile row are CONSECUTIVE PAGES OF ONE BANK: one NoC read of n pages from the
// address of page (k, c) returns all n (the read probe, optimisation/ttnn-op/mlp_gateup/readprobe*, checked this on a card by copying every tile it read back and comparing).
// The stock 1D reader hands a core neighbouring columns, which sit in different banks, so it issues one 576-byte request per tile; this kernel hands a worker the `chunk`
// columns first, first + banks, ... of ONE bank and reads them as one request, which the probe measured at 403 GB/s against the stock pattern's 294.
//
// WHAT A WORKER READS. Runtime arguments: the gate, up and output addresses, `first` (the tile column of its first tile, bank + banks * first group) and `valid` (the columns it
// owns, at most chunk; the last worker of a bank may own fewer). For each K tile row k it reads `valid` gate tiles and `valid` up tiles, each as ONE request of valid * 576 bytes
// from the page k * pair_columns + first, into the block layout the native matmul compute kernel reads with TWO input subblocks of `chunk` columns each:
//     [K tile][gate tile 0 .. chunk-1][up tile 0 .. chunk-1]
// (tile slot inner * 2 * chunk + i for the gate, + chunk for the up). The compute kernel's subblock 0 is then the gate columns, subblock 1 the up columns, and the same column
// index in the two is the same output column; the SwiGLU product of column i is written to the output page row * pair_columns + first + banks * i.
//
// THE PADDING FIX. A worker with fewer than `chunk` valid columns has padding slots that the compute kernel still reads. They are zeroed ONCE, before the K loop, in every block slot of
// the weight circular buffer (`depth` blocks, 2 by default): this kernel never writes them again and the compute kernel never writes this buffer, so they stay zero (zeroing them per K
// block, as the first fused kernel did, cost 20 blocks of 144-word stores per padding tile on the reader RISC, the p3/p5/p7 pathology of the first fused op).
//
// DEPTH. The circular buffer holds `depth` K blocks (default 2: the reader fills one block while the compute kernel consumes another). A deeper buffer changes nothing this kernel does
// per block; it lets the reader run further ahead of the compute kernel and of the lock-stepped activation multicast.
static inline void zero_tile(uint32_t address) {
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    for (uint32_t word = 0; word < 144; ++word) {   // 576 bytes of a bfloat4_b tile
        words[word] = 0;
    }
}

void kernel_main() {
    const uint32_t gate_address = get_arg_val<uint32_t>(0);
    const uint32_t up_address = get_arg_val<uint32_t>(1);
    const uint32_t output_address = get_arg_val<uint32_t>(2);
    const uint32_t first = get_arg_val<uint32_t>(3);
    const uint32_t valid = get_arg_val<uint32_t>(4);
    constexpr uint32_t chunk = get_named_compile_time_arg_val("chunk");
    constexpr uint32_t k_blocks = get_named_compile_time_arg_val("k_blocks");
    constexpr uint32_t block_tiles = get_named_compile_time_arg_val("block_tiles");
    constexpr uint32_t row_tiles = get_named_compile_time_arg_val("row_tiles");
    constexpr uint32_t pair_columns = get_named_compile_time_arg_val("pair_columns");
    constexpr uint32_t banks = get_named_compile_time_arg_val("banks");
    constexpr uint32_t depth = get_named_compile_time_arg_val("depth");
    constexpr auto gate_args = TensorAccessorArgs<0>();
    constexpr auto up_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<up_args.next_compile_time_args_offset()>();
    const auto gate = TensorAccessor(gate_args, gate_address, 576);
    const auto up = TensorAccessor(up_args, up_address, 576);
    const auto output = TensorAccessor(output_args, output_address, 2048);
    constexpr uint32_t row_slots = 2 * chunk;                    // tile slots of one K tile row: the gate run then the up run
    constexpr uint32_t block_slots = block_tiles * row_slots;    // tile slots of one K block (the circular buffer holds `depth`)
    if (valid < chunk) {
        const uint32_t base = get_write_ptr(1);                  // nothing is reserved or pushed yet: the base of the buffer
        for (uint32_t slot_row = 0; slot_row < depth * block_tiles; ++slot_row) {     // every block slot
            for (uint32_t pair = valid; pair < chunk; ++pair) {
                const uint32_t gate_tile = base + (slot_row * row_slots + pair) * 576;
                zero_tile(gate_tile);
                zero_tile(gate_tile + chunk * 576);
            }
        }
    }
    for (uint32_t block = 0; block < k_blocks; ++block) {
        cb_reserve_back(1, block_slots);
        const uint32_t destination = get_write_ptr(1);
        for (uint32_t inner = 0; inner < block_tiles; ++inner) {
            const uint32_t page = (block * block_tiles + inner) * pair_columns + first;
            const uint32_t gate_run = destination + inner * row_slots * 576;
            noc_async_read(gate.get_noc_addr(page), gate_run, valid * 576);
            noc_async_read(up.get_noc_addr(page), gate_run + chunk * 576, valid * 576);
        }
        noc_async_read_barrier();
        cb_push_back(1, block_slots);
    }
    for (uint32_t row = 0; row < row_tiles; ++row) {
        for (uint32_t pair = 0; pair < chunk; ++pair) {
            cb_wait_front(4, 1);
            if (pair < valid) {
                noc_async_write_tile(row * pair_columns + first + banks * pair, output, get_read_ptr(4));
                noc_async_write_barrier();
            }
            cb_pop_front(4, 1);
        }
    }
}
