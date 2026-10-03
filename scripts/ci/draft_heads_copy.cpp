// Tile copier behind the drafter's head split and head merge (tp4/samp-draft, QWEN_FAST_TP4_DRAFT_HEADS, default off).
//
// nlp_create_qkv_heads (transpose_k_heads=False) and nlp_concat_heads on tiled bf16 whose head_dim is a whole number of tiles
// (128 = four) move no element inside a tile: each is a permutation of whole 32 x 32 tiles. So both are this one launch: a task
// copies ONE tile from a source page to a destination page, tasks are dealt over a few cores, and the tile bytes are untouched.
//
// Runtime args: [tasks, then per task: source address, source page, destination address, destination page], zero padded to
// 1 + 4 * CAPACITY words. Compile-time args: the source accessor, the destination accessor (every source of a launch is an
// interleaved DRAM bf16 tile tensor, and so is every destination, so one layout describes all of them), then CAPACITY, the most
// tasks any core carries: generic_op's program cache does not hash runtime-arg lengths, so a capacity that is a compile-time arg
// keeps launches with different list lengths in different cache entries (the gdn rows mover's rule, gdn_rows_dma_tp.cpp).
// Eight tiles are in flight per barrier.
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t LANES = 8;
constexpr uint32_t TASK_WORDS = 4;
constexpr uint32_t TILE_BYTES = 2048;

void kernel_main() {
    constexpr auto source_args = TensorAccessorArgs<0>();
    constexpr auto destination_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    constexpr uint32_t CAPACITY = get_compile_time_arg_val(destination_args.next_compile_time_args_offset());
    const uint32_t given = get_arg_val<uint32_t>(0);
    const uint32_t tasks = given < CAPACITY ? given : CAPACITY;  // never reads past 1 + TASK_WORDS * CAPACITY
    const uint32_t scratch = get_write_ptr(0);
    for (uint32_t first = 0; first < tasks; first += LANES) {
        const uint32_t lanes = tasks - first < LANES ? tasks - first : LANES;
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            const uint32_t base = 1 + (first + lane) * TASK_WORDS;
            const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(base), TILE_BYTES);
            noc_async_read_tile(get_arg_val<uint32_t>(base + 1), source, scratch + lane * TILE_BYTES);
        }
        noc_async_read_barrier();
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            const uint32_t base = 1 + (first + lane) * TASK_WORDS;
            const auto destination = TensorAccessor(destination_args, get_arg_val<uint32_t>(base + 2), TILE_BYTES);
            noc_async_write_tile(get_arg_val<uint32_t>(base + 3), destination, scratch + lane * TILE_BYTES);
        }
        noc_async_write_barrier();
    }
}
