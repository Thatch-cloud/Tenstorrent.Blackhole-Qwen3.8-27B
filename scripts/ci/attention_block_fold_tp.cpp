// Multi-task sibling of attention_fold_dma_tp.cpp (QWEN_FAST_TP4_ATTN_FOLD): the same fold, run for every (user, group)
// of the packed block in ONE launch. Each core walks its own task list; a task is one output tile of the served
// launch, with its source and destination buffers, source page base and destination page base given, so the same
// kernel reads the block query at a group's token offset (forward) or an SDPA result at a group's batch (inverse) and
// writes a user's stacked query or the block output. The tile body below is attention_fold_dma_tp.cpp's, line for line.
//
// Runtime args: [inverse, tasks, then per task: source address, destination address, rows, source base, destination
// base, task], zero padded to 2 + 6 * CAPACITY words (CAPACITY: the last compile-time arg, the most tasks any core
// carries, so launches of different list lengths never share a program cache entry). `task` is the served launch's worker index (tile row * 8 + column); the pages read are
// (source base + tile) * 8 + column, the page written is destination base + task.
#ifndef QWEN_FOLD_HEAD_ROWS
#error "QWEN_FOLD_HEAD_ROWS is defined by the launch builder (tp_kernels.fold_defines)"
#endif
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto source_args = TensorAccessorArgs<0>();
    constexpr auto destination_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    const bool inverse = get_arg_val<uint32_t>(0) != 0;
    constexpr uint32_t CAPACITY = get_compile_time_arg_val(destination_args.next_compile_time_args_offset());
    const uint32_t given = get_arg_val<uint32_t>(1);
    const uint32_t tasks = given < CAPACITY ? given : CAPACITY;  // never reads past 2 + 6 * CAPACITY
    for (uint32_t index = 0; index < tasks; index++) {
        const uint32_t base = 2 + index * 6;
        const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(base), 2048);
        const auto destination = TensorAccessor(destination_args, get_arg_val<uint32_t>(base + 1), 2048);
        const uint32_t rows = get_arg_val<uint32_t>(base + 2);
        const uint32_t source_base = get_arg_val<uint32_t>(base + 3);
        const uint32_t destination_base = get_arg_val<uint32_t>(base + 4);
        const uint32_t task = get_arg_val<uint32_t>(base + 5);
        const uint32_t column = task % 8;
        const uint32_t staged = get_write_ptr(0);
        const uint32_t output = staged + (rows > 4 ? rows : 4) * 2048;
        const uint32_t source_tiles = inverse ? (rows * QWEN_FOLD_HEAD_ROWS + 31) / 32 : rows;
        for (uint32_t tile = 0; tile < source_tiles; tile++) {
            const uint32_t page = (tile + source_base) * 8 + column;
            noc_async_read_tile(page, source, staged + tile * 2048);
        }
        noc_async_read_barrier();
        auto words = reinterpret_cast<volatile uint32_t*>(output);
        for (uint32_t word = 0; word < 512; word++) { words[word] = 0; }
        for (uint32_t target_row = 0; target_row < 32; target_row++) {
            uint32_t source_tile;
            uint32_t source_head;
            if (inverse) {
                if (target_row >= QWEN_FOLD_HEAD_ROWS) { continue; }
                const uint32_t token = task / 8;
                const uint32_t head = (target_row / 6) * rows * 6 + token * 6 + target_row % 6;
                source_tile = head / 32;
                source_head = head % 32;
            } else {
                const uint32_t head = (task / 8) * 32 + target_row;
                if (head >= rows * QWEN_FOLD_HEAD_ROWS) { continue; }
                const uint32_t remainder = head % (rows * 6);
                source_tile = remainder / 6;
                source_head = (head / (rows * 6)) * 6 + remainder % 6;
            }
            const uint32_t source_offset = ((source_head / 16) * 512 + (source_head % 16) * 16) * 2;
            const uint32_t target_offset = ((target_row / 16) * 512 + (target_row % 16) * 16) * 2;
            for (uint32_t face = 0; face < 2; face++) {
                const auto input = reinterpret_cast<volatile const uint32_t*>(staged + source_tile * 2048 + source_offset + face * 512);
                auto target = reinterpret_cast<volatile uint32_t*>(output + target_offset + face * 512);
                for (uint32_t word = 0; word < 8; word++) { target[word] = input[word]; }
            }
        }
        asm volatile("" ::: "memory");
        noc_async_write_tile(destination_base + task, destination, output);
        noc_async_write_barrier();
    }
}
