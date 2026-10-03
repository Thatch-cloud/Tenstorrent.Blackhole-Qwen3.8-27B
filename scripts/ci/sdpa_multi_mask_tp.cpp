// The narrow tail mask of the multi-user SDPA launch (QWEN_FAST_TP4_SDPA=multi): one (U, 1, 96, 256) bf16 TILE mask for U users'
// 16-token entries, written by one launch. It is attention_mask_replay_tp.cpp's tile body (capacity 256, offset 0) with the
// three differences the one-entry-per-user layout needs, and no others:
//   - each entry is ONE user, so its first word is that user's own positions tensor (start & 255, word 0), read from the
//     address given for the entry's batch; the served bundle shared one word between its two 8-token groups;
//   - an entry holds 16 tokens (96 head rows), so the token of a row is head / 6 and the position is word + token; the served
//     entry b added b * rows (b * 8) for its group's first token, which is the same position for the same token;
//   - a core walks a task list (CAPACITY tasks at most), because U * 24 tile tasks can exceed the cores one tile each.
// A task is one output tile: batch = task / (head_tiles * 8), head_tile = (task / 8) % head_tiles, column_tile = task % 8, and
// the page written is the task id (capacity 256: (batch * head_tiles + head_tile) * 8 + column_tile).
//
// Runtime args: [mask address, rows, tasks, positions address of batch 0 ... 7, then the task ids], zero padded to
// 11 + CAPACITY words. CAPACITY (the last compile-time arg) is the most tasks any core carries, so launches of different
// lengths never share a program cache entry.
#ifndef QWEN_FOLD_HEAD_ROWS
#error "QWEN_FOLD_HEAD_ROWS is defined by the launch builder (tp_kernels.fold_defines)"
#endif
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto position_args = TensorAccessorArgs<0>();
    constexpr auto mask_args = TensorAccessorArgs<position_args.next_compile_time_args_offset()>();
    constexpr uint32_t CAPACITY = get_compile_time_arg_val(mask_args.next_compile_time_args_offset());
    const auto mask = TensorAccessor(mask_args, get_arg_val<uint32_t>(0), 2048);
    const uint32_t rows = get_arg_val<uint32_t>(1);
    const uint32_t given = get_arg_val<uint32_t>(2);
    const uint32_t tasks = given < CAPACITY ? given : CAPACITY;  // never reads past 11 + CAPACITY
    const uint32_t head_tiles = (rows * QWEN_FOLD_HEAD_ROWS + 31) / 32;
    for (uint32_t index = 0; index < tasks; index++) {
        const uint32_t task = get_arg_val<uint32_t>(11 + index);
        const uint32_t batch = task / (head_tiles * 8);
        const uint32_t head_tile = (task / 8) % head_tiles;
        const uint32_t column_tile = task % 8;
        const auto positions = TensorAccessor(position_args, get_arg_val<uint32_t>(3 + batch), 32);
        const uint32_t staged = get_write_ptr(0);
        noc_async_read(get_noc_addr(0, positions), staged, 32);
        noc_async_read_barrier();
        const uint32_t start = *reinterpret_cast<volatile uint32_t*>(staged);
        const uint32_t output = staged + 2048;
        auto words = reinterpret_cast<volatile uint16_t*>(output);
        for (uint32_t row = 0; row < 32; row++) {
            const uint32_t head = head_tile * 32 + row;
            const uint32_t position = start + head / 6;
            for (uint32_t column = 0; column < 32; column++) {
                const uint32_t index_in_tile = (row / 16) * 512 + (column / 16) * 256 + (row % 16) * 16 + column % 16;
                const uint32_t cache_position = column_tile * 32 + column;
                words[index_in_tile] = head >= rows * QWEN_FOLD_HEAD_ROWS || cache_position > position ? 0xff80 : 0;
            }
        }
        asm volatile("" ::: "memory");
        noc_async_write_tile(task, mask, output);
        noc_async_write_barrier();
    }
}
