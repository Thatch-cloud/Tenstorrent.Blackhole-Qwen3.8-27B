// The page table and cur_pos of the multi-user SDPA launch (QWEN_FAST_TP4_SDPA=multi), gathered from the users' own lent
// storage. The packed block lends each user a (B, width) int32 table and a (B,) int32 cur_pos per bundle (the pool keeps them
// current every round); one launch over U users needs ONE (U, width) table and ONE (U,) cur_pos. Rather than change the pool
// (a pinned seam), this launch copies, inside the trace, once per forward:
//   task t < users   page 0 of user t's lent table (every row of a lent table is the user's row) -> page t of the stacked table
//   task == users    word 0 of page 0 of each user's lent cur_pos (E - 1, the same word in every row) -> word u of the stacked
//                    cur_pos page, the rest of the page zero
// The copies are whole pages (the aligned page size is a compile-time arg, so the stride is the buffer's own), so the stacked
// table is the users' rows byte for byte.
//
// Runtime args: [task, users, stacked table address, stacked cur_pos address, then per user u (0 .. 7): lent table address,
// lent cur_pos address] (zero padded to 20 words). Compile-time args: the four accessors (lent table, stacked table, lent
// cur_pos, stacked cur_pos), then the aligned page bytes of a table row, of a lent cur_pos page and of the stacked cur_pos page.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto table_in_args = TensorAccessorArgs<0>();
    constexpr auto table_out_args = TensorAccessorArgs<table_in_args.next_compile_time_args_offset()>();
    constexpr auto position_in_args = TensorAccessorArgs<table_out_args.next_compile_time_args_offset()>();
    constexpr auto position_out_args = TensorAccessorArgs<position_in_args.next_compile_time_args_offset()>();
    constexpr uint32_t first = position_out_args.next_compile_time_args_offset();
    constexpr uint32_t TABLE_BYTES = get_compile_time_arg_val(first);
    constexpr uint32_t POSITION_IN_BYTES = get_compile_time_arg_val(first + 1);
    constexpr uint32_t POSITION_OUT_BYTES = get_compile_time_arg_val(first + 2);
    const uint32_t task = get_arg_val<uint32_t>(0);
    const uint32_t users = get_arg_val<uint32_t>(1);
    const uint32_t staged = get_write_ptr(0);
    if (task < users) {
        const auto source = TensorAccessor(table_in_args, get_arg_val<uint32_t>(4 + 2 * task), TABLE_BYTES);
        const auto destination = TensorAccessor(table_out_args, get_arg_val<uint32_t>(2), TABLE_BYTES);
        noc_async_read(get_noc_addr(0, source), staged, TABLE_BYTES);
        noc_async_read_barrier();
        noc_async_write(staged, get_noc_addr(task, destination), TABLE_BYTES);
        noc_async_write_barrier();
    } else {
        const auto destination = TensorAccessor(position_out_args, get_arg_val<uint32_t>(3), POSITION_OUT_BYTES);
        const uint32_t scratch = staged + ((POSITION_OUT_BYTES + 63) / 64) * 64;
        auto words = reinterpret_cast<volatile uint32_t*>(staged);
        for (uint32_t word = 0; word < POSITION_OUT_BYTES / 4; word++) { words[word] = 0; }
        for (uint32_t user = 0; user < users; user++) {
            const auto source = TensorAccessor(position_in_args, get_arg_val<uint32_t>(5 + 2 * user), POSITION_IN_BYTES);
            noc_async_read(get_noc_addr(0, source), scratch, POSITION_IN_BYTES);
            noc_async_read_barrier();
            words[user] = *reinterpret_cast<volatile uint32_t*>(scratch);
        }
        asm volatile("" ::: "memory");
        noc_async_write(staged, get_noc_addr(0, destination), POSITION_OUT_BYTES);
        noc_async_write_barrier();
    }
}
