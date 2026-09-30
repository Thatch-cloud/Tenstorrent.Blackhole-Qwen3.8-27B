// Four-card pipelined sibling of gdn_commit_dma_tp.cpp (QWEN_FAST_TP4_COMMIT_LANES): the same bytes to the same
// destinations, eight tiles in flight per barrier instead of one. Derived from the unqualified two-card
// gdn_commit_batched_dma.cpp with its page counts taken from the launch builder's defines (tp_kernels.defines) and one
// trim: each lane's output scratch is zeroed once, because only bytes [0,32) and [512,544) of it are ever rewritten.
#ifndef QWEN_STATE_PAGES
#error "QWEN_STATE_PAGES is defined by the launch builder (tp_kernels.defines)"
#endif
#ifndef QWEN_CONV_TASKS
#error "QWEN_CONV_TASKS is defined by the launch builder (tp_kernels.defines)"
#endif
#ifndef QWEN_CONV_PAGES
#error "QWEN_CONV_PAGES is defined by the launch builder (tp_kernels.defines)"
#endif
static_assert(QWEN_STATE_PAGES % 16 == 0, "state pages must fill whole two-worker eight-lane batches");
static_assert(QWEN_CONV_TASKS % 16 == 0, "conv tasks must fill whole two-worker eight-lane batches");
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto entry_rec_args = TensorAccessorArgs<0>();
    constexpr auto entry_conv_args = TensorAccessorArgs<entry_rec_args.next_compile_time_args_offset()>();
    constexpr auto history_rec_args = TensorAccessorArgs<entry_conv_args.next_compile_time_args_offset()>();
    constexpr auto history_conv_args = TensorAccessorArgs<history_rec_args.next_compile_time_args_offset()>();
    constexpr auto native_rec_args = TensorAccessorArgs<history_conv_args.next_compile_time_args_offset()>();
    constexpr auto native_conv_args = TensorAccessorArgs<native_rec_args.next_compile_time_args_offset()>();
    constexpr auto checkpoint_rec_args = TensorAccessorArgs<native_conv_args.next_compile_time_args_offset()>();
    constexpr auto checkpoint_conv_args = TensorAccessorArgs<checkpoint_rec_args.next_compile_time_args_offset()>();
    const uint32_t prefix = get_arg_val<uint32_t>(20);
    const uint32_t worker = get_arg_val<uint32_t>(21);
    const uint32_t scratch = get_write_ptr(0);
    const auto entry_rec = TensorAccessor(entry_rec_args, get_arg_val<uint32_t>(0), 2048);
    const auto history_rec = TensorAccessor(history_rec_args, get_arg_val<uint32_t>(5), 2048);
    const auto native_rec = TensorAccessor(native_rec_args, get_arg_val<uint32_t>(10), 2048);
    const auto checkpoint_rec = TensorAccessor(checkpoint_rec_args, get_arg_val<uint32_t>(15), 2048);
    for (uint32_t batch = 0; batch < QWEN_STATE_PAGES; batch += 16) {
        for (uint32_t lane = 0; lane < 8; ++lane) {
            const uint32_t page = batch + lane * 2 + worker;
            const uint32_t staged = scratch + lane * 4096;
            if (prefix == 0) {
                noc_async_read_tile(page, entry_rec, staged);
            } else {
                noc_async_read_tile((prefix - 1) * QWEN_STATE_PAGES + page, history_rec, staged);
            }
        }
        noc_async_read_barrier();
        for (uint32_t lane = 0; lane < 8; ++lane) {
            const uint32_t page = batch + lane * 2 + worker;
            const uint32_t staged = scratch + lane * 4096;
            noc_async_write_tile(page, native_rec, staged);
            noc_async_write_tile(page, checkpoint_rec, staged);
        }
        noc_async_write_barrier();
    }
    const uint32_t token = prefix == 0 ? 0 : prefix - 1;
    const uint32_t offset = ((token / 16) * 512 + (token % 16) * 16) * 2;
    for (uint32_t lane = 0; lane < 8; ++lane) {
        auto words = reinterpret_cast<volatile uint32_t*>(scratch + lane * 4096 + 2048);
        for (uint32_t word = 0; word < 512; ++word) { words[word] = 0; }
    }
    for (uint32_t batch = 0; batch < QWEN_CONV_TASKS; batch += 16) {
        for (uint32_t lane = 0; lane < 8; ++lane) {
            const uint32_t task = batch + lane * 2 + worker;
            const uint32_t slot = task / QWEN_CONV_PAGES;
            const uint32_t page = task % QWEN_CONV_PAGES;
            const uint32_t staged = scratch + lane * 4096;
            const auto entry = TensorAccessor(entry_conv_args, get_arg_val<uint32_t>(1 + slot), 2048);
            const auto history = TensorAccessor(history_conv_args, get_arg_val<uint32_t>(6 + slot), 2048);
            // Whole-tile reads, as the served kernel does: a 32-byte DRAM read at a token offset is not 64-byte aligned.
            if (prefix == 0) {
                noc_async_read_tile(page, entry, staged);
            } else {
                noc_async_read_tile(page, history, staged);
            }
        }
        noc_async_read_barrier();
        for (uint32_t lane = 0; lane < 8; ++lane) {
            const uint32_t task = batch + lane * 2 + worker;
            const uint32_t slot = task / QWEN_CONV_PAGES;
            const uint32_t page = task % QWEN_CONV_PAGES;
            const uint32_t staged = scratch + lane * 4096;
            const uint32_t output = staged + 2048;
            const auto native = TensorAccessor(native_conv_args, get_arg_val<uint32_t>(11 + slot), 2048);
            const auto checkpoint = TensorAccessor(checkpoint_conv_args, get_arg_val<uint32_t>(16 + slot), 2048);
            for (uint32_t face = 0; face < 2; ++face) {
                const auto source = reinterpret_cast<volatile const uint32_t*>(staged + offset + face * 512);
                auto destination = reinterpret_cast<volatile uint32_t*>(output + face * 512);
                for (uint32_t word = 0; word < 8; ++word) { destination[word] = source[word]; }
            }
            asm volatile("" ::: "memory");
            noc_async_write_tile(page, checkpoint, output);
            noc_async_write(output, native.get_noc_addr(page, 0), 32);
            noc_async_write(output + 512, native.get_noc_addr(page, 512), 32);
        }
        noc_async_write_barrier();
    }
}
