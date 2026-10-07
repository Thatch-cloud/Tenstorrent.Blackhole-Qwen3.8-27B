// The bit-for-bit compare of the multi-user SDPA audit (QWEN_FAST_TP4_SDPA_AUDIT=1, gate profiles only). Each launch compares
// the block output of the served per-user launches (A) with the multi launch's (B), tile page by tile page, as 32-bit words
// (so -0 and +0, or any one differing bit, count), and writes one counter page per core:
//   word 0 = words that differ, word 1 = words of A that are not zero (the liveness count: a compare of two zero tensors proves
//   nothing), word 2 = words compared, the rest zero.
// The host reads the counters after the replay (never inside it). Cores whose tile range is empty still write their page (all
// zero), so a counter page is always this launch's and never an earlier one's.
//
// Runtime args: [A address, B address, counters address, counter page, first tile, tile count]. Compile-time args: the accessors
// of A, B and the counters, then the counters' aligned page bytes.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto left_args = TensorAccessorArgs<0>();
    constexpr auto right_args = TensorAccessorArgs<left_args.next_compile_time_args_offset()>();
    constexpr auto counter_args = TensorAccessorArgs<right_args.next_compile_time_args_offset()>();
    constexpr uint32_t COUNTER_BYTES = get_compile_time_arg_val(counter_args.next_compile_time_args_offset());
    const auto left = TensorAccessor(left_args, get_arg_val<uint32_t>(0), 2048);
    const auto right = TensorAccessor(right_args, get_arg_val<uint32_t>(1), 2048);
    const auto counters = TensorAccessor(counter_args, get_arg_val<uint32_t>(2), COUNTER_BYTES);
    const uint32_t counter_page = get_arg_val<uint32_t>(3);
    const uint32_t first_tile = get_arg_val<uint32_t>(4);
    const uint32_t tile_count = get_arg_val<uint32_t>(5);
    const uint32_t staged = get_write_ptr(0);
    uint32_t differing = 0;
    uint32_t live = 0;
    uint32_t compared = 0;
    for (uint32_t tile = first_tile; tile < first_tile + tile_count; tile++) {
        noc_async_read_tile(tile, left, staged);
        noc_async_read_tile(tile, right, staged + 2048);
        noc_async_read_barrier();
        auto a = reinterpret_cast<volatile uint32_t*>(staged);
        auto b = reinterpret_cast<volatile uint32_t*>(staged + 2048);
        for (uint32_t word = 0; word < 512; word++) {
            const uint32_t x = a[word];
            if (x != b[word]) { differing++; }
            if (x != 0) { live++; }
            compared++;
        }
    }
    auto out = reinterpret_cast<volatile uint32_t*>(staged + 4096);
    for (uint32_t word = 0; word < COUNTER_BYTES / 4; word++) { out[word] = 0; }
    out[0] = differing;
    out[1] = live;
    out[2] = compared;
    asm volatile("" ::: "memory");
    noc_async_write(staged + 4096, get_noc_addr(counter_page, counters), COUNTER_BYTES);
    noc_async_write_barrier();
}
