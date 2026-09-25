// mlp_c1e_pack (one per RISC: RISCV_0 on CB 0, RISCV_1 on CB 1). See mlp_c1e_pack.py.
//
// Rebuilds one layer's served tile-pair-interleaved [gate|up] prefill weight (mlp.py
// _build_gate_up: prepare_for_fused_swiglu, then ShardTensorToMesh on dim -1) from that layer's
// separate w1 (gate) and w3 (up) shards, page for page:
//
//     packed page (r * 2C + 2c + 0) = w1 page (r * C + c)
//     packed page (r * 2C + 2c + 1) = w3 page (r * C + c)
//
// C = the N tiles of one projection (272 at TP2), r = the K tile row. A bfloat4_b tile is one
// self-contained page (512 mantissa bytes, 64 shared-exponent bytes), so the copy moves whole
// pages and changes no byte: the fused op then reads exactly the bytes it reads from the served
// w_gate_up (packed_weight_check.cpp checks this equality with the inverse index map, and run
// 33996306217 found it on all 64 layers x 2 chips of the real model). Full-page DRAM accesses only.
//
// Worker = (core, RISC): a contiguous range of pair indices [start, start + count) (the per-core
// runtime args). Per batch: read up to `batch` pairs (2 pages each) into L1, one read barrier,
// write them to their packed pages, wait for the writes to leave L1 before the buffer is reused,
// and one write barrier before the kernel returns.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#ifndef C1E_SRC_SHA
#error "mlp_c1e_pack.py passes C1E_SRC_SHA (a stale JIT binary must never be reused)"
#endif

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t columns = get_compile_time_arg_val(1);
    constexpr uint32_t batch = get_compile_time_arg_val(2);
    constexpr uint32_t cb_index = get_compile_time_arg_val(3);
    constexpr auto gate_args = TensorAccessorArgs<4>();
    constexpr auto up_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();
    constexpr auto packed_args = TensorAccessorArgs<up_args.next_compile_time_args_offset()>();

    const auto gate = TensorAccessor(gate_args, get_common_arg_val<uint32_t>(0), page_bytes);
    const auto up = TensorAccessor(up_args, get_common_arg_val<uint32_t>(1), page_bytes);
    const auto packed = TensorAccessor(packed_args, get_common_arg_val<uint32_t>(2), page_bytes);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    // 64-byte aligned staging (the CB is sized with 64 bytes of slack for this).
    const uint32_t staging = (get_write_ptr(cb_index) + 63) & ~63u;

    const uint32_t end = start + count;
    for (uint32_t pair = start; pair < end; pair += batch) {
        const uint32_t n = (end - pair) < batch ? (end - pair) : batch;
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read_tile(pair + i, gate, staging + (2 * i) * page_bytes);
            noc_async_read_tile(pair + i, up, staging + (2 * i + 1) * page_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t source = pair + i;
            const uint32_t target = (source / columns) * (2 * columns) + (source % columns) * 2;
            noc_async_write_tile(target, packed, staging + (2 * i) * page_bytes);
            noc_async_write_tile(target + 1, packed, staging + (2 * i + 1) * page_bytes);
        }
        noc_async_writes_flushed();  // the staging pages have left L1: the next batch may overwrite them
    }
    noc_async_write_barrier();
}
