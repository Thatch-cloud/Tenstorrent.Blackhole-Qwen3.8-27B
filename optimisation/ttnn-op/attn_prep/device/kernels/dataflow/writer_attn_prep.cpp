// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// Writer for the fused attention decode prologue: per instance (b, kind), the HDt finished tiles
// go to pages b*HDt + t of the kind's output -- q and gate interleaved [1,B,NH,HD], k and v in the
// KV update's height-sharded config [1,B,32,HD]; the accessor resolves either. Full-page writes.
//
// Compile args: {HDt} + accessors (q, gate, k, v). Runtime args: {inst_start, n_inst, q, gate, k, v}.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 2;

void kernel_main() {
    constexpr uint32_t HDt = get_compile_time_arg_val(0);
    constexpr auto q_a = TensorAccessorArgs<1>();
    constexpr auto g_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto k_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();

    const uint32_t inst_start = get_arg_val<uint32_t>(0);
    const uint32_t n_inst = get_arg_val<uint32_t>(1);
    const uint32_t q_addr = get_arg_val<uint32_t>(2);
    const uint32_t g_addr = get_arg_val<uint32_t>(3);
    const uint32_t k_addr = get_arg_val<uint32_t>(4);
    const uint32_t v_addr = get_arg_val<uint32_t>(5);

    const uint32_t tb = get_tile_size(cb_out);
    const auto q_acc = TensorAccessor(q_a, q_addr, tb);
    const auto g_acc = TensorAccessor(g_a, g_addr, tb);
    const auto k_acc = TensorAccessor(k_a, k_addr, tb);
    const auto v_acc = TensorAccessor(v_a, v_addr, tb);

    Noc noc;
    for (uint32_t inst = inst_start; inst < inst_start + n_inst; ++inst) {
        const uint32_t b = inst / 4;
        const uint32_t kind = inst % 4;
        CircularBuffer cb(cb_out);
        cb.wait_front(HDt);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
        for (uint32_t t = 0; t < HDt; t++) {
            const uint32_t page = b * HDt + t;
            if (kind == 0) {
                noc.async_write(src, q_acc, tb, {.offset_bytes = t * tb}, {.page_id = page});
            } else if (kind == 1) {
                noc.async_write(src, k_acc, tb, {.offset_bytes = t * tb}, {.page_id = page});
            } else if (kind == 2) {
                noc.async_write(src, v_acc, tb, {.offset_bytes = t * tb}, {.page_id = page});
            } else {
                noc.async_write(src, g_acc, tb, {.offset_bytes = t * tb}, {.page_id = page});
            }
        }
        noc.async_write_barrier();
        cb.pop_front(HDt);
    }
}
