// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// Writer for the fused GDN decode conv + gates. Device 2.0 API.
//
// Per conv instance (page p): store conv_out page p, then the advanced shift-register --
// the window tiles the compute passed through -- into conv_states[0..K-1] page p, IN PLACE.
// This core's reader consumed those same pages before the compute could produce the
// window, and no other core touches page p, so the in-place write is ordered by the CB
// dependency alone. Gate instances store beta and g page gi. All writes are full tile
// pages.
//
// Compile args: {K, Ct, Nvt} + accessor args (conv_out, st0..st3, beta, g).
// Runtime args: {inst_start, n_inst, g_n, conv_out, st0..st3, beta, g}.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 4, cb_shift = 5, cb_beta = 11, cb_g = 12;

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    static_assert(K == 4, "writer is written for K == 4 taps");

    constexpr auto o_a = TensorAccessorArgs<3>();
    constexpr auto s0_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();
    constexpr auto s1_a = TensorAccessorArgs<s0_a.next_compile_time_args_offset()>();
    constexpr auto s2_a = TensorAccessorArgs<s1_a.next_compile_time_args_offset()>();
    constexpr auto s3_a = TensorAccessorArgs<s2_a.next_compile_time_args_offset()>();
    constexpr auto beta_a = TensorAccessorArgs<s3_a.next_compile_time_args_offset()>();
    constexpr auto g_a = TensorAccessorArgs<beta_a.next_compile_time_args_offset()>();

    const uint32_t inst_start = get_arg_val<uint32_t>(0);
    const uint32_t n_inst = get_arg_val<uint32_t>(1);
    const uint32_t g_n = get_arg_val<uint32_t>(2);
    const uint32_t o_addr = get_arg_val<uint32_t>(3);
    const uint32_t s0_addr = get_arg_val<uint32_t>(4);
    const uint32_t s1_addr = get_arg_val<uint32_t>(5);
    const uint32_t s2_addr = get_arg_val<uint32_t>(6);
    const uint32_t s3_addr = get_arg_val<uint32_t>(7);
    const uint32_t beta_addr = get_arg_val<uint32_t>(8);
    const uint32_t g_addr = get_arg_val<uint32_t>(9);

    const uint32_t tb = get_tile_size(cb_out);
    const auto o_acc = TensorAccessor(o_a, o_addr, tb);
    const auto s0_acc = TensorAccessor(s0_a, s0_addr, tb);
    const auto s1_acc = TensorAccessor(s1_a, s1_addr, tb);
    const auto s2_acc = TensorAccessor(s2_a, s2_addr, tb);
    const auto s3_acc = TensorAccessor(s3_a, s3_addr, tb);
    const auto beta_acc = TensorAccessor(beta_a, beta_addr, tb);
    const auto g_acc = TensorAccessor(g_a, g_addr, tb);

    Noc noc;

    for (uint32_t inst = inst_start; inst < inst_start + n_inst; ++inst) {
        {
            CircularBuffer cb(cb_out);
            cb.wait_front(1);
            auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
            noc.async_write(src, o_acc, tb, {.offset_bytes = 0}, {.page_id = inst});
            noc.async_write_barrier();
            cb.pop_front(1);
        }
        {
            // window slots 0..3 = new st0..st3 (= old st1, st2, st3, x-masked)
            CircularBuffer cb(cb_shift);
            cb.wait_front(K);
            auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
            noc.async_write(src, s0_acc, tb, {.offset_bytes = 0 * tb}, {.page_id = inst});
            noc.async_write(src, s1_acc, tb, {.offset_bytes = 1 * tb}, {.page_id = inst});
            noc.async_write(src, s2_acc, tb, {.offset_bytes = 2 * tb}, {.page_id = inst});
            noc.async_write(src, s3_acc, tb, {.offset_bytes = 3 * tb}, {.page_id = inst});
            noc.async_write_barrier();
            cb.pop_front(K);
        }
    }

    for (uint32_t gi = 0; gi < g_n; ++gi) {
        {
            CircularBuffer cb(cb_beta);
            cb.wait_front(1);
            auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
            noc.async_write(src, beta_acc, tb, {.offset_bytes = 0}, {.page_id = gi});
            noc.async_write_barrier();
            cb.pop_front(1);
        }
        {
            CircularBuffer cb(cb_g);
            cb.wait_front(1);
            auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
            noc.async_write(src, g_acc, tb, {.offset_bytes = 0}, {.page_id = gi});
            noc.async_write_barrier();
            cb.pop_front(1);
        }
    }
}
