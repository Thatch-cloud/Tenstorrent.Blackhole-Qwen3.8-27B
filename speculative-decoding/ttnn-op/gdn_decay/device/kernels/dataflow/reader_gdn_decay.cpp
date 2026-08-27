// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader: the initial state once into c_0, then per-token g/k/v/beta/q streamed for T tokens.
// g and beta are per-head-per-token SCALARS: one tile per token each, not Kt*Vt / Mt*Vt.
//
// Runtime args: 0 h, 1 g, 2 k, 3 v, 4 beta, 5 q, 6 Kt, 7 Vt, 8 Mt, 9 T, 10 head_idx
//
// Every page_id is head-relative. The tensors fold batch*heads into dim 0, so for an
// interleaved TILE tensor [d0,d1,R,C] the page is ((d0*D1 + d1)*Rt + i)*Ct + j and head h's
// slice starts at h * (d1_extent * tiles_per_token). Note the `* T` in the g/k/v bases: drop
// it and the op is still exactly right at T=1 and wrong at T>=2.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t h_addr = get_arg_val<uint32_t>(0);
    const uint32_t g_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_arg_val<uint32_t>(3);
    const uint32_t b_addr = get_arg_val<uint32_t>(4);
    const uint32_t q_addr = get_arg_val<uint32_t>(5);
    const uint32_t Kt = get_arg_val<uint32_t>(6);
    const uint32_t Vt = get_arg_val<uint32_t>(7);
    const uint32_t Mt = get_arg_val<uint32_t>(8);
    const uint32_t T = get_arg_val<uint32_t>(9);
    const uint32_t head_idx = get_arg_val<uint32_t>(10);

    constexpr uint32_t f32 = get_tile_size(tt::CBIndex::c_0);
    constexpr auto a0 = TensorAccessorArgs<0>();
    const auto h_gen = TensorAccessor(a0, h_addr, f32);
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    const auto g_gen = TensorAccessor(a1, g_addr, f32);
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    const auto k_gen = TensorAccessor(a2, k_addr, f32);
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    const auto v_gen = TensorAccessor(a3, v_addr, f32);
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    const auto b_gen = TensorAccessor(a4, b_addr, f32);
    constexpr auto a5 = TensorAccessorArgs<a4.next_compile_time_args_offset()>();
    const auto q_gen = TensorAccessor(a5, q_addr, f32);

    Noc noc;
    CircularBuffer cbh(tt::CBIndex::c_0), cbg(tt::CBIndex::c_1), cbk(tt::CBIndex::c_3);
    CircularBuffer cbv(tt::CBIndex::c_4), cbb(tt::CBIndex::c_5), cbq(tt::CBIndex::c_6);

    const uint32_t st = Kt * Vt, kt = Mt * Kt, vt = Mt * Vt;
    const uint32_t h_base = head_idx * st;       // h    [BH,1,K,V]
    const uint32_t k_base = head_idx * T * kt;   // k,q  [BH,T,M,K]
    const uint32_t v_base = head_idx * T * vt;   // v    [BH,T,M,V]
    const uint32_t s_base = head_idx * T;        // g, beta [BH,T,1,1] -- one tile per token

    cbh.reserve_back(st);
    for (uint32_t t = 0; t < st; t++) {
        noc.async_read(h_gen, cbh, f32, {.page_id = h_base + t}, {.offset_bytes = t * f32});
    }
    noc.async_read_barrier();
    cbh.push_back(st);

    for (uint32_t i = 0; i < T; i++) {
        cbg.reserve_back(1);
        noc.async_read(g_gen, cbg, f32, {.page_id = s_base + i}, {.offset_bytes = 0});
        cbk.reserve_back(kt);
        cbq.reserve_back(kt);
        for (uint32_t t = 0; t < kt; t++) {
            noc.async_read(k_gen, cbk, f32, {.page_id = k_base + i * kt + t}, {.offset_bytes = t * f32});
            noc.async_read(q_gen, cbq, f32, {.page_id = k_base + i * kt + t}, {.offset_bytes = t * f32});
        }
        cbv.reserve_back(vt);
        cbb.reserve_back(1);
        for (uint32_t t = 0; t < vt; t++) {
            noc.async_read(v_gen, cbv, f32, {.page_id = v_base + i * vt + t}, {.offset_bytes = t * f32});
        }
        noc.async_read(b_gen, cbb, f32, {.page_id = s_base + i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cbg.push_back(1);
        cbk.push_back(kt);
        cbq.push_back(kt);
        cbv.push_back(vt);
        cbb.push_back(1);
    }
}
