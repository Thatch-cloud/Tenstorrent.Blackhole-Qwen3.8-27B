// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer: per token, drain c_17 -> states[t] and c_16 -> o[t].
//
// Runtime args: 0 o_addr, 1 s_addr, 2 o_tiles_per_token, 3 state_tiles_per_token, 4 T, 5 head_idx
//
// Head-relative destinations, matching the reader. If only ONE of the two kernels is made
// head-relative, every core computes head 0 and scatters it to BH distinct destinations:
// well-formed, self-consistent, and wrong for heads 1..BH-1.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t s_addr = get_arg_val<uint32_t>(1);
    const uint32_t ot = get_arg_val<uint32_t>(2);
    const uint32_t sst = get_arg_val<uint32_t>(3);
    const uint32_t T = get_arg_val<uint32_t>(4);
    const uint32_t head_idx = get_arg_val<uint32_t>(5);

    constexpr uint32_t f32 = get_tile_size(tt::CBIndex::c_16);
    constexpr auto oa = TensorAccessorArgs<0>();
    const auto o_gen = TensorAccessor(oa, o_addr, f32);
    constexpr auto sa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto s_gen = TensorAccessor(sa, s_addr, f32);

    Noc noc;
    CircularBuffer cbo(tt::CBIndex::c_16), cbs(tt::CBIndex::c_17);

    const uint32_t o_base = head_idx * T * ot;   // o      [BH,T,M,V]
    const uint32_t s_base = head_idx * T * sst;  // states [BH,T,K,V]

    for (uint32_t i = 0; i < T; i++) {
        cbs.wait_front(sst);
        for (uint32_t t = 0; t < sst; t++) {
            noc.async_write(cbs, s_gen, f32, {.offset_bytes = t * f32}, {.page_id = s_base + i * sst + t});
        }
        noc.async_write_barrier();
        cbs.pop_front(sst);

        cbo.wait_front(ot);
        for (uint32_t t = 0; t < ot; t++) {
            noc.async_write(cbo, o_gen, f32, {.offset_bytes = t * f32}, {.page_id = o_base + i * ot + t});
        }
        noc.async_write_barrier();
        cbo.pop_front(ot);
    }
}
