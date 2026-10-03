// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// Writer for the fused GDN decode output norm + gate: per instance (bt, h, ct), the one output
// tile goes to page bt*(H*Vt) + h*Vt + ct of the [1,B,H*V] TILE output -- a full-page write.
//
// Compile args: {Vt, H} + accessor args (out). Runtime args: {inst_start, n_inst, out}.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 13;

void kernel_main() {
    constexpr uint32_t Vt = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr auto out_a = TensorAccessorArgs<2>();

    const uint32_t inst_start = get_arg_val<uint32_t>(0);
    const uint32_t n_inst = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);

    const uint32_t tb = get_tile_size(cb_out);
    const auto out_acc = TensorAccessor(out_a, out_addr, tb);

    Noc noc;
    for (uint32_t inst = inst_start; inst < inst_start + n_inst; ++inst) {
        const uint32_t ct = inst % Vt;
        const uint32_t hh = (inst / Vt) % H;
        const uint32_t bt = inst / (Vt * H);
        CircularBuffer cb(cb_out);
        cb.wait_front(1);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
        noc.async_write(src, out_acc, tb, {.offset_bytes = 0}, {.page_id = bt * (H * Vt) + hh * Vt + ct});
        noc.async_write_barrier();
        cb.pop_front(1);
    }
}
