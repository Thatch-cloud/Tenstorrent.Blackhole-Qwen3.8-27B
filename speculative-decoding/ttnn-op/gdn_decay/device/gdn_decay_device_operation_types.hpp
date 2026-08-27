// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

struct GdnDecayParams {
    uint32_t Kt = 0;  // state rows in tiles
    uint32_t Vt = 0;  // state cols in tiles
    uint32_t Mt = 0;  // token rows in tiles (1)
    uint32_t T = 0;   // tokens per dispatch
    uint32_t BH = 0;  // batch*heads, the parallel axis (one head per core)
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct GdnDecayInputs {
    Tensor h;
    Tensor g;
    Tensor k;
    Tensor v;
    Tensor beta;
    Tensor q;
};

}  // namespace ttnn::prim
