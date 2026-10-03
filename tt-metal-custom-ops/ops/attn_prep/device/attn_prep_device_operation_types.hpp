// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct AttnPrepParams {
    uint32_t B;    // active batch rows
    uint32_t NH;   // query heads (per device)
    uint32_t NKV;  // kv heads (per device)
    uint32_t HD;   // head dim (multiple of 32)
    uint32_t RD;   // rotary dims (multiple of 64: two tile-aligned halves)
    uint32_t Wt;   // qkv width in tiles
    uint32_t eps_bits;    // fp32 bits of HD*epsilon
    uint32_t scale_bits;  // fp32 bits of sqrt(HD)
    tt::tt_metal::MemoryConfig output_mem_config;  // q, gate
    tt::tt_metal::MemoryConfig kv_mem_config;      // k, v (height-sharded for the KV update)
};

struct AttnPrepInputs {
    Tensor qkv;       // [1,1,B,W] TILE
    Tensor cos;       // [1,B,1,RD] TILE
    Tensor sin;       // [1,B,1,RD] TILE
    Tensor q_norm_w;  // [1,1,HD] TILE
    Tensor k_norm_w;  // [1,1,HD] TILE
};

}  // namespace ttnn::prim
