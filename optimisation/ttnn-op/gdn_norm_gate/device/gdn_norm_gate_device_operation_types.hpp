// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// Types for the fused GDN decode output norm + gate op (K's third kernel).

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct GdnNormGateParams {
    uint32_t B;         // active batch rows
    uint32_t H;         // heads
    uint32_t V;         // head dim (multiple of 32)
    uint32_t Bz;        // z's logical rows (>= B)
    uint32_t W;         // z's logical width (>= z_off + H*V)
    uint32_t z_off;     // column of z where head 0 starts (multiple of 32)
    float epsilon;      // rms_norm epsilon
    tt::tt_metal::MemoryConfig output_mem_config;
};

// o is ROW_MAJOR: page bh = head (b,h)'s [V] stick, exactly as decode_gated_delta_rule writes
// it. z and norm_w are TILE. The output is TILE [1,B,H*V] in z's dtype.
struct GdnNormGateInputs {
    Tensor o;       // [B,1,H,V] ROW_MAJOR
    Tensor z;       // [1,Bz,W] TILE
    Tensor norm_w;  // [1,1,V]  TILE
};

}  // namespace ttnn::prim
