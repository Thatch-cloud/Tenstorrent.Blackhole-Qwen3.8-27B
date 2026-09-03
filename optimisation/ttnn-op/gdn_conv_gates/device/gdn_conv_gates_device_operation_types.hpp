// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// Types for the fused GDN decode conv + gates op (K's first kernel).

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct GdnConvGatesParams {
    uint32_t B;     // active rows of x (rows >= B enter the shift register as zeros)
    uint32_t Bx;    // x's logical row count (>= B)
    uint32_t Bmax;  // rows of each conv state
    uint32_t C;     // channels (multiple of 32)
    uint32_t Nv;    // value heads on this device (gate width)
    uint32_t K;     // conv taps (== conv_states.size() == taps.size())
    tt::tt_metal::MemoryConfig output_mem_config;
};

// All TILE, one dtype (bf16 or fp32), on device. conv_states are read AND written by the
// program (shift-register advance); the caller keeps the same buffers, which is what the
// decode trace's fixed-address discipline needs.
struct GdnConvGatesInputs {
    Tensor x;                         // [1,Bx,C]
    std::vector<Tensor> conv_states;  // K x [1,Bmax,C]
    std::vector<Tensor> taps;         // K x [1,1,C]
    Tensor a;                         // [1,B,Nv]
    Tensor b;                         // [1,B,Nv]
    Tensor dt_bias;                   // [1,1,Nv]
    Tensor neg_exp_A;                 // [1,1,Nv]
};

}  // namespace ttnn::prim
