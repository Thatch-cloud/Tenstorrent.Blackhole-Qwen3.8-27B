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
    // Direct read of the fused projection output (milestone 4): x may be wider than C (pages
    // index by Wt = its real width in tiles) and a / b may be column windows of a wider
    // tensor at any element offset -- the gate reader gathers them per element.
    uint32_t Wt = 0;      // x's width in tiles (== C/32 when x is exactly the conv channels)
    bool gates_packed = false;
    uint32_t AWt = 0;     // a's width in tiles
    uint32_t a_col = 0;   // element column of head 0 in a
    uint32_t BWt = 0;     // b's width in tiles
    uint32_t b_col = 0;   // element column of head 0 in b
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
