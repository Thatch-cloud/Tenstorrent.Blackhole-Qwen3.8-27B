// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::transformer {

/**
 * Fused GDN decode conv + gates: the first kernel of "K" (docs/optimisation-plan.md §4).
 *
 * Replaces, per layer per decode step, the conv shift-register (K copies), the K-tap
 * FIR + SiLU (multiply, K-1 mac, silu) and the two gates (sigmoid; add+softplus, multiply)
 * with one program:
 *
 *   window   = [conv_states[1], ..., conv_states[K-1], x]      (shift-register advanced)
 *   conv_out = silu( sum_j window[j] * taps[j] )               (taps broadcast over batch)
 *   beta     = sigmoid(b)
 *   g        = neg_exp_A * softplus(a + dt_bias)               (softplus beta=1, threshold=20)
 *   conv_states[j] <- window[j]                                 (written IN PLACE)
 *
 *   x            [1, B, C]      TILE bf16/fp32, the new qkv projection (B active rows)
 *   conv_states  K x [1, Bmax, C]  TILE, same dtype; persistent, updated in place
 *   taps         K x [1, 1, C]     TILE, same dtype (row 0 holds the tap)
 *   a, b         [1, B, Nv]     TILE
 *   dt_bias, neg_exp_A  [1, 1, Nv]  TILE
 *
 * Returns conv_out [1, B, C], beta [1, B, Nv], g [1, B, Nv] (all TILE, x's dtype).
 *
 * Direct read (milestone 4): x may be the whole fused projection output [1, B, W] with
 * `channels` = C (the conv reads its first C columns), and a / b may be that same tensor
 * with `a_col` / `b_col` naming the element column where head 0 sits -- no slices needed.
 *
 * Rows of x at or beyond `batch` (default: x's row count) are treated as zero when they
 * enter the shift register, which is what bucketed decode (B < Bmax) needs; idle rows'
 * conv_out is don't-care. One Tensix core per (batch-tile, channel-tile); the gates run
 * on one additional core. Every DRAM access is a full tile page.
 */
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> gdn_decode_conv_gates(
    const ttnn::Tensor& x,
    const std::vector<ttnn::Tensor>& conv_states,
    const std::vector<ttnn::Tensor>& taps,
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const ttnn::Tensor& dt_bias,
    const ttnn::Tensor& neg_exp_A,
    std::optional<uint32_t> batch = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    std::optional<uint32_t> channels = std::nullopt,
    uint32_t a_col = 0,
    uint32_t b_col = 0);

}  // namespace ttnn::transformer
