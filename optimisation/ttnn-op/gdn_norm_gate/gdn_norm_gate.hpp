// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::transformer {

/**
 * Fused GDN decode output norm + gate: the third kernel of "K" (docs/optimisation-plan.md §4).
 *
 * Takes the recurrence op's `o` exactly as it writes it (ROW_MAJOR, one [V] stick per
 * (batch, head)) and produces, per layer per decode step,
 *
 *   gated[b, h*V + j] = rms_norm(o[b,h,:])[j] * norm_w[j] * silu(z[b, z_off + h*V + j])
 *   rms_norm(x) = x / sqrt(mean(x^2) + eps)
 *
 * directly in the [1, B, H*V] TILE layout the out-projection consumes. Replaces the
 * ROW_MAJOR->TILE relayout of o, the [B*H,V]->[B,H*V] reshape, rms_norm, silu and multiply.
 *
 *   o       [B, 1, H, V]  ROW_MAJOR, bf16 or fp32 (page = one head's [V] stick)
 *   z       [1, Bz, W]    TILE (Bz >= B, W >= z_off + H*V); z_off lets z be a column
 *                         window of the fused projection output, so no slice is needed
 *   norm_w  [1, 1, V]     TILE
 *
 * Returns gated [1, B, H*V] TILE in z's dtype. One Tensix core per (batch-tile, head):
 * it reads 32 sticks (full pages), assembles them into a 32-row tile block in L1, and
 * writes V/32 full output tiles. Rows at or beyond B are zero.
 */
ttnn::Tensor gdn_decode_norm_gate(
    const ttnn::Tensor& o,
    const ttnn::Tensor& z,
    const ttnn::Tensor& norm_w,
    uint32_t num_heads,
    std::optional<uint32_t> batch = std::nullopt,
    uint32_t z_col_offset = 0,
    float epsilon = 1e-6f,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::transformer
