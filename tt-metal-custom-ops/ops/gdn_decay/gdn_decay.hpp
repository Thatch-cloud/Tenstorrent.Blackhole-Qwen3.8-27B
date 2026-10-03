// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::transformer {

/**
 * One step of the Gated DeltaNet recurrent decode, fused:
 *
 *   h1     = h * exp(g)
 *   v_read = k @ h1
 *   delta  = (v - v_read) * beta
 *   h_new  = h1 + k^T @ delta
 *   o      = q @ h_new
 *
 * All tensors float32, TILE_LAYOUT, DRAM. T tokens per dispatch, BH = batch*heads folded
 * into dim 0 and mapped one head per core.
 *   h          [BH, 1, K, V]    initial state
 *   g          [BH, T, 1, 1]    per-head-per-token scalar; exp() is applied on device
 *   k, q       [BH, T, M, K]    M = 32 (one token row, tile-padded)
 *   v          [BH, T, M, V]
 *   beta       [BH, T, 1, 1]    per-head-per-token scalar, broadcast on device
 * Returns (o [BH,T,M,V], states [BH,T,K,V]) -- every intermediate state, for speculative
 * rollback. At BH=1 the contract is byte-identical to the single-head version.
 */
std::tuple<ttnn::Tensor, ttnn::Tensor> gdn_decay(
    const ttnn::Tensor& h,
    const ttnn::Tensor& g,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& q,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::transformer
