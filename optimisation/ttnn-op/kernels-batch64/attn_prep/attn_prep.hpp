// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::transformer {

/**
 * Fused attention decode prologue (docs/optimisation-plan.md, the attention layer's small ops).
 *
 * From the fused [q | k | v | gate] projection output, per decode step and layer, produces in one
 * program everything the paged SDPA-decode and the paged KV update consume:
 *
 *   q     [1, B, NH, HD]  TILE = rope(rms_norm(q_h) * q_norm_w)     (rotate-half on the first RD dims)
 *   gate  [1, B, NH, HD]  TILE = the gate block, head-split
 *   k     [1, B, 32, HD]  TILE = rope(rms_norm(k_h) * k_norm_w), heads padded to 32, in kv_memory_config
 *   v     [1, B, 32, HD]  TILE = v head-split, heads padded to 32, in kv_memory_config
 *
 * replacing the two projection slices, the head-split op and its three reshards, the two
 * rms_norms and weight multiplies, the two partial-RoPE sequences and the two KV pad+reshards.
 *
 *   qkv       [1, 1, B, W]  TILE bf16, W >= (NH + 2*NKV + NH) * HD, columns q | k | v | gate head-major
 *   cos, sin  [1, B, 1, RD] TILE bf16 (HF layout: the two halves duplicated)
 *   q_norm_w, k_norm_w  [1, 1, HD] TILE bf16 (the +1 already folded in)
 *
 * Every DRAM access is a full tile page: the reader gathers head h's row b out of the shared
 * projection tiles into row h of a 32-row block (rows >= the head count stay zero), and the
 * writer stores whole tiles; the KV outputs' height-sharded layout is addressed through the
 * tensor accessor. One core per (batch row, output kind).
 */
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> attn_decode_prep(
    const ttnn::Tensor& qkv,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& q_norm_w,
    const ttnn::Tensor& k_norm_w,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t rope_dim,
    const ttnn::MemoryConfig& kv_memory_config,
    std::optional<uint32_t> batch = std::nullopt,
    float epsilon = 1e-6f,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::transformer
