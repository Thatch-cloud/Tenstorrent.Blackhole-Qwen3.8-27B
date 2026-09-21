// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#include "attn_prep.hpp"

#include "device/attn_prep_device_operation.hpp"

namespace ttnn::transformer {

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
    std::optional<uint32_t> batch,
    float epsilon,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    const uint32_t B = batch.value_or(qkv.logical_shape()[-2]);
    auto r = ttnn::prim::attn_prep(
        qkv,
        cos,
        sin,
        q_norm_w,
        k_norm_w,
        B,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_dim,
        epsilon,
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        kv_memory_config);
    return {r[0], r[1], r[2], r[3]};
}

}  // namespace ttnn::transformer
