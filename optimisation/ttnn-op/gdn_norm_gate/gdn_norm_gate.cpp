// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#include "gdn_norm_gate.hpp"

#include "device/gdn_norm_gate_device_operation.hpp"

namespace ttnn::transformer {

ttnn::Tensor gdn_decode_norm_gate(
    const ttnn::Tensor& o,
    const ttnn::Tensor& z,
    const ttnn::Tensor& norm_w,
    uint32_t num_heads,
    std::optional<uint32_t> batch,
    uint32_t z_col_offset,
    float epsilon,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    const auto& os = o.logical_shape();
    const uint32_t V = os[-1];
    const uint32_t sticks = o.logical_volume() / V;
    const uint32_t B = batch.value_or(sticks / num_heads);
    auto results = ttnn::prim::gdn_norm_gate(
        o, z, norm_w, B, num_heads, z_col_offset, epsilon, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));
    return results[0];
}

}  // namespace ttnn::transformer
