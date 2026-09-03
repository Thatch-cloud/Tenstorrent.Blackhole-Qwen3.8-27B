// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#include "gdn_conv_gates.hpp"

#include "device/gdn_conv_gates_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> gdn_decode_conv_gates(
    const ttnn::Tensor& x,
    const std::vector<ttnn::Tensor>& conv_states,
    const std::vector<ttnn::Tensor>& taps,
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const ttnn::Tensor& dt_bias,
    const ttnn::Tensor& neg_exp_A,
    std::optional<uint32_t> batch,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    const uint32_t B = batch.value_or(x.logical_shape()[1]);
    auto results = ttnn::prim::gdn_conv_gates(
        x, conv_states, taps, a, b, dt_bias, neg_exp_A, B, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));
    return {results[0], results[1], results[2]};
}

}  // namespace ttnn::transformer
