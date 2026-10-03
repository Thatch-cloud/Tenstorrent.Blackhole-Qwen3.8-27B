// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/gdn_decay/gdn_decay.hpp"
#include "ttnn/operations/transformer/gdn_decay/device/gdn_decay_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/device.hpp"

namespace ttnn::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor> gdn_decay(
    const ttnn::Tensor& h,
    const ttnn::Tensor& g,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& q,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    auto mc = memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
    auto kc = init_device_compute_kernel_config(
        h.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, false, true, false);
    auto r = ttnn::prim::gdn_decay(h, g, k, v, beta, q, mc, kc);
    return {r[0], r[1]};
}

}  // namespace ttnn::transformer
