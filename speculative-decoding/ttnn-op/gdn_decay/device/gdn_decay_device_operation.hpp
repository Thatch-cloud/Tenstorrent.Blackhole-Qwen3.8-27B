// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "ttnn/operations/transformer/gdn_decay/device/gdn_decay_device_operation_types.hpp"
#include "ttnn/operations/transformer/gdn_decay/device/gdn_decay_program_factory.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct GdnDecayDeviceOperation {
    using operation_attributes_t = GdnDecayParams;
    using tensor_args_t = GdnDecayInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<GdnDecayProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> gdn_decay(
    const Tensor& h,
    const Tensor& g,
    const Tensor& k,
    const Tensor& v,
    const Tensor& beta,
    const Tensor& q,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
