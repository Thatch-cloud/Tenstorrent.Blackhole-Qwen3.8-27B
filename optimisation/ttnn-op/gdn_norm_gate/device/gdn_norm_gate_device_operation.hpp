// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"

#include "gdn_norm_gate_device_operation_types.hpp"
#include "gdn_norm_gate_program_factory.hpp"

namespace ttnn::prim {

// Returns {gated [1,B,H*V] TILE}.
struct GdnNormGateDeviceOperation {
    using operation_attributes_t = GdnNormGateParams;
    using tensor_args_t = GdnNormGateInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<GdnNormGateProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> gdn_norm_gate(
    const Tensor& o,
    const Tensor& z,
    const Tensor& norm_w,
    uint32_t batch,
    uint32_t num_heads,
    uint32_t z_col_offset,
    float epsilon,
    const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::prim
