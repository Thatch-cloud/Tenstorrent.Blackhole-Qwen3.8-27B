// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"

#include "attn_prep_device_operation_types.hpp"
#include "attn_prep_program_factory.hpp"

namespace ttnn::prim {

// Returns {q [1,B,NH,HD], gate [1,B,NH,HD], k [1,B,32,HD], v [1,B,32,HD]}.
struct AttnPrepDeviceOperation {
    using operation_attributes_t = AttnPrepParams;
    using tensor_args_t = AttnPrepInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<AttnPrepProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> attn_prep(
    const Tensor& qkv,
    const Tensor& cos,
    const Tensor& sin,
    const Tensor& q_norm_w,
    const Tensor& k_norm_w,
    uint32_t batch,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t rope_dim,
    float epsilon,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::MemoryConfig& kv_mem_config);

}  // namespace ttnn::prim
