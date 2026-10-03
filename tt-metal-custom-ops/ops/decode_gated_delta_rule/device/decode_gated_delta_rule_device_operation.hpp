// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"

#include "decode_gated_delta_rule_device_operation_types.hpp"
#include "decode_gated_delta_rule_program_factory.hpp"

namespace ttnn::prim {

// Device operation returning two tensors: {o [B,1,H,V], new_state [B,H,K,V]}.
// new_state is the caller's initial_state tensor itself when inplace_state.
struct DecodeGatedDeltaRuleDeviceOperation {
    using operation_attributes_t = DecodeGatedDeltaRuleParams;
    using tensor_args_t = DecodeGatedDeltaRuleInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<DecodeGatedDeltaRuleProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// Low-level primitive dispatch.
std::vector<Tensor> decode_gated_delta_rule(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& beta,
    const Tensor& g,
    const std::optional<Tensor>& initial_state,
    bool inplace_state,
    float scale,
    const tt::tt_metal::MemoryConfig& output_mem_config);

// Packed-layout dispatch: qkv [1,B,C] (= q|k|v head-major), beta/g [1,B,H].
std::vector<Tensor> decode_gated_delta_rule_packed(
    const Tensor& qkv,
    const Tensor& beta,
    const Tensor& g,
    const std::optional<Tensor>& initial_state,
    bool inplace_state,
    float scale,
    uint32_t num_k_heads,
    uint32_t num_v_heads,
    uint32_t head_k,
    uint32_t head_v,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<Tensor>& z = std::nullopt,
    const std::optional<Tensor>& norm_w = std::nullopt,
    uint32_t z_col_offset = 0,
    float epsilon = 1e-6f);

}  // namespace ttnn::prim
