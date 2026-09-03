// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"

#include "gdn_conv_gates_device_operation_types.hpp"
#include "gdn_conv_gates_program_factory.hpp"

namespace ttnn::prim {

// Returns {conv_out [1,B,C], beta [1,B,Nv], g [1,B,Nv]}; conv_states are updated in place.
struct GdnConvGatesDeviceOperation {
    using operation_attributes_t = GdnConvGatesParams;
    using tensor_args_t = GdnConvGatesInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<GdnConvGatesProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> gdn_conv_gates(
    const Tensor& x,
    const std::vector<Tensor>& conv_states,
    const std::vector<Tensor>& taps,
    const Tensor& a,
    const Tensor& b,
    const Tensor& dt_bias,
    const Tensor& neg_exp_A,
    uint32_t batch,
    const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::prim
