// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#include "gdn_norm_gate_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

GdnNormGateDeviceOperation::program_factory_t GdnNormGateDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return GdnNormGateProgramFactory{};
}

void GdnNormGateDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    TT_FATAL(in.o.layout() == Layout::ROW_MAJOR, "gdn_decode_norm_gate: o must be ROW_MAJOR (the recurrence op's o)");
    TT_FATAL(in.z.layout() == Layout::TILE && in.norm_w.layout() == Layout::TILE, "gdn_decode_norm_gate: z and norm_w must be TILE");
    for (const Tensor* t : {&in.o, &in.z, &in.norm_w}) {
        TT_FATAL(t->buffer() != nullptr, "gdn_decode_norm_gate: inputs must be on device");
        TT_FATAL(
            t->dtype() == DataType::BFLOAT16 || t->dtype() == DataType::FLOAT32,
            "gdn_decode_norm_gate: dtypes must be bf16 or fp32");
    }
    TT_FATAL(attrs.V % TILE_WIDTH == 0, "gdn_decode_norm_gate: V must be a multiple of 32");
    TT_FATAL(
        in.o.dtype() == DataType::BFLOAT16,
        "gdn_decode_norm_gate: o must be bf16 (the plain tilize path does not carry fp32 row-major input)");
    TT_FATAL(attrs.z_off % TILE_WIDTH == 0, "gdn_decode_norm_gate: z_col_offset must be a multiple of 32");
    const auto& os = in.o.logical_shape();
    TT_FATAL(os[-1] == attrs.V, "gdn_decode_norm_gate: o's last dim must be V");
    TT_FATAL(
        in.o.logical_volume() / attrs.V >= attrs.B * attrs.H,
        "gdn_decode_norm_gate: o must hold at least B*H sticks");
    const auto& zs = in.z.logical_shape();
    TT_FATAL(
        zs.rank() == 3 && zs[0] == 1 && zs[1] == attrs.Bz && zs[2] == attrs.W && attrs.Bz >= attrs.B &&
            attrs.W >= attrs.z_off + attrs.H * attrs.V,
        "gdn_decode_norm_gate: z must be [1,Bz,W] with Bz >= B and W >= z_col_offset + H*V");
    const auto& ws = in.norm_w.logical_shape();
    TT_FATAL(ws[-1] == attrs.V && in.norm_w.logical_volume() == attrs.V, "gdn_decode_norm_gate: norm_w must be [1,1,V]");
}

GdnNormGateDeviceOperation::spec_return_value_t GdnNormGateDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const auto layout = TensorLayout(in.z.dtype(), PageConfig(Layout::TILE), attrs.output_mem_config);
    return {TensorSpec(ttnn::Shape({1, attrs.B, attrs.H * attrs.V}), layout)};
}

GdnNormGateDeviceOperation::tensor_return_value_t GdnNormGateDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    return {create_device_tensor(specs[0], in.o.device())};
}

std::vector<Tensor> gdn_norm_gate(
    const Tensor& o,
    const Tensor& z,
    const Tensor& norm_w,
    uint32_t batch,
    uint32_t num_heads,
    uint32_t z_col_offset,
    float epsilon,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = GdnNormGateDeviceOperation;
    const auto& os = o.logical_shape();
    const auto& zs = z.logical_shape();
    TT_FATAL(zs.rank() == 3, "gdn_decode_norm_gate: z must be rank 3 [1,Bz,W]");
    auto attrs = OperationType::operation_attributes_t{
        .B = batch,
        .H = num_heads,
        .V = static_cast<uint32_t>(os[-1]),
        .Bz = static_cast<uint32_t>(zs[1]),
        .W = static_cast<uint32_t>(zs[2]),
        .z_off = z_col_offset,
        .epsilon = epsilon,
        .output_mem_config = output_mem_config,
    };
    auto tensor_args = OperationType::tensor_args_t{.o = o, .z = z, .norm_w = norm_w};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
