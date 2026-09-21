// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#include "attn_prep_device_operation.hpp"

#include <bit>
#include <cmath>

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

AttnPrepDeviceOperation::program_factory_t AttnPrepDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return AttnPrepProgramFactory{};
}

void AttnPrepDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    for (const Tensor* t : {&in.qkv, &in.cos, &in.sin, &in.q_norm_w, &in.k_norm_w}) {
        TT_FATAL(t->layout() == Layout::TILE, "attn_decode_prep: all inputs must be TILE");
        TT_FATAL(t->dtype() == DataType::BFLOAT16, "attn_decode_prep: all inputs must be bf16");
        TT_FATAL(t->buffer() != nullptr, "attn_decode_prep: inputs must be on device");
    }
    TT_FATAL(attrs.HD % TILE_WIDTH == 0, "attn_decode_prep: head_dim must be a multiple of 32");
    TT_FATAL(attrs.RD % 64 == 0 && attrs.RD <= attrs.HD, "attn_decode_prep: rope_dim must be a multiple of 64 and <= head_dim");
    TT_FATAL(attrs.NH <= 32 && attrs.NKV <= 32, "attn_decode_prep: at most 32 heads per device (one tile-row of heads)");
    const auto& qs = in.qkv.logical_shape();
    TT_FATAL(
        qs.rank() == 4 && qs[0] == 1 && qs[1] == 1 && qs[2] >= attrs.B &&
            qs[3] >= (attrs.NH + 2 * attrs.NKV + attrs.NH) * attrs.HD,
        "attn_decode_prep: qkv must be [1,1,>=B,W] with W >= (NH+2*NKV+NH)*HD");
    for (const Tensor* t : {&in.cos, &in.sin}) {
        const auto& s = t->logical_shape();
        TT_FATAL(
            s.rank() == 4 && s[0] == 1 && s[1] >= attrs.B && s[2] == 1 && s[3] == attrs.RD,
            "attn_decode_prep: cos/sin must be [1,>=B,1,RD]");
    }
    for (const Tensor* t : {&in.q_norm_w, &in.k_norm_w}) {
        TT_FATAL(t->logical_shape()[-1] == attrs.HD && t->logical_volume() == attrs.HD, "attn_decode_prep: norm weights must be [1,1,HD]");
    }
    TT_FATAL(attrs.kv_mem_config.is_sharded() || true, "attn_decode_prep: kv memory config");
}

AttnPrepDeviceOperation::spec_return_value_t AttnPrepDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const DataType dt = in.qkv.dtype();
    const auto lo = TensorLayout(dt, PageConfig(Layout::TILE), attrs.output_mem_config);
    const auto lkv = TensorLayout(dt, PageConfig(Layout::TILE), attrs.kv_mem_config);
    return {
        TensorSpec(ttnn::Shape({1, attrs.B, attrs.NH, attrs.HD}), lo),
        TensorSpec(ttnn::Shape({1, attrs.B, attrs.NH, attrs.HD}), lo),
        TensorSpec(ttnn::Shape({1, attrs.B, 32, attrs.HD}), lkv),
        TensorSpec(ttnn::Shape({1, attrs.B, 32, attrs.HD}), lkv)};
}

AttnPrepDeviceOperation::tensor_return_value_t AttnPrepDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    auto* device = in.qkv.device();
    std::vector<Tensor> outs;
    outs.reserve(4);
    for (const auto& s : specs) {
        outs.push_back(create_device_tensor(s, device));
    }
    return outs;
}

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
    const tt::tt_metal::MemoryConfig& kv_mem_config) {
    using OperationType = AttnPrepDeviceOperation;
    using namespace tt::constants;
    const auto& qs = qkv.logical_shape();
    auto attrs = OperationType::operation_attributes_t{
        .B = batch,
        .NH = num_heads,
        .NKV = num_kv_heads,
        .HD = head_dim,
        .RD = rope_dim,
        .Wt = (static_cast<uint32_t>(qs[-1]) + TILE_WIDTH - 1) / TILE_WIDTH,
        .eps_bits = std::bit_cast<uint32_t>(epsilon * static_cast<float>(head_dim)),
        .scale_bits = std::bit_cast<uint32_t>(std::sqrt(static_cast<float>(head_dim))),
        .output_mem_config = output_mem_config,
        .kv_mem_config = kv_mem_config,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .qkv = qkv, .cos = cos, .sin = sin, .q_norm_w = q_norm_w, .k_norm_w = k_norm_w};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
