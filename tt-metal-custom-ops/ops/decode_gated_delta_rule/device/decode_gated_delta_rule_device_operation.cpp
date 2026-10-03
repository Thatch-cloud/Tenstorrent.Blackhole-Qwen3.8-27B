// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "decode_gated_delta_rule_device_operation.hpp"

#include <bit>
#include <cmath>
#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

DecodeGatedDeltaRuleDeviceOperation::program_factory_t DecodeGatedDeltaRuleDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return DecodeGatedDeltaRuleProgramFactory{};
}

void DecodeGatedDeltaRuleDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    const DataType dt = in.q.dtype();
    auto check = [&](const Tensor& t, const char* name) {
        TT_FATAL(t.layout() == Layout::TILE, "decode_gated_delta_rule: {} must be TILE layout", name);
        TT_FATAL(t.dtype() == dt, "decode_gated_delta_rule: all inputs must share one dtype", name);
        TT_FATAL(
            dt == DataType::BFLOAT16 || dt == DataType::FLOAT32, "decode_gated_delta_rule: dtype must be bf16 or fp32");
        TT_FATAL(t.buffer() != nullptr, "decode_gated_delta_rule: {} must be on device", name);
    };
    check(in.q, "q");
    check(in.k, "k");
    check(in.v, "v");
    check(in.beta, "beta");
    check(in.g, "g");
    TT_FATAL(attrs.K % TILE_WIDTH == 0, "decode_gated_delta_rule: K must be a multiple of 32");
    TT_FATAL(attrs.V % TILE_WIDTH == 0, "decode_gated_delta_rule: V must be a multiple of 32");
    if (attrs.packed) {
        const auto& qs = in.q.logical_shape();  // [1,B,C]
        TT_FATAL(
            qs.rank() == 3 && qs[0] == 1 && qs[1] == attrs.B && qs[2] == attrs.Ct * TILE_WIDTH,
            "decode_gated_delta_rule (packed): qkv must be [1,B,C] with C == Ct*32");
        TT_FATAL(
            attrs.v_off_t * TILE_WIDTH + attrs.H * attrs.V == attrs.Ct * TILE_WIDTH,
            "decode_gated_delta_rule (packed): C must equal 2*Nk*K + H*V");
        TT_FATAL(
            attrs.rf >= 1 && attrs.H % attrs.rf == 0, "decode_gated_delta_rule (packed): H must be a multiple of rf");
        const auto& bs = in.beta.logical_shape();
        const auto& gs = in.g.logical_shape();
        TT_FATAL(
            bs.rank() == 3 && bs[0] == 1 && bs[1] == attrs.B && bs[2] == attrs.H && gs == bs,
            "decode_gated_delta_rule (packed): beta and g must be [1,B,H]");
        if (attrs.fuse_ng) {
            TT_FATAL(in.z.has_value() && in.norm_w.has_value(), "decode_gated_delta_rule (fused norm+gate): z and norm_w required");
            check(*in.z, "z");
            check(*in.norm_w, "norm_w");
            const auto& zs = in.z->logical_shape();
            TT_FATAL(
                zs.rank() == 3 && zs[0] == 1 && zs[1] >= attrs.B && zs[2] >= attrs.z_off_t * TILE_WIDTH + attrs.H * attrs.V,
                "decode_gated_delta_rule (fused norm+gate): z must be [1,>=B,W] with W >= z_col_offset + H*V");
            const auto& ws = in.norm_w->logical_shape();
            TT_FATAL(ws[-1] == attrs.V && in.norm_w->logical_volume() == attrs.V, "decode_gated_delta_rule (fused norm+gate): norm_w must be [1,1,V]");
        }
    } else {
        TT_FATAL(!attrs.fuse_ng, "decode_gated_delta_rule: fused norm+gate requires packed mode");
        TT_FATAL(in.q.logical_shape()[1] == 1, "decode_gated_delta_rule: q must be T=1 [B,1,H,K]");
        TT_FATAL(in.v.logical_shape()[2] == attrs.H, "decode_gated_delta_rule: v heads must equal q heads (no GQA)");
    }
    if (in.initial_state.has_value()) {
        check(*in.initial_state, "initial_state");
        const auto& s = in.initial_state->logical_shape();
        TT_FATAL(
            s[0] == attrs.B && s[1] == attrs.H && s[2] == attrs.K && s[3] == attrs.V,
            "decode_gated_delta_rule: initial_state must be [B,H,K,V]");
    } else {
        TT_FATAL(!attrs.inplace_state, "decode_gated_delta_rule: inplace_state requires initial_state");
    }
}

DecodeGatedDeltaRuleDeviceOperation::spec_return_value_t DecodeGatedDeltaRuleDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const DataType dt = in.q.dtype();
    // o is ROW_MAJOR on purpose: its flat 2D is [B*H, V], so page bh is head
    // bh's own [V] stick and the writer kernel issues ONE full-page write per
    // head (page-exclusive). A TILE o would share each page across 32 heads,
    // forcing sub-page writes, which silently no-op on this stack (red-state
    // pcc_o=0.000). Callers wanting TILE o pass it through ttnn.to_layout.
    const auto layout_rm = TensorLayout(dt, PageConfig(Layout::ROW_MAJOR), attrs.output_mem_config);
    const auto layout_tile = TensorLayout(dt, PageConfig(Layout::TILE), attrs.output_mem_config);
    if (attrs.fuse_ng) {
        // gated [1,B,H*V] TILE: the assembler cores write full tiles (see the writer).
        return {
            tt::tt_metal::TensorSpec(ttnn::Shape({1, attrs.B, attrs.H * attrs.V}), layout_tile),
            tt::tt_metal::TensorSpec(ttnn::Shape({attrs.B, attrs.H, attrs.K, attrs.V}), layout_tile)};
    }
    return {
        tt::tt_metal::TensorSpec(ttnn::Shape({attrs.B, 1, attrs.H, attrs.V}), layout_rm),
        tt::tt_metal::TensorSpec(ttnn::Shape({attrs.B, attrs.H, attrs.K, attrs.V}), layout_tile)};
}

DecodeGatedDeltaRuleDeviceOperation::tensor_return_value_t DecodeGatedDeltaRuleDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    auto* device = in.q.device();
    std::vector<Tensor> outs;
    outs.reserve(2);
    outs.push_back(create_device_tensor(specs[0], device));
    if (attrs.inplace_state) {
        outs.push_back(*in.initial_state);  // write new state back into the caller's buffer
    } else {
        outs.push_back(create_device_tensor(specs[1], device));
    }
    return outs;
}

std::vector<Tensor> decode_gated_delta_rule(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& beta,
    const Tensor& g,
    const std::optional<Tensor>& initial_state,
    bool inplace_state,
    float scale,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = DecodeGatedDeltaRuleDeviceOperation;

    const auto& qs = q.logical_shape();  // [B,1,H,K]
    const auto& vs = v.logical_shape();  // [B,1,H,V]

    auto attrs = OperationType::operation_attributes_t{
        .B = qs[0],
        .H = qs[2],
        .BH = qs[0] * qs[2],
        .K = qs[3],
        .V = vs[3],
        .has_initial_state = initial_state.has_value(),
        .inplace_state = inplace_state,
        .scale = scale,
        .output_mem_config = output_mem_config,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .q = q,
        .k = k,
        .v = v,
        .beta = beta,
        .g = g,
        .initial_state = initial_state,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

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
    const std::optional<Tensor>& z,
    const std::optional<Tensor>& norm_w,
    uint32_t z_col_offset,
    float epsilon) {
    using OperationType = DecodeGatedDeltaRuleDeviceOperation;
    using namespace tt::constants;
    const auto& qs = qkv.logical_shape();  // [1,B,C]
    TT_FATAL(qs.rank() == 3, "decode_gated_delta_rule (packed): qkv must be rank 3 [1,B,C]");
    TT_FATAL(
        num_k_heads >= 1 && num_v_heads % num_k_heads == 0, "decode_gated_delta_rule (packed): H must be a multiple of Nk");
    TT_FATAL(
        (num_k_heads * head_k) % TILE_WIDTH == 0, "decode_gated_delta_rule (packed): Nk*K must be a multiple of 32");
    const uint32_t B = qs[1];
    auto attrs = OperationType::operation_attributes_t{
        .B = B,
        .H = num_v_heads,
        .BH = B * num_v_heads,
        .K = head_k,
        .V = head_v,
        .has_initial_state = initial_state.has_value(),
        .inplace_state = inplace_state,
        .scale = scale,
        .output_mem_config = output_mem_config,
        .packed = true,
        .Ct = static_cast<uint32_t>(qs[2]) / TILE_WIDTH,
        .q_off_t = 0,
        .k_off_t = (num_k_heads * head_k) / TILE_WIDTH,
        .v_off_t = (2 * num_k_heads * head_k) / TILE_WIDTH,
        .rf = num_v_heads / num_k_heads,
        .Nvt = (num_v_heads + TILE_WIDTH - 1) / TILE_WIDTH,
    };
    if (z.has_value()) {
        TT_FATAL(z_col_offset % TILE_WIDTH == 0, "decode_gated_delta_rule (fused norm+gate): z_col_offset must be a multiple of 32");
        attrs.fuse_ng = true;
        attrs.Wt_z = (static_cast<uint32_t>(z->logical_shape()[2]) + TILE_WIDTH - 1) / TILE_WIDTH;
        attrs.z_off_t = z_col_offset / TILE_WIDTH;
        attrs.ng_eps_bits = std::bit_cast<uint32_t>(epsilon * static_cast<float>(head_v));
        attrs.ng_scale_bits = std::bit_cast<uint32_t>(std::sqrt(static_cast<float>(head_v)));
    }
    auto tensor_args = OperationType::tensor_args_t{
        .q = qkv,
        .k = qkv,
        .v = qkv,
        .beta = beta,
        .g = g,
        .initial_state = initial_state,
        .z = z,
        .norm_w = norm_w,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
