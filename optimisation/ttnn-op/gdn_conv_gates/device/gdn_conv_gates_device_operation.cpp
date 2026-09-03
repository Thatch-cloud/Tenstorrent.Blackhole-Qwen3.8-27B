// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#include "gdn_conv_gates_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

GdnConvGatesDeviceOperation::program_factory_t GdnConvGatesDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return GdnConvGatesProgramFactory{};
}

void GdnConvGatesDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    const DataType dt = in.x.dtype();
    auto check = [&](const Tensor& t, const char* name) {
        TT_FATAL(t.layout() == Layout::TILE, "gdn_decode_conv_gates: {} must be TILE layout", name);
        TT_FATAL(t.dtype() == dt, "gdn_decode_conv_gates: {} must share x's dtype", name);
        TT_FATAL(
            dt == DataType::BFLOAT16 || dt == DataType::FLOAT32, "gdn_decode_conv_gates: dtype must be bf16 or fp32");
        TT_FATAL(t.buffer() != nullptr, "gdn_decode_conv_gates: {} must be on device", name);
    };
    check(in.x, "x");
    check(in.a, "a");
    check(in.b, "b");
    check(in.dt_bias, "dt_bias");
    check(in.neg_exp_A, "neg_exp_A");
    TT_FATAL(attrs.K == 4, "gdn_decode_conv_gates: only K=4 conv taps are supported (got {})", attrs.K);
    TT_FATAL(in.conv_states.size() == attrs.K && in.taps.size() == attrs.K, "gdn_decode_conv_gates: need K states and K taps");
    TT_FATAL(attrs.C % TILE_WIDTH == 0, "gdn_decode_conv_gates: C must be a multiple of 32");
    TT_FATAL(attrs.B >= 1 && attrs.B <= attrs.Bx, "gdn_decode_conv_gates: batch must be in [1, rows of x]");
    TT_FATAL(attrs.Bx <= attrs.Bmax, "gdn_decode_conv_gates: x has more rows than the conv states");
    const auto& xs = in.x.logical_shape();
    TT_FATAL(xs.rank() == 3 && xs[0] == 1 && xs[2] == attrs.C, "gdn_decode_conv_gates: x must be [1,B,C]");
    for (uint32_t j = 0; j < attrs.K; j++) {
        check(in.conv_states[j], "conv_state");
        check(in.taps[j], "tap");
        const auto& ss = in.conv_states[j].logical_shape();
        TT_FATAL(
            ss.rank() == 3 && ss[0] == 1 && ss[1] == attrs.Bmax && ss[2] == attrs.C,
            "gdn_decode_conv_gates: conv_states[{}] must be [1,Bmax,C]",
            j);
        const auto& ts = in.taps[j].logical_shape();
        TT_FATAL(
            ts.rank() == 3 && ts[0] == 1 && ts[1] == 1 && ts[2] == attrs.C,
            "gdn_decode_conv_gates: taps[{}] must be [1,1,C]",
            j);
    }
    const auto& as = in.a.logical_shape();
    const auto& bs = in.b.logical_shape();
    TT_FATAL(
        as.rank() == 3 && as[0] == 1 && as[1] == attrs.B && as[2] == attrs.Nv && bs == as,
        "gdn_decode_conv_gates: a and b must be [1,B,Nv] with B == batch");
    const auto& ds = in.dt_bias.logical_shape();
    const auto& ns = in.neg_exp_A.logical_shape();
    TT_FATAL(
        ds.rank() == 3 && ds[0] == 1 && ds[1] == 1 && ds[2] == attrs.Nv && ns == ds,
        "gdn_decode_conv_gates: dt_bias and neg_exp_A must be [1,1,Nv]");
}

GdnConvGatesDeviceOperation::spec_return_value_t GdnConvGatesDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const DataType dt = in.x.dtype();
    const auto layout = TensorLayout(dt, PageConfig(Layout::TILE), attrs.output_mem_config);
    return {
        TensorSpec(ttnn::Shape({1, attrs.Bx, attrs.C}), layout),
        TensorSpec(ttnn::Shape({1, attrs.B, attrs.Nv}), layout),
        TensorSpec(ttnn::Shape({1, attrs.B, attrs.Nv}), layout)};
}

GdnConvGatesDeviceOperation::tensor_return_value_t GdnConvGatesDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    auto* device = in.x.device();
    std::vector<Tensor> outs;
    outs.reserve(3);
    for (const auto& s : specs) {
        outs.push_back(create_device_tensor(s, device));
    }
    return outs;
}

std::vector<Tensor> gdn_conv_gates(
    const Tensor& x,
    const std::vector<Tensor>& conv_states,
    const std::vector<Tensor>& taps,
    const Tensor& a,
    const Tensor& b,
    const Tensor& dt_bias,
    const Tensor& neg_exp_A,
    uint32_t batch,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = GdnConvGatesDeviceOperation;
    const auto& xs = x.logical_shape();  // [1,Bx,C]
    TT_FATAL(!conv_states.empty(), "gdn_decode_conv_gates: conv_states is empty");
    const auto& ss = conv_states[0].logical_shape();
    const auto& as = a.logical_shape();
    auto attrs = OperationType::operation_attributes_t{
        .B = batch,
        .Bx = xs[1],
        .Bmax = ss[1],
        .C = xs[2],
        .Nv = as[2],
        .K = static_cast<uint32_t>(conv_states.size()),
        .output_mem_config = output_mem_config,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .x = x,
        .conv_states = conv_states,
        .taps = taps,
        .a = a,
        .b = b,
        .dt_bias = dt_bias,
        .neg_exp_A = neg_exp_A,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
