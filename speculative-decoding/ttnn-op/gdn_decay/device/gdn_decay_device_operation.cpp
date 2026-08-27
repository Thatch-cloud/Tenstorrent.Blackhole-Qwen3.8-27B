// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/gdn_decay/device/gdn_decay_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/constants.hpp>
#include <utility>

using namespace tt::tt_metal;

namespace ttnn::prim {

// Op-unique name: Unity builds concatenate translation units, so even a `static` helper
// collides with an identically-named one in a neighbouring op.
static void validate_gdn_decay_tensor(const Tensor& t, const std::string& name) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(t.buffer() != nullptr, "{} must be allocated", name);
    TT_FATAL(t.buffer()->buffer_type() == BufferType::DRAM, "{} must be in DRAM", name);
    TT_FATAL(t.layout() == Layout::TILE, "{} must be tiled", name);
    TT_FATAL(t.dtype() == DataType::FLOAT32, "{} must be float32, got {}", name, t.dtype());
}

void GdnDecayDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& in) {
    validate_gdn_decay_tensor(in.h, "h");
    validate_gdn_decay_tensor(in.g, "g");
    validate_gdn_decay_tensor(in.k, "k");
    validate_gdn_decay_tensor(in.v, "v");
    validate_gdn_decay_tensor(in.beta, "beta");
    validate_gdn_decay_tensor(in.q, "q");
    const auto hs = in.h.logical_shape();
    const auto ks = in.k.logical_shape();
    const auto vs = in.v.logical_shape();
    const uint32_t T = static_cast<uint32_t>(in.g.logical_shape()[1]);
    TT_FATAL(in.g.logical_shape()[2] == 1 && in.g.logical_shape()[3] == 1,
             "g must be a per-head scalar [BH,T,1,1], got [...,{},{}]",
             in.g.logical_shape()[2], in.g.logical_shape()[3]);
    TT_FATAL(static_cast<uint32_t>(in.k.logical_shape()[1]) == T, "k must carry T tokens");
    TT_FATAL(static_cast<uint32_t>(in.v.logical_shape()[1]) == T, "v must carry T tokens");
    TT_FATAL(static_cast<uint32_t>(in.beta.logical_shape()[1]) == T, "beta must carry T tokens");
    TT_FATAL(in.beta.logical_shape()[2] == 1 && in.beta.logical_shape()[3] == 1,
             "beta must be a per-head scalar [BH,T,1,1], got [...,{},{}]",
             in.beta.logical_shape()[2], in.beta.logical_shape()[3]);
    TT_FATAL(static_cast<uint32_t>(in.q.logical_shape()[1]) == T, "q must carry T tokens");
    TT_FATAL(ks[3] == hs[2], "k inner dim {} must equal state rows {}", ks[3], hs[2]);
    TT_FATAL(vs[3] == hs[3], "v inner dim {} must equal state cols {}", vs[3], hs[3]);
    TT_FATAL(ks[2] == vs[2], "k and v must share the token-row count");

    // Head/batch axis lives in dim 0 and must agree across every input: the program
    // factory launches exactly BH cores off h alone, so a shorter g/k/v/beta/q would
    // have cores reading past the end of their tensor with no error.
    const uint32_t BH = static_cast<uint32_t>(hs[0]);
    TT_FATAL(BH >= 1, "BH must be at least 1, got {}", BH);
    TT_FATAL(hs[1] == 1, "h carries no token axis, expected dim 1 == 1, got {}", hs[1]);
    const std::pair<const Tensor*, const char*> gdn_decay_bh_checked[] = {
        {&in.g, "g"}, {&in.k, "k"}, {&in.v, "v"}, {&in.beta, "beta"}, {&in.q, "q"}};
    for (const auto& [t, name] : gdn_decay_bh_checked) {
        TT_FATAL(static_cast<uint32_t>(t->logical_shape()[0]) == BH,
                 "{} batch*heads {} must equal h's {}", name, t->logical_shape()[0], BH);
    }
}

void GdnDecayDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& a, const tensor_args_t& in) {
    validate_on_program_cache_miss(a, in);
}

GdnDecayDeviceOperation::spec_return_value_t GdnDecayDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const auto hs = in.h.logical_shape();
    const auto ks = in.k.logical_shape();
    auto layout =
        tt::tt_metal::TensorLayout(DataType::FLOAT32, tt::tt_metal::PageConfig(Layout::TILE), attrs.output_mem_config);
    const auto T = in.g.logical_shape()[1];
    return {
        TensorSpec(ttnn::Shape({hs[0], T, ks[2], hs[3]}), layout),  // o      [BH,T,M,V]
        TensorSpec(ttnn::Shape({hs[0], T, hs[2], hs[3]}), layout),  // states [BH,T,K,V]
    };
}

GdnDecayDeviceOperation::tensor_return_value_t GdnDecayDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    return {create_device_tensor(specs[0], in.h.device()), create_device_tensor(specs[1], in.h.device())};
}

std::vector<Tensor> gdn_decay(
    const Tensor& h,
    const Tensor& g,
    const Tensor& k,
    const Tensor& v,
    const Tensor& beta,
    const Tensor& q,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    const auto hs = h.logical_shape();
    const auto ks = k.logical_shape();
    return ttnn::device_operation::launch<GdnDecayDeviceOperation>(
        GdnDecayParams{
            .Kt = static_cast<uint32_t>(hs[2] / tt::constants::TILE_HEIGHT),
            .Vt = static_cast<uint32_t>(hs[3] / tt::constants::TILE_WIDTH),
            .Mt = static_cast<uint32_t>(ks[2] / tt::constants::TILE_HEIGHT),
            .T = static_cast<uint32_t>(g.logical_shape()[1]),
            .BH = static_cast<uint32_t>(hs[0]),
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        GdnDecayInputs{h, g, k, v, beta, q});
}

}  // namespace ttnn::prim
