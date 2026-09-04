// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for the fused attention decode prologue (Device 2.0 descriptor style).
//
// Work unit ("instance") = one (batch row b, output kind): kind 0 = q, 1 = k, 2 = v, 3 = gate.
// The reader assembles the kind's [32 rows (heads, zero beyond the head count) x HD] block from
// the projection output (row b of the shared tiles, head h into row h), plus this batch row's
// cos/sin row-0 tiles; the compute normalises, weights, rounds to bf16 and ropes q and k (v and
// gate are pure copies the reader lands straight in the output ring); the writer stores the HDt
// full tiles at page b*HDt + t of the kind's output -- interleaved for q/gate, the KV update's
// height-sharded config for k/v (addressed through the tensor accessor).

#include "attn_prep_program_factory.hpp"

#include <set>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {

namespace cbap {
constexpr uint32_t src = tt::CBIndex::c_0;    // [HDt] io   staging: one head's HDt source tiles
constexpr uint32_t blk = tt::CBIndex::c_1;    // [HDt] io   assembled 32 x HD block (q, k)
constexpr uint32_t out = tt::CBIndex::c_2;    // [HDt] io   finished block for the writer (all kinds)
constexpr uint32_t cos = tt::CBIndex::c_3;    // [RDt] io   this row's cos (row 0)
constexpr uint32_t sin = tt::CBIndex::c_4;    // [RDt] io   this row's sin (row 0)
constexpr uint32_t wq = tt::CBIndex::c_5;     // [HDt] io   q_norm_w row-0 tiles (once per core)
constexpr uint32_t wk = tt::CBIndex::c_6;     // [HDt] io   k_norm_w row-0 tiles (once per core)
constexpr uint32_t wqf = tt::CBIndex::c_7;    // [HDt] fp32 persistent
constexpr uint32_t wkf = tt::CBIndex::c_8;    // [HDt] fp32 persistent
constexpr uint32_t ones = tt::CBIndex::c_9;   // 1 fp32
constexpr uint32_t xf = tt::CBIndex::c_10;    // [HDt] fp32 block
constexpr uint32_t sq = tt::CBIndex::c_11;    // [HDt] fp32 x*x
constexpr uint32_t sc = tt::CBIndex::c_12;    // 1 fp32 row sums
constexpr uint32_t fac = tt::CBIndex::c_13;   // 1 fp32 rms factor
constexpr uint32_t xn = tt::CBIndex::c_14;    // [HDt] fp32 x*factor
constexpr uint32_t xw = tt::CBIndex::c_15;    // [HDt] fp32 xn*w
constexpr uint32_t rt = tt::CBIndex::c_16;    // [HDt] io   bf16 round-trip
constexpr uint32_t xwr = tt::CBIndex::c_17;   // [HDt] fp32 bf16-valued xw
constexpr uint32_t cosf = tt::CBIndex::c_18;  // [RDt] fp32
constexpr uint32_t sinf = tt::CBIndex::c_19;  // [RDt] fp32
constexpr uint32_t ta = tt::CBIndex::c_20;    // 1 fp32 rope temp
constexpr uint32_t tb = tt::CBIndex::c_21;    // 1 fp32 rope temp
constexpr uint32_t xnr = tt::CBIndex::c_22;   // [HDt] fp32 bf16-valued xn
}  // namespace cbap

tt::tt_metal::ProgramDescriptor AttnPrepProgramFactory::create_descriptor(
    const AttnPrepParams& attrs, const AttnPrepInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t HDt = attrs.HD / TILE_WIDTH;
    const uint32_t RDt = attrs.RD / TILE_WIDTH;
    const uint32_t n_inst = attrs.B * 4;
    const tt::DataFormat df_io = tt::DataFormat::Float16_b;
    const tt::DataFormat f32 = tt::DataFormat::Float32;

    auto* device = in.qkv.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t grid_y = grid.y;
    const uint32_t ncores = grid.x * grid.y;
    const uint32_t per_core = (n_inst + ncores - 1) / ncores;
    const uint32_t n_cores = (n_inst + per_core - 1) / per_core;
    std::vector<CoreCoord> active_cores;
    std::set<CoreRange> core_set;
    for (uint32_t c = 0; c < n_cores; c++) {
        active_cores.push_back(CoreCoord{c / grid_y, c % grid_y});
        core_set.insert(CoreRange{active_cores.back(), active_cores.back()});
    }
    CoreRangeSet cores{core_set};

    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t idx, uint32_t n_tiles, uint32_t nbuf, tt::DataFormat fmt) {
        const uint32_t ts = tt::tile_size(fmt);
        desc.cbs.push_back(CBDescriptor{
            .total_size = n_tiles * nbuf * ts,
            .core_ranges = cores,
            .format_descriptors = {
                {CBFormatDescriptor{.buffer_index = static_cast<uint8_t>(idx), .data_format = fmt, .page_size = ts}}}});
    };
    add_cb(cbap::src, HDt * 4, 1, df_io);  // staging: 4 heads' tiles per barrier (reader HG)
    add_cb(cbap::blk, HDt, 2, df_io);
    add_cb(cbap::out, HDt, 2, df_io);
    add_cb(cbap::cos, RDt, 2, df_io);
    add_cb(cbap::sin, RDt, 2, df_io);
    add_cb(cbap::wq, HDt, 1, df_io);
    add_cb(cbap::wk, HDt, 1, df_io);
    add_cb(cbap::wqf, HDt, 1, f32);
    add_cb(cbap::wkf, HDt, 1, f32);
    add_cb(cbap::ones, 1, 1, f32);
    add_cb(cbap::xf, HDt, 1, f32);
    add_cb(cbap::sq, HDt, 1, f32);
    add_cb(cbap::sc, 1, 1, f32);
    add_cb(cbap::fac, 1, 1, f32);
    add_cb(cbap::xn, HDt, 1, f32);
    add_cb(cbap::xw, HDt, 1, f32);
    add_cb(cbap::rt, HDt, 1, df_io);
    add_cb(cbap::xwr, HDt, 1, f32);
    add_cb(cbap::cosf, RDt, 1, f32);
    add_cb(cbap::sinf, RDt, 1, f32);
    add_cb(cbap::ta, 1, 1, f32);
    add_cb(cbap::tb, 1, 1, f32);
    add_cb(cbap::xnr, HDt, 1, f32);

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/attn_prep/device/kernels/";
    const uint32_t q_off_t = 0;
    const uint32_t k_off_t = (attrs.NH * attrs.HD) / TILE_WIDTH;
    const uint32_t v_off_t = ((attrs.NH + attrs.NKV) * attrs.HD) / TILE_WIDTH;
    const uint32_t g_off_t = ((attrs.NH + 2 * attrs.NKV) * attrs.HD) / TILE_WIDTH;

    // Reader: {HDt, RDt, NH, NKV, Wt, q_off_t, k_off_t, v_off_t, g_off_t} + accessors (qkv, cos, sin, wq, wk)
    std::vector<uint32_t> reader_ct = {HDt, RDt, attrs.NH, attrs.NKV, attrs.Wt, q_off_t, k_off_t, v_off_t, g_off_t};
    TensorAccessorArgs(*in.qkv.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.cos.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.sin.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.q_norm_w.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.k_norm_w.buffer()).append_to(reader_ct);
    // Writer: {HDt} + accessors (q, gate, k, v)
    std::vector<uint32_t> writer_ct = {HDt};
    for (int i = 0; i < 4; i++) {
        TensorAccessorArgs(*outputs[i].buffer()).append_to(writer_ct);
    }
    // Compute: {HDt, RDt, EPS_BITS, SCALE_BITS}
    const std::vector<uint32_t> compute_ct = {HDt, RDt, attrs.eps_bits, attrs.scale_bits};

    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_attn_prep.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.config = ReaderConfigDescriptor{};
    reader.runtime_args.reserve(n_cores);

    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_attn_prep.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};
    writer.runtime_args.reserve(n_cores);

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/attn_prep.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = compute_ct;
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false};
    compute.runtime_args.reserve(n_cores);

    using RtArg = std::variant<uint32_t, Buffer*>;
    Buffer* qkv_buf = in.qkv.buffer();
    Buffer* cos_buf = in.cos.buffer();
    Buffer* sin_buf = in.sin.buffer();
    Buffer* wq_buf = in.q_norm_w.buffer();
    Buffer* wk_buf = in.k_norm_w.buffer();
    for (uint32_t c = 0; c < n_cores; c++) {
        const auto& core = active_cores[c];
        const uint32_t start = c * per_core;
        const uint32_t n = std::min(per_core, n_inst - start);
        // Instance inst = b*4 + kind; the kernels derive (b, kind) from the instance number.
        reader.emplace_runtime_args(core, std::vector<RtArg>{start, n, qkv_buf, cos_buf, sin_buf, wq_buf, wk_buf});
        writer.emplace_runtime_args(
            core,
            std::vector<RtArg>{start, n, outputs[0].buffer(), outputs[1].buffer(), outputs[2].buffer(), outputs[3].buffer()});
        compute.emplace_runtime_args(core, {start, n});
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::prim
