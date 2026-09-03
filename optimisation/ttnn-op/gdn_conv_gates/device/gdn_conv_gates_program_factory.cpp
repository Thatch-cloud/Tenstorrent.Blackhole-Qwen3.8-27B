// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for the fused GDN decode conv + gates op (Device 2.0 descriptor style,
// mirroring decode_gated_delta_rule).
//
// Work unit ("instance") = one (batch-tile, channel-tile) of the [1,Bmax,C] TILE tensors,
// numbered inst = bt*Ct + cc, which is also its page id in every [1,Bmax,C] tensor. Each
// active core owns a contiguous instance range; per instance the reader DMAs the K-1 newest
// conv states + x (masked to the active batch rows) and the K taps, the compute forms the
// four broadcast products, sums them, applies SiLU, and passes the window through for the
// writer, which stores conv_out and the advanced shift-register (IN PLACE, same pages the
// reader just consumed on this core -- one owner per page, so no cross-core hazard).
//
// The gates (one tile per (batch-tile, Nv-tile)) run on one extra core when the grid has
// one spare, else on the last conv core after its conv instances.

#include "gdn_conv_gates_program_factory.hpp"

#include <bit>
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

// CB index plan (io = input dtype). Kept in sync with the kernels. Unique namespace: the
// transformer ops are unity-built, so per-file `namespace cb` blocks collide.
namespace cbcg {
constexpr uint32_t win = tt::CBIndex::c_0;    // [K] io   window: st[1..K-1], x(masked)
constexpr uint32_t tap = tt::CBIndex::c_1;    // [K] io   taps (row 0)
constexpr uint32_t prod = tt::CBIndex::c_2;   // [K] fp32 window[j] * tap[j]
constexpr uint32_t acc = tt::CBIndex::c_3;    // 1 fp32 (2-page ring) running sum
constexpr uint32_t out = tt::CBIndex::c_4;    // 1 io   silu(sum)
constexpr uint32_t shift = tt::CBIndex::c_5;  // [K] io window pass-through for the state write-back
constexpr uint32_t a = tt::CBIndex::c_6;      // 1 io
constexpr uint32_t b = tt::CBIndex::c_7;      // 1 io
constexpr uint32_t dtb = tt::CBIndex::c_8;    // 1 io   dt_bias (row 0)
constexpr uint32_t nega = tt::CBIndex::c_9;   // 1 io   neg_exp_A (row 0)
constexpr uint32_t sp = tt::CBIndex::c_10;    // 1 io   softplus(a + dt_bias)
constexpr uint32_t beta = tt::CBIndex::c_11;  // 1 io
constexpr uint32_t g = tt::CBIndex::c_12;     // 1 io
}  // namespace cbcg

tt::tt_metal::ProgramDescriptor GdnConvGatesProgramFactory::create_descriptor(
    const GdnConvGatesParams& attrs, const GdnConvGatesInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t K = attrs.K;
    const uint32_t Ct = attrs.C / TILE_WIDTH;
    const uint32_t Bt = (attrs.Bmax + TILE_HEIGHT - 1) / TILE_HEIGHT;   // state tile-rows
    const uint32_t xBt = (attrs.Bx + TILE_HEIGHT - 1) / TILE_HEIGHT;    // x tile-rows
    const uint32_t gBt = (attrs.B + TILE_HEIGHT - 1) / TILE_HEIGHT;     // a/b/beta/g tile-rows
    const uint32_t Nvt = (attrs.Nv + TILE_WIDTH - 1) / TILE_WIDTH;
    const uint32_t n_conv = Bt * Ct;
    const uint32_t n_gate = gBt * Nvt;

    const tt::DataFormat df_io =
        (in.x.dtype() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;

    constexpr uint32_t one_bits = 0x3F800000u;     // 1.0f: softplus beta and 1/beta
    constexpr uint32_t twenty_bits = 0x41A00000u;  // 20.0f: softplus threshold (python SOFTPLUS(1.0, 20.0))

    auto* device = in.x.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t grid_y = grid.y;
    const uint32_t ncores = grid.x * grid.y;

    // Conv instances over as many cores as needed; gates on the next free core if any.
    const uint32_t per_core = (n_conv + ncores - 1) / ncores;
    const uint32_t n_conv_cores = (n_conv + per_core - 1) / per_core;
    const bool own_gate_core = n_conv_cores < ncores;
    const uint32_t n_cores = own_gate_core ? n_conv_cores + 1 : n_conv_cores;
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
    add_cb(cbcg::win, K, 2, df_io);
    add_cb(cbcg::tap, K, 2, df_io);
    add_cb(cbcg::prod, K, 1, tt::DataFormat::Float32);
    add_cb(cbcg::acc, 1, 2, tt::DataFormat::Float32);
    add_cb(cbcg::out, 1, 2, df_io);
    add_cb(cbcg::shift, K, 2, df_io);
    add_cb(cbcg::a, 1, 2, df_io);
    add_cb(cbcg::b, 1, 2, df_io);
    add_cb(cbcg::dtb, 1, 2, df_io);
    add_cb(cbcg::nega, 1, 2, df_io);
    add_cb(cbcg::sp, 1, 2, df_io);
    add_cb(cbcg::beta, 1, 2, df_io);
    add_cb(cbcg::g, 1, 2, df_io);

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/gdn_conv_gates/device/kernels/";

    // Reader compile args: {K, Ct, Nvt, B, xBt} + accessors (x, st0..3, tap0..3, a, b, dt_bias, neg_exp_A).
    std::vector<uint32_t> reader_ct = {K, Ct, Nvt, attrs.B, xBt};
    TensorAccessorArgs(*in.x.buffer()).append_to(reader_ct);
    for (uint32_t j = 0; j < K; j++) {
        TensorAccessorArgs(*in.conv_states[j].buffer()).append_to(reader_ct);
    }
    for (uint32_t j = 0; j < K; j++) {
        TensorAccessorArgs(*in.taps[j].buffer()).append_to(reader_ct);
    }
    TensorAccessorArgs(*in.a.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.b.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.dt_bias.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.neg_exp_A.buffer()).append_to(reader_ct);

    // Writer compile args: {K, Ct, Nvt} + accessors (conv_out, st0..3, beta, g).
    std::vector<uint32_t> writer_ct = {K, Ct, Nvt};
    TensorAccessorArgs(*outputs[0].buffer()).append_to(writer_ct);
    for (uint32_t j = 0; j < K; j++) {
        TensorAccessorArgs(*in.conv_states[j].buffer()).append_to(writer_ct);
    }
    TensorAccessorArgs(*outputs[1].buffer()).append_to(writer_ct);
    TensorAccessorArgs(*outputs[2].buffer()).append_to(writer_ct);

    const std::vector<uint32_t> compute_ct = {K, Nvt, one_bits, twenty_bits};

    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_gdn_conv_gates.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.config = ReaderConfigDescriptor{};
    reader.runtime_args.reserve(n_cores);

    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_gdn_conv_gates.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};
    writer.runtime_args.reserve(n_cores);

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/gdn_conv_gates.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = compute_ct;
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false};
    compute.runtime_args.reserve(n_cores);

    // Runtime args carry Buffer* (not raw addresses) so the framework re-resolves them on a
    // program-cache hit: the decode trace replays this program against whatever buffers the
    // caller holds, and a baked-in address would silently go stale.
    using RtArg = std::variant<uint32_t, Buffer*>;
    Buffer* x_buf = in.x.buffer();
    std::vector<Buffer*> st_buf, tap_buf;
    for (uint32_t j = 0; j < K; j++) {
        st_buf.push_back(in.conv_states[j].buffer());
        tap_buf.push_back(in.taps[j].buffer());
    }
    Buffer* a_buf = in.a.buffer();
    Buffer* b_buf = in.b.buffer();
    Buffer* dtb_buf = in.dt_bias.buffer();
    Buffer* nega_buf = in.neg_exp_A.buffer();
    Buffer* out_buf = outputs[0].buffer();
    Buffer* beta_buf = outputs[1].buffer();
    Buffer* g_buf = outputs[2].buffer();

    for (uint32_t c = 0; c < n_cores; c++) {
        const auto& core = active_cores[c];
        const bool conv_core = c < n_conv_cores;
        const uint32_t start = conv_core ? c * per_core : 0;
        const uint32_t n_inst = conv_core ? std::min(per_core, n_conv - start) : 0;
        const bool gate_core = (c == n_cores - 1);
        const uint32_t g_n = gate_core ? n_gate : 0;

        std::vector<RtArg> r = {start, n_inst, g_n, x_buf};
        for (auto* p : st_buf) {
            r.emplace_back(p);
        }
        for (auto* p : tap_buf) {
            r.emplace_back(p);
        }
        r.insert(r.end(), {RtArg{a_buf}, RtArg{b_buf}, RtArg{dtb_buf}, RtArg{nega_buf}});
        reader.emplace_runtime_args(core, r);

        std::vector<RtArg> w = {start, n_inst, g_n, out_buf};
        for (auto* p : st_buf) {
            w.emplace_back(p);
        }
        w.insert(w.end(), {RtArg{beta_buf}, RtArg{g_buf}});
        writer.emplace_runtime_args(core, w);

        compute.emplace_runtime_args(core, {n_inst, g_n});
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::prim
