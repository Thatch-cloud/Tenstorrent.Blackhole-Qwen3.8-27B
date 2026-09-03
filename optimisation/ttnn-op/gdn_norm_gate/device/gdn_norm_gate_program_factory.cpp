// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for the fused GDN decode output norm + gate (Device 2.0 descriptor style).
//
// Work unit ("instance") = one (batch-tile bt, head h, column-tile ct): the core DMAs the
// 32 sticks o[b, h, :] for b in [bt*32, bt*32+32) contiguously into a row-major block
// (rows at or beyond B zeroed), the compute TILIZES that block into Vt tiles (the unpacker
// does the relayout; no core-side scatter), takes the per-row rms factor from all Vt
// tiles, and finishes only ITS column tile: scale, norm_w, silu(z), multiply. The writer
// stores that one tile at (bt, h*Vt + ct). Every DRAM access is a full page. Bt*H*Vt
// instances (96 at this model's shape) over the grid.

#include "gdn_norm_gate_program_factory.hpp"

#include <bit>
#include <cmath>
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

// CB index plan (unique namespace: the transformer ops are unity-built).
namespace cbng {
constexpr uint32_t orm = tt::CBIndex::c_0;   // [Vt] io_o   row-major 32 x V block of o (32 sticks)
constexpr uint32_t z = tt::CBIndex::c_1;     // 1 io_z      z tile (bt, z_off_t + h*Vt + ct)
constexpr uint32_t w = tt::CBIndex::c_2;     // 1 io_w      norm_w row-0 tile ct
constexpr uint32_t ones = tt::CBIndex::c_3;  // 1 fp32      all-ones (row-sum contraction)
constexpr uint32_t ot = tt::CBIndex::c_4;    // [Vt] io_o   tilized o
constexpr uint32_t sq = tt::CBIndex::c_5;    // [Vt] fp32   o*o
constexpr uint32_t sc = tt::CBIndex::c_6;    // 1 fp32      row sums of squares
constexpr uint32_t fac = tt::CBIndex::c_7;   // 1 fp32      per-row rms factor
constexpr uint32_t of = tt::CBIndex::c_8;    // 1 fp32      this column tile of o
constexpr uint32_t xn = tt::CBIndex::c_9;    // 1 fp32      of * factor
constexpr uint32_t wf = tt::CBIndex::c_10;   // 1 fp32      norm_w tile
constexpr uint32_t xw = tt::CBIndex::c_11;   // 1 fp32      xn * norm_w
constexpr uint32_t zs = tt::CBIndex::c_12;   // 1 fp32      silu(z)
constexpr uint32_t out = tt::CBIndex::c_13;  // 1 io_z      xw * silu(z)
}  // namespace cbng

static tt::DataFormat df_of(DataType dt) {
    return dt == DataType::FLOAT32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
}

tt::tt_metal::ProgramDescriptor GdnNormGateProgramFactory::create_descriptor(
    const GdnNormGateParams& attrs, const GdnNormGateInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t H = attrs.H;
    const uint32_t V = attrs.V;
    const uint32_t Vt = V / TILE_WIDTH;
    const uint32_t Bt = (attrs.B + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t Wt = (attrs.W + TILE_WIDTH - 1) / TILE_WIDTH;  // z tiles per tile-row
    const uint32_t z_off_t = attrs.z_off / TILE_WIDTH;
    const uint32_t n_inst = Bt * H * Vt;

    const tt::DataFormat df_o = df_of(in.o.dtype());
    const tt::DataFormat df_z = df_of(in.z.dtype());
    const tt::DataFormat df_w = df_of(in.norm_w.dtype());

    // rms_norm(x) = x / sqrt(sum/V + eps) = x * sqrt(V) / sqrt(sum + V*eps)
    const float eps_scaled = attrs.epsilon * static_cast<float>(V);
    const float scale = std::sqrt(static_cast<float>(V));
    const uint32_t eps_bits = std::bit_cast<uint32_t>(eps_scaled);
    const uint32_t scale_bits = std::bit_cast<uint32_t>(scale);

    auto* device = in.o.device();
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
    // The row-major block is Vt tile-sized pages: 32 sticks of V elements == Vt tiles of bytes.
    add_cb(cbng::orm, Vt, 2, df_o);
    add_cb(cbng::z, 1, 2, df_z);
    add_cb(cbng::w, 1, 2, df_w);
    add_cb(cbng::ones, 1, 1, tt::DataFormat::Float32);
    add_cb(cbng::ot, Vt, 1, df_o);
    add_cb(cbng::sq, Vt, 1, tt::DataFormat::Float32);
    add_cb(cbng::sc, 1, 1, tt::DataFormat::Float32);
    add_cb(cbng::fac, 1, 1, tt::DataFormat::Float32);
    add_cb(cbng::of, 1, 1, tt::DataFormat::Float32);
    add_cb(cbng::xn, 1, 1, tt::DataFormat::Float32);
    add_cb(cbng::wf, 1, 1, tt::DataFormat::Float32);
    add_cb(cbng::xw, 1, 1, tt::DataFormat::Float32);
    add_cb(cbng::zs, 1, 1, tt::DataFormat::Float32);
    add_cb(cbng::out, 1, 2, df_z);

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/gdn_norm_gate/device/kernels/";
    const uint32_t o_page = static_cast<uint32_t>(in.o.buffer()->page_size());

    // Reader: {Vt, H, B, Wt, z_off_t, o_page} + accessors (o, z, w)
    std::vector<uint32_t> reader_ct = {Vt, H, attrs.B, Wt, z_off_t, o_page};
    TensorAccessorArgs(*in.o.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.z.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.norm_w.buffer()).append_to(reader_ct);
    // Writer: {Vt, H} + accessor (out)
    std::vector<uint32_t> writer_ct = {Vt, H};
    TensorAccessorArgs(*outputs[0].buffer()).append_to(writer_ct);
    const std::vector<uint32_t> compute_ct = {Vt, eps_bits, scale_bits};

    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_gdn_norm_gate.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.config = ReaderConfigDescriptor{};
    reader.runtime_args.reserve(n_cores);

    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_gdn_norm_gate.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};
    writer.runtime_args.reserve(n_cores);

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/gdn_norm_gate.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = compute_ct;
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false};
    compute.runtime_args.reserve(n_cores);

    using RtArg = std::variant<uint32_t, Buffer*>;
    Buffer* o_buf = in.o.buffer();
    Buffer* z_buf = in.z.buffer();
    Buffer* w_buf = in.norm_w.buffer();
    Buffer* out_buf = outputs[0].buffer();
    for (uint32_t c = 0; c < n_cores; c++) {
        const auto& core = active_cores[c];
        const uint32_t start = c * per_core;
        const uint32_t n = std::min(per_core, n_inst - start);
        reader.emplace_runtime_args(core, std::vector<RtArg>{start, n, o_buf, z_buf, w_buf});
        writer.emplace_runtime_args(core, std::vector<RtArg>{start, n, out_buf});
        compute.emplace_runtime_args(core, {n, start});
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::prim
