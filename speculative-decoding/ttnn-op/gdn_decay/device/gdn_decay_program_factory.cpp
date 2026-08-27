// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/gdn_decay/device/gdn_decay_program_factory.hpp"

#include <algorithm>
#include <set>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

static constexpr const char* kBase = "ttnn/cpp/ttnn/operations/transformer/gdn_decay/device/kernels/";

// One core per head. Within a head the state is carried across every stage, so splitting its
// tiles across cores would need a reduction; across heads the recurrences are independent,
// so head->core is the natural (and only) parallel axis. Column-major, matching the
// neighbouring gated_delta_attn.
GdnDecayProgramFactory::cached_program_t GdnDecayProgramFactory::create(
    const GdnDecayParams& attrs, const GdnDecayInputs& in, std::vector<Tensor>& out) {
    Program program{};
    auto* device = in.h.device();
    const uint32_t tb = in.h.tensor_spec().tile().get_tile_size(tt::DataFormat::Float32);
    const uint32_t Kt = attrs.Kt, Vt = attrs.Vt, Mt = attrs.Mt, T = attrs.T;

    const uint32_t BH = attrs.BH;
    const CoreCoord compute_grid = device->compute_with_storage_grid_size();
    const uint32_t grid_y = compute_grid.y;
    TT_FATAL(
        BH <= compute_grid.x * grid_y,
        "BH {} exceeds the compute grid {}x{}={}",
        BH,
        compute_grid.x,
        grid_y,
        compute_grid.x * grid_y);
    std::vector<CoreCoord> head_cores(BH);
    for (uint32_t h = 0; h < BH; h++) {
        head_cores[h] = CoreCoord{h / grid_y, h % grid_y};
    }
    std::set<CoreRange> core_set;
    for (const auto& c : head_cores) {
        core_set.insert(CoreRange{c, c});
    }
    CoreRangeSet cores(core_set);
    auto mk = [&](uint32_t idx, uint32_t pages) {
        CircularBufferConfig c =
            CircularBufferConfig(pages * tb, {{idx, tt::DataFormat::Float32}}).set_page_size(idx, tb);
        CreateCircularBuffer(program, cores, c);
    };
    // Per-token input CBs hold exactly ONE token (not double-buffered -- the dataflow kernels
    // address tiles at an absolute `t * f32` offset, which only agrees with the CB's relative
    // addressing at one slot), so the reader cannot run ahead. The state CB (c_11) needs two
    // slots: a new state is reserved while the old one is still the front. Sizes are per-core
    // and do not grow with BH -- each core owns exactly one head.
    // g (c_1), beta (c_5) and exp(g) (c_12) are single 32x32 fp32 tiles: they carry one
    // scalar per token, SCALAR-broadcast on device. A full tile page is required even though
    // only lane [0][0] is read -- llk_unpack_AB_init cross-checks face_r_dim/num_faces
    // between the two operands.
    mk(tt::CBIndex::c_0, Kt * Vt);       // initial h
    mk(tt::CBIndex::c_1, 1);             // g (per-head-per-token scalar tile)
    mk(tt::CBIndex::c_2, Kt * Vt);       // h1
    mk(tt::CBIndex::c_3, Mt * Kt);       // k
    mk(tt::CBIndex::c_4, Mt * Vt);       // v
    mk(tt::CBIndex::c_5, 1);             // beta (per-head-per-token scalar tile)
    mk(tt::CBIndex::c_6, Mt * Kt);       // q
    mk(tt::CBIndex::c_7, Mt * Vt);       // v_read
    mk(tt::CBIndex::c_8, Mt * Vt);       // delta
    mk(tt::CBIndex::c_9, Kt * Mt);       // k transposed
    mk(tt::CBIndex::c_10, Kt * Vt);      // kT @ delta, before += h1
    mk(tt::CBIndex::c_11, 2 * Kt * Vt);  // running state
    mk(tt::CBIndex::c_12, 1);            // exp(g), one scalar tile
    mk(tt::CBIndex::c_13, Mt * Vt);      // v - v_read, before the beta scale
    mk(tt::CBIndex::c_16, Mt * Vt);      // o
    mk(tt::CBIndex::c_17, Kt * Vt);      // states

    std::vector<uint32_t> rct;
    for (const auto* t : {&in.h, &in.g, &in.k, &in.v, &in.beta, &in.q}) {
        TensorAccessorArgs(*t->buffer()).append_to(rct);
    }
    std::vector<uint32_t> wct;
    TensorAccessorArgs(*out[0].buffer()).append_to(wct);
    TensorAccessorArgs(*out[1].buffer()).append_to(wct);

    auto reader = CreateKernel(
        program, std::string(kBase) + "dataflow/reader_gdn_decay.cpp", cores, ReaderDataMovementConfig(rct));
    auto writer = CreateKernel(
        program, std::string(kBase) + "dataflow/writer_gdn_decay.cpp", cores, WriterDataMovementConfig(wct));
    auto [mf, approx, fp32acc, l1acc, dst_full] =
        get_compute_kernel_config_args(device->arch(), attrs.compute_kernel_config);
    auto compute = CreateKernel(
        program,
        std::string(kBase) + "compute/gdn_decay.cpp",
        cores,
        ComputeConfig{.math_fidelity = mf, .fp32_dest_acc_en = fp32acc, .math_approx_mode = approx, .compile_args = {}});

    for (uint32_t head = 0; head < BH; head++) {
        const CoreCoord& core = head_cores[head];
        SetRuntimeArgs(
            program,
            reader,
            core,
            {in.h.buffer()->address(),
             in.g.buffer()->address(),
             in.k.buffer()->address(),
             in.v.buffer()->address(),
             in.beta.buffer()->address(),
             in.q.buffer()->address(),
             Kt,
             Vt,
             Mt,
             T,
             head});
        SetRuntimeArgs(
            program,
            writer,
            core,
            {out[0].buffer()->address(), out[1].buffer()->address(), Mt * Vt, Kt * Vt, T, head});
        // Identical on every core: the compute kernel is head-agnostic.
        SetRuntimeArgs(program, compute, core, {Kt, Vt, Mt, T});
    }
    return {
        std::move(program),
        GdnDecaySharedVars{
            .reader_kernel_id = reader,
            .writer_kernel_id = writer,
            .compute_kernel_id = compute,
            .cores = cores,
            .num_cores = BH,
            .head_cores = std::move(head_cores)}};
}

void GdnDecayProgramFactory::override_runtime_arguments(
    cached_program_t& cached, const GdnDecayParams&, const GdnDecayInputs& in, std::vector<Tensor>& out) {
    auto& v = cached.shared_variables;
    // EVERY core, not just {0,0}. On a program-cache hit the tensors are new but the cached
    // program still holds the first dispatch's addresses; refreshing only head 0 leaves heads
    // 1..BH-1 reading freed DRAM. That fails silently -- plausible numbers, no error -- and a
    // single-dispatch test cannot see it.
    for (const auto& core : v.head_cores) {
        auto& ra = GetRuntimeArgs(cached.program, v.reader_kernel_id, core);
        ra[0] = in.h.buffer()->address();
        ra[1] = in.g.buffer()->address();
        ra[2] = in.k.buffer()->address();
        ra[3] = in.v.buffer()->address();
        ra[4] = in.beta.buffer()->address();
        ra[5] = in.q.buffer()->address();
        auto& wa = GetRuntimeArgs(cached.program, v.writer_kernel_id, core);
        wa[0] = out[0].buffer()->address();
        wa[1] = out[1].buffer()->address();
    }
}

}  // namespace ttnn::prim
