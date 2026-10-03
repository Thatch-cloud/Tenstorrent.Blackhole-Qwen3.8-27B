// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <tt-metalium/core_coord.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/gdn_decay/device/gdn_decay_device_operation_types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::prim {

struct GdnDecaySharedVars {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    tt::tt_metal::KernelHandle compute_kernel_id{};
    tt::tt_metal::CoreRangeSet cores{};
    uint32_t num_cores = 0;
    // The actual head->core list, not a grid_y to recompute from: override_runtime_arguments
    // must walk exactly the cores create() launched, and recomputing the mapping in two
    // places is how they drift apart.
    std::vector<tt::tt_metal::CoreCoord> head_cores{};
};

struct GdnDecayProgramFactory {
    using shared_variables_t = GdnDecaySharedVars;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const GdnDecayParams&, const GdnDecayInputs&, std::vector<Tensor>& out);
    static void override_runtime_arguments(
        cached_program_t&, const GdnDecayParams&, const GdnDecayInputs&, std::vector<Tensor>& out);
};

}  // namespace ttnn::prim
