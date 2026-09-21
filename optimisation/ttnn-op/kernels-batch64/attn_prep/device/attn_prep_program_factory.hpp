// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "attn_prep_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct AttnPrepProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AttnPrepParams& attrs, const AttnPrepInputs& in, std::vector<Tensor>& outputs);
};

}  // namespace ttnn::prim
