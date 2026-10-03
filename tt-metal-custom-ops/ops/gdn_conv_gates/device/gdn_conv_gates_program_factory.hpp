// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "gdn_conv_gates_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct GdnConvGatesProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GdnConvGatesParams& attrs, const GdnConvGatesInputs& in, std::vector<Tensor>& outputs);
};

}  // namespace ttnn::prim
