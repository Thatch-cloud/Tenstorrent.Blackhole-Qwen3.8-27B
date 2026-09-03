// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>
namespace nb = nanobind;

namespace ttnn::operations::transformer {

void bind_gdn_conv_gates(nb::module_& mod);

}  // namespace ttnn::operations::transformer
