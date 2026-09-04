// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>
namespace nb = nanobind;

namespace ttnn::operations::transformer {

void bind_attn_prep(nb::module_& mod);

}  // namespace ttnn::operations::transformer
