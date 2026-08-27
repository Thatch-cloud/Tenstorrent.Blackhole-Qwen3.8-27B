// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <nanobind/nanobind.h>

namespace ttnn::transformer {
void bind_gdn_decay(nanobind::module_& mod);
}  // namespace ttnn::transformer
