// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#include "gdn_norm_gate_nanobind.hpp"
#include "gdn_norm_gate.hpp"

#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>

namespace ttnn::operations::transformer {

void bind_gdn_norm_gate(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Fused GDN decode output norm + gate: rms_norm(o) * norm_w * silu(z), written straight
        into the [1, B, H*V] TILE layout the out-projection consumes. Takes o as the fused
        recurrence writes it (ROW_MAJOR sticks) so no relayout/reshape/norm/silu/multiply ops.

        Args:
            o (ttnn.Tensor):      [B, 1, H, V] ROW_MAJOR (bf16 or fp32)
            z (ttnn.Tensor):      [1, Bz, W] TILE, Bz >= B, W >= z_col_offset + H*V
            norm_w (ttnn.Tensor): [1, 1, V] TILE
            num_heads (int):      H

        Keyword Args:
            batch (int, optional): B (default: o's stick count / H)
            z_col_offset (int): column of z where head 0 starts (multiple of 32); default 0
            epsilon (float): rms_norm epsilon; default 1e-6
            memory_config (ttnn.MemoryConfig, optional)

        Returns:
            ttnn.Tensor: gated [1, B, H*V] TILE, z's dtype.
        )doc";

    ttnn::bind_function<"gdn_decode_norm_gate", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::gdn_decode_norm_gate,
        nb::arg("o").noconvert(),
        nb::arg("z").noconvert(),
        nb::arg("norm_w").noconvert(),
        nb::arg("num_heads"),
        nb::kw_only(),
        nb::arg("batch") = nb::none(),
        nb::arg("z_col_offset") = 0,
        nb::arg("epsilon") = 1e-6f,
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::transformer
