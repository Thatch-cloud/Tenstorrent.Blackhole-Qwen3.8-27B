// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#include "gdn_conv_gates_nanobind.hpp"
#include "gdn_conv_gates.hpp"

#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace ttnn::operations::transformer {

void bind_gdn_conv_gates(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Fused GDN decode conv + gates (one program): advances the conv shift-register in
        place, computes silu(sum_j window[j] * taps[j]) over the K-tap window, and the two
        gates beta = sigmoid(b), g = neg_exp_A * softplus(a + dt_bias).

        Args:
            x (ttnn.Tensor):            [1, B, C] TILE, the new qkv projection
            conv_states (list[Tensor]): K x [1, Bmax, C] TILE, updated IN PLACE
            taps (list[Tensor]):        K x [1, 1, C] TILE
            a, b (ttnn.Tensor):         [1, B, Nv] TILE
            dt_bias, neg_exp_A:         [1, 1, Nv] TILE

        Keyword Args:
            batch (int, optional): active rows of x (default: all). Rows at or beyond it
                enter the shift register as zeros (bucketed decode).
            memory_config (ttnn.MemoryConfig, optional): placement of the three outputs.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]: conv_out [1,B,C], beta [1,B,Nv], g [1,B,Nv].
        )doc";

    ttnn::bind_function<"gdn_decode_conv_gates", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::gdn_decode_conv_gates,
        nb::arg("x").noconvert(),
        nb::arg("conv_states"),
        nb::arg("taps"),
        nb::arg("a").noconvert(),
        nb::arg("b").noconvert(),
        nb::arg("dt_bias").noconvert(),
        nb::arg("neg_exp_A").noconvert(),
        nb::kw_only(),
        nb::arg("batch") = nb::none(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::transformer
