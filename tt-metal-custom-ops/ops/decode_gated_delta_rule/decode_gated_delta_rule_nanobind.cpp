// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "decode_gated_delta_rule_nanobind.hpp"
#include "decode_gated_delta_rule.hpp"

#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

namespace ttnn::operations::transformer {

void bind_decode_gated_delta_rule(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Fused T=1 (decode step) Gated Delta Rule forward: one reader/compute/writer
        program, one core per head, replacing the ~12-kernel python decode graph.

        Args:
            q (ttnn.Tensor):    [B, 1, H, K] TILE
            k (ttnn.Tensor):    [B, 1, H, K] TILE
            v (ttnn.Tensor):    [B, 1, H, V] TILE
            beta (ttnn.Tensor): [B, 1, H] TILE
            g (ttnn.Tensor):    [B, 1, H] TILE, log-space decay

        Keyword Args:
            scale (float, optional): defaults to K**-0.5.
            initial_state (ttnn.Tensor, optional): [B, H, K, V] TILE, same dtype; zeros if absent.
            inplace_state (bool): default False. When True (requires initial_state),
                new_state is written into initial_state's buffer and initial_state is
                returned as new_state (trace-safe, no allocation).
            memory_config (ttnn.MemoryConfig, optional).

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]: o [B,1,H,V] TILE, new_state [B,H,K,V] TILE.
        )doc";

    ttnn::bind_function<"decode_gated_delta_rule", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::decode_gated_delta_rule,
        nb::arg("q").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("beta").noconvert(),
        nb::arg("g").noconvert(),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("initial_state") = nb::none(),
        nb::arg("inplace_state") = false,
        nb::arg("memory_config") = nb::none());

    const auto* doc_packed =
        R"doc(
        Packed-layout variant of decode_gated_delta_rule: takes the conv+gates kernel's
        outputs directly (qkv [1,B,C] TILE with channels [q|k|v] head-major, beta/g [1,B,H]
        TILE) and gathers each head's rows in the reader, GQA included. Same outputs.

        Args:
            qkv (ttnn.Tensor):  [1, B, C] TILE, C = 2*num_k_heads*head_k + num_v_heads*head_v
            beta, g (ttnn.Tensor): [1, B, num_v_heads] TILE
            num_k_heads, num_v_heads, head_k, head_v (int)

        Keyword Args:
            scale, initial_state, inplace_state, memory_config: as decode_gated_delta_rule.
            z (ttnn.Tensor, optional): [1, Bz, W] TILE. With norm_w, fuses the output norm and
                gate: the first output becomes gated [1, B, H*V] TILE =
                rms_norm(o) * norm_w * silu(z[:, :, z_col_offset + h*V ...]).
            norm_w (ttnn.Tensor, optional): [1, 1, V] TILE.
            z_col_offset (int): column of head 0 in z (multiple of 32). epsilon (float): 1e-6.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]: o [B,1,H,V] ROW_MAJOR (or gated [1,B,H*V] TILE),
            new_state [B,H,K,V] TILE.
        )doc";

    ttnn::bind_function<"decode_gated_delta_rule_packed", "ttnn.transformer.">(
        mod,
        doc_packed,
        &ttnn::transformer::decode_gated_delta_rule_packed,
        nb::arg("qkv").noconvert(),
        nb::arg("beta").noconvert(),
        nb::arg("g").noconvert(),
        nb::arg("num_k_heads"),
        nb::arg("num_v_heads"),
        nb::arg("head_k"),
        nb::arg("head_v"),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("initial_state") = nb::none(),
        nb::arg("inplace_state") = false,
        nb::arg("memory_config") = nb::none(),
        nb::arg("z") = nb::none(),
        nb::arg("norm_w") = nb::none(),
        nb::arg("z_col_offset") = 0,
        nb::arg("epsilon") = 1e-6f);
}

}  // namespace ttnn::operations::transformer
