// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0

#include "attn_prep_nanobind.hpp"
#include "attn_prep.hpp"

#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

namespace ttnn::operations::transformer {

void bind_attn_prep(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Fused attention decode prologue: from the fused [q|k|v|gate] projection output, one
        program produces q (rms_norm * q_norm_w, partial rotate-half RoPE) [1,B,NH,HD], the gate
        block [1,B,NH,HD], and k (normed + roped) and v padded to 32 heads in the KV update's
        memory config [1,B,32,HD].

        Args:
            qkv (ttnn.Tensor):  [1, 1, B, W] TILE, W >= (NH + 2*NKV + NH) * HD
            cos, sin (ttnn.Tensor): [1, B, 1, RD] TILE
            q_norm_w, k_norm_w (ttnn.Tensor): [1, 1, HD] TILE
            num_heads, num_kv_heads, head_dim, rope_dim (int)
            kv_memory_config (ttnn.MemoryConfig): where k and v go (the height-sharded KV update config)

        Keyword Args:
            batch (int, optional): B (default: qkv's row count)
            epsilon (float): rms_norm epsilon; default 1e-6
            memory_config (ttnn.MemoryConfig, optional): q and gate placement (default DRAM)

        Returns:
            tuple: q, gate, k, v
        )doc";

    ttnn::bind_function<"attn_decode_prep", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::attn_decode_prep,
        nb::arg("qkv").noconvert(),
        nb::arg("cos").noconvert(),
        nb::arg("sin").noconvert(),
        nb::arg("q_norm_w").noconvert(),
        nb::arg("k_norm_w").noconvert(),
        nb::arg("num_heads"),
        nb::arg("num_kv_heads"),
        nb::arg("head_dim"),
        nb::arg("rope_dim"),
        nb::arg("kv_memory_config"),
        nb::kw_only(),
        nb::arg("batch") = nb::none(),
        nb::arg("epsilon") = 1e-6f,
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::transformer
