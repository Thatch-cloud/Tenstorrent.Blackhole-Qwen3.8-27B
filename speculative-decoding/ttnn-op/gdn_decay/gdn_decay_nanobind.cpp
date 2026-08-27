// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gdn_decay_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/transformer/gdn_decay/gdn_decay.hpp"

namespace ttnn::transformer {

void bind_gdn_decay(nanobind::module_& mod) {
    namespace nb = nanobind;
    // Python-facing name is the accurate one; the C++ directory stays gdn_decay so the
    // registration in sources.cmake / CMakeLists / transformer_nanobind is untouched.
    mod.def(
        "gdn_recurrent_step",
        [](const ttnn::Tensor& h,
           const ttnn::Tensor& g,
           const ttnn::Tensor& k,
           const ttnn::Tensor& v,
           const ttnn::Tensor& beta,
           const ttnn::Tensor& q,
           const std::optional<ttnn::MemoryConfig>& memory_config,
           const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
            return ttnn::transformer::gdn_decay(h, g, k, v, beta, q, memory_config, compute_kernel_config);
        },
        nb::arg("h"), nb::arg("g"), nb::arg("k"), nb::arg("v"), nb::arg("beta"), nb::arg("q"),
        nb::kw_only(),
        nb::arg("memory_config") = std::nullopt,
        nb::arg("compute_kernel_config") = std::nullopt,
        R"doc(T fused Gated DeltaNet recurrent decode steps, BH heads in parallel (one per core).

h [BH,1,K,V], k/q [BH,T,M,K], v [BH,T,M,V], and g/beta [BH,T,1,1] -- per-head-per-token
scalars broadcast on device (g's exp is applied there too). All float32/TILE/DRAM.
Returns (o [BH,T,M,V], states [BH,T,K,V]) -- all T intermediate states.)doc");
}

}  // namespace ttnn::transformer
