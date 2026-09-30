// Syntax-check stubs for the sdpa_decode dataflow kernels: what the decode reader and writer use beyond
// ../../../sdpa_prefill_chain/stubcheck/include/stub_tt_api.hpp (the prefill reader's surface). Force-included
// by stub_compile.py before each kernel. NOT tt-metal: declarations shaped after the decode call sites, so
// g++ -fsyntax-only can type-check the served stage-3 reader and stock writer and the K64i slice kernels
// against the same surface. A pass is evidence; the JIT compile on card B is the proof.
#pragma once

#include <algorithm>  // the real dataflow API pulls it in; rt_args_common.hpp uses std::min
#include <climits>
#include <cstdint>
#include <optional>
#include <tuple>

#include "stub_tt_api.hpp"

namespace tt {
enum CBIndex : uint8_t {
    c_0 = 0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15,
    c_16, c_17, c_18, c_19, c_20, c_21, c_22, c_23, c_24, c_25, c_26, c_27, c_28, c_29, c_30, c_31,
};
}  // namespace tt

uint32_t get_arg_addr(int index);
uint32_t get_common_arg_addr(int index);

namespace ckernel {
enum class PoolType { SUM, AVG, MAX };
enum class ReduceDim { REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR };
}  // namespace ckernel
