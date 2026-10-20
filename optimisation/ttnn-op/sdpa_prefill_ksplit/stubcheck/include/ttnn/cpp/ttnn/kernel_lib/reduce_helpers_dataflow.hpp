// Syntax-check stub of ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp: the scaler call of the pinned writer.
#pragma once
#include "stub_tt_api.hpp"

namespace ckernel {
enum class PoolType { SUM, AVG, MAX };
enum class ReduceDim { REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR };
}  // namespace ckernel
namespace dataflow_kernel_lib {
constexpr uint32_t SUM_AND_MAX_REDUCE_FACTOR = 1;
template <uint32_t cb, ckernel::PoolType pool, ckernel::ReduceDim dim, uint32_t factor>
void calculate_and_prepare_reduce_scaler();
}  // namespace dataflow_kernel_lib
