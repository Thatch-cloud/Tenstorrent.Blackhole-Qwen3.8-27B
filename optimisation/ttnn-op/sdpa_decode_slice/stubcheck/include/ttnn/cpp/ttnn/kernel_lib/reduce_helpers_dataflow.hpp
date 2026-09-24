// stub: the two dataflow_kernel_lib helpers writer_decode_all.cpp calls (shaped after its call sites)
#pragma once
#include "stub_decode_api.hpp"

namespace dataflow_kernel_lib {
constexpr uint32_t SUM_AND_MAX_REDUCE_FACTOR = 1;
template <uint32_t cb_id, ckernel::PoolType pool_type, ckernel::ReduceDim reduce_dim, uint32_t reduce_factor>
void calculate_and_prepare_reduce_scaler();
template <uint32_t cb_id>
void prepare_zero_tile();
}  // namespace dataflow_kernel_lib
