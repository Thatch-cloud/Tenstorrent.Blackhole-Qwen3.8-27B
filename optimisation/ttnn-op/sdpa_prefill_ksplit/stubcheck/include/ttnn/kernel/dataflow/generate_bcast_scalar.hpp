// Syntax-check stub of ttnn/kernel/dataflow/generate_bcast_scalar.hpp: the one call the pinned writer makes (and the K-split writer makes once more).
#pragma once
#include "stub_tt_api.hpp"

void generate_bcast_col_scalar(const CircularBuffer& cb, uint32_t packed_bf16_pair);
