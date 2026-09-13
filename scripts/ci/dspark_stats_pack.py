"""Scoped draft SDPA explicit maximum-difference pack format."""

from contextlib import contextmanager
from unittest.mock import patch

import native_draft_sdpa


BEFORE = '    sub_init(in0_cb, in1_cb);\n    exp_tile_init<EXP_APPROX_MODE>();'
AFTER = '    sub_init(in0_cb, in1_cb);\n    pack_reconfig_data_format(out_cb);\n    exp_tile_init<EXP_APPROX_MODE>();'
SELECTOR_ASSERT = '''static_assert(!QWEN_DRAFT_EXP_APPROX,
    "8K diagnostic must execute the precise draft specialization");
static_assert(get_compile_time_arg_val(3) == 266 && get_compile_time_arg_val(8) == 2,
    "8K diagnostic must use 8512 keys and 64-key chunks");
'''
SUM_BEFORE = '        matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);'
SUM_AFTER = '''        if constexpr (!QWEN_DRAFT_EXP_APPROX) {
            CircularBuffer qwen_sum(alias_prev_sum);
            CircularBuffer(cb_identity_scale_in).wait_front(1);
            qwen_sum.wait_front(Sq_chunk_t);
            reconfig_data_format(cb_identity_scale_in, alias_prev_sum);
            pack_reconfig_data_format(alias_prev_sum);
            for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
                tile_regs_acquire();
                reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(
                    alias_prev_sum, cb_identity_scale_in, alias_prev_sum);
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(
                    alias_prev_sum, cb_identity_scale_in, 0, 0, 0);
                reduce_uninit();
                tile_regs_commit();
                qwen_sum.pop_front(1);
                qwen_sum.reserve_back(1);
                tile_regs_wait();
                pack_tile(0, alias_prev_sum);
                tile_regs_release();
                qwen_sum.push_back(1);
            }
        } else {
            matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);
        }'''


@contextmanager
def scoped_stats_pack():
    original = native_draft_sdpa.replacements

    def replacements():
        substitutions = original()
        before, after = substitutions['sdpa.cpp'][0]
        after = after.replace('#include "compute_common.hpp"', SELECTOR_ASSERT + '#include "compute_common.hpp"')
        substitutions['sdpa.cpp'] = ((before, after),)
        substitutions['compute_common.hpp'] += ((BEFORE, AFTER), (SUM_BEFORE, SUM_AFTER))
        return substitutions

    with patch.object(native_draft_sdpa, 'replacements', replacements):
        yield
