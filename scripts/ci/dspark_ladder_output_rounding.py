"""Final-output BF16 rounding experiment; no intermediate-format changes."""

from contextlib import contextmanager
from unittest.mock import patch

import native_draft_sdpa


TEMPLATE = 'template <uint32_t rows, uint32_t cols, bool immediate_pop, bool pack_accumulate>\nvoid mul_block_bcast_cols('
INIT = '''    mul_bcast_cols_init(in0_cb, in1_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(rows);'''
MULTIPLY = '                    mul_tiles_bcast_cols(in0_cb, in1_cb, in0_index, i, j);'
FINAL = '            mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(alias_mm2_prev_out, alias_prev_sum, cb_out);'
CAST = 'static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)'


@contextmanager
def final_output_rounding():
    original = native_draft_sdpa.replacements

    def replacements():
        substitutions = original()
        substitutions['compute_common.hpp'] += (
            ('#include <cstdint>', '#include <cstdint>\n#include "api/compute/eltwise_unary/typecast.h"'),
            (TEMPLATE, TEMPLATE.replace('bool pack_accumulate>', 'bool pack_accumulate, bool round_output = false>')),
            (INIT, INIT + f'\n    if constexpr (round_output) typecast_tile_init<{CAST}>();'),
            (MULTIPLY, MULTIPLY + f'\n                    if constexpr (round_output) typecast_tile<{CAST}>(j);'),
            (FINAL, FINAL.replace('false, false>',
                'false, false, !QWEN_DRAFT_EXP_APPROX && get_compile_time_arg_val(3) == 2112>')))
        return substitutions

    with patch.object(native_draft_sdpa, 'replacements', replacements):
        yield
