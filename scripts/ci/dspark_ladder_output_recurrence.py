"""Ladder-only diagnostic replacing output L1 pack accumulation with explicit addition."""

from contextlib import contextmanager
from unittest.mock import patch

import native_draft_sdpa


BEFORE = '''                mul_block_bcast_cols<Sq_chunk_t, vDHt, false, true>(
                    alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out);'''
AFTER = '''                if constexpr (!QWEN_DRAFT_EXP_APPROX) {
                    mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(alias_mm2_prev_out, cb_exp_max_diff);
                    add_block_inplace(alias_mm2_cur_out, alias_mm2_prev_out, out_chunk_tiles);
                } else {
''' + BEFORE + '''
                }'''


@contextmanager
def explicit_output_recurrence():
    original = native_draft_sdpa.replacements

    def replacements():
        substitutions = original()
        substitutions['compute_common.hpp'] += ((BEFORE, AFTER),)
        return substitutions

    with patch.object(native_draft_sdpa, 'replacements', replacements):
        yield
