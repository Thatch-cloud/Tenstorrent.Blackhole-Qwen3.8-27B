"""Read-only native stage snapshots for the bounded ladder diagnostic, not timing evidence."""

from contextlib import contextmanager
from unittest.mock import patch

import native_draft_sdpa


INCLUDE = '#include <cstdint>'
INCLUDE_AFTER = INCLUDE + '\n#include "api/debug/dprint.h"'
REDUCE = '        matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);'
OUTPUT_UPDATE = '''                mul_block_bcast_cols<Sq_chunk_t, vDHt, false, true>(
                    alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out);'''
OUTPUT_BEFORE = '''
                if (!QWEN_DRAFT_EXP_APPROX && processed_k_chunks == 65) {
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
                    CircularBuffer(alias_mm2_prev_out).wait_front(out_chunk_tiles);
                    CircularBuffer(alias_mm2_cur_out).wait_front(out_chunk_tiles);
                    CircularBuffer(cb_exp_max_diff).wait_front(Sq_chunk_t);
                    DEVICE_PRINT("QWEN_OUTPUT_BEFORE q={} chunk={} previous={:.9f} partial={:.9f} correction={:.9f}\\n",
                        local_q_start + q_iter - iter_q_start, processed_k_chunks,
                        TSLICE(alias_mm2_prev_out, 0,
                            (SliceRange{.h0=2, .h1=3, .hs=1, .w0=5, .w1=6, .ws=1}), true, true),
                        TSLICE(alias_mm2_cur_out, 0,
                            (SliceRange{.h0=2, .h1=3, .hs=1, .w0=5, .w1=6, .ws=1}), true, true),
                        TSLICE(cb_exp_max_diff, 0,
                            (SliceRange{.h0=2, .h1=3, .hs=1, .w0=0, .w1=1, .ws=1}), true, true));
#endif
                }
'''
EXP_UPDATE = '''            sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, scale_fp32, true>(
                alias_cur_max, alias_cur_sum, Sk_chunk_t);'''
MASK_UPDATE = '                    add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);'
QK_SNAPSHOT = '''
            if (!QWEN_DRAFT_EXP_APPROX && processed_k_chunks == 65) {
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
                CircularBuffer(cb_qk_im).wait_front(Sk_chunk_t * Sq_chunk_t);
                DEVICE_PRINT("QWEN_QK_STAGE q={} values={:.9f}\\n",
                    local_q_start + q_iter - iter_q_start,
                    TSLICE(cb_qk_im, 0,
                        (SliceRange{.h0=2, .h1=3, .hs=1, .w0=0, .w1=15, .ws=1}), true, true));
#endif
            }
'''
PARTIAL_SNAPSHOT = '''
        if constexpr (!QWEN_DRAFT_EXP_APPROX) {
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
                CircularBuffer(alias_prev_sum).wait_front(Sq_chunk_t);
                DEVICE_PRINT("QWEN_PARTIAL_SUM q={} low={:.9f} high={:.9f}\\n",
                    local_q_start + q_iter - iter_q_start,
                    TSLICE(alias_prev_sum, 0,
                        (SliceRange{.h0=2, .h1=3, .hs=1, .w0=0, .w1=16, .ws=1}),
                        true, true),
                    TSLICE(alias_prev_sum, 0,
                        (SliceRange{.h0=2, .h1=3, .hs=1, .w0=16, .w1=32, .ws=1}),
                        true, true));
#endif
        }
'''
SNAPSHOT = '''
        if constexpr (!QWEN_DRAFT_EXP_APPROX) {
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
                CircularBuffer(alias_prev_sum).wait_front(Sq_chunk_t);
                CircularBuffer(alias_prev_max).wait_front(Sq_chunk_t);
                CircularBuffer(alias_mm2_prev_out).wait_front(out_chunk_tiles);
                DEVICE_PRINT("QWEN_STAGE q={} numerator={:.9f} denominator={:.9f} maximum={:.9f}\\n",
                    local_q_start + q_iter - iter_q_start,
                    TSLICE(alias_mm2_prev_out, 0,
                        (SliceRange{.h0=2, .h1=3, .hs=1, .w0=5, .w1=6, .ws=1}),
                        true, true),
                    TSLICE(alias_prev_sum, 0,
                        (SliceRange{.h0=2, .h1=3, .hs=1, .w0=0, .w1=1, .ws=1}),
                        true, true),
                    TSLICE(alias_prev_max, 0,
                        (SliceRange{.h0=2, .h1=3, .hs=1, .w0=0, .w1=1, .ws=1}),
                        true, true));
#endif
        }
'''

RECIPROCAL = '''        } else {
            /* cb_cur_sum = 1.0 / cb_cur_sum */
            recip_block_inplace(alias_prev_sum, Sq_chunk_t);'''
RECIPROCAL_SNAPSHOT = '''
            if constexpr (!QWEN_DRAFT_EXP_APPROX) {
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
                CircularBuffer(alias_prev_sum).wait_front(Sq_chunk_t);
                DEVICE_PRINT("QWEN_RECIP q={} reciprocal={:.9f}\\n",
                    local_q_start + q_iter - iter_q_start,
                    TSLICE(alias_prev_sum, 0,
                        (SliceRange{.h0=2, .h1=3, .hs=1, .w0=0, .w1=1, .ws=1}),
                        true, true));
#endif
            }
'''


@contextmanager
def stage_snapshots(*, row=2, column=5):
    if type(row) is not int or type(column) is not int or not 0 <= row < 15 or not 0 <= column < 128:
        raise ValueError('One live proposal row and output channel required')
    tile, lane = divmod(column, 32)
    snapshot = SNAPSHOT.replace('.h0=2, .h1=3', f'.h0={row}, .h1={row + 1}').replace(
        '.w0=5, .w1=6', f'.w0={lane}, .w1={lane + 1}').replace(
        'TSLICE(alias_mm2_prev_out, 0,', f'TSLICE(alias_mm2_prev_out, {tile},')
    reciprocal = RECIPROCAL_SNAPSHOT.replace('.h0=2, .h1=3', f'.h0={row}, .h1={row + 1}')
    partial = PARTIAL_SNAPSHOT.replace('.h0=2, .h1=3', f'.h0={row}, .h1={row + 1}')
    qk_snapshot = QK_SNAPSHOT.replace('.h0=2, .h1=3', f'.h0={row}, .h1={row + 1}')
    output_before = OUTPUT_BEFORE.replace('.h0=2, .h1=3', f'.h0={row}, .h1={row + 1}').replace(
        '.w0=5, .w1=6', f'.w0={lane}, .w1={lane + 1}')
    for name in ('alias_mm2_prev_out', 'alias_mm2_cur_out'):
        output_before = output_before.replace(f'TSLICE({name}, 0,', f'TSLICE({name}, {tile},')
    original = native_draft_sdpa.replacements

    def replacements():
        substitutions = original()
        substitutions['compute_common.hpp'] += (
            (INCLUDE, INCLUDE_AFTER), (REDUCE, partial + REDUCE + snapshot),
            (OUTPUT_UPDATE, output_before + OUTPUT_UPDATE),
            (EXP_UPDATE, qk_snapshot.replace('QWEN_QK_STAGE', 'QWEN_QK_SCORES') + EXP_UPDATE
                + qk_snapshot.replace('QWEN_QK_STAGE', 'QWEN_QK_EXP')),
            (MASK_UPDATE, qk_snapshot.replace('QWEN_QK_STAGE', 'QWEN_QK_PRE_MASK') + MASK_UPDATE
                + qk_snapshot.replace('QWEN_QK_STAGE', 'QWEN_QK_POST_MASK')),
            (RECIPROCAL, RECIPROCAL + reciprocal))
        return substitutions

    with patch.object(native_draft_sdpa, 'replacements', replacements):
        yield
