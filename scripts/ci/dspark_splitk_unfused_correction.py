"""Simulator-only decomposition of native cross-core softmax correction."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path


HEADER = 'ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp'
START = '                    correction_block<scale_fp32, vector_mode>('
END = '                        Sq_chunk_t);'
REPLACEMENT = '''                    reconfig_data_format(cb_prev_max, cb_m_in);
                    pack_reconfig_data_format(cb_cur_max);
                    max_block<vector_mode>(cb_prev_max, cb_m_in, cb_cur_max, Sq_chunk_t);
                    reconfig_data_format(cb_prev_max, cb_cur_max);
                    pack_reconfig_data_format(cb_exp_max_diff);
                    sub_exp_block<scale_fp32>(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
                    reconfig_data_format(cb_m_in, cb_cur_max);
                    pack_reconfig_data_format(cb_exp_max_diff_2);
                    sub_exp_block<scale_fp32>(cb_m_in, cb_cur_max, cb_exp_max_diff_2, Sq_chunk_t);
                    reconfig_data_format(cb_prev_sum, cb_exp_max_diff);
                    pack_reconfig_data_format(cb_prev_sum);
                    mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);
                    reconfig_data_format(cb_prev_sum_2, cb_exp_max_diff_2);
                    pack_reconfig_data_format(cb_prev_sum_2);
                    mul_block_inplace(cb_prev_sum_2, cb_exp_max_diff_2, Sq_chunk_t);
                    add_block_inplace<true>(cb_prev_sum, cb_prev_sum_2, Sq_chunk_t);
                    reconfig_data_format(cb_prev_sum, cb_prev_sum);
                    pack_reconfig_data_format(cb_cur_sum);
                    move_block<true>(cb_prev_sum, cb_cur_sum, Sq_chunk_t);'''


def transform(source):
    if source.count(START) != 1:
        raise ValueError('Unique native correction call required')
    begin = source.index(START)
    end = source.index(END, begin) + len(END)
    block = source[begin:end]
    if block.count('cb_exp_max_diff_2') != 1 or block.count('cb_prev_sum_2') != 1:
        raise ValueError('Native correction arguments changed')
    result = source[:begin] + REPLACEMENT + source[end:]
    reciprocal = '            recip_block_inplace(cb_prev_sum, Sq_chunk_t);'
    precise = '''            {
                CircularBuffer denominator(cb_prev_sum);
                reconfig_data_format_srca(cb_prev_sum);
                copy_tile_to_dst_init_short(cb_prev_sum);
                recip_tile_init();
                pack_reconfig_data_format(cb_prev_sum);
                denominator.wait_front(Sq_chunk_t);
                for (uint32_t tile = 0; tile < Sq_chunk_t; ++tile) {
                    tile_regs_acquire();
                    copy_tile(cb_prev_sum, tile, 0);
                    MATH((recip_tile_first_column<false>(0)));
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_prev_sum);
                    tile_regs_release();
                }
                denominator.pop_front(Sq_chunk_t);
                denominator.reserve_back(Sq_chunk_t);
                denominator.push_back(Sq_chunk_t);
            }'''
    if result.count(reciprocal) != 1:
        raise ValueError('Unique final decode reciprocal required')
    result = result.replace(reciprocal, precise)
    normalize = '            mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_prev_sum);'
    sfpu_normalize = '''            {
                CircularBuffer numerator(cb_out_accumulate_im);
                CircularBuffer reciprocal(cb_prev_sum);
                numerator.wait_front(out_chunk_tiles);
                reciprocal.wait_front(Sq_chunk_t);
                pack_reconfig_data_format(cb_out_accumulate_im);
                PACK((llk_pack_reconfig_l1_acc(false)));
                for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
                    for (uint32_t column = 0; column < vDHt; ++column) {
                        tile_regs_acquire();
                        reconfig_data_format_srca(cb_out_accumulate_im);
                        copy_tile_to_dst_init_short(cb_out_accumulate_im);
                        copy_tile(cb_out_accumulate_im, 0, 0);
                        reconfig_data_format_srca(cb_prev_sum);
                        copy_tile_to_dst_init_short(cb_prev_sum);
                        copy_tile(cb_prev_sum, row, 1);
                        sfpu_mul_bcast_col_init();
                        sfpu_mul_bcast_col(0, 1);
                        tile_regs_commit();
                        numerator.pop_front(1);
                        numerator.reserve_back(1);
                        tile_regs_wait();
                        pack_tile(0, cb_out_accumulate_im);
                        tile_regs_release();
                        numerator.push_back(1);
                    }
                }
                reciprocal.pop_front(Sq_chunk_t);
            }'''
    include = '#include "api/compute/eltwise_unary/recip.h"'
    if result.count(normalize) != 1 or result.count(include) != 1:
        raise ValueError('Unique native final normalization and include required')
    return result.replace(normalize, sfpu_normalize).replace(include,
        include + '\n#include "api/compute/sfpu_binary_bcast.h"')


@contextmanager
def unfused_correction_scope():
    if os.environ.get('QWEN_PRECISE_DRAFT_ACTIVE') != '1':
        yield
        return
    if os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR'):
        raise ValueError('Unfused correction remains simulator-only')
    path = Path(os.environ['TT_METAL_HOME']) / HEADER
    original = path.read_bytes()
    replacement = transform(original.decode()).encode()
    print(json.dumps(dict(stage='splitk-unfused-correction',
        original_sha256=hashlib.sha256(original).hexdigest(),
        patched_sha256=hashlib.sha256(replacement).hexdigest())), flush=True)
    try:
        path.write_bytes(replacement)
        yield
    finally:
        path.write_bytes(original)
