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
    return source[:begin] + REPLACEMENT + source[end:]


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
