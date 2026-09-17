"""Unqualified mode-aware destination stride for native Blackhole correction."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path


HEADER = 'tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h'
BEFORE = '''    constexpr uint32_t prev_max_base_idx = 0;
    constexpr uint32_t worker_max_base_idx = 32;
    constexpr uint32_t cur_max_base_idx = 64;
    constexpr uint32_t prev_sum_base_idx = 96;
    constexpr uint32_t worker_sum_base_idx = 128;'''
AFTER = '''    constexpr uint32_t correction_tile_stride = DST_ACCUM_MODE ? 64 : 32;
    constexpr uint32_t prev_max_base_idx = 0;
    constexpr uint32_t worker_max_base_idx = correction_tile_stride;
    constexpr uint32_t cur_max_base_idx = 2 * correction_tile_stride;
    constexpr uint32_t prev_sum_base_idx = 3 * correction_tile_stride;
    constexpr uint32_t worker_sum_base_idx = 4 * correction_tile_stride;'''


def transform(source):
    if source.count(BEFORE) != 1 or source.count('inline void calculate_fused_max_sub_exp_add_tile(') != 1:
        raise ValueError('Exact native fused correction source required')
    return source.replace(BEFORE, AFTER)


@contextmanager
def correction_stride_scope():
    if os.environ.get('QWEN_PRECISE_DRAFT_ACTIVE') != '1':
        yield
        return
    if os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR'):
        raise ValueError('Correction stride remains simulator-only')
    path = Path(os.environ['TT_METAL_HOME']) / HEADER
    original = path.read_bytes()
    replacement = transform(original.decode()).encode()
    print(json.dumps(dict(stage='splitk-correction-stride',
        original_sha256=hashlib.sha256(original).hexdigest(),
        patched_sha256=hashlib.sha256(replacement).hexdigest())), flush=True)
    try:
        path.write_bytes(replacement)
        yield
    finally:
        path.write_bytes(original)
