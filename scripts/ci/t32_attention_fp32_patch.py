"""Pinned source transformation for a disposable T32 draft attention runtime experiment."""

import hashlib


SOURCE_SHA256 = 'a263559fe23cdf6fa8194604b238a939d299a356592eae1c7b2df11868383ebc'
SOURCE_PATH = 'ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp'
BEFORE = '''    tt::DataFormat im_df =
        tt::DataFormat::Float16_b;  // Keep most intermediates in bf16 to save L1; opt-in fp32 per-CB below.'''
AFTER = '''    const bool t32_draft_fp32_intermediates =
        B == 1 && NQH == 16 && NKH == 4 && Sq == 32 && DH == 128 &&
        !is_causal && use_provided_mask && !use_streaming_compute && fp32_dest_acc_en &&
        q_df == tt::DataFormat::Float16_b && k_df == tt::DataFormat::Float16_b &&
        v_df == tt::DataFormat::Float16_b;
    tt::DataFormat im_df = t32_draft_fp32_intermediates ?
        tt::DataFormat::Float32 : tt::DataFormat::Float16_b;'''


def patched_bytes(source, *, variant='fp32'):
    if variant not in ('fp32', 'output-only'):
        raise ValueError('Explicit FP32 intermediate variant required')
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError('Exact pinned SDPA program factory required')
    if source.count(BEFORE.encode()) != 1:
        raise ValueError('Unique intermediate format anchor required')
    result = source.replace(BEFORE.encode(), AFTER.encode())
    if variant == 'output-only':
        before = b'    tt::DataFormat stats_df = im_df;'
        if result.count(before) != 1:
            raise ValueError('Unique statistics format anchor required')
        result = result.replace(before, b'    tt::DataFormat stats_df = tt::DataFormat::Float16_b;')
    return result
