"""Unqualified, geometry-scoped SDPA factory transform for simulator experiments."""

import hashlib


SOURCE = 'ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp'
SOURCE_SHA256 = 'a263559fe23cdf6fa8194604b238a939d299a356592eae1c7b2df11868383ebc'
ANCHOR = '''    tt::DataFormat im_df =
        tt::DataFormat::Float16_b;  // Keep most intermediates in bf16 to save L1; opt-in fp32 per-CB below.
    tt::DataFormat stats_df = im_df;'''
REPLACEMENT = '''    const bool qwen_draft_fp32_intermediates =
        B == 1 && NQH == 16 && NKH == 4 && DHt == 4 && vDHt == 4 &&
        Skt == 266 && Sq_chunk_t == 1 && Sk_chunk_t == 2 &&
        !is_causal && compute_use_provided_mask && !is_chunked &&
        !use_attention_sink && !is_windowed && !use_streaming_compute &&
        fp32_dest_acc_en && !exp_approx_mode;
    tt::DataFormat im_df = qwen_draft_fp32_intermediates ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat stats_df = im_df;'''


def transform(source):
    if not isinstance(source, bytes) or hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError('Exact pinned SDPA factory required')
    if source.count(ANCHOR.encode()) != 1:
        raise ValueError('Unique intermediate-format anchor required')
    return source.replace(ANCHOR.encode(), REPLACEMENT.encode())
