"""Unqualified ladder sum-buffer direct-unpack candidate; affects all sum consumers."""

from dspark_ladder_factory import transform as ladder_transform


ANCHOR = '    KernelDescriptor compute_desc;'
INSERT = '''    ComputeConfigDescriptor::UnpackToDestModes qwen_sum_unpack_modes;
    if (qwen_draft_fp32_intermediates) {
        TT_FATAL(sum_df == tt::DataFormat::Float32, "Ladder direct sum unpack requires FP32 storage");
        qwen_sum_unpack_modes.resize(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
        qwen_sum_unpack_modes.at(cb_ids.sum_A) = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        qwen_sum_unpack_modes.at(cb_ids.sum_B) = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

'''
CONFIG = '        .dst_full_sync_en = dst_full_sync_en,\n        .math_approx_mode = math_approx_mode,'
CONFIG_AFTER = '        .dst_full_sync_en = dst_full_sync_en,\n        .unpack_to_dest_mode = qwen_sum_unpack_modes,\n        .math_approx_mode = math_approx_mode,'


def transform(source):
    candidate = ladder_transform(source)
    if candidate.count(ANCHOR.encode()) != 1 or candidate.count(CONFIG.encode()) != 1:
        raise ValueError('Exact unique SDPA compute descriptor anchors required')
    return candidate.replace(ANCHOR.encode(), (INSERT + ANCHOR).encode()).replace(
        CONFIG.encode(), CONFIG_AFTER.encode())


def remove_unpack_transform(candidate):
    if (candidate.count((INSERT + ANCHOR).encode()) != 1
            or candidate.count(CONFIG_AFTER.encode()) != 1):
        raise ValueError('Complete unique sum-unpack transformation required')
    return candidate.replace((INSERT + ANCHOR).encode(), ANCHOR.encode()).replace(
        CONFIG_AFTER.encode(), CONFIG.encode())
