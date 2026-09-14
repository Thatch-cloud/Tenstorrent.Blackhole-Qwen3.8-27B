"""Unqualified FP32 SFPU score-centering candidate, isolated from defaults."""

from contextlib import contextmanager
from unittest.mock import patch

import native_draft_sdpa


@contextmanager
def factory_scope():
    import dspark_ladder_build

    original = dspark_ladder_build.factory_transform

    def transformed(source, *, reverse=False):
        if reverse:
            return original(factory_transform(source, reverse=True), reverse=True)
        return factory_transform(original(source))

    with patch.object(dspark_ladder_build, 'factory_transform', transformed):
        yield


@contextmanager
def kernel_scope():
    import dspark_ladder_score_center

    original = dspark_ladder_score_center.scalar_score_center

    @contextmanager
    def centered(key_tiles=2112):
        with original(key_tiles=key_tiles), sfpu_score_center(key_tiles=16 if key_tiles == 40 else key_tiles):
            yield

    with patch.object(dspark_ladder_score_center, 'scalar_score_center', centered):
        yield


HELPER = r'''
void qwen_copy_fp32_init(uint32_t source_cb) {
    reconfig_data_format_srca(source_cb);
    state_configure(source_cb, __builtin_LINE());
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(0, 0, source_cb)));
    MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE>(source_cb)));
}

void qwen_copy_fp32(uint32_t source_cb, uint32_t destination) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(source_cb, 0)));
    MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, true>(destination, source_cb)));
}

void qwen_stage_score_tile(uint32_t source_cb, uint32_t scratch_cb) {
    CircularBuffer(source_cb).wait_front(1);
    CircularBuffer(scratch_cb).reserve_back(1);
    reconfig_data_format_srca(source_cb);
    copy_tile_init(source_cb);
    pack_reconfig_data_format(scratch_cb);
    tile_regs_acquire();
    copy_tile(source_cb, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, scratch_cb);
    tile_regs_release();
    CircularBuffer(scratch_cb).push_back(1);
    CircularBuffer(scratch_cb).wait_front(1);
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
    const auto source_address = get_local_cb_interface(source_cb).fifo_rd_ptr << cb_addr_shift;
    const auto scratch_address = get_local_cb_interface(scratch_cb).fifo_rd_ptr << cb_addr_shift;
    const volatile uint32_t* source = reinterpret_cast<const volatile uint32_t*>(source_address);
    volatile uint32_t* scratch = reinterpret_cast<volatile uint32_t*>(scratch_address);
    for (uint32_t index = 0; index < 1024; ++index) {
        scratch[index] = source[index];
    }
#endif
}

void qwen_prepare_center_scratch(uint32_t maxima_cb, uint32_t scratch_cb) {
    DEVICE_PRINT("QWEN_SFPU_CENTER_PREPARE\n");
    CircularBuffer(maxima_cb).wait_front(1);
    CircularBuffer(scratch_cb).reserve_back(1);
    reconfig_data_format_srca(maxima_cb);
    copy_tile_init(maxima_cb);
    pack_reconfig_data_format(scratch_cb);
    tile_regs_acquire();
    copy_tile(maxima_cb, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, scratch_cb);
    tile_regs_release();
    CircularBuffer(scratch_cb).push_back(1);
    CircularBuffer(scratch_cb).wait_front(1);
    DEVICE_PRINT("QWEN_SFPU_CENTER_SCRATCH_READY\n");
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
    const auto source_address = get_local_cb_interface(maxima_cb).fifo_rd_ptr << cb_addr_shift;
    const auto scratch_address = get_local_cb_interface(scratch_cb).fifo_rd_ptr << cb_addr_shift;
    const volatile uint32_t* source = reinterpret_cast<const volatile uint32_t*>(source_address);
    volatile uint32_t* scratch = reinterpret_cast<volatile uint32_t*>(scratch_address);
    for (uint32_t index = 0; index < 1024; ++index) {
        scratch[index] = 0;
    }
    for (uint32_t row = 0; row < 32; ++row) {
        const uint32_t first = row < 16 ? row * 16 : 512 + (row - 16) * 16;
        const uint32_t maximum = source[first];
        scratch[first] = maximum == 0xff800000U ? 0U : maximum;
    }
#endif
}
'''


def factory_transform(source, *, reverse=False):
    changes = (
        ('    } else if (qwen_draft_fp32_intermediates && Skt == 2112) {',
         '    } else if (qwen_draft_fp32_intermediates && (Skt == 2112 || Skt == 16)) {'),
        ('    if (qwen_draft_fp32_intermediates && Skt == 2112) {',
         '    if (qwen_draft_fp32_intermediates && (Skt == 2112 || Skt == 16)) {'),
        ('        qwen_normalization_modes.at(cb_ids.recip_scratch) = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;',
         '        qwen_normalization_modes.at(cb_ids.recip_scratch) = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;\n'
         '        qwen_normalization_modes.at(qwen_score_scratch_cb) = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;'),
        ('    cb_ids.qk_im = allocate_tile_cb(qk_tiles, qk_im_tile_size, qk_im_df);',
         '    uint32_t qwen_score_scratch_cb = cb_ids.q_in;\n'
         '    if (qwen_draft_fp32_intermediates && (Skt == 2112 || Skt == 16)) {\n'
         '        qwen_score_scratch_cb = allocate_tile_cb(1, tt::tile_size(tt::DataFormat::Float32), tt::DataFormat::Float32);\n'
         '    }\n    cb_ids.qk_im = allocate_tile_cb(qk_tiles, qk_im_tile_size, qk_im_df);'),
        ('    compute_desc.defines = defines;',
         '    compute_desc.defines = defines;\n'
         '    compute_desc.defines["QWEN_SCORE_SCRATCH_CB"] = std::to_string(qwen_score_scratch_cb);'))
    for before, after in (reversed(changes) if reverse else changes):
        if reverse:
            before, after = after, before
        if source.count(before.encode()) != 1:
            raise ValueError('Exact scratch allocation and FP32 unpack anchors required')
        source = source.replace(before.encode(), after.encode())
    return source


@contextmanager
def sfpu_score_center(*, key_tiles):
    if key_tiles not in (16, 2112):
        raise ValueError('Explicit small simulator or 64K candidate geometry required')
    original = native_draft_sdpa.replacements
    condition = f'!QWEN_DRAFT_EXP_APPROX && get_compile_time_arg_val(3) == {key_tiles}'

    def replacements():
        result = original()
        substitutions = []
        replaced_center = replaced_copy = 0
        for before, after in result['compute_common.hpp']:
            if before == '    sub_bcast_cols_init(in0_cb, in1_cb);':
                call = '        qwen_scalar_score_transform(in0_cb, in1_cb, rows * cols, cols, false);'
                if after.count(call) != 1:
                    raise ValueError('Existing scalar centering branch required')
                after = after.replace(call,
                    '        static_assert(rows == 1);\n'
                    '        DEVICE_PRINT("QWEN_SFPU_CENTER_ENTER\\n");\n'
                    '        qwen_prepare_center_scratch(in1_cb, get_compile_time_arg_val(42));')
                replaced_center += 1
            if before == '                sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);':
                call = '                    copy_tile(in0_cb, j, j);'
                if after.count(call) != 1:
                    raise ValueError('Existing scalar corrected score reload required')
                after = after.replace(call, '''                    qwen_copy_fp32_init(QWEN_SCORE_SCRATCH_CB);
                    qwen_copy_fp32(QWEN_SCORE_SCRATCH_CB, j);
                    qwen_copy_fp32_init(get_compile_time_arg_val(42));
                    qwen_copy_fp32(get_compile_time_arg_val(42), 1);
                    sfpu_sub_bcast_col(j, 1);
                    exp_tile_init<QWEN_DRAFT_EXP_APPROX, scale_fp32, InputClamping::None>();''')
                replaced_copy += 1
            substitutions.append((before, after))
        if (replaced_center, replaced_copy) != (1, 1):
            raise ValueError('Exactly one scalar centering and reload site required')
        substitutions.extend((
            ('            tile_regs_release();\n            if constexpr (do_reduce) {',
             '            tile_regs_release();\n'
             f'            if constexpr ({condition}) {{ CircularBuffer(QWEN_SCORE_SCRATCH_CB).pop_front(1); }}\n'
             '            if constexpr (do_reduce) {'),
            ('#include "api/compute/bcast.h"',
             '#include "api/compute/bcast.h"\n#include "api/compute/sfpu_binary_bcast.h"'),
            ('void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {',
             HELPER + '\nvoid recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {'),
            ('    PACK((llk_pack_relu_config(ReluConfig::none())));\n}',
             f'    if constexpr ({condition}) {{ CircularBuffer(get_compile_time_arg_val(42)).pop_front(1); }}\n'
             '    PACK((llk_pack_relu_config(ReluConfig::none())));\n}'),
            ('    for (uint32_t i = 0; i < rows; ++i) {\n        for (uint32_t u = 0; u < granularity; u++) {\n            tile_regs_acquire();',
             f'    if constexpr ({condition}) {{ dst_tiles = 1; granularity = cols; }}\n'
             '    for (uint32_t i = 0; i < rows; ++i) {\n        for (uint32_t u = 0; u < granularity; u++) {\n'
             f'            if constexpr ({condition}) {{ qwen_stage_score_tile(in0_cb, QWEN_SCORE_SCRATCH_CB); }}\n'
             f'            if constexpr ({condition}) {{ sfpu_sub_bcast_col_init(); }}\n'
             '            tile_regs_acquire();')))
        result['compute_common.hpp'] = tuple(substitutions)
        return result

    with patch.object(native_draft_sdpa, 'replacements', replacements):
        yield
