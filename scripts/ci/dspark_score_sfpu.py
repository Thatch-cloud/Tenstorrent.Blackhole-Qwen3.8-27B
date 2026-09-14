"""Unqualified FP32 SFPU score-centering candidate, isolated from defaults."""

from contextlib import contextmanager
from unittest.mock import patch

import native_draft_sdpa


HELPER = r'''
void qwen_prepare_center_scratch(uint32_t maxima_cb, uint32_t scratch_cb) {
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
         '        qwen_normalization_modes.at(cb_ids.qk_im) = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;'))
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
                    '        qwen_prepare_center_scratch(in1_cb, get_compile_time_arg_val(42));')
                replaced_center += 1
            if before == '                sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);':
                call = '                    copy_tile(in0_cb, j, j);'
                if after.count(call) != 1:
                    raise ValueError('Existing scalar corrected score reload required')
                after = after.replace(call, '''                    reconfig_data_format_srca(in0_cb);
                    copy_tile_init(in0_cb);
                    copy_tile(in0_cb, j, j);
                    reconfig_data_format_srca(get_compile_time_arg_val(42));
                    copy_tile_init(get_compile_time_arg_val(42));
                    copy_tile(get_compile_time_arg_val(42), 0, 1);
                    sfpu_sub_bcast_col(j, 1);
                    exp_tile_init<QWEN_DRAFT_EXP_APPROX, scale_fp32, InputClamping::None>();''')
                replaced_copy += 1
            substitutions.append((before, after))
        if (replaced_center, replaced_copy) != (1, 1):
            raise ValueError('Exactly one scalar centering and reload site required')
        substitutions.extend((
            ('#include "api/compute/bcast.h"',
             '#include "api/compute/bcast.h"\n#include "api/compute/sfpu_binary_bcast.h"'),
            ('enum SDPAType {', HELPER + '\nenum SDPAType {'),
            ('    PACK((llk_pack_relu_config(ReluConfig::none())));\n}',
             f'    if constexpr ({condition}) {{ CircularBuffer(get_compile_time_arg_val(42)).pop_front(1); }}\n'
             '    PACK((llk_pack_relu_config(ReluConfig::none())));\n}'),
            ('    for (uint32_t i = 0; i < rows; ++i) {\n        for (uint32_t u = 0; u < granularity; u++) {\n            tile_regs_acquire();',
             f'    if constexpr ({condition}) {{ dst_tiles = 1; granularity = cols; }}\n'
             '    for (uint32_t i = 0; i < rows; ++i) {\n        for (uint32_t u = 0; u < granularity; u++) {\n'
             f'            if constexpr ({condition}) {{ sfpu_sub_bcast_col_init(); }}\n'
             '            tile_regs_acquire();')))
        result['compute_common.hpp'] = tuple(substitutions)
        return result

    with patch.object(native_draft_sdpa, 'replacements', replacements):
        yield
