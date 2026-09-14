"""Unqualified dedicated-scratch SFPU normalization candidate."""

from contextlib import contextmanager
from unittest.mock import patch

import native_draft_sdpa


def factory_transform(source, *, reverse=False):
    substitutions = (
        ('    if (use_streaming_compute) {\n        cb_ids.recip_scratch = allocate_tile_cb(1, im_tile_size, im_df);\n    }',
         '    if (use_streaming_compute) {\n        cb_ids.recip_scratch = allocate_tile_cb(1, im_tile_size, im_df);\n'
         '    } else if (qwen_draft_fp32_intermediates && Skt == 2112) {\n'
         '        cb_ids.recip_scratch = allocate_tile_cb(1, tt::tile_size(tt::DataFormat::Float32), tt::DataFormat::Float32);\n    }'),
        ('    compute_desc.config = ComputeConfigDescriptor{',
         '    std::vector<tt::tt_metal::UnpackToDestMode> qwen_normalization_modes;\n'
         '    if (qwen_draft_fp32_intermediates && Skt == 2112) {\n'
         '        qwen_normalization_modes.resize(64, tt::tt_metal::UnpackToDestMode::Default);\n'
         '        qwen_normalization_modes.at(cb_ids.recip_scratch) = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;\n'
         '    }\n    compute_desc.config = ComputeConfigDescriptor{'),
        ('        .dst_full_sync_en = dst_full_sync_en,\n        .math_approx_mode = math_approx_mode,',
         '        .dst_full_sync_en = dst_full_sync_en,\n'
         '        .unpack_to_dest_mode = qwen_normalization_modes,\n        .math_approx_mode = math_approx_mode,'))
    for before, after in (reversed(substitutions) if reverse else substitutions):
        if reverse:
            before, after = after, before
        if source.count(before.encode()) != 1:
            raise ValueError('Unique normalization factory anchor required')
        source = source.replace(before.encode(), after.encode())
    return source


HELPER = r'''
template <uint32_t columns>
void qwen_normalize_scratch(uint32_t numerator_cb, uint32_t reciprocal_cb,
                            uint32_t scratch_cb, uint32_t output_cb) {
    CircularBuffer(reciprocal_cb).wait_front(1);
    CircularBuffer(scratch_cb).reserve_back(1);
    reconfig_data_format_srca(reciprocal_cb);
    copy_tile_init(reciprocal_cb);
    pack_reconfig_data_format(scratch_cb);
    tile_regs_acquire();
    copy_tile(reciprocal_cb, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, scratch_cb);
    tile_regs_release();
    CircularBuffer(scratch_cb).push_back(1);
    CircularBuffer(scratch_cb).wait_front(1);
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
    const auto source_address = get_local_cb_interface(reciprocal_cb).fifo_rd_ptr << cb_addr_shift;
    const auto scratch_address = get_local_cb_interface(scratch_cb).fifo_rd_ptr << cb_addr_shift;
    const volatile uint32_t* source = reinterpret_cast<const volatile uint32_t*>(source_address);
    volatile uint32_t* scratch = reinterpret_cast<volatile uint32_t*>(scratch_address);
    for (uint32_t index = 0; index < 1024; ++index) {
        scratch[index] = source[index];
    }
#endif
    CircularBuffer(numerator_cb).wait_front(columns);
    CircularBuffer(output_cb).reserve_back(columns);
    pack_reconfig_data_format(output_cb);
    PACK((llk_pack_reconfig_l1_acc(false)));
    sfpu_mul_bcast_col_init();
    for (uint32_t tile = 0; tile < columns; ++tile) {
        tile_regs_acquire();
        reconfig_data_format_srca(numerator_cb);
        copy_tile_init(numerator_cb);
        copy_tile(numerator_cb, tile, 0);
        reconfig_data_format_srca(scratch_cb);
        copy_tile_init(scratch_cb);
        copy_tile(scratch_cb, 0, 1);
        sfpu_mul_bcast_col(0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, output_cb);
        tile_regs_release();
    }
    CircularBuffer(numerator_cb).pop_front(columns);
    CircularBuffer(reciprocal_cb).pop_front(1);
    CircularBuffer(scratch_cb).pop_front(1);
    CircularBuffer(output_cb).push_back(columns);
}
'''


@contextmanager
def scratch_normalization():
    original = native_draft_sdpa.replacements

    def replacements():
        substitutions = original()
        final = '            mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(alias_mm2_prev_out, alias_prev_sum, cb_out);'
        substitutions['compute_common.hpp'] += (
            ('#include "api/compute/bcast.h"',
             '#include "api/compute/bcast.h"\n#include "api/compute/sfpu_binary_bcast.h"'),
            ('enum SDPAType {', HELPER + '\nenum SDPAType {'),
            (final, '            if constexpr (!QWEN_DRAFT_EXP_APPROX && get_compile_time_arg_val(3) == 2112) {\n'
             '                static_assert(Sq_chunk_t == 1);\n'
             '                qwen_normalize_scratch<vDHt>(alias_mm2_prev_out, alias_prev_sum, get_compile_time_arg_val(42), cb_out);\n'
             '            } else {\n' + final + '\n            }'))
        return substitutions

    with patch.object(native_draft_sdpa, 'replacements', replacements):
        yield
