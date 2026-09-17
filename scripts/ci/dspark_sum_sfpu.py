"""Unqualified SFPU online-sum update; requires the qualified score-centering scope."""

from contextlib import contextmanager
from unittest.mock import patch

import dspark_ladder_sum_update
import dspark_score_sfpu
import native_draft_sdpa


HELPER = r'''
void qwen_sum_update_sfpu(uint32_t previous_cb, uint32_t current_cb, uint32_t factor_cb) {
    constexpr uint32_t previous_scratch = QWEN_SCORE_SCRATCH_CB;
    constexpr uint32_t factor_scratch = get_compile_time_arg_val(42);
    constexpr uint32_t current_scratch = QWEN_SUM_SCRATCH_CB;
    qwen_stage_score_tile(previous_cb, previous_scratch);
    qwen_stage_score_tile(factor_cb, factor_scratch);
    qwen_stage_score_tile(current_cb, current_scratch);
    tile_regs_acquire();
    qwen_copy_fp32_init(previous_scratch);
    qwen_copy_fp32(previous_scratch, 0);
    qwen_copy_fp32_init(factor_scratch);
    qwen_copy_fp32(factor_scratch, 1);
    sfpu_mul_bcast_col_init();
    sfpu_mul_bcast_col(0, 1);
    qwen_copy_fp32_init(current_scratch);
    qwen_copy_fp32(current_scratch, 1);
    add_binary_tile_init();
    add_binary_tile(1, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer(previous_scratch).pop_front(1);
    CircularBuffer(previous_scratch).reserve_back(1);
    pack_reconfig_data_format(previous_scratch);
    PACK((llk_pack_reconfig_l1_acc(false)));
    pack_tile(0, previous_scratch);
    tile_regs_release();
    CircularBuffer(previous_scratch).push_back(1);
    CircularBuffer(previous_scratch).wait_front(1);
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
    const auto source_address = get_local_cb_interface(previous_scratch).fifo_rd_ptr << cb_addr_shift;
    const auto destination_address = get_local_cb_interface(current_cb).fifo_rd_ptr << cb_addr_shift;
    const volatile uint32_t* source = reinterpret_cast<const volatile uint32_t*>(source_address);
    volatile uint32_t* destination = reinterpret_cast<volatile uint32_t*>(destination_address);
    for (uint32_t index = 0; index < 1024; ++index) {
        destination[index] = source[index];
    }
#endif
    CircularBuffer(previous_scratch).pop_front(1);
    CircularBuffer(factor_scratch).pop_front(1);
    CircularBuffer(current_scratch).pop_front(1);
    CircularBuffer(previous_cb).pop_front(1);
}
'''


def factory_transform(source, *, reverse=False):
    substitutions = (
        ('    uint32_t qwen_score_scratch_cb = cb_ids.q_in;',
         '    uint32_t qwen_score_scratch_cb = cb_ids.q_in;\n    uint32_t qwen_sum_scratch_cb = cb_ids.q_in;'),
        ('        qwen_score_scratch_cb = allocate_tile_cb(1, tt::tile_size(tt::DataFormat::Float32), tt::DataFormat::Float32);',
         '        qwen_score_scratch_cb = allocate_tile_cb(1, tt::tile_size(tt::DataFormat::Float32), tt::DataFormat::Float32);\n'
         '        qwen_sum_scratch_cb = allocate_tile_cb(1, tt::tile_size(tt::DataFormat::Float32), tt::DataFormat::Float32);'),
        ('        qwen_normalization_modes.at(qwen_score_scratch_cb) = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;',
         '        qwen_normalization_modes.at(qwen_score_scratch_cb) = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;\n'
         '        qwen_normalization_modes.at(qwen_sum_scratch_cb) = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;'),
        ('    compute_desc.defines.emplace_back("QWEN_SCORE_SCRATCH_CB", std::to_string(qwen_score_scratch_cb));',
         '    compute_desc.defines.emplace_back("QWEN_SCORE_SCRATCH_CB", std::to_string(qwen_score_scratch_cb));\n'
         '    compute_desc.defines.emplace_back("QWEN_SUM_SCRATCH_CB", std::to_string(qwen_sum_scratch_cb));'))
    for before, after in reversed(substitutions) if reverse else substitutions:
        if reverse:
            before, after = after, before
        if source.count(before.encode()) != 1:
            raise ValueError('Unique sum-scratch factory anchor required')
        source = source.replace(before.encode(), after.encode())
    return source


@contextmanager
def sum_scope():
    original_factory = dspark_score_sfpu.factory_transform
    original_replacements = native_draft_sdpa.replacements

    def factory(source, *, reverse=False):
        if reverse:
            return original_factory(factory_transform(source, reverse=True), reverse=True)
        return factory_transform(original_factory(source))

    def replacements():
        substitutions = original_replacements()
        substitutions['compute_common.hpp'] += (
            ('#include "api/compute/bcast.h"',
             '#include "api/compute/bcast.h"\n#include "api/compute/eltwise_binary_sfpu.h"'),
            ('enum SDPAType {', HELPER + '\nenum SDPAType {'))
        return substitutions

    before = dspark_ladder_sum_update.BEFORE
    after = '''                if constexpr (!QWEN_DRAFT_EXP_APPROX &&
                    (get_compile_time_arg_val(3) == 2112 || get_compile_time_arg_val(3) == 16)) {
                    static_assert(Sq_chunk_t == 1);
                    qwen_sum_update_sfpu(alias_prev_sum, alias_cur_sum, cb_exp_max_diff);
                } else {
''' + before + '\n                }'
    with patch.object(dspark_score_sfpu, 'factory_transform', factory), \
            patch.object(dspark_ladder_sum_update, 'AFTER', after), \
            patch.object(native_draft_sdpa, 'replacements', replacements):
        yield
