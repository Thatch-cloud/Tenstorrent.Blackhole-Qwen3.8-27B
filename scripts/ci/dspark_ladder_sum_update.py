"""64K-only scalar sum-update diagnostic preserving native buffer formats."""

from contextlib import contextmanager
from unittest.mock import patch

import native_draft_sdpa


BEFORE = '''                mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);

                /* cb_cur_sum += cb_prev_sum */
                add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);'''
BODY = '''
                    CircularBuffer(alias_prev_sum).wait_front(Sq_chunk_t);
                    CircularBuffer(alias_cur_sum).wait_front(Sq_chunk_t);
                    CircularBuffer(cb_exp_max_diff).wait_front(Sq_chunk_t);
                    for (uint32_t tile = 0; tile < Sq_chunk_t; ++tile) {
                        const auto previous_cb = get_local_cb_interface(alias_prev_sum);
                        const auto current_cb = get_local_cb_interface(alias_cur_sum);
                        const auto factor_cb = get_local_cb_interface(cb_exp_max_diff);
                        const volatile float* previous = reinterpret_cast<const volatile float*>(
                            (previous_cb.fifo_rd_ptr + tile * previous_cb.fifo_page_size) << cb_addr_shift);
                        volatile float* current = reinterpret_cast<volatile float*>(
                            (current_cb.fifo_rd_ptr + tile * current_cb.fifo_page_size) << cb_addr_shift);
                        const volatile float* factor = reinterpret_cast<const volatile float*>(
                            (factor_cb.fifo_rd_ptr + tile * factor_cb.fifo_page_size) << cb_addr_shift);
                        for (uint32_t row = 0; row < 32; ++row) {
                            const uint32_t first = row < 16 ? row * 16 : 512 + (row - 16) * 16;
                            const float correction = factor[first];
                            for (uint32_t column = 0; column < 32; ++column) {
                                const uint32_t offset = first + (column < 16 ? column : 256 + column - 16);
                                current[offset] = current[offset] + previous[offset] * correction;
                            }
                        }
                    }
                    CircularBuffer(alias_prev_sum).pop_front(Sq_chunk_t);
'''
AFTER = '''                if constexpr (!QWEN_DRAFT_EXP_APPROX && get_compile_time_arg_val(3) == 2112) {
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
''' + BODY + '''#endif
                } else {
''' + BEFORE + '''
                }'''


@contextmanager
def scalar_sum_update():
    original = native_draft_sdpa.replacements

    def replacements():
        substitutions = original()
        substitutions['compute_common.hpp'] += ((BEFORE, AFTER),)
        return substitutions

    with patch.object(native_draft_sdpa, 'replacements', replacements):
        yield
