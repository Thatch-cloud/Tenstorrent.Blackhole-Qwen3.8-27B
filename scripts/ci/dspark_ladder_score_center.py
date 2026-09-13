"""Diagnostic FP32 masking and centering before native exponential reload."""

from contextlib import contextmanager
from unittest.mock import patch

import native_draft_sdpa


HELPER_ANCHOR = 'void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {'
HELPER = '''
void qwen_scalar_score_transform(uint32_t scores_cb, uint32_t operand_cb, uint32_t tiles, uint32_t columns, bool mask) {
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
    CircularBuffer(scores_cb).wait_front(tiles);
    CircularBuffer(operand_cb).wait_front(mask ? tiles : tiles / columns);
    const auto scores_interface = get_local_cb_interface(scores_cb);
    const auto operand_interface = get_local_cb_interface(operand_cb);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        volatile float* scores = reinterpret_cast<volatile float*>(
            (scores_interface.fifo_rd_ptr + tile * scores_interface.fifo_page_size) << cb_addr_shift);
        const uint32_t operand_address = (operand_interface.fifo_rd_ptr +
            (mask ? tile : tile / columns) * operand_interface.fifo_page_size) << cb_addr_shift;
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t first = row < 16 ? row * 16 : 512 + (row - 16) * 16;
            for (uint32_t column = 0; column < 32; ++column) {
                const uint32_t offset = first + (column < 16 ? column : 256 + column - 16);
                if (mask) {
                    const volatile uint16_t* masks = reinterpret_cast<const volatile uint16_t*>(operand_address);
                    union { uint32_t bits; float value; } converted;
                    converted.bits = static_cast<uint32_t>(masks[offset]) << 16;
                    scores[offset] = scores[offset] + converted.value;
                } else {
                    const volatile float* maxima = reinterpret_cast<const volatile float*>(operand_address);
                    scores[offset] = scores[offset] - maxima[first];
                }
            }
        }
    }
    if (mask) CircularBuffer(operand_cb).pop_front(tiles);
#endif
}
'''
MASK = '                    add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);'
MASK_AFTER = '''                    if constexpr (!QWEN_DRAFT_EXP_APPROX && get_compile_time_arg_val(3) == 2112) {
                        qwen_scalar_score_transform(cb_qk_im, cb_mask_in, qk_chunk_tiles, Sk_chunk_t, true);
                    } else {
''' + MASK + '''
                    }'''
INIT = '    sub_bcast_cols_init(in0_cb, in1_cb);'
INIT_AFTER = '''    if constexpr (!QWEN_DRAFT_EXP_APPROX && get_compile_time_arg_val(3) == 2112) {
        qwen_scalar_score_transform(in0_cb, in1_cb, rows * cols, cols, false);
        copy_tile_to_dst_init_short(in0_cb);
    } else {
''' + INIT + '''
    }'''
SUBTRACT = '                sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);'
SUBTRACT_AFTER = '''                if constexpr (!QWEN_DRAFT_EXP_APPROX && get_compile_time_arg_val(3) == 2112) {
                    copy_tile(in0_cb, j, j);
                } else {
''' + SUBTRACT + '''
                }'''


@contextmanager
def scalar_score_center():
    original = native_draft_sdpa.replacements

    def replacements():
        substitutions = original()
        substitutions['compute_common.hpp'] += ((HELPER_ANCHOR, HELPER + HELPER_ANCHOR),
            (MASK, MASK_AFTER), (INIT, INIT_AFTER), (SUBTRACT, SUBTRACT_AFTER))
        return substitutions

    with patch.object(native_draft_sdpa, 'replacements', replacements):
        yield
