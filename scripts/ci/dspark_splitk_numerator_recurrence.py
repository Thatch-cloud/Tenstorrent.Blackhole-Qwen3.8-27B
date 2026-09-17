"""Experimental single-pack FP32 numerator recurrence using native SFPU broadcast."""


MULTIPLY = '''                    mul_block_bcast_cols<Sq_chunk_t, vDHt, true, false>(
                        cb_out_accumulate_im, cb_exp_max_diff, cb_out_accumulate_im);'''
ADD = '                    add_block_inplace<true>(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);'
HELPER = '''template <uint32_t rows, uint32_t columns>
void qwen_splitk_numerator_recurrence(uint32_t accumulator, uint32_t partial, uint32_t correction) {
    constexpr uint32_t tiles = rows * columns;
    CircularBuffer(accumulator).wait_front(tiles);
    CircularBuffer(partial).wait_front(tiles);
    CircularBuffer(correction).wait_front(rows);
    pack_reconfig_data_format(accumulator);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < columns; ++column) {
            sfpu_mul_bcast_col_init();
            tile_regs_acquire();
            qwen_splitk_copy_fp32_init(accumulator);
            qwen_splitk_copy_fp32(accumulator, 0, 0);
            reconfig_data_format_srca(correction);
            copy_tile_to_dst_init_short(correction);
            copy_tile(correction, row, 1);
            sfpu_mul_bcast_col(0, 1);
            qwen_splitk_copy_fp32_init(partial);
            qwen_splitk_copy_fp32(partial, row * columns + column, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            CircularBuffer(accumulator).pop_front(1);
            CircularBuffer(accumulator).reserve_back(1);
            tile_regs_wait();
            pack_tile(0, accumulator);
            tile_regs_release();
            CircularBuffer(accumulator).push_back(1);
        }
    }
    CircularBuffer(partial).pop_front(tiles);
    CircularBuffer(correction).pop_front(rows);
}

'''


def transform(source):
    include = '#include "api/compute/eltwise_binary_sfpu.h"'
    for anchor in (MULTIPLY, ADD, include, 'void kernel_main() {'):
        if source.count(anchor) != 1:
            raise ValueError('Exact local numerator recurrence required')
    return source.replace(MULTIPLY,
        '                    qwen_splitk_numerator_recurrence<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_out_im, cb_exp_max_diff);').replace(
        ADD, '').replace(include, include + '\n#include "api/compute/sfpu_binary_bcast.h"').replace(
        'void kernel_main() {', HELPER + 'void kernel_main() {')
