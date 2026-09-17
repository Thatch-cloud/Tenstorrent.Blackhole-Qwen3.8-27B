"""Experimental FP32 local denominator recurrence; not yet device-qualified."""


MULTIPLY = '''                    /* PREV_SUM *= EXP_MAX_DIFF */
                    reconfig_data_format(cb_prev_sum, cb_exp_max_diff);
                    pack_reconfig_data_format(cb_prev_sum);
                    mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);'''
ADD = '                    add_block_inplace<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);'
MOVE = '''                // PREV_SUM <- CUR_SUM
                reconfig_data_format_srca(cb_cur_sum);
                pack_reconfig_data_format(cb_prev_sum);
                move_block<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);'''
HELPER = '''void qwen_splitk_denominator_recurrence(uint32_t current, uint32_t previous, uint32_t correction, uint32_t tiles) {
    CircularBuffer(current).wait_front(tiles);
    CircularBuffer(previous).wait_front(tiles);
    CircularBuffer(correction).wait_front(tiles);
    pack_reconfig_data_format(current);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        qwen_splitk_copy_fp32_init(previous);
        qwen_splitk_copy_fp32(previous, tile, 0);
        reconfig_data_format_srca(correction);
        copy_tile_to_dst_init_short(correction);
        copy_tile(correction, tile, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        qwen_splitk_copy_fp32_init(current);
        qwen_splitk_copy_fp32(current, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        CircularBuffer(current).pop_front(1);
        CircularBuffer(current).reserve_back(1);
        tile_regs_wait();
        pack_tile(0, current);
        tile_regs_release();
        CircularBuffer(current).push_back(1);
    }
    CircularBuffer(previous).pop_front(tiles);
}

void qwen_splitk_denominator_move(uint32_t source, uint32_t destination, uint32_t tiles) {
    CircularBuffer(source).wait_front(tiles);
    CircularBuffer(destination).reserve_back(tiles);
    pack_reconfig_data_format(destination);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        qwen_splitk_copy_fp32_init(source);
        qwen_splitk_copy_fp32(source, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, destination);
        tile_regs_release();
        CircularBuffer(destination).push_back(1);
    }
    CircularBuffer(source).pop_front(tiles);
}

'''


def transform(source):
    for anchor in (MULTIPLY, ADD, MOVE, 'void kernel_main() {'):
        if source.count(anchor) != 1:
            raise ValueError('Exact qualified local denominator recurrence required')
    return source.replace(MULTIPLY,
        '                    qwen_splitk_denominator_recurrence(cb_cur_sum, cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);').replace(ADD, '').replace(
        MOVE, '                qwen_splitk_denominator_move(cb_cur_sum, cb_prev_sum, Sq_chunk_t);').replace(
        'void kernel_main() {', HELPER + 'void kernel_main() {')
