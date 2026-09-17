"""Experimental exact FP32 denominator transport and cross-worker reduction."""

from dspark_splitk_unfused_correction import REPLACEMENT


START = '                    reconfig_data_format(cb_prev_sum, cb_exp_max_diff);'
TREE_ARITHMETIC = REPLACEMENT[REPLACEMENT.index(START):].replace(
    '                    move_block<true>(cb_prev_sum, cb_cur_sum, Sq_chunk_t);',
    '                    reconfig_data_format_srca(cb_prev_sum);\n'
    '                    pack_reconfig_data_format(cb_cur_sum);\n'
    '                    move_block<true>(cb_prev_sum, cb_cur_sum, Sq_chunk_t);')
HELPER = '''void qwen_splitk_tree_denominator(uint32_t left, uint32_t right, uint32_t left_scale,
    uint32_t right_scale, uint32_t output, uint32_t tiles) {
    CircularBuffer(left).wait_front(tiles);
    CircularBuffer(right).wait_front(tiles);
    CircularBuffer(left_scale).wait_front(tiles);
    CircularBuffer(right_scale).wait_front(tiles);
    CircularBuffer(output).reserve_back(tiles);
    pack_reconfig_data_format(output);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        qwen_splitk_copy_fp32_init(left);
        qwen_splitk_copy_fp32(left, tile, 0);
        reconfig_data_format_srca(left_scale);
        copy_tile_to_dst_init_short(left_scale);
        copy_tile(left_scale, tile, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        qwen_splitk_copy_fp32_init(right);
        qwen_splitk_copy_fp32(right, tile, 1);
        reconfig_data_format_srca(right_scale);
        copy_tile_to_dst_init_short(right_scale);
        copy_tile(right_scale, tile, 2);
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, output);
        tile_regs_release();
        CircularBuffer(output).push_back(1);
    }
    CircularBuffer(left).pop_front(tiles);
    CircularBuffer(right).pop_front(tiles);
}

'''


def transform(source):
    if source.count(TREE_ARITHMETIC) != 1 or source.count('void kernel_main() {') != 1:
        raise ValueError('Exact decomposed tree denominator required')
    source = source.replace(TREE_ARITHMETIC,
        '                    qwen_splitk_tree_denominator(cb_prev_sum, cb_prev_sum_2, '
        'cb_exp_max_diff, cb_exp_max_diff_2, cb_cur_sum, Sq_chunk_t);')
    for arguments in ('cb_l_in, cb_prev_sum_2', 'cb_cur_sum, cb_prev_sum', 'cb_prev_sum, cb_out_l'):
        before = 'move_block<true>(' + arguments + ', Sq_chunk_t);'
        if source.count(before) != 1:
            raise ValueError('Exact FP32 tree denominator transport required: ' + arguments)
        source = source.replace(before, 'qwen_splitk_denominator_move(' + arguments + ', Sq_chunk_t);')
    return source.replace('void kernel_main() {', HELPER + 'void kernel_main() {')
