"""Keep cross-worker softmax correction on the local softmax scale."""


HELPER = '''template <uint32_t scale_fp32>
void qwen_splitk_merge_exp(uint32_t left, uint32_t right, uint32_t output, uint32_t tiles) {
    reconfig_data_format(left, right);
    sub_init(left, right);
    binop_with_scalar_tile_init();
    exp_tile_init<false>();
    pack_reconfig_data_format(output);
    CircularBuffer(left).wait_front(tiles);
    CircularBuffer(right).wait_front(tiles);
    CircularBuffer(output).reserve_back(tiles);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        sub_tiles(left, right, tile, tile, 0);
        mul_unary_tile(0, scale_fp32);
        exp_tile<false>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, output);
        tile_regs_release();
        CircularBuffer(output).push_back(1);
    }
}

'''


def transform(source):
    marker = 'TREE REDUCTION LOGIC'
    if source.count(marker) != 1:
        raise ValueError('Unique tree reduction boundary required')
    prefix, suffix = source.split(marker)
    local_call = 'sub_exp_block<scale_fp32>(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);'
    if prefix.count(local_call) != 1:
        raise ValueError('Unique local recurrence exponent required')
    prefix = prefix.replace(local_call, local_call.replace('sub_exp_block', 'qwen_splitk_merge_exp'))
    for arguments in ('cb_prev_max, cb_cur_max, cb_exp_max_diff',
            'cb_m_in, cb_cur_max, cb_exp_max_diff_2'):
        before = '                    sub_exp_block<scale_fp32>(' + arguments + ', Sq_chunk_t);'
        if suffix.count(before) != 1:
            raise ValueError('Unique cross-worker exponent call required')
        suffix = suffix.replace(before, before.replace('sub_exp_block', 'qwen_splitk_merge_exp'))
    source = prefix + marker + suffix
    entry = 'void kernel_main() {'
    if source.count(entry) != 1:
        raise ValueError('Unique kernel entry required')
    return source.replace(entry, HELPER + entry)
