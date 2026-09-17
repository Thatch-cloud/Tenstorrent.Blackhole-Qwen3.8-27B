"""Format handling for explicitly FP32 local denominator buffers."""


MULTIPLY = '''                    /* PREV_SUM *= EXP_MAX_DIFF */
                    mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);'''


def transform(source):
    replacements = (
        (MULTIPLY, '''                    /* PREV_SUM *= EXP_MAX_DIFF */
                    reconfig_data_format(cb_prev_sum, cb_exp_max_diff);
                    pack_reconfig_data_format(cb_prev_sum);
                    mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);'''),
        ('                copy_tile_to_dst_init_short(cb_prev_sum);',
         '                qwen_splitk_copy_fp32_init(cb_prev_sum);'),
        ('                    copy_tile(cb_prev_sum, tile, 0);',
         '                    qwen_splitk_copy_fp32(cb_prev_sum, tile, 0);'))
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('Exact local denominator format anchor required')
        source = source.replace(before, after)
    return source
