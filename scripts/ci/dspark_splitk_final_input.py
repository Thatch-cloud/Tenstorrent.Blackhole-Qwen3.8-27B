"""Simulator diagnostic: normalize a BF16 copy of the FP32 numerator."""


def transform(source):
    start = '            /* OUT_ACC *= 1/SUM */'
    stop = '        } else if (has_parent) {'
    if source.count(start) != 1 or source.count(stop) != 1:
        raise ValueError('Exact root normalization boundaries required')
    begin = source.index(start)
    end = source.index(stop, begin)
    block = source[begin:end]
    if 'mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_prev_sum);' not in block:
        raise ValueError('Native final broadcast required')
    copy = '''            reconfig_data_format_srca(cb_out_accumulate_im);
            pack_reconfig_data_format(cb_out_o);
            move_block<true>(cb_out_accumulate_im, cb_out_o, out_chunk_tiles);
'''
    before_multiply = snapshot('reciprocal', 'cb_prev_sum', 'Sq_chunk_t') + snapshot(
        'numerator-copy', 'cb_out_o', 'out_chunk_tiles')
    block = block.replace('cb_out_accumulate_im', 'cb_out_o')
    multiply = '            mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_o, cb_prev_sum);'
    block = block.replace(multiply, multiply + snapshot('normalized', 'cb_out_o', 'out_chunk_tiles'))
    return source[:begin] + copy + before_multiply + block + source[end:]


def snapshot(stage, buffer, tiles):
    return ('\n#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0\n'
        f'            CircularBuffer({buffer}).wait_front({tiles});\n'
        f'            DEVICE_PRINT("QWEN_SPLITK_FINAL {stage}={{:.9f}}\\n",\n'
        f'                TSLICE({buffer}, 0,\n'
        '                    (SliceRange{.h0=0, .h1=4, .hs=1, .w0=0, .w1=4, .ws=1}), true, true));\n'
        '#endif\n')
