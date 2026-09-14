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
    return source[:begin] + copy + block.replace('cb_out_accumulate_im', 'cb_out_o') + source[end:]
