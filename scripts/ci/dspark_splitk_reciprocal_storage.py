"""Keep the corrected reciprocal in FP32 without changing sum storage."""


def transform(source):
    declaration = '    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;'
    start = '                CircularBuffer denominator(cb_prev_sum);'
    finish = '                denominator.push_back(Sq_chunk_t);'
    root_end = '        } else if (has_parent) {'
    for anchor in (declaration, start, finish, root_end):
        if source.count(anchor) != 1:
            raise ValueError('Exact split-K reciprocal-storage anchors required')
    begin = source.index(start)
    end = source.index(finish, begin) + len(finish)
    block = source[begin:end]
    before = '                pack_reconfig_data_format(cb_prev_sum);'
    if block.count(before) != 1 or block.count('pack_tile(0, cb_prev_sum);') != 1:
        raise ValueError('Exact reciprocal output pack required')
    block = block.replace(before, '                pack_reconfig_data_format(cb_precise_reciprocal);\n'
        '                CircularBuffer(cb_precise_reciprocal).reserve_back(Sq_chunk_t);')
    block = block.replace('pack_tile(0, cb_prev_sum);', 'pack_tile(0, cb_precise_reciprocal);\n'
        '                    CircularBuffer(cb_precise_reciprocal).push_back(1);')
    block = block.replace('                denominator.reserve_back(Sq_chunk_t);\n' + finish, '')
    stop = source.index(root_end, end)
    result = source[:begin] + block + source[end:stop].replace('cb_prev_sum', 'cb_precise_reciprocal') + source[stop:]
    return result.replace(declaration, declaration + '\n    constexpr uint32_t cb_precise_reciprocal = tt::CBIndex::c_33;')
