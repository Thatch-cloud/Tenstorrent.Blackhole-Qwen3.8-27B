"""Isolate FP32-input row reduction using a separate BF16 probability copy."""


def transform(source):
    declaration = '    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;'
    before = '''                reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                pack_reconfig_data_format(cb_cur_sum);'''
    after = '''                reconfig_data_format_srca(cb_qk_im);
                pack_reconfig_data_format(cb_exponent_sum);
                move_block<false>(cb_qk_im, cb_exponent_sum, qk_chunk_tiles_dynamic);
                reconfig_data_format(cb_exponent_sum, cb_identity_scale_in);
                pack_reconfig_data_format(cb_cur_sum);'''
    reduce = '''                reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
                    cb_cur_sum, cb_cur_sum, Sk_chunk_t_dynamic, false);'''
    for anchor in (declaration, before, reduce):
        if source.count(anchor) != 1:
            raise ValueError('Exact decode sum-input anchors required')
    return source.replace(declaration, declaration + '\n    constexpr uint32_t cb_exponent_sum = tt::CBIndex::c_32;').replace(
        before, after).replace(reduce, reduce.replace('cb_qk_im', 'cb_exponent_sum') +
        '\n                CircularBuffer(cb_exponent_sum).pop_front(qk_chunk_tiles_dynamic);')
