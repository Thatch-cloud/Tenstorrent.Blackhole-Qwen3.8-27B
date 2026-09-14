"""Isolate FP32-input row reduction using a separate BF16 probability copy."""


def transform(source):
    declaration = '    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;'
    before = '''                reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                pack_reconfig_data_format(cb_cur_sum);'''
    after = '''                CircularBuffer(cb_qk_im).wait_front(qk_chunk_tiles_dynamic);'''
    reduce = '''                reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
                    cb_cur_sum, cb_cur_sum, Sk_chunk_t_dynamic, false);'''
    for anchor in (declaration, before, reduce):
        if source.count(anchor) != 1:
            raise ValueError('Exact decode sum-input anchors required')
    replacement = '''                for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
                    reconfig_data_format_srca(cb_qk_im);
                    copy_tile_to_dst_init_short(cb_qk_im);
                    pack_reconfig_data_format(cb_exponent_sum);
                    CircularBuffer(cb_exponent_sum).reserve_back(Sk_chunk_t_dynamic);
                    for (uint32_t column = 0; column < Sk_chunk_t_dynamic; ++column) {
                        tile_regs_acquire();
                        copy_tile(cb_qk_im, row * Sk_chunk_t_dynamic + column, 0);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile(0, cb_exponent_sum);
                        tile_regs_release();
                        CircularBuffer(cb_exponent_sum).push_back(1);
                    }
                    reconfig_data_format(cb_exponent_sum, cb_identity_scale_in);
                    pack_reconfig_data_format(cb_cur_sum);
                    reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_exponent_sum, cb_identity_scale_in, 1, vector_mode>(
                        cb_cur_sum, cb_cur_sum, Sk_chunk_t_dynamic, false);
                    CircularBuffer(cb_exponent_sum).pop_front(Sk_chunk_t_dynamic);
                }'''
    return source.replace(declaration, declaration + '\n    constexpr uint32_t cb_exponent_sum = tt::CBIndex::c_32;').replace(
        before, after).replace(reduce, replacement)
