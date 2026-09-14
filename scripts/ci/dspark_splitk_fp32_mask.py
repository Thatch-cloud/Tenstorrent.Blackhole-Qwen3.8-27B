"""Preserve FP32 scores through explicit SFPU mask addition in decode."""


def transform(source):
    before = '                            add_block_inplace<true>(cb_qk_im, cb_mask_in, qk_chunk_tiles_dynamic);'
    after = '''                            {
                                CircularBuffer scores(cb_qk_im);
                                CircularBuffer mask(cb_mask_in);
                                scores.wait_front(qk_chunk_tiles_dynamic);
                                mask.wait_front(qk_chunk_tiles_dynamic);
                                pack_reconfig_data_format(cb_qk_im);
                                PACK((llk_pack_reconfig_l1_acc(false)));
                                for (uint32_t tile = 0; tile < qk_chunk_tiles_dynamic; ++tile) {
                                    tile_regs_acquire();
                                    reconfig_data_format_srca(cb_qk_im);
                                    copy_tile_to_dst_init_short(cb_qk_im);
                                    copy_tile(cb_qk_im, 0, 0);
                                    reconfig_data_format_srca(cb_mask_in);
                                    copy_tile_to_dst_init_short(cb_mask_in);
                                    copy_tile(cb_mask_in, tile, 1);
                                    add_binary_tile_init();
                                    add_binary_tile(0, 1, 0);
                                    tile_regs_commit();
                                    scores.pop_front(1);
                                    scores.reserve_back(1);
                                    tile_regs_wait();
                                    pack_tile(0, cb_qk_im);
                                    tile_regs_release();
                                    scores.push_back(1);
                                }
                                mask.pop_front(qk_chunk_tiles_dynamic);
                            }'''
    include = '#include "api/compute/eltwise_binary.h"'
    if source.count(before) != 1 or source.count(include) != 1:
        raise ValueError('Exact explicit decode mask path required')
    result = source.replace(before, after).replace(include,
        include + '\n#include "api/compute/eltwise_binary_sfpu.h"')
    markers = (
        ('                if (!add_mask_fusion) {', 'scores-ready', 'cb_qk_im', 'qk_chunk_tiles_dynamic'),
        ('                reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(',
         'masked-scores-ready', 'cb_qk_im', 'qk_chunk_tiles_dynamic'),
        ('                /* QK -= cb_cur_max */', 'maximum-ready', 'cb_cur_max', 'Sq_chunk_t'),
        ('                CircularBuffer(cb_qk_im).wait_front(qk_chunk_tiles_dynamic);',
         'exponents-ready', 'cb_qk_im', 'qk_chunk_tiles_dynamic'),
        ('                /* OUT_IM = QK @ V_CHUNK */', 'sum-ready', 'cb_cur_sum', 'Sq_chunk_t'),
    )
    for anchor, stage, buffer, tiles in sorted(markers, key=lambda entry: result.index(entry[0]), reverse=True):
        if result.count(anchor) != 1:
            raise ValueError('Unique split-K stage marker anchor required')
        marker = ('\n#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0\n'
            f'                CircularBuffer({buffer}).wait_front({tiles});\n'
            f'                DEVICE_PRINT("QWEN_SPLITK_STAGE {stage}\\n");\n'
            '#endif\n')
        result = result.replace(anchor, marker + anchor)
    return result
