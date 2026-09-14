"""Preserve FP32 scores through explicit SFPU mask addition in decode."""


def transform(source, *, fp32_mask=False):
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
                                    qwen_splitk_copy_fp32_init(cb_qk_im);
                                    qwen_splitk_copy_fp32(cb_qk_im, 0, 0);
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
    result = (source.replace(before, after) if fp32_mask else source).replace(include,
        include + '\n#include "api/compute/eltwise_binary_sfpu.h"')
    entry = 'void kernel_main() {'
    helper = '''void qwen_splitk_copy_fp32_init(uint32_t source_cb) {
    reconfig_data_format_srca(source_cb);
    state_configure(source_cb, __builtin_LINE());
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(0, 0, source_cb)));
    MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE>(source_cb)));
}

void qwen_splitk_copy_fp32(uint32_t source_cb, uint32_t tile, uint32_t destination) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(source_cb, tile)));
    MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, true>(destination, source_cb)));
}

'''
    if result.count(entry) != 1:
        raise ValueError('Unique decode kernel entry required')
    result = result.replace(entry, helper + entry)
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
            '                if (k_chunk == k_chunk_start) {\n'
            f'                    DEVICE_PRINT("QWEN_SPLITK_VALUES {stage}={{:.9f}}\\n",\n'
            f'                        TSLICE({buffer}, 0,\n'
            '                            (SliceRange{.h0=0, .h1=4, .hs=1, .w0=0, .w1=4, .ws=1}), true, true));\n'
            '                }\n'
            '#endif\n')
        result = result.replace(anchor, marker + anchor)
    return result
