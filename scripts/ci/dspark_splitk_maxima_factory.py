"""Unqualified FP32 local-maxima ablation on the experimental split-K factory."""


def transform(source):
    if source.count('const bool qwen_splitk_fp32 =') != 1:
        raise ValueError('Apply only after the guarded split-K factory transform')
    for index in (27, 28):
        if f'add_cb(CBIndex::c_{index}, statistics_tiles * im_tile_size' in source:
            raise ValueError('Maximum allocation already transformed')
        before = (f'    add_cb(CBIndex::c_{index}, statistics_tiles * stats_tile_size, '
            'stats_df, stats_tile_size, &stats_tile);')
        if source.count(before) != 1:
            raise ValueError('Exact unchanged local maximum allocation required')
        source = source.replace(before,
            '    if (qwen_splitk_fp32) {\n'
            f'        add_cb(CBIndex::c_{index}, statistics_tiles * im_tile_size, im_df, im_tile_size, &im_tile);\n'
            '    } else {\n' + before + '\n    }')
    return source
