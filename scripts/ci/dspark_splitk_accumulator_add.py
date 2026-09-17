"""Preserve FP32 numerator tiles through local accumulation."""


BEFORE = 'add_block_inplace<true>(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);'
HELPER = '''void qwen_splitk_accumulator_add(uint32_t accumulator, uint32_t partial, uint32_t tiles) {
    CircularBuffer(accumulator).wait_front(tiles);
    CircularBuffer(partial).wait_front(tiles);
    pack_reconfig_data_format(accumulator);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        tile_regs_acquire();
        qwen_splitk_copy_fp32_init(accumulator);
        qwen_splitk_copy_fp32(accumulator, 0, 0);
        qwen_splitk_copy_fp32_init(partial);
        qwen_splitk_copy_fp32(partial, tile, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        CircularBuffer(accumulator).pop_front(1);
        CircularBuffer(accumulator).reserve_back(1);
        tile_regs_wait();
        pack_tile(0, accumulator);
        tile_regs_release();
        CircularBuffer(accumulator).push_back(1);
    }
    CircularBuffer(partial).pop_front(tiles);
}

'''


def transform(source):
    entry = 'void kernel_main() {'
    if source.count(BEFORE) != 1 or source.count(entry) != 1:
        raise ValueError('Unique local numerator accumulation and kernel entry required')
    return source.replace(BEFORE,
        'qwen_splitk_accumulator_add(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);').replace(entry, HELPER + entry)
