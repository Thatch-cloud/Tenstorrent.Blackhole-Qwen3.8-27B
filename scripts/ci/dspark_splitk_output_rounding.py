"""Explicit final BF16 rounding for the simulator split-K candidate."""


def transform(source):
    before = '                move_block<true>(cb_out_accumulate_im, cb_out_final, out_chunk_tiles);'
    include = '#include "api/compute/eltwise_unary/recip.h"'
    if source.count(before) != 1 or source.count(include) != 1:
        raise ValueError('Exact tiled final-output copy required')
    after = '''                {
                    reconfig_data_format_srca(cb_out_accumulate_im);
                    copy_tile_to_dst_init_short(cb_out_accumulate_im);
                    pack_reconfig_data_format(cb_out_final);
                    typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>();
                    CircularBuffer(cb_out_accumulate_im).wait_front(out_chunk_tiles);
                    CircularBuffer(cb_out_final).reserve_back(out_chunk_tiles);
                    for (uint32_t tile = 0; tile < out_chunk_tiles; ++tile) {
                        tile_regs_acquire();
                        copy_tile(cb_out_accumulate_im, tile, 0);
                        typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile(0, cb_out_final);
                        tile_regs_release();
                        CircularBuffer(cb_out_final).push_back(1);
                    }
                    CircularBuffer(cb_out_accumulate_im).pop_front(out_chunk_tiles);
                }'''
    return source.replace(before, after).replace(include,
        include + '\n#include "api/compute/eltwise_unary/typecast.h"')
