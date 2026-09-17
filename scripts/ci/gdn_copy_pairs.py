"""Experimental paired register handoffs for recurrence tile conversions only."""

from gdn_multitoken import replace_once
from gdn_vsplit_norm_batch import load_kernels as baseline_kernels


ORIGINAL = '''    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }'''

PAIRED = '''    for (uint32_t tile = 0; tile < n; tile += 2) {
        const bool paired = tile + 1 < n;
        tile_regs_acquire();
        copy_tile(in, tile, 0);
        if (paired) { copy_tile(in, tile + 1, 1); }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, tile);
        if (paired) { pack_tile(1, o, tile + 1); }
        tile_regs_release();
    }'''


def transform(source):
    start = 'void copy_tiles(uint32_t in, uint32_t o, uint32_t n) {'
    finish = '    cb_push_back(o, n);\n}'
    if source.count(start) != 1:
        raise ValueError('Expected exactly one native copy helper')
    offset = source.index(start)
    end = source.index(finish, offset) + len(finish)
    original = source[offset:end]
    candidate = replace_once(original, ORIGINAL, PAIRED)
    return source[:offset] + candidate + source[end:]


def load_kernels(root):
    kernels = baseline_kernels(root)
    kernels['recurrence']['compute'] = transform(kernels['recurrence']['compute'])
    return kernels
