"""Unqualified in-register outer-product plus state addition for T16 GDN."""

from gdn_multitoken import replace_once
from gdn_vsplit_norm_batch import load_kernels as baseline_kernels


HELPER = '''void outer_add(uint32_t delta, uint32_t key, uint32_t state, uint32_t output,
               uint32_t key_tiles, uint32_t value_tiles) {
    cb_reserve_back(output, key_tiles * value_tiles);
    pack_reconfig_data_format(output);
    for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
        for (uint32_t value_tile = 0; value_tile < value_tiles; ++value_tile) {
            const uint32_t tile = key_tile * value_tiles + value_tile;
            reconfig_data_format(delta, key);
            mul_bcast_cols_init(delta, key);
            tile_regs_acquire();
            mul_tiles_bcast_cols(delta, key, value_tile, key_tile, 0);
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(state);
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(state, tile, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, output, tile);
            tile_regs_release();
        }
    }
    cb_push_back(output, key_tiles * value_tiles);
}

'''

ORIGINAL = '''        outer_bcast(cb_vread, cb_kcol, cb_outer, Kt, Vt);  // outer[i,j] = D'_j * kcol_i[:,0]
        WAIT(cb_outer, kv);
        POP(cb_vread, Vt);
        POP(cb_kcol, Kt);
        ew(cb_sdec, cb_outer, cb_snew, kv, 0);  // fp32 new state
        WAIT(cb_snew, kv);
        POP(cb_sdec, kv);
        POP(cb_outer, kv);'''

FUSED = '''        outer_add(cb_vread, cb_kcol, cb_sdec, cb_snew, Kt, Vt);
        WAIT(cb_snew, kv);
        POP(cb_vread, Vt);
        POP(cb_kcol, Kt);
        POP(cb_sdec, kv);'''


def transform(source):
    source = replace_once(source, 'void kernel_main() {', HELPER + 'void kernel_main() {')
    return replace_once(source, ORIGINAL, FUSED)


def load_kernels(root):
    kernels = baseline_kernels(root)
    kernels['recurrence']['compute'] = transform(kernels['recurrence']['compute'])
    return kernels
