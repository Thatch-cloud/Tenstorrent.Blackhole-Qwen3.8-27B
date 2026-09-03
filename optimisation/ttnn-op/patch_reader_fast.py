"""Recurrence op: drop the reader's per-tile row zeroing (kernel-only).

The reader gathers each head's q/k/v row by DMA-ing the full TILE page and then zeroing the
other 31 rows with RISC word stores -- ~1,000 words per tile, twelve tiles per head, which
timed at ~17 us per layer (99 -> 81.5 us at B=8, 62 -> 45 us at B=1 with the zeroing
removed). The zeros were only ever needed by the rank-1 outer product, whose matmul
contracts over rows: outer[i,j] = sum_r kcol[i,r] * delta[r,j]. Computed as two broadcasts
instead -- D'[i,j] = delta[0,j] (row-0 broadcast against the all-ones tile), then
outer[i,j] = D'[i,j] * kcol[i,0] (column-0 broadcast) -- it reads only the wanted row and
column, every other consumer of the gathered tiles is row-independent, and the neighbouring
rows (real finite values from the same tensor page) can stay. Same math, one product per
element, no accumulation over zeros.

Applies on top of patch_fuse_ng.py. Idempotent; run from kwork/.
"""
import io
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decode_gated_delta_rule")


def rw(rel, fn):
    p = os.path.join(ROOT, rel)
    s = io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    s2 = fn(s)
    assert s2 != s, p
    io.open(p, "w", encoding="utf-8", newline="\n").write(s2)
    print("patched", rel)


def reader_cpp(s):
    old = """            if (r != 0) {
                copy_chunk(p + src_e0 * elem, p);               // row 0 cols 0-15
                copy_chunk(p + src_e1 * elem, p + 256 * elem);  // row 0 cols 16-31
            }
            // zero everything except row 0's two 16-element face chunks
            zero(p + 16 * elem, (256 - 16) * elem / 4);
            zero(p + (256 + 16) * elem, (1024 - 272) * elem / 4);
        }"""
    new = """            if (r != 0) {
                copy_chunk(p + src_e0 * elem, p);               // row 0 cols 0-15
                copy_chunk(p + src_e1 * elem, p + 256 * elem);  // row 0 cols 16-31
            }
            // Rows 1..31 keep the page's real (finite) neighbouring rows: every consumer in the
            // compute is row-independent now that the rank-1 write is two broadcasts rather
            // than a row-contracting matmul (see the compute kernel). Zeroing them cost ~17 us
            // per layer of RISC word stores.
        }"""
    assert s.count(old) == 1, "gather_row zero anchor"
    s = s.replace(old, new, 1)
    old = """        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        const uint32_t base = cb.get_write_ptr();
        zero(base, tb_io / 4);
        noc.async_read(acc, cb, tb_io, {.page_id = page}, {.offset_bytes = 0});"""
    new = """        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        const uint32_t base = cb.get_write_ptr();
        // No zeroing: only element [0,0] of a scalar tile is ever read (scalar broadcasts); the
        // rest of the page holds the tensor's own finite values.
        noc.async_read(acc, cb, tb_io, {.page_id = page}, {.offset_bytes = 0});"""
    assert s.count(old) == 1, "gather_scalar zero anchor"
    return s.replace(old, new, 1)


def compute_cpp(s):
    old = """        // ---- rank-1 write: new_h = h + (kn)^T @ (beta*delta) ----
        transpose_row(cb_kn, cb_kcol, Kt);  // kn -> column form
        WAIT(cb_kcol, Kt);
        POP(cb_kn, Kt);
        mm(cb_kcol, cb_delta, cb_outer, Kt, 1, Vt, false);  // [K,1]@[1,V]
        WAIT(cb_outer, kv);
        POP(cb_delta, Vt);
        POP(cb_kcol, Kt);"""
    new = """        // ---- rank-1 write: new_h = h + (kn)^T (x) (beta*delta) ----
        // As two broadcasts rather than a [K,1]@[1,V] matmul: the matmul contracts over the
        // 32 rows of its operands and so needed rows 1..31 of the gathered tiles to be exact
        // zeros (17 us of reader zeroing per layer). D'[i,j] = delta[0,j] via a row-0 broadcast
        // against the all-ones tile, then outer[i,j] = D'[i,j] * kcol[i,0] via a column-0
        // broadcast: only the wanted row and column are read. One product per element, so the
        // result is the matmul's to the bit.
        transpose_row(cb_kn, cb_kcol, Kt);  // kn -> column form (column 0 is what matters)
        WAIT(cb_kcol, Kt);
        POP(cb_kn, Kt);
        rowbcast_delta(cb_delta, cb_vread, Vt);  // D'_j = delta row 0, every row (reuses vread's ring)
        WAIT(cb_vread, Vt);
        POP(cb_delta, Vt);
        outer_bcast(cb_vread, cb_kcol, cb_outer, Kt, Vt);  // outer[i,j] = D'_j * kcol_i[:,0]
        WAIT(cb_outer, kv);
        POP(cb_vread, Vt);
        POP(cb_kcol, Kt);"""
    assert s.count(old) == 1, "outer product anchor"
    s = s.replace(old, new, 1)
    old = """// rowsum_k: o[1 tile] = in[1,Kt] @ ones"""
    new = """// o[Vt] : tile j = ones * delta_j with delta's row 0 broadcast down every row, i.e.
// D'_j[i,c] = delta_j[0,c]. Only row 0 of delta is read.
void rowbcast_delta(uint32_t delta, uint32_t o, uint32_t Vt) {
    cb_reserve_back(o, Vt);
    pack_reconfig_data_format(o);
    reconfig_data_format(cb_ones, delta);  // bcast(a=ones, b=delta): a->srcA, b->srcB
    mul_bcast_rows_init(cb_ones, delta);
    for (uint32_t j = 0; j < Vt; j++) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(cb_ones, delta, 0, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, j);
        tile_regs_release();
    }
    cb_push_back(o, Vt);
}

// o[Kt,Vt] : tile (i,j) = D'_j * kcol_i's column 0 broadcast across columns, i.e.
// outer[i*32+r, j*32+c] = delta[0, j*32+c] * kn[0, i*32+r]. Only column 0 of kcol is read.
void outer_bcast(uint32_t dprime, uint32_t kcol, uint32_t o, uint32_t Kt, uint32_t Vt) {
    cb_reserve_back(o, Kt * Vt);
    pack_reconfig_data_format(o);
    reconfig_data_format(dprime, kcol);  // bcast(a=D', col=kcol): a->srcA, col->srcB
    mul_bcast_cols_init(dprime, kcol);
    for (uint32_t i = 0; i < Kt; i++) {
        for (uint32_t j = 0; j < Vt; j++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(dprime, kcol, j, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, i * Vt + j);
            tile_regs_release();
        }
    }
    cb_push_back(o, Kt * Vt);
}

// rowsum_k: o[1 tile] = in[1,Kt] @ ones"""
    assert s.count(old) == 1, "helper anchor"
    return s.replace(old, new, 1)


if __name__ == "__main__":
    marker = os.path.join(ROOT, "device", "kernels", "compute", "decode_gated_delta_rule.cpp")
    if "outer_bcast" in io.open(marker, encoding="utf-8").read():
        print("already applied")
        sys.exit(0)
    rw("device/kernels/dataflow/reader_decode_gated_delta_rule.cpp", reader_cpp)
    rw("device/kernels/compute/decode_gated_delta_rule.cpp", compute_cpp)
    print("all patched")
