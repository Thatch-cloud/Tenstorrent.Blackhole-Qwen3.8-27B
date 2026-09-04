"""[NOT ADOPTED — kept as the record of a measured negative; see docs/optimisation-plan.md]

Recurrence op: skip both state format conversions (compute + factory).

Per head the compute converted the 16-tile state io -> fp32 on the way in (copy_tiles into
cb_sf) and fp32 -> io on the way out (copy_tiles snew -> sout): 32 of its ~100 tile passes.
Neither is needed:
  * in: the decay multiply reads the state straight from cb_state. Its scalar exp(g) is packed
    in the io format too (cb_gexp's format becomes df_io), so the srcA/srcB pair is uniform --
    bf16 x bf16 (exact product in fp32 DST) or fp32 x fp32 -- and the state was bf16-valued
    already; only exp(g) picks up a bf16 rounding (2^-9 relative on the decay), which is what the
    composed bf16 GDN step (lever C) had.
  * out: new_h = sdec + outer is packed twice from the same DST tile, once fp32 (cb_snew, the
    o = qn @ new_h operand) and once io (cb_sout, the writer), with a pack reconfig between.

Applies on top of patch_reader_fast.py. Idempotent; run from kwork/.
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


def compute_cpp(s):
    # 1. no state -> fp32 copy; the state stays in cb_state until the decay multiply consumes it
    old = """        copy_tiles(cb_beta, cb_betaf, 1);
        copy_tiles(cb_state, cb_sf, kv);
        POP(cb_q, Kt);
        POP(cb_k, Kt);
        POP(cb_v, Vt);
        POP(cb_g, 1);
        POP(cb_beta, 1);
        POP(cb_state, kv);
"""
    new = """        copy_tiles(cb_beta, cb_betaf, 1);
        // the state is NOT converted: the decay multiply reads cb_state directly (io x io)
        POP(cb_q, Kt);
        POP(cb_k, Kt);
        POP(cb_v, Vt);
        POP(cb_g, 1);
        POP(cb_beta, 1);
"""
    assert s.count(old) == 1, "state copy anchor"
    s = s.replace(old, new, 1)
    old = """        WAIT(cb_betaf, 1);
        WAIT(cb_sf, kv);
"""
    new = """        WAIT(cb_betaf, 1);
"""
    assert s.count(old) == 1, "sf wait anchor"
    s = s.replace(old, new, 1)
    # 2. decay: state (io) x exp(g) (io) -> sdec fp32
    old = """        expc(cb_gf, cb_gexp, 1);  // [0,0] = exp(g_h)
        WAIT(cb_gexp, 1);
        POP(cb_gf, 1);
        bcast_scalar_mul(cb_sf, cb_gexp, cb_sdec, kv);
        WAIT(cb_sdec, kv);
        POP(cb_sf, kv);
        POP(cb_gexp, 1);
"""
    new = """        expc(cb_gf, cb_gexp, 1);  // [0,0] = exp(g_h), packed in the io format (cb_gexp is df_io)
        WAIT(cb_gexp, 1);
        POP(cb_gf, 1);
        bcast_scalar_mul(cb_state, cb_gexp, cb_sdec, kv);  // io x io -> fp32 (uniform srcA/srcB)
        WAIT(cb_sdec, kv);
        POP(cb_state, kv);
        POP(cb_gexp, 1);
"""
    assert s.count(old) == 1, "decay anchor"
    s = s.replace(old, new, 1)
    # 3. new state packed twice: fp32 for the o matmul, io for the writer
    old = """        ew(cb_sdec, cb_outer, cb_snew, kv, 0);  // fp32 new state
        WAIT(cb_snew, kv);
        POP(cb_sdec, kv);
        POP(cb_outer, kv);
"""
    new = """        add2out(cb_sdec, cb_outer, cb_snew, cb_sout, kv);  // new state: fp32 (o matmul) + io (writer)
        WAIT(cb_snew, kv);
        POP(cb_sdec, kv);
        POP(cb_outer, kv);
"""
    assert s.count(old) == 1, "snew anchor"
    s = s.replace(old, new, 1)
    old = """        // ---- new state -> io for the writer ----
        copy_tiles(cb_snew, cb_sout, kv);
        POP(cb_snew, kv);
"""
    new = """        // (new state already packed to cb_sout for the writer)
        POP(cb_snew, kv);
"""
    assert s.count(old) == 1, "sout copy anchor"
    s = s.replace(old, new, 1)
    # helper
    old = "void ew(uint32_t a, uint32_t b, uint32_t o, uint32_t n, int op) {"
    new = """// o1[n] = o2[n] = a + b, packed twice from the same DST tile (o1 and o2 may differ in format).
void add2out(uint32_t a, uint32_t b, uint32_t o1, uint32_t o2, uint32_t n) {
    cb_reserve_back(o1, n);
    cb_reserve_back(o2, n);
    reconfig_data_format(a, b);
    add_init(a, b);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        add_tiles(a, b, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(o1);
        pack_tile(0, o1, i);
        pack_reconfig_data_format(o2);
        pack_tile(0, o2, i);
        tile_regs_release();
    }
    cb_push_back(o1, n);
    cb_push_back(o2, n);
}

void ew(uint32_t a, uint32_t b, uint32_t o, uint32_t n, int op) {"""
    assert s.count(old) == 1, "ew anchor"
    return s.replace(old, new, 1)


def factory_cpp(s):
    old = "    add_cb(cbd::gexp, 1, 1, tt::DataFormat::Float32);\n"
    new = "    add_cb(cbd::gexp, 1, 1, df_io);  // exp(g) in the io format: uniform io x io decay multiply\n"
    assert s.count(old) == 1, "gexp cb anchor"
    return s.replace(old, new, 1)


if __name__ == "__main__":
    marker = os.path.join(ROOT, "device", "kernels", "compute", "decode_gated_delta_rule.cpp")
    if "add2out" in io.open(marker, encoding="utf-8").read():
        print("already applied")
        sys.exit(0)
    rw("device/kernels/compute/decode_gated_delta_rule.cpp", compute_cpp)
    rw("device/decode_gated_delta_rule_program_factory.cpp", factory_cpp)
    print("all patched")
