"""[NOT ADOPTED — kept as the record of a measured negative; see docs/optimisation-plan.md]

Recurrence reader: two NOC barriers per head instead of seven.

Each gather (q, k, v, beta, g, state, and z in fold mode) did reserve -> DMA -> barrier ->
row copy -> push, so every head paid seven DRAM round-trip latencies in series, and the
reader is this op's critical path (removing 17 us of its zeroing paid out in full while
removing 32 compute passes paid nothing). Now every read of the head is issued first and one
barrier covers them all; the fold's z row shares v's ring and can only be reserved after v is
pushed (the ring's write pointer moves on push), so it keeps a second round.

Applies on top of patch_reader_fast.py. Idempotent; run from kwork/.
"""
import io
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decode_gated_delta_rule")
P = os.path.join(ROOT, "device", "kernels", "dataflow", "reader_decode_gated_delta_rule.cpp")


def patch(s):
    # 1. split gather_row into issue (reserve + DMA, no barrier) and finish (row copy + push)
    old = """    auto gather_row = [&](const auto& acc, uint32_t cb_id, uint32_t n_tiles, uint32_t first_page, uint32_t r) {
        const uint32_t frow = r % 16;   // row within a face
        const uint32_t fhalf = r / 16;  // 0: rows 0-15 (faces 0,1), 1: rows 16-31 (faces 2,3)
        const uint32_t src_e0 = (fhalf * 2 + 0) * 256 + frow * 16;
        const uint32_t src_e1 = (fhalf * 2 + 1) * 256 + frow * 16;
        CircularBuffer cb(cb_id);
        cb.reserve_back(n_tiles);
        const uint32_t base = cb.get_write_ptr();
        for (uint32_t t = 0; t < n_tiles; t++) {
            noc.async_read(acc, cb, tb_io, {.page_id = first_page + t}, {.offset_bytes = t * tb_io});
        }
        noc.async_read_barrier();
        for (uint32_t t = 0; t < n_tiles; t++) {"""
    new = """    // issue_row: reserve the target pages and issue the DMAs (no barrier). finish_row: after the
    // caller's barrier, extract row r into row 0 and push. Two barriers per head instead of seven.
    auto issue_row = [&](const auto& acc, uint32_t cb_id, uint32_t n_tiles, uint32_t first_page) -> uint32_t {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n_tiles);
        const uint32_t base = cb.get_write_ptr();
        for (uint32_t t = 0; t < n_tiles; t++) {
            noc.async_read(acc, cb, tb_io, {.page_id = first_page + t}, {.offset_bytes = t * tb_io});
        }
        return base;
    };
    auto finish_row = [&](uint32_t cb_id, uint32_t base, uint32_t n_tiles, uint32_t r) {
        const uint32_t frow = r % 16;   // row within a face
        const uint32_t fhalf = r / 16;  // 0: rows 0-15 (faces 0,1), 1: rows 16-31 (faces 2,3)
        const uint32_t src_e0 = (fhalf * 2 + 0) * 256 + frow * 16;
        const uint32_t src_e1 = (fhalf * 2 + 1) * 256 + frow * 16;
        CircularBuffer cb(cb_id);
        for (uint32_t t = 0; t < n_tiles; t++) {"""
    assert s.count(old) == 1, "gather_row anchor"
    s = s.replace(old, new, 1)
    # 2. gather_scalar -> issue_scalar / finish_scalar
    old = """    auto gather_scalar = [&](const auto& acc, uint32_t cb_id, uint32_t page, uint32_t r, uint32_t cc) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        const uint32_t base = cb.get_write_ptr();
        // No zeroing: only element [0,0] of a scalar tile is ever read (scalar broadcasts); the
        // rest of the page holds the tensor's own finite values.
        noc.async_read(acc, cb, tb_io, {.page_id = page}, {.offset_bytes = 0});
        noc.async_read_barrier();
        const uint32_t eoff"""
    new = """    auto issue_scalar = [&](const auto& acc, uint32_t cb_id, uint32_t page) -> uint32_t {
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        const uint32_t base = cb.get_write_ptr();
        // No zeroing: only element [0,0] of a scalar tile is ever read (scalar broadcasts); the
        // rest of the page holds the tensor's own finite values.
        noc.async_read(acc, cb, tb_io, {.page_id = page}, {.offset_bytes = 0});
        return base;
    };
    auto finish_scalar = [&](uint32_t cb_id, uint32_t base, uint32_t r, uint32_t cc) {
        CircularBuffer cb(cb_id);
        const uint32_t eoff"""
    assert s.count(old) == 1, "gather_scalar anchor"
    s = s.replace(old, new, 1)
    # 3. the instance loop: issue everything, one barrier, finish everything; z in a second round
    old = """        if constexpr (PACKED) {
            // [1,B,C] TILE: row b of tile-row b/32; q/k from GQA source head h/RF.
            const uint32_t row_page0 = (b / 32) * Ct;
            const uint32_t r = b % 32;
            const uint32_t hq = h / RF;
            gather_row(q_acc, cb_q, Kt, row_page0 + QOT + hq * Kt, r);
            gather_row(k_acc, cb_k, Kt, row_page0 + KOT + hq * Kt, r);
            gather_row(v_acc, cb_v, Vt, row_page0 + VOT + h * Vt, r);
            // beta/g [1,B,H] TILE: page (b/32)*NVT + h/32, row b%32, col h%32.
            gather_scalar(beta_acc, cb_beta, (b / 32) * NVT + h / 32, r, h % 32);
            gather_scalar(g_acc, cb_g, (b / 32) * NVT + h / 32, r, h % 32);
        } else {
            // Head (b,h)'s physical row in the [B,1,H,D] TILE tensors.
            const uint32_t physrow = b * padH + h;
            gather_row(q_acc, cb_q, Kt, (physrow / 32) * Kt, physrow % 32);
            gather_row(k_acc, cb_k, Kt, (physrow / 32) * Kt, physrow % 32);
            gather_row(v_acc, cb_v, Vt, (physrow / 32) * Vt, physrow % 32);
            // beta/g [B,1,H]: (b,hcol) at page b*ppb + hcol/32, row 0, col hcol%32.
            gather_scalar(beta_acc, cb_beta, b * ppb + h / 32, 0, h % 32);
            gather_scalar(g_acc, cb_g, b * ppb + h / 32, 0, h % 32);
        }

        // State [B,H,K,V] flat 2D is [B*H*K, V] (K,V are 32-multiples: no
        // per-batch padding): head bh owns full row-tiles (bh*Kt .. bh*Kt+Kt).
        CircularBuffer cbs(cb_state);
        cbs.reserve_back(kv);
        if (has_s0) {
            const uint32_t base_page = bh * kv;
            for (uint32_t t = 0; t < kv; t++) {
                noc.async_read(s0_acc, cbs, tb_io, {.page_id = base_page + t}, {.offset_bytes = t * tb_io});
            }
            noc.async_read_barrier();
        } else {
            zero(cbs.get_write_ptr(), kv * tb_io / 4);
        }
        cbs.push_back(kv);

        if constexpr (FNG) {
            // z's row for head (b,h): [1,Bz,W] TILE, row b, tile column ZOT + h*Vt -- the same
            // gather as v's, into the v ring (the compute has consumed v by the time it needs z).
            gather_row(z_acc, cb_v, Vt, (b / 32) * WTZ + ZOT + h * Vt, b % 32);
        }
    }"""
    new = """        // ---- round 1: issue every read of this head, one barrier, then extract + push ----
        uint32_t rq, rk, rv, rs;  // source rows within the tiles
        uint32_t bq, bk, bv, bbeta, bg;
        uint32_t cc;
        if constexpr (PACKED) {
            // [1,B,C] TILE: row b of tile-row b/32; q/k from GQA source head h/RF.
            const uint32_t row_page0 = (b / 32) * Ct;
            const uint32_t r = b % 32;
            const uint32_t hq = h / RF;
            rq = rk = rv = r;
            rs = r;
            cc = h % 32;
            bq = issue_row(q_acc, cb_q, Kt, row_page0 + QOT + hq * Kt);
            bk = issue_row(k_acc, cb_k, Kt, row_page0 + KOT + hq * Kt);
            bv = issue_row(v_acc, cb_v, Vt, row_page0 + VOT + h * Vt);
            // beta/g [1,B,H] TILE: page (b/32)*NVT + h/32, row b%32, col h%32.
            bbeta = issue_scalar(beta_acc, cb_beta, (b / 32) * NVT + h / 32);
            bg = issue_scalar(g_acc, cb_g, (b / 32) * NVT + h / 32);
        } else {
            // Head (b,h)'s physical row in the [B,1,H,D] TILE tensors.
            const uint32_t physrow = b * padH + h;
            rq = rk = rv = physrow % 32;
            rs = 0;
            cc = h % 32;
            bq = issue_row(q_acc, cb_q, Kt, (physrow / 32) * Kt);
            bk = issue_row(k_acc, cb_k, Kt, (physrow / 32) * Kt);
            bv = issue_row(v_acc, cb_v, Vt, (physrow / 32) * Vt);
            // beta/g [B,1,H]: (b,hcol) at page b*ppb + hcol/32, row 0, col hcol%32.
            bbeta = issue_scalar(beta_acc, cb_beta, b * ppb + h / 32);
            bg = issue_scalar(g_acc, cb_g, b * ppb + h / 32);
        }
        // State [B,H,K,V] flat 2D is [B*H*K, V] (K,V are 32-multiples: no
        // per-batch padding): head bh owns full row-tiles (bh*Kt .. bh*Kt+Kt).
        CircularBuffer cbs(cb_state);
        cbs.reserve_back(kv);
        if (has_s0) {
            const uint32_t base_page = bh * kv;
            for (uint32_t t = 0; t < kv; t++) {
                noc.async_read(s0_acc, cbs, tb_io, {.page_id = base_page + t}, {.offset_bytes = t * tb_io});
            }
        } else {
            zero(cbs.get_write_ptr(), kv * tb_io / 4);
        }
        noc.async_read_barrier();
        finish_row(cb_q, bq, Kt, rq);
        finish_row(cb_k, bk, Kt, rk);
        finish_row(cb_v, bv, Vt, rv);
        finish_scalar(cb_beta, bbeta, rs, cc);
        finish_scalar(cb_g, bg, rs, cc);
        cbs.push_back(kv);

        if constexpr (FNG) {
            // ---- round 2: z's row for head (b,h): [1,Bz,W] TILE, row b, tile column ZOT + h*Vt --
            // the same gather as v's, into the v ring (its pages come after v's, so it can only be
            // reserved once v is pushed: the ring's write pointer moves on push).
            const uint32_t bz = issue_row(z_acc, cb_v, Vt, (b / 32) * WTZ + ZOT + h * Vt);
            noc.async_read_barrier();
            finish_row(cb_v, bz, Vt, b % 32);
        }
    }"""
    assert s.count(old) == 1, "instance loop anchor"
    return s.replace(old, new, 1)


if __name__ == "__main__":
    s = io.open(P, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    if "issue_row" in s:
        print("already applied")
        sys.exit(0)
    s2 = patch(s)
    io.open(P, "w", encoding="utf-8", newline="\n").write(s2)
    print("patched reader: two barriers per head")
