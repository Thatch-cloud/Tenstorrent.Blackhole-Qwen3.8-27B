#!/usr/bin/env python3
"""Fused output norm + gate inside decode_gated_delta_rule_packed: correctness + device time.

Reference: the same packed op WITHOUT z (returns o ROW_MAJOR), then torch
rms_norm(o)*norm_w*silu(z) laid out [1, B, H*V]. Cases cover one batch tile with all rows
(B=32: 768 heads, 7 per core), bucketed rows (B=3, B=1), the model's B=8, and z as a column
window of a wider tensor. Then a trace-replayed timing: fused op vs packed op + the six
composed ops (to_layout, reshape, rms_norm, reshape, silu, multiply).
"""
import sys
import time

import torch
import ttnn

NK, NV, DK, DV = 8, 24, 128, 128
C = 2 * NK * DK + NV * DV
N = 32


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def tt(dev, t, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)


def run_case(dev, B, W, z_off, seed):
    torch.manual_seed(seed)
    qkv = torch.randn(1, B, C)
    beta = torch.rand(1, B, NV)
    g = -torch.rand(1, B, NV) * 0.5
    state = torch.randn(B, NV, DK, DV) * 0.1
    z = torch.randn(1, B, W) * 2
    w = torch.rand(1, 1, DV) + 0.5
    r = lambda t: t.to(torch.bfloat16).float()

    o_rm, _ = ttnn.transformer.decode_gated_delta_rule_packed(
        tt(dev, qkv), tt(dev, beta), tt(dev, g), NK, NV, DK, DV, initial_state=tt(dev, state))
    o = ttnn.to_torch(ttnn.to_layout(o_rm, ttnn.TILE_LAYOUT)).float().reshape(B, NV, DV)
    rms = o / torch.sqrt((o * o).mean(-1, keepdim=True) + 1e-6)
    ref = (rms * r(w).reshape(1, 1, DV)).reshape(1, B, NV * DV) * torch.nn.functional.silu(r(z)[:, :, z_off : z_off + NV * DV])

    t0 = time.perf_counter()
    gated, h = ttnn.transformer.decode_gated_delta_rule_packed(
        tt(dev, qkv), tt(dev, beta), tt(dev, g), NK, NV, DK, DV, initial_state=tt(dev, state),
        z=tt(dev, z), norm_w=tt(dev, w), z_col_offset=z_off, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) * 1000
    got = ttnn.to_torch(gated).float()
    assert list(got.shape) == [1, B, NV * DV], got.shape
    maxerr = (got - ref).abs().max().item()
    rel = maxerr / max(ref.abs().max().item(), 1e-3)
    res = {"pcc": pcc(got, ref), "maxerr": round(maxerr, 4), "rel_to_max": round(rel, 5), "first_call_ms": round(dt, 1)}
    ok = res["pcc"] > 0.9999 and rel < 0.01
    print(f"CASE B={B} W={W} z_off={z_off}: {'PASS' if ok else 'FAIL'} {res}", flush=True)
    if not ok:
        d = (got - ref).abs().reshape(B, NV, DV).amax(-1)
        bad = (d > 0.05 * max(ref.abs().max().item(), 1)).nonzero().tolist()
        print(f"   bad (b,h) count={len(bad)} first={bad[:12]}")
    return ok


def timed_trace(dev, fn, label, B):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(N):
        fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    reps = 10
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    us = (time.perf_counter() - t0) / (reps * N) * 1e6
    ttnn.release_trace(dev, tid)
    print(f"TRACE {label}: {us:.1f} us per layer-step (device, B={B})", flush=True)
    return us


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=128 * 1024 * 1024)
    try:
        ok = True
        ok &= run_case(dev, 8, NV * DV, 0, 0)
        ok &= run_case(dev, 3, NV * DV, 0, 1)
        ok &= run_case(dev, 1, NV * DV, 0, 2)
        ok &= run_case(dev, 8, 5120 + NV * DV + 64, 5120, 3)
        ok &= run_case(dev, 32, NV * DV, 0, 4)

        for B in (8, 1):
            torch.manual_seed(9)
            L1 = ttnn.L1_MEMORY_CONFIG
            qkv = tt(dev, torch.randn(1, B, C), L1)
            beta = tt(dev, torch.rand(1, B, NV), L1)
            g = tt(dev, -torch.rand(1, B, NV) * 0.5, L1)
            st = tt(dev, torch.randn(B, NV, DK, DV) * 0.1)
            z = tt(dev, torch.randn(1, B, NV * DV), L1)
            w = tt(dev, torch.rand(1, 1, DV) + 0.5)

            def composed():
                o, _ = ttnn.transformer.decode_gated_delta_rule_packed(qkv, beta, g, NK, NV, DK, DV, initial_state=st, inplace_state=True)
                o = ttnn.to_layout(o, ttnn.TILE_LAYOUT)
                out_r = ttnn.reshape(o, (B, NV, DV))
                out_n = ttnn.rms_norm(out_r, weight=w, epsilon=1e-6, memory_config=L1)
                out_f = ttnn.reshape(out_n, (1, B, NV * DV))
                return ttnn.multiply(out_f, ttnn.silu(z, memory_config=L1), memory_config=L1)

            def fused():
                gated, _ = ttnn.transformer.decode_gated_delta_rule_packed(
                    qkv, beta, g, NK, NV, DK, DV, initial_state=st, inplace_state=True, z=z, norm_w=w, memory_config=L1)
                return gated

            tf = timed_trace(dev, fused, f"packed op with fused norm+gate (B={B})", B)
            tc = timed_trace(dev, composed, f"packed op + 6 composed ops (B={B})", B)
            print(f"BENCH B={B}: composed {tc:.1f} us -> fused {tf:.1f} us per layer; x48 = {(tc - tf) * 48 / 1000:.2f} ms/step", flush=True)
        print("RESULT", "PASS" if ok else "FAIL", flush=True)
        sys.exit(0 if ok else 1)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
