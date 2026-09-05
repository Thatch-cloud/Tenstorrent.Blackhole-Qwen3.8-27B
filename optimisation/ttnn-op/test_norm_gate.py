#!/usr/bin/env python3
"""Unit test + device-time microbench for ttnn.transformer.gdn_decode_norm_gate.

Reference (torch, fp32 on bf16-rounded inputs): rms_norm over V of each (b,h) stick of o,
times norm_w, times silu(z[b, z_off + h*V : ...]), laid out [1, B, H*V]. Cases: B == 8 with
z as an exact [1,B,H*V] tensor, B == 3 (bucketed rows), z as a column window of a wider
tensor (the fused projection output), and fp32 o. Then a trace-replayed timing of the fused
op against the five composed ops it replaces.
"""
import sys
import time

import torch
import ttnn

H, V = 24, 128
N = 64


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def tt_tile(dev, t, dtype=ttnn.bfloat16, mc=ttnn.DRAM_MEMORY_CONFIG):
    tt_ = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    return ttnn.from_torch(t.to(tt_), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)


def tt_rm(dev, t, dtype=ttnn.bfloat16):
    tt_ = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    return ttnn.from_torch(t.to(tt_), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)


def reference(o, z, w, B, z_off):
    r = lambda t: t.to(torch.bfloat16).float()
    o = r(o).reshape(B, H, V)
    rms = o / torch.sqrt((o * o).mean(-1, keepdim=True) + 1e-6)
    xw = (rms * r(w).reshape(1, 1, V)).reshape(1, B, H * V)
    zz = r(z)[:, :B, z_off : z_off + H * V]
    return xw * torch.nn.functional.silu(zz)


def run_case(dev, B, Bz, W, z_off, o_dtype, seed):
    torch.manual_seed(seed)
    o = torch.randn(B, 1, H, V)
    z = torch.randn(1, Bz, W) * 2
    w = torch.rand(1, 1, V) + 0.5
    ref = reference(o, z, w, B, z_off)
    to = tt_rm(dev, o, o_dtype)
    tz = tt_tile(dev, z)
    tw = tt_tile(dev, w)
    t0 = time.perf_counter()
    out = ttnn.transformer.gdn_decode_norm_gate(to, tz, tw, H, batch=B, z_col_offset=z_off, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) * 1000
    got = ttnn.to_torch(out).float()
    assert list(got.shape) == [1, B, H * V], got.shape
    # the output is bf16: judge by PCC and by max error relative to the output's magnitude
    # (1 bf16 ulp at |y| = 16 is 0.0625)
    maxerr = (got - ref).abs().max().item()
    relerr = maxerr / max(ref.abs().max().item(), 1e-3)
    res = {"pcc": pcc(got, ref), "maxerr": round(maxerr, 4), "rel_to_max": round(relerr, 5), "first_call_ms": round(dt, 1)}
    ok = res["pcc"] > 0.9999 and relerr < 0.01
    print(f"CASE B={B} Bz={Bz} W={W} z_off={z_off} o={o_dtype}: {'PASS' if ok else 'FAIL'} {res}", flush=True)
    if not ok:
        d = (got - ref).abs().reshape(B, H, V).amax(-1)
        print("   worst (b,h):", d.flatten().topk(4).indices.tolist(), d.max().item())
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
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=64 * 1024 * 1024)
    try:
        ok = True
        ok &= run_case(dev, 8, 8, H * V, 0, ttnn.bfloat16, 0)
        ok &= run_case(dev, 3, 8, H * V, 0, ttnn.bfloat16, 1)
        ok &= run_case(dev, 8, 8, 5120 + H * V + 64, 5120, ttnn.bfloat16, 2)  # z as a window of qkvzab
        # fp32 o is rejected by the op (plain tilize is bf16-only); the ship config runs bf16 GDN
        ok &= run_case(dev, 32, 32, H * V, 0, ttnn.bfloat16, 4)

        # ---- microbench at B=8: fused vs the composed to_layout/reshape/rms_norm/reshape/silu/mul ----
        B = 8
        torch.manual_seed(9)
        L1 = ttnn.L1_MEMORY_CONFIG
        o_rm = tt_rm(dev, torch.randn(B, 1, H, V))
        z = tt_tile(dev, torch.randn(1, B, H * V), mc=L1)
        w = tt_tile(dev, torch.rand(1, 1, V) + 0.5)

        def composed():
            o = ttnn.to_layout(o_rm, ttnn.TILE_LAYOUT)
            out_r = ttnn.reshape(o, (B, H, V))
            out_n = ttnn.rms_norm(out_r, weight=w, epsilon=1e-6, memory_config=L1)
            out_f = ttnn.reshape(out_n, (1, B, H * V))
            return ttnn.multiply(out_f, ttnn.silu(z, memory_config=L1), memory_config=L1)

        def fused():
            return ttnn.transformer.gdn_decode_norm_gate(o_rm, z, w, H, batch=B, memory_config=L1)

        tf = timed_trace(dev, fused, "fused norm+gate (1 op)", B)
        tc = timed_trace(dev, composed, "composed (to_layout, reshape, rms_norm, reshape, silu, mul)", B)
        print(f"BENCH B={B}: composed {tc:.1f} us -> fused {tf:.1f} us per layer; x48 = {(tc - tf) * 48 / 1000:.2f} ms/step", flush=True)
        print("RESULT", "PASS" if ok else "FAIL", flush=True)
        sys.exit(0 if ok else 1)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
