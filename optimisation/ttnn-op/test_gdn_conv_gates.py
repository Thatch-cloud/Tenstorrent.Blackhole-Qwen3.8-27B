#!/usr/bin/env python3
"""Unit test for ttnn.transformer.gdn_decode_conv_gates against a torch reference.

Shapes are Qwen3.8-27B's GDN at TP=2: C = 2*8*128 + 24*128 = 5120 channels, Nv = 24,
K = 4 taps. Checks conv_out (active rows), beta, g against fp32 torch math on the
bf16-rounded inputs, and that the shift-register advanced exactly (old st1 -> st0, ...,
x-masked -> st3). Runs B == Bmax and the bucketed B < Bmax case.
"""
import os
import sys
import time

import torch
import ttnn

C, NV, K = 5120, 24, 4


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    if a.std() == 0 or b.std() == 0:
        return float(torch.allclose(a, b))
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def tt(dev, t):
    return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def run_case(dev, B, Bmax, seed):
    torch.manual_seed(seed)
    x = torch.randn(1, B, C)
    st = [torch.randn(1, Bmax, C) for _ in range(K)]
    taps = [torch.randn(1, 1, C) * 0.5 for _ in range(K)]
    a = torch.randn(1, B, NV) * 2
    b = torch.randn(1, B, NV) * 2
    dtb = torch.randn(1, 1, NV)
    nega = -torch.exp(torch.randn(1, 1, NV))

    # bf16-rounded inputs are what the device sees
    r = lambda t: t.to(torch.bfloat16).float()
    x_m = r(x).clone()
    x_pad = torch.zeros(1, Bmax, C)
    x_pad[:, :B] = x_m
    win = [r(st[1]), r(st[2]), r(st[3]), x_pad]
    conv_ref = torch.nn.functional.silu(sum(win[j] * r(taps[j]) for j in range(K)))
    beta_ref = torch.sigmoid(r(b))
    g_ref = r(nega) * torch.nn.functional.softplus(r(a) + r(dtb), beta=1.0, threshold=20.0)

    tx = tt(dev, x)
    tst = [tt(dev, s) for s in st]
    ttaps = [tt(dev, t) for t in taps]
    ta, tb_, tdtb, tnega = tt(dev, a), tt(dev, b), tt(dev, dtb), tt(dev, nega)
    addrs = [s.buffer_address() for s in tst]

    t0 = time.perf_counter()
    out, beta, g = ttnn.transformer.gdn_decode_conv_gates(
        tx, tst, ttaps, ta, tb_, tdtb, tnega, batch=B, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) * 1000

    out_t = ttnn.to_torch(out).float()[:, :B]
    beta_t = ttnn.to_torch(beta).float()[:, :B]
    g_t = ttnn.to_torch(g).float()[:, :B]
    st_after = [ttnn.to_torch(s).float() for s in tst]

    res = {
        "conv_pcc": pcc(out_t, conv_ref[:, :B]),
        "conv_maxerr": (out_t - conv_ref[:, :B]).abs().max().item(),
        "beta_pcc": pcc(beta_t, beta_ref),
        "beta_maxerr": (beta_t - beta_ref).abs().max().item(),
        "g_pcc": pcc(g_t, g_ref),
        "g_maxerr": (g_t - g_ref).abs().max().item(),
        "shift_exact": all(torch.equal(st_after[j], win[j]) for j in range(K)),
        "state_addr_stable": [s.buffer_address() for s in tst] == addrs,
        "first_call_ms": dt,
    }
    ok = res["conv_pcc"] > 0.999 and res["beta_pcc"] > 0.999 and res["g_pcc"] > 0.999 and res["shift_exact"]
    print(f"CASE B={B} Bmax={Bmax}: {'PASS' if ok else 'FAIL'} {res}", flush=True)
    if not res["shift_exact"]:
        for j in range(K):
            d = (st_after[j] - win[j]).abs()
            print(f"   st{j}: maxdiff={d.max().item():.4g} rows_bad={(d.amax(dim=-1) > 0).sum().item()}")
    return ok


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        ok = True
        ok &= run_case(dev, 8, 8, 0)
        ok &= run_case(dev, 1, 8, 1)
        ok &= run_case(dev, 3, 8, 2)
        ok &= run_case(dev, 32, 32, 3)
        # steady-state timing: op cost when the program is cached
        torch.manual_seed(9)
        B = 8
        tx = tt(dev, torch.randn(1, B, C))
        tst = [tt(dev, torch.randn(1, B, C)) for _ in range(K)]
        ttaps = [tt(dev, torch.randn(1, 1, C)) for _ in range(K)]
        ta, tb_ = tt(dev, torch.randn(1, B, NV)), tt(dev, torch.randn(1, B, NV))
        tdtb, tnega = tt(dev, torch.randn(1, 1, NV)), tt(dev, -torch.rand(1, 1, NV))
        for _ in range(3):
            ttnn.transformer.gdn_decode_conv_gates(tx, tst, ttaps, ta, tb_, tdtb, tnega, batch=B)
        ttnn.synchronize_device(dev)
        n = 50
        t0 = time.perf_counter()
        for _ in range(n):
            o, be, gg = ttnn.transformer.gdn_decode_conv_gates(tx, tst, ttaps, ta, tb_, tdtb, tnega, batch=B)
        ttnn.synchronize_device(dev)
        print(f"TIMING cached op: {(time.perf_counter() - t0) / n * 1e6:.1f} us per call (host-dispatched, B={B})", flush=True)
        print("RESULT", "PASS" if ok else "FAIL", flush=True)
        sys.exit(0 if ok else 1)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
