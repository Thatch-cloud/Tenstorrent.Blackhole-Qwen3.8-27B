#!/usr/bin/env python3
"""Device-time microbench: the fused conv+gates op vs the 12 composed ops it replaces.

Both variants are captured into a trace of N iterations and the trace is replayed, so the
number is device time per layer-step with no host dispatch in it (the same way decode runs).
Shapes are Qwen3.8-27B at TP=2, B=8: C=5120, Nv=24, K=4.
"""
import sys
import time

import torch
import ttnn

C, NV, K = 5120, 24, 4
B = int(sys.argv[1]) if len(sys.argv) > 1 else 8
N = 64


def tt(dev, t, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)


def softplus_add(a, bias):
    return ttnn.add(a, bias, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)])


def composed(qkv, st, taps, a, b, dtb, nega, L1):
    # exactly gdn/tp.py forward_decode's conv + gates at B == Bmax
    for j in range(K - 1):
        ttnn.copy(st[j + 1], st[j])
    ttnn.copy(qkv, st[K - 1])
    conv = ttnn.multiply(st[0], taps[0], memory_config=L1)
    for j in range(1, K):
        conv = ttnn.mac(st[j], taps[j], conv)
    conv = ttnn.silu(conv, memory_config=L1)
    beta = ttnn.sigmoid(b, memory_config=L1)
    g = ttnn.multiply(nega, softplus_add(a, dtb), memory_config=L1)
    return conv, beta, g


def fused(qkv, st, taps, a, b, dtb, nega, L1):
    return ttnn.transformer.gdn_decode_conv_gates(qkv, st, taps, a, b, dtb, nega, batch=B, memory_config=L1)


def timed_trace(dev, fn, label):
    # warm (compiles + program cache), then capture N calls and replay the trace
    for _ in range(2):
        fn()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(N):
        fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)  # once untimed
    reps = 10
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    us = (time.perf_counter() - t0) / (reps * N) * 1e6
    ttnn.release_trace(dev, tid)
    print(f"TRACE {label}: {us:.1f} us per layer-step (device, B={B}, {N}x{reps} replays)", flush=True)
    return us


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=64 * 1024 * 1024)
    try:
        L1 = ttnn.L1_MEMORY_CONFIG
        torch.manual_seed(0)
        qkv = tt(dev, torch.randn(1, B, C), L1)
        st = [tt(dev, torch.randn(1, B, C)) for _ in range(K)]
        taps = [tt(dev, torch.randn(1, 1, C) * 0.5) for _ in range(K)]
        a, b = tt(dev, torch.randn(1, B, NV), L1), tt(dev, torch.randn(1, B, NV), L1)
        dtb, nega = tt(dev, torch.randn(1, 1, NV)), tt(dev, -torch.rand(1, 1, NV))
        # trace-safe: every persistent buffer allocated above; the composed path allocates its
        # intermediates inside the trace region like decode does
        tf = timed_trace(dev, lambda: fused(qkv, st, taps, a, b, dtb, nega, L1), "fused conv+gates (1 op)")
        tc = timed_trace(dev, lambda: composed(qkv, st, taps, a, b, dtb, nega, L1), "composed (12 ops)")
        print(f"RESULT B={B}: composed {tc:.1f} us -> fused {tf:.1f} us per layer; x48 layers = {(tc - tf) * 48 / 1000:.2f} ms/step saved", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
