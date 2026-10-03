#!/usr/bin/env python3
"""Unit test + device-time microbench for ttnn.transformer.attn_decode_prep.

Shapes are Qwen3.8-27B's attention at TP=2: NH=12, NKV=2, HD=256, RD=64 (partial rotary 0.25),
projection width W = (12 + 4 + 12) * 256 = 7168. Reference: torch on the bf16-rounded inputs --
rms_norm over HD per head, times the (1+w) weight, rounded to bf16, then HF rotate-half on the
first 64 dims. The composed chain (slices, nlp_create_qkv_heads_decode + reshards, two rms_norms
and multiplies, two partial-rope sequences, two pads + reshards) is timed in a replayed trace
against the fused op at B=8 and B=1.
"""
import sys
import time

import torch
import ttnn

NH, NKV, HD, RD = 12, 2, 256, 64
W = (NH + 2 * NKV + NH) * HD
N = 32


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def tt(dev, t, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)


def kv_cfg(B):
    cols = next(c for c in range(min(8, B), 0, -1) if B % c == 0)
    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, HD),
        core_grid=ttnn.CoreGrid(x=cols, y=B // cols),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def rope_tables(B, positions):
    theta = 10000000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, RD, 2).float() / RD))
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().reshape(1, B, 1, RD), emb.sin().reshape(1, B, 1, RD)


def reference(qkv, cos, sin, wq, wk, B):
    r = lambda t: t.to(torch.bfloat16).float()
    x = r(qkv)[0, 0, :B]  # [B, W]
    q = x[:, : NH * HD].reshape(B, NH, HD)
    k = x[:, NH * HD : (NH + NKV) * HD].reshape(B, NKV, HD)
    v = x[:, (NH + NKV) * HD : (NH + 2 * NKV) * HD].reshape(B, NKV, HD)
    g = x[:, (NH + 2 * NKV) * HD :].reshape(B, NH, HD)

    def norm_rope(t, w):
        y = t / torch.sqrt((t * t).mean(-1, keepdim=True) + 1e-6) * r(w).reshape(1, 1, HD)
        y = y.to(torch.bfloat16).float()
        c = r(cos)[0, :B, 0, :].unsqueeze(1)  # [B,1,RD]
        s = r(sin)[0, :B, 0, :].unsqueeze(1)
        xr = y[..., :RD]
        x1, x2 = xr[..., : RD // 2], xr[..., RD // 2 :]
        rot = torch.cat([-x2, x1], dim=-1)
        roped = xr * c + rot * s
        return torch.cat([roped, y[..., RD:]], dim=-1)

    return norm_rope(q, wq), g, norm_rope(k, wk), v


def run_case(dev, B, seed):
    torch.manual_seed(seed)
    qkv = torch.randn(1, 1, B, W)
    positions = torch.randint(0, 4000, (B,))
    cos, sin = rope_tables(B, positions)
    wq = torch.rand(1, 1, HD) + 0.5
    wk = torch.rand(1, 1, HD) + 0.5
    q_ref, g_ref, k_ref, v_ref = reference(qkv, cos, sin, wq, wk, B)
    t0 = time.perf_counter()
    q, g, k, v = ttnn.transformer.attn_decode_prep(
        tt(dev, qkv), tt(dev, cos), tt(dev, sin), tt(dev, wq), tt(dev, wk), NH, NKV, HD, RD, kv_cfg(B), batch=B)
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) * 1000
    q_t = ttnn.to_torch(q).float()[0]          # [B, NH, HD]
    g_t = ttnn.to_torch(g).float()[0]
    k_t = ttnn.to_torch(ttnn.sharded_to_interleaved(k, ttnn.DRAM_MEMORY_CONFIG)).float()[0]  # [B, 32, HD]
    v_t = ttnn.to_torch(ttnn.sharded_to_interleaved(v, ttnn.DRAM_MEMORY_CONFIG)).float()[0]
    res = {
        "q_pcc": pcc(q_t, q_ref), "q_rel": ((q_t - q_ref).abs().max() / q_ref.abs().max()).item(),
        "k_pcc": pcc(k_t[:, :NKV], k_ref), "k_rel": ((k_t[:, :NKV] - k_ref).abs().max() / k_ref.abs().max()).item(),
        "k_pad_zero": bool((k_t[:, NKV:] == 0).all()),
        "v_exact": torch.equal(v_t[:, :NKV], v_ref), "v_pad_zero": bool((v_t[:, NKV:] == 0).all()),
        "g_exact": torch.equal(g_t, g_ref), "first_call_ms": round(dt, 1),
    }
    ok = res["q_pcc"] > 0.9999 and res["k_pcc"] > 0.9999 and res["q_rel"] < 0.01 and res["k_rel"] < 0.01 \
        and res["v_exact"] and res["g_exact"] and res["k_pad_zero"] and res["v_pad_zero"]
    print(f"CASE B={B}: {'PASS' if ok else 'FAIL'} {res}", flush=True)
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


def partial_rope(x, cos_tt, sin_tt, n_heads, B):
    L1 = ttnn.L1_MEMORY_CONFIG
    x_rope = ttnn.slice(x, (0, 0, 0, 0), (1, B, n_heads, RD))
    x_rope_t = ttnn.transpose(x_rope, 1, 2)
    cos_p = ttnn.reshape(cos_tt, (1, 1, B, RD))
    sin_p = ttnn.reshape(sin_tt, (1, 1, B, RD))
    roped_t = ttnn.experimental.rotary_embedding_hf(x_rope_t, cos_p, sin_p, is_decode_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    roped = ttnn.to_memory_config(ttnn.transpose(roped_t, 1, 2), ttnn.DRAM_MEMORY_CONFIG)
    x_pass = ttnn.to_memory_config(ttnn.slice(x, (0, 0, 0, RD), (1, B, n_heads, HD)), ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.concat([roped, x_pass], dim=-1)


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=128 * 1024 * 1024)
    try:
        ok = True
        ok &= run_case(dev, 8, 0)
        ok &= run_case(dev, 1, 1)
        ok &= run_case(dev, 3, 2)
        ok &= run_case(dev, 32, 3)

        for B in (8, 1):
            torch.manual_seed(9)
            L1 = ttnn.L1_MEMORY_CONFIG
            qkv = tt(dev, torch.randn(1, 1, B, W))
            cos, sin = rope_tables(B, torch.randint(0, 4000, (B,)))
            cos_tt, sin_tt = tt(dev, cos), tt(dev, sin)
            wq, wk = tt(dev, torch.rand(1, 1, HD) + 0.5), tt(dev, torch.rand(1, 1, HD) + 0.5)
            cfg = kv_cfg(B)
            qkv3_dim = (NH + 2 * NKV) * HD

            def composed():
                qkv3 = ttnn.slice(qkv, (0, 0, 0, 0), (1, 1, B, qkv3_dim))
                gate = ttnn.slice(qkv, (0, 0, 0, qkv3_dim), (1, 1, B, qkv3_dim + NH * HD))
                qkv_l1 = ttnn.to_memory_config(qkv3, L1)
                q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(qkv_l1, num_heads=NH, num_kv_heads=NKV, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
                q = ttnn.sharded_to_interleaved(q, L1)
                k = ttnn.sharded_to_interleaved(k, L1)
                v = ttnn.sharded_to_interleaved(v, L1)
                g = ttnn.reshape(gate, (1, B, NH, HD))
                q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6, memory_config=L1), wq, memory_config=L1)
                k = ttnn.multiply(ttnn.rms_norm(k, epsilon=1e-6, memory_config=L1), wk, memory_config=L1)
                q = partial_rope(q, cos_tt, sin_tt, NH, B)
                k = partial_rope(k, cos_tt, sin_tt, NKV, B)
                k_p = ttnn.pad(k, [1, B, 32, HD], [0, 0, 0, 0], 0.0, memory_config=L1)
                v_p = ttnn.pad(v, [1, B, 32, HD], [0, 0, 0, 0], 0.0, memory_config=L1)
                k_sh = ttnn.to_memory_config(k_p, cfg)
                v_sh = ttnn.to_memory_config(v_p, cfg)
                return q, g, k_sh, v_sh

            def fused():
                return ttnn.transformer.attn_decode_prep(qkv, cos_tt, sin_tt, wq, wk, NH, NKV, HD, RD, cfg, batch=B)

            tf = timed_trace(dev, fused, f"attn_decode_prep (B={B})", B)
            tc = timed_trace(dev, composed, f"composed prologue (B={B})", B)
            print(f"BENCH B={B}: composed {tc:.1f} us -> fused {tf:.1f} us per layer; x16 = {(tc - tf) * 16 / 1000:.2f} ms/step", flush=True)
        print("RESULT", "PASS" if ok else "FAIL", flush=True)
        sys.exit(0 if ok else 1)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
