#!/usr/bin/env python3
"""Equivalence test: decode_gated_delta_rule_packed vs decode_gated_delta_rule.

The packed op reads q/k/v out of the conv+gates kernel's [1,B,C] layout (GQA in the reader)
and beta/g from [1,B,H]. The reference builds the [B,1,H,*] tensors on the host exactly as
gdn/tp.py does (slice, reshape, repeat_interleave) and runs the already-validated unpacked
op. Same kernel math on identical inputs, so outputs should agree to bf16 exactness.
"""
import sys
import time

import torch
import ttnn

NK, NV, DK, DV = 8, 24, 128, 128
C = 2 * NK * DK + NV * DV  # 5120
RF = NV // NK


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def tt(dev, t, dtype):
    return ttnn.from_torch(t.to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)


def run_case(dev, B, dtype, seed):
    torch.manual_seed(seed)
    qkv = torch.randn(1, B, C)
    beta = torch.rand(1, B, NV)
    g = -torch.rand(1, B, NV) * 0.5
    state = torch.randn(B, NV, DK, DV) * 0.1
    # host-side split exactly as tp.py forward_decode
    kd = NK * DK
    q = qkv[:, :, :kd].reshape(B, NK, DK).repeat_interleave(RF, dim=1).reshape(B, 1, NV, DK)
    k = qkv[:, :, kd : 2 * kd].reshape(B, NK, DK).repeat_interleave(RF, dim=1).reshape(B, 1, NV, DK)
    v = qkv[:, :, 2 * kd :].reshape(B, NV, DV).reshape(B, 1, NV, DV)
    beta_r = beta.reshape(B, 1, NV)
    g_r = g.reshape(B, 1, NV)

    t_state_a = tt(dev, state, dtype)
    t_state_b = tt(dev, state, dtype)
    o_ref, h_ref = ttnn.transformer.decode_gated_delta_rule(
        tt(dev, q, dtype), tt(dev, k, dtype), tt(dev, v, dtype), tt(dev, beta_r, dtype), tt(dev, g_r, dtype),
        initial_state=t_state_a, inplace_state=False,
    )
    t0 = time.perf_counter()
    o_p, h_p = ttnn.transformer.decode_gated_delta_rule_packed(
        tt(dev, qkv, dtype), tt(dev, beta, dtype), tt(dev, g, dtype), NK, NV, DK, DV,
        initial_state=t_state_b, inplace_state=False,
    )
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) * 1000
    o_ref_t = ttnn.to_torch(ttnn.to_layout(o_ref, ttnn.TILE_LAYOUT)).float()
    o_p_t = ttnn.to_torch(ttnn.to_layout(o_p, ttnn.TILE_LAYOUT)).float()
    h_ref_t = ttnn.to_torch(h_ref).float()
    h_p_t = ttnn.to_torch(h_p).float()
    res = {
        "o_pcc": pcc(o_p_t, o_ref_t),
        "o_maxdiff": (o_p_t - o_ref_t).abs().max().item(),
        "o_identical": torch.equal(o_p_t, o_ref_t),
        "h_pcc": pcc(h_p_t, h_ref_t),
        "h_maxdiff": (h_p_t - h_ref_t).abs().max().item(),
        "h_identical": torch.equal(h_p_t, h_ref_t),
        "first_call_ms": round(dt, 1),
    }
    ok = res["o_pcc"] > 0.9999 and res["h_pcc"] > 0.9999 and res["o_maxdiff"] < 1e-2 and res["h_maxdiff"] < 1e-2
    print(f"CASE B={B} dtype={dtype}: {'PASS' if ok else 'FAIL'} {res}", flush=True)
    if not ok:
        # which heads / rows disagree
        d = (o_p_t - o_ref_t).abs().reshape(B, NV, DV).amax(dim=-1)
        bad = (d > 1e-2).nonzero().tolist()
        print(f"   bad (b,h) pairs: {bad[:20]} of {B*NV}")
    return ok


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        ok = True
        ok &= run_case(dev, 8, ttnn.bfloat16, 0)
        ok &= run_case(dev, 1, ttnn.bfloat16, 1)
        ok &= run_case(dev, 8, ttnn.float32, 2)
        ok &= run_case(dev, 32, ttnn.bfloat16, 3)
        # in-place variant on the packed path
        torch.manual_seed(5)
        B = 8
        st = tt(dev, torch.randn(B, NV, DK, DV) * 0.1, ttnn.bfloat16)
        addr = st.buffer_address()
        o, h = ttnn.transformer.decode_gated_delta_rule_packed(
            tt(dev, torch.randn(1, B, C), ttnn.bfloat16), tt(dev, torch.rand(1, B, NV), ttnn.bfloat16),
            tt(dev, -torch.rand(1, B, NV), ttnn.bfloat16), NK, NV, DK, DV, initial_state=st, inplace_state=True,
        )
        inplace_ok = h.buffer_address() == addr
        print(f"CASE inplace packed: {'PASS' if inplace_ok else 'FAIL'} same_buffer={inplace_ok}", flush=True)
        ok &= inplace_ok
        print("RESULT", "PASS" if ok else "FAIL", flush=True)
        sys.exit(0 if ok else 1)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
