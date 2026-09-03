#!/usr/bin/env python3
"""Section 3.2 arm D: DRAM-sharded vs tuned 1D at the model's real bf8 decode projections (TP=2)."""
import sys, torch, ttnn
sys.path.insert(0, "/opt/tt-metal")
import models.demos.blackhole.qwen36.tt.tp_common as tpc
dev = ttnn.open_device(device_id=0); M = 32
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True)
plan = []
def run(label, x_t, w_t, pc, mc):
    try:
        for _ in range(8):
            o = ttnn.linear(x_t, w_t, program_config=pc, compute_kernel_config=ckc, memory_config=mc); ttnn.deallocate(o)
        ttnn.synchronize_device(dev); plan.append(label); print(f"CFG {label} ok", flush=True)
    except Exception as e: print(f"CFG {label} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
# (name, K, N, tuned 1D num_cores) -- widths from model_config for TP=2; N values are per-device
SHAPES = [("gdn_in",  5120, 8256, 44), ("attn_qkv", 5120, 7168, 64), ("mlp_down", 8704, 5120, 33), ("gdn_out", 3072, 5120, 33)]
for name, K, N, nc in SHAPES:
    x_il = ttnn.from_torch(torch.randn(1,1,M,K,dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    wt = torch.randn(K,N,dtype=torch.bfloat16)
    w1 = ttnn.from_torch(wt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    run(f"{name}_1d{nc}", x_il, w1, tpc.create_matmul_1d_decode_progcfg(M, K, N, num_cores=nc, grid_w=11), ttnn.L1_MEMORY_CONFIG); ttnn.deallocate(w1)
    try:
        x_sh = ttnn.to_memory_config(x_il, tpc.create_activation_shard_config(K))
        w_sh = ttnn.from_torch(wt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=tpc.create_dram_sharded_mem_config(K, N))
        run(f"{name}_dramsh", x_sh, w_sh, tpc.create_dram_sharded_matmul_program_config(M, K, N), ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
        ttnn.deallocate(w_sh); ttnn.deallocate(x_sh)
    except Exception as e: print(f"CFG {name}_dramsh FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
    ttnn.deallocate(x_il)
print("PLAN " + ",".join(plan), flush=True); ttnn.close_device(dev)
