"""Section 3.2 v2. Under `python -m tracy -r`; parse MatmulDeviceOperation rows in PLAN order (8 each)."""
import sys, torch, ttnn
sys.path.insert(0, "/opt/tt-metal")
import models.demos.blackhole.qwen36.tt.tp_common as tpc
dev = ttnn.open_device(device_id=0)
K, M = 5120, 32
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True)
WARM, REPS = 2, 6
plan = []
def run(label, x_t, w_t, pc, mc=ttnn.L1_MEMORY_CONFIG):
    try:
        for _ in range(WARM + REPS):
            o = ttnn.linear(x_t, w_t, program_config=pc, compute_kernel_config=ckc, memory_config=mc); ttnn.deallocate(o)
        ttnn.synchronize_device(dev); plan.append(label); print(f"CFG {label} ok", flush=True)
    except Exception as e:
        print(f"CFG {label} FAILED {type(e).__name__}: {str(e)[:220]}", flush=True)
x_il = ttnn.from_torch(torch.randn(1,1,M,K,dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
def W(N, dt, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(torch.randn(K,N,dtype=torch.bfloat16), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
# A. per-tile vs per-op: bf4 across N on the tuned builder (fixed cost shows as a nonzero intercept vs tiles)
for N in (4352, 8704, 17408):
    pc = tpc.create_matmul_1d_decode_progcfg(M, K, N, num_cores=44, grid_w=11); w = W(N, ttnn.bfloat4_b)
    run(f"nsweep_bf4_N{N}", x_il, w, pc); ttnn.deallocate(w)
# B. in0_block_w sweep at N=8704, bf4: copy every field of the tuned config, vary only in0_block_w
N = 8704; base = tpc.create_matmul_1d_decode_progcfg(M, K, N, num_cores=44, grid_w=11); w4 = W(N, ttnn.bfloat4_b)
print(f"tuned cfg: in0_block_w={base.in0_block_w} per_core_N={base.per_core_N} sub_h={base.out_subblock_h} sub_w={base.out_subblock_w}", flush=True)
for ib in (1, 2, 4, 5, 8, 10, 16, 20, 32, 40):
    if (K // 32) % ib: continue
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=base.compute_with_storage_grid_size, in0_block_w=ib,
        out_subblock_h=base.out_subblock_h, out_subblock_w=base.out_subblock_w,
        per_core_M=base.per_core_M, per_core_N=base.per_core_N, fuse_batch=True, fused_activation=None, mcast_in0=True)
    run(f"ib{ib}_bf4", x_il, w4, pc)
ttnn.deallocate(w4)
# C. DRAM-width-sharded weight + the model's DRAM-sharded progcfg, with the L1-sharded activation it requires
try:
    x_sh = ttnn.to_memory_config(x_il, tpc.create_activation_shard_config(K))
    w_sh = W(N, ttnn.bfloat4_b, tpc.create_dram_sharded_mem_config(K, N))
    run("dramsharded_bf4", x_sh, w_sh, tpc.create_dram_sharded_matmul_program_config(M, K, N), ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
    w_sh8 = W(N, ttnn.bfloat8_b, tpc.create_dram_sharded_mem_config(K, N))
    run("dramsharded_bf8", x_sh, w_sh8, tpc.create_dram_sharded_matmul_program_config(M, K, N), ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
except Exception as e:
    print(f"CFG dramsharded setup FAILED {type(e).__name__}: {str(e)[:220]}", flush=True)
print("PLAN " + ",".join(plan), flush=True)
ttnn.close_device(dev)
