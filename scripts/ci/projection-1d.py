"""Projection-only, real-weight two-chip gate/up correctness and paired trace timing."""

import json
import os
from pathlib import Path
import statistics
import time

from fused_1d import FusedProjection


def main():
    if os.environ.get("QWEN_HARDWARE_TESTS") != "1" or os.environ.get("QWEN_CARDS_ALLOCATED") != "1":
        raise RuntimeError("Explicit hardware allocation required")
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tests.test_factory import load_mlp_layer, compute_pcc
    from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

    report = dict(passed=False, checks=[], blocks=[], seeds=[123, 456, 789],
                  scope="Projection-only B1 TP2; same 39 N-workers, 7 gate/up pairs each, multicast input, K-block 8; not full MLP or serving throughput")
    root = Path("/experiment/results")
    mesh = None
    traces = {}
    outputs = {}
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=268435456)
        mesh.enable_program_cache()
        args = Qwen36ModelArgs(mesh, max_batch_size=8, max_seq_len=65536)
        if (args.dim, args.hidden_dim, args.num_devices) != (5120, 17408, 2):
            raise AssertionError("Unreviewed model dimensions")
        state = load_mlp_layer(args.CKPT_DIR, 0)
        mlp = Qwen36MLP(mesh, state, None, args=args, tt_ccl=TT_CCL(mesh))
        combined = torch.cat([state["gate_proj.weight"].bfloat16().T.contiguous(),
                              state["up_proj.weight"].bfloat16().T.contiguous()], dim=-1)
        packed = prepare_for_fused_swiglu(combined, ndev=2, gate_is_first=True)
        weight = ttnn.from_torch(packed, device=mesh, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))
        for chip, (packed_shard, gate, up) in enumerate(zip(ttnn.get_device_tensors(weight),
                ttnn.get_device_tensors(mlp.weights.w1), ttnn.get_device_tensors(mlp.weights.w3))):
            tiles = ttnn.to_torch(packed_shard).reshape(5120, 272, 2, 32)
            if not torch.equal(tiles[:, :, 0, :].reshape(5120, 8704), ttnn.to_torch(gate).reshape(5120, 8704)):
                raise AssertionError("Packed gate quantization changed")
            if not torch.equal(tiles[:, :, 1, :].reshape(5120, 8704), ttnn.to_torch(up).reshape(5120, 8704)):
                raise AssertionError("Packed up quantization changed")
            report["checks"].append(dict(chip=chip, packed_weight_exact=True))
        del tiles, combined, packed
        fused = FusedProjection(mesh, weight)
        diagnostic = FusedProjection(mesh, weight, intermediates=True)
        report["kernel"] = fused.manifest
        report["extra_packed_weight_bytes_per_chip"] = weight.buffer_num_pages() * weight.buffer_aligned_page_size()
        def control(value, intermediates=False):
            gate = ttnn.linear(value, mlp.weights.w1, compute_kernel_config=mlp.compute_kernel_config_decode,
                program_config=args.mlp_w1_decode_1d_progcfg, memory_config=ttnn.L1_MEMORY_CONFIG)
            up = ttnn.linear(value, mlp.weights.w3, compute_kernel_config=mlp.compute_kernel_config_decode,
                program_config=args.mlp_w3_decode_1d_progcfg, memory_config=ttnn.L1_MEMORY_CONFIG)
            if intermediates:
                return gate, up
            hidden = ttnn.mul(gate, up, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            return hidden
        inputs = []
        for seed in report["seeds"]:
            torch.manual_seed(seed)
            inputs.append(ttnn.from_torch(torch.randn(1, 1, 1, 5120, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)))
        value = ttnn.to_device(inputs[0], mesh, memory_config=ttnn.L1_MEMORY_CONFIG)
        expected = []
        def read(tensor):
            return [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(tensor)]
        def check(actual, reference, seed, mode):
            if len(actual) != 2 or len(reference) != 2:
                raise AssertionError("Both TP2 shards must be checked")
            exact = True
            numerical = True
            for chip, (result, target) in enumerate(zip(actual, reference)):
                equal = torch.equal(result, target)
                pcc = compute_pcc(target, result)
                report["checks"].append(dict(mode=mode, seed=seed, chip=chip, exact=equal, control_pcc=pcc,
                    mismatches=int(torch.count_nonzero(result != target)), max_abs=float((result - target).abs().max())))
                numerical &= bool(torch.isfinite(result).all()) and pcc >= .97
                exact &= equal
            if not exact:
                torch.save(dict(actual=actual, expected=reference), root / f"projection-{mode}-{seed}.pt")
            if not numerical:
                raise AssertionError("Projection numerical gate failed")
            return exact
        exact = True
        for seed, host in zip(report["seeds"], inputs):
            ttnn.copy_host_to_device_tensor(host, value)
            gate, up = control(value, intermediates=True)
            reference_gate, reference_up = read(gate), read(up)
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            intermediate = diagnostic(value)
            pairs = [shard.reshape(1, 1, 1, 272, 2, 32) for shard in read(intermediate)]
            check([shard[..., 0, :].reshape(1, 1, 1, 8704) for shard in pairs], reference_gate, seed, "gate")
            check([shard[..., 1, :].reshape(1, 1, 1, 8704) for shard in pairs], reference_up, seed, "up")
            ttnn.deallocate(intermediate)
            baseline = control(value)
            expected.append(read(baseline))
            ttnn.deallocate(baseline)
            candidate = fused(value)
            exact &= check(read(candidate), expected[-1], seed, "eager")
            ttnn.deallocate(candidate)
            print(json.dumps(dict(seed=seed, eager_exact=exact)), flush=True)
        for name, operation in (("control", control), ("fused", fused)):
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            outputs[name] = operation(value)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            traces[name] = trace
        for seed, host, reference in zip(report["seeds"], inputs, expected):
            ttnn.copy_host_to_device_tensor(host, value)
            for name in ("control", "fused"):
                ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
                matches = check(read(outputs[name]), reference, seed, "trace-" + name)
                if name == "control" and not matches:
                    raise AssertionError("Control trace differs from eager")
                exact &= matches
        def timed(name):
            ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            started = time.perf_counter()
            for repeat in range(40):
                ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            return (time.perf_counter() - started) * 1000 / 40
        ttnn.copy_host_to_device_tensor(inputs[0], value)
        for block in range(3):
            samples = [timed(name) for name in ("control", "fused", "fused", "control")]
            control_ms = statistics.mean((samples[0], samples[3]))
            fused_ms = statistics.mean(samples[1:3])
            report["blocks"].append(dict(samples_ms=samples, control_ms=control_ms, fused_ms=fused_ms,
                                          latency_change=fused_ms / control_ms - 1))
        report.update(passed=True, exact_control=exact,
            eligible_for_mlp_gate=exact and all(block["latency_change"] < -.02 for block in report["blocks"]))
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (root / "projection-1d.json").write_text(json.dumps(report, indent=2))
        if mesh is not None:
            for trace in traces.values():
                ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
