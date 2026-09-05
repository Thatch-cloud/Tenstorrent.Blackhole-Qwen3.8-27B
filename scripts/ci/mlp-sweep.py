"""Real-weight TP2 MLP grid sweep; no serving defaults or precision changes."""

import json
import argparse
import math
import os
from pathlib import Path
import statistics
import time


def candidates(packing=False, fusion=False):
    result = [dict(name="control", gate=44, down=33, block=8)]
    if packing and fusion:
        raise ValueError("Select one experiment")
    if fusion:
        return result + [dict(name=f"fused-nblock-{block}", nblock=block) for block in (8, 16, 32)]
    if packing:
        result += [dict(name=f"gate-tiles-{tiles}", gate=44, down=33, block=8, gate_tiles=tiles) for tiles in (8, 12, 16)]
        result += [dict(name=f"down-tiles-{tiles}", gate=44, down=33, block=8, down_tiles=tiles) for tiles in (8, 12, 16)]
        return result
    result += [dict(name=f"gate-{cores}", gate=cores, down=33, block=8) for cores in (22, 33, 55, 66, 88, 110)]
    result += [dict(name=f"down-{cores}", gate=44, down=cores, block=8) for cores in (22, 44, 55, 66, 88, 110)]
    result += [dict(name=f"block-{block}", gate=44, down=33, block=block) for block in (4, 16)]
    return result


def geometry(inner, output, cores, block, per_core=None):
    if cores not in range(11, 111, 11) or inner % (32 * block):
        raise ValueError("Unreviewed grid or K blocking")
    minimum = math.ceil(math.ceil(output / 32) / cores)
    per_core = minimum if per_core is None else per_core
    if per_core < minimum or type(per_core) is not int:
        raise ValueError("Insufficient output tile coverage")
    subblock = max(value for value in range(1, 5) if per_core % value == 0)
    return dict(compute_with_storage_grid_size=(11, cores // 11), in0_block_w=block,
                out_subblock_h=1, out_subblock_w=subblock, per_core_M=1, per_core_N=per_core,
                fuse_batch=True, mcast_in0=True)


def paired_summary(blocks):
    changes = [block["candidate_ms"] / block["control_ms"] - 1 for block in blocks]
    return dict(mean_latency_change=statistics.mean(changes), min_latency_change=min(changes),
                max_latency_change=max(changes))


def projection_candidates(report):
    if report.get("passed") is not True or report.get("seeds") != [123, 456, 789]:
        raise ValueError("Completed three-seed projection prerequisite required")
    result = [dict(name="control", gate=44, down=33, block=8)]
    for name, candidate in report.get("candidates", {}).items():
        if candidate.get("exact_control") is not True or candidate.get("eligible_for_mlp_gate") is not True:
            continue
        blocks = candidate.get("blocks", [])
        if len(blocks) != 3 or not all(
            math.isfinite(block["control_ms"]) and math.isfinite(block["fused_ms"])
            and 0 < block["fused_ms"] < .98 * block["control_ms"] for block in blocks
        ):
            raise ValueError("Projection latency gate is inconsistent")
        required = {(mode + name, seed, chip) for mode in ("eager-", "trace-")
                    for seed in report["seeds"] for chip in (0, 1)}
        verified = {(check.get("mode"), check.get("seed"), check.get("chip"))
                    for check in report["checks"] if check.get("exact") is True}
        if not required <= verified:
            raise ValueError("Missing exact projection comparisons on both chips")
        kernel = candidate["kernel"]
        if (kernel["pairs_per_worker"], kernel["workers"]) not in ((7, 39), (5, 55), (4, 68), (3, 91)):
            raise ValueError("Unreviewed projection mapping")
        result.append(dict(name=name, pairs_per_worker=kernel["pairs_per_worker"], projection_kernel=kernel))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--fusion", action="store_true")
    parser.add_argument("--projection-report", type=Path)
    options = parser.parse_args()
    if os.environ.get("QWEN_HARDWARE_TESTS") != "1" or os.environ.get("QWEN_CARDS_ALLOCATED") != "1":
        raise RuntimeError("Explicit hardware allocation required")
    prerequisite = None
    if options.projection_report:
        if options.fusion or options.packing:
            raise ValueError("Select one experiment")
        prerequisite = json.loads(options.projection_report.read_text())
        configurations = projection_candidates(prerequisite)
        if len(configurations) == 1:
            skipped = dict(passed=False, skipped=True, reason="No projection candidate passed exactness and all timing blocks")
            Path("/experiment/results/mlp-sweep.json").write_text(json.dumps(skipped, indent=2))
            print(json.dumps(skipped), flush=True)
            return
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tests.test_factory import load_mlp_layer, compute_pcc
    from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
    from models.tt_transformers.tt.ccl import TT_CCL

    root = Path("/experiment/results")
    report = dict(passed=False, results=[], scope="Single real-weight MLP layer, B1, TP2, replicated DRAM input; not full-model throughput",
                  layer=0, seeds=[123, 456, 789], repeats_per_sample=20,
                  precision="Existing BF4 gate/up, BF8 down, LoFi FP32 destination, packer L1 accumulation enabled")
    mesh = None
    traces = {}
    outputs = {}
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=268435456)
        mesh.enable_program_cache()
        args = Qwen36ModelArgs(mesh, max_batch_size=8, max_seq_len=65536)
        if (args.dim, args.hidden_dim, args.num_devices, args.decode_grid_w) != (5120, 17408, 2, 11):
            raise AssertionError("Unexpected model/grid")
        state = load_mlp_layer(args.CKPT_DIR, 0)
        mlp = Qwen36MLP(mesh, state, None, args=args, tt_ccl=TT_CCL(mesh))
        frozen = (args.mlp_w1_decode_1d_progcfg, args.mlp_w3_decode_1d_progcfg, args.mlp_w2_decode_1d_progcfg)
        gate_weight = state["gate_proj.weight"].float()
        up_weight = state["up_proj.weight"].float()
        down_weight = state["down_proj.weight"].float()
        vectors = []
        references = []
        for seed in report["seeds"]:
            torch.manual_seed(seed)
            vector = torch.randn(1, 1, 1, args.dim, dtype=torch.bfloat16)
            vectors.append(vector)
            activation = vector.float().reshape(1, args.dim)
            references.append((torch.nn.functional.silu(activation @ gate_weight.T) * (activation @ up_weight.T)) @ down_weight.T)
        host_inputs = [ttnn.from_torch(vector, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                      mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)) for vector in vectors]
        device_input = ttnn.from_torch(vectors[0], device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        if prerequisite is None:
            configurations = candidates(options.packing, options.fusion)
        report["packing_sweep"] = options.packing
        report["fusion_sweep"] = options.fusion
        report["projection_prerequisite"] = str(options.projection_report) if prerequisite is not None else None
        packed_weight = None
        if options.fusion or prerequisite is not None:
            from models.tt_dit.utils.tensor import prepare_for_fused_swiglu
            gate_up = torch.cat([state["gate_proj.weight"].bfloat16().T.contiguous(),
                                 state["up_proj.weight"].bfloat16().T.contiguous()], dim=-1)
            packed = prepare_for_fused_swiglu(gate_up, ndev=2, gate_is_first=True)
            restored = packed.reshape(args.dim, 2, args.hidden_dim // 64, 2, 32).permute(0, 3, 1, 2, 4).reshape_as(gate_up)
            if not torch.equal(restored, gate_up):
                raise AssertionError("Gate/up tile packing changed host weights")
            packed_weight = ttnn.from_torch(packed, device=mesh, dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))
            report["packed_weights"] = dict(extra_elements_per_chip=args.dim * args.hidden_dim,
                raw_four_bit_bytes_per_chip=args.dim * args.hidden_dim // 2,
                scope="Additional single-layer experiment allocation; raw bytes exclude tile metadata. No cache writes or full-model allocation claim")
        controls = []
        forward = mlp.forward
        def configure(candidate):
            nonlocal forward
            forward = mlp.forward
            if candidate["name"] == "control":
                args.mlp_w1_decode_1d_progcfg, args.mlp_w3_decode_1d_progcfg, args.mlp_w2_decode_1d_progcfg = frozen
                return
            if options.fusion or prerequisite is not None:
                from models.tt_transformers.tt.ccl import tt_all_reduce
                if prerequisite is not None:
                    from fused_1d import FusedProjection
                    projection = FusedProjection(mesh, packed_weight, pairs_per_worker=candidate["pairs_per_worker"])
                    if projection.manifest != candidate["projection_kernel"]:
                        raise ValueError("Projection implementation changed since prerequisite")
                else:
                    config = ttnn.MinimalMatmulConfig(M_block_size=1, K_block_size=8,
                        N_block_size=candidate["nblock"], subblock_h=1, subblock_w=4,
                        compute_with_storage_grid_size=ttnn.CoreCoord(11, 2))
                def fused_forward(value):
                    local = ttnn.to_memory_config(value, ttnn.L1_MEMORY_CONFIG)
                    if prerequisite is not None:
                        hidden = projection(local)
                    else:
                        hidden = ttnn.experimental.minimal_matmul(local, packed_weight, config=config,
                            memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                            compute_kernel_config=mlp.compute_kernel_config_decode, fuse_swiglu=True)
                    ttnn.deallocate(local)
                    partial = ttnn.linear(hidden, mlp.weights.w2,
                        compute_kernel_config=mlp.compute_kernel_config_decode,
                        program_config=frozen[2], memory_config=ttnn.L1_MEMORY_CONFIG)
                    ttnn.deallocate(hidden)
                    return tt_all_reduce(partial, mesh, mlp.tt_ccl, cluster_axis=0, dim=3,
                        topology=args.ccl_topology(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                forward = fused_forward
                return
            gate = geometry(args.dim, args.hidden_dim // 2, candidate["gate"], candidate["block"], candidate.get("gate_tiles"))
            down = geometry(args.hidden_dim // 2, args.dim, candidate["down"], candidate["block"], candidate.get("down_tiles"))
            args.mlp_w1_decode_1d_progcfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(**gate, fused_activation=ttnn.UnaryOpType.SILU)
            args.mlp_w3_decode_1d_progcfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(**gate, fused_activation=None)
            args.mlp_w2_decode_1d_progcfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(**down, fused_activation=None)
        def read(tensor):
            return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3)).reshape(1, args.dim).float()
        for candidate in configurations:
            configure(candidate)
            result = dict(**candidate, exact_control=True, correctness=[], traced_correctness=[], blocks=[])
            report["results"].append(result)
            for index, host in enumerate(host_inputs):
                ttnn.copy_host_to_device_tensor(host, device_input)
                output = forward(device_input)
                actual = read(output)
                ttnn.deallocate(output)
                pcc = compute_pcc(references[index], actual)
                if candidate["name"] == "control":
                    controls.append(actual.clone())
                exact = torch.equal(actual, controls[index])
                result["correctness"].append(dict(seed=report["seeds"][index], torch_pcc=pcc, exact_control=exact))
                result["exact_control"] &= exact
                if not math.isfinite(pcc) or pcc < .97:
                    raise AssertionError(f"Existing MLP reference threshold failed: {candidate['name']}")
            print(json.dumps(dict(compiled=candidate["name"], exact_control=result["exact_control"])), flush=True)
        ttnn.copy_host_to_device_tensor(host_inputs[0], device_input)
        for candidate in configurations:
            configure(candidate)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            outputs[candidate["name"]] = forward(device_input)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            traces[candidate["name"]] = trace
        for index, host in enumerate(host_inputs):
            ttnn.copy_host_to_device_tensor(host, device_input)
            for result in report["results"]:
                ttnn.execute_trace(mesh, traces[result["name"]], cq_id=0, blocking=False)
                actual = read(outputs[result["name"]])
                exact = torch.equal(actual, controls[index])
                if not exact:
                    result["exact_control"] = False
                pcc = compute_pcc(references[index], actual)
                result["traced_correctness"].append(dict(seed=report["seeds"][index], torch_pcc=pcc, exact_control=exact))
                if not math.isfinite(pcc) or pcc < .97:
                    raise AssertionError("Traced MLP reference gate failed")
        ttnn.copy_host_to_device_tensor(host_inputs[0], device_input)
        def timed(name):
            ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            started = time.perf_counter()
            for repeat in range(20):
                ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            return (time.perf_counter() - started) * 1000 / 20
        for result in report["results"][1:]:
            for block in range(3):
                samples = [timed(name) for name in ("control", result["name"], result["name"], "control")]
                result["blocks"].append(dict(block=block, samples_ms=samples,
                    control_ms=statistics.mean((samples[0], samples[3])), candidate_ms=statistics.mean(samples[1:3])))
            result.update(paired_summary(result["blocks"]))
            result["eligible_for_full_model_gate"] = result["exact_control"] and result["max_latency_change"] < -.02
            print(json.dumps(result), flush=True)
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (root / "mlp-sweep.json").write_text(json.dumps(report, indent=2))
        if mesh is not None:
            for trace in traces.values():
                ttnn.release_trace(mesh, trace)
            for output in outputs.values():
                ttnn.deallocate(output)
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
