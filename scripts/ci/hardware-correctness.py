"""Actual hardware operator and fabric correctness gates, no throughput claims."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("audit", "kernels", "fabric"), required=True)
    parser.add_argument("--device-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("QWEN_HARDWARE_TESTS") != "1" or os.environ.get("TT_METAL_SIMULATOR") or os.environ.get("TT_METAL_SLOW_DISPATCH_MODE"):
        raise RuntimeError("Explicit hardware authorization and fast dispatch required; simulator prohibited")
    import torch
    import ttnn

    source = Path(os.environ["TT_METAL_HOME"])
    report = {"passed": False, "suite": args.suite, "backend": "hardware", "device_index": args.device_index,
              "ttnn_path": ttnn.__file__, "torch_version": torch.__version__, "results": [],
              "scope": "Correctness only; first-call timings include compilation and are not decode throughput"}
    try:
        report["tt_metal_revision"] = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        report["extension_sha256"] = hashlib.sha256(Path(ttnn._ttnn.__file__).read_bytes()).hexdigest()
        names = ("gdn_decode_norm_gate", "gdn_decode_conv_gates", "attn_decode_prep", "decode_gated_delta_rule")
        report["bindings"] = {name: hasattr(ttnn.transformer, name) for name in names}
        for directory in (source, Path("/opt/vllm-tt-plugin")):
            report[str(directory) + "_diff"] = subprocess.check_output(["git", "-C", str(directory), "diff", "--stat"], text=True)
        if report["tt_metal_revision"] != "9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9":
            raise RuntimeError("Unreviewed image TT-Metal revision")
        if not all(report["bindings"].values()):
            raise RuntimeError("Required fused op binding absent from serving image")
        if args.suite == "kernels":
            device = ttnn.open_device(device_id=args.device_index, l1_small_size=24576)
            try:
                device.enable_program_cache()
                root = Path(__file__).resolve().parents[2] / "optimisation/ttnn-op"
                for name in ("norm_gate", "gdn_conv_gates", "attn_prep"):
                    path = root / f"test_{name}.py"
                    spec = importlib.util.spec_from_file_location(name, path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    for batch in (1, 3, 8, 32):
                        initial_cache = None
                        for seed in (0, 1):
                            if name == "norm_gate":
                                passed = module.run_case(device, batch, max(8, batch), 8256, 5120, ttnn.bfloat16, seed)
                            elif name == "gdn_conv_gates":
                                passed = module.run_case(device, batch, max(8, batch), seed)
                                passed = module.run_direct_case(device, batch, max(8, batch), seed + 100) and passed
                            else:
                                passed = module.run_case(device, batch, seed)
                            cache = device.num_program_cache_entries()
                            result = dict(op=name, batch=batch, seed=seed, passed=bool(passed), program_cache_entries=cache)
                            report["results"].append(result)
                            if not passed or (initial_cache is not None and initial_cache != cache):
                                raise AssertionError(result)
                            initial_cache = cache
            finally:
                ttnn.close_device(device)
        elif args.suite == "fabric":
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=16777216)
            trace = None
            try:
                mesh.enable_program_cache()
                expected = torch.cat([torch.ones(1, 1, 32, 32), torch.full((1, 1, 32, 32), 2.0)], dim=3).bfloat16()
                tensor = ttnn.from_torch(expected, device=mesh, layout=ttnn.TILE_LAYOUT,
                                         mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3))
                def operation():
                    return ttnn.all_gather(tensor, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
                def verify(output, label):
                    shards = ttnn.get_device_tensors(output)
                    if len(shards) != 2:
                        raise AssertionError("Expected output from both physical chips")
                    for index, shard in enumerate(shards):
                        torch.testing.assert_close(ttnn.to_torch(shard), expected, rtol=0, atol=0)
                        report["results"].append(dict(check=label, chip=index, passed=True))
                warm = operation()
                verify(warm, "fabric_all_gather_eager")
                ttnn.deallocate(warm)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                output = operation()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                for repeat in range(3):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    verify(output, f"fabric_all_gather_trace_{repeat}")
            finally:
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                ttnn.close_mesh_device(mesh)
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
