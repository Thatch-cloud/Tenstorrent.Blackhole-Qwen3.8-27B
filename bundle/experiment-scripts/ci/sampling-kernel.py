"""TP2 force-argmax correctness and warmed latency probe, not model throughput."""

import json
import os
from pathlib import Path
import statistics
import time

import torch


def logits_case(vocab, rows, kind):
    if vocab < 64 or vocab % 2:
        raise ValueError("Expected an even vocabulary of at least 64")
    torch.manual_seed(123)
    if kind == "random":
        return torch.randn(1, 1, rows, vocab).bfloat16()
    logits = torch.full((1, 1, rows, vocab), -10., dtype=torch.bfloat16)
    boundaries = (0, 31, 32, vocab // 2 - 1, vocab // 2, vocab - 1)
    for row in range(rows):
        if kind == "boundaries":
            logits[0, 0, row, boundaries[row % len(boundaries)]] = 100.
        elif kind == "cross-shard-tie":
            logits[0, 0, row, 31] = 100.
            logits[0, 0, row, vocab // 2] = 100.
        elif kind == "near-tie":
            logits[0, 0, row, 31] = 1.
            logits[0, 0, row, vocab // 2] = 1.0078125
        elif kind == "all-equal":
            logits.zero_()
            break
        else:
            raise ValueError(kind)
    return logits


def main():
    if os.environ.get("QWEN_HARDWARE_TESTS") != "1" or os.environ.get("QWEN_CARDS_ALLOCATED") != "1":
        raise RuntimeError("Explicit hardware allocation required")
    import ttnn
    from models.common.sampling.generator import SamplingGenerator
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
    from models.tt_transformers.tt.ccl import TT_CCL

    report = dict(passed=False, cases=[], scope="32-row TP2 sampler probe; not full-model or single-stream decode speed")
    root = Path("/experiment/results")
    mesh = None
    sampler = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=67108864)
        mesh.enable_program_cache()
        args = Qwen36ModelArgs(mesh, max_batch_size=8, max_seq_len=65536)
        report.update(vocab=args.vocab_size, padded_vocab=args.padded_vocab_size,
                      mesh=[int(dim) for dim in mesh.shape])
        if args.vocab_size != 248320 or args.padded_vocab_size != args.vocab_size:
            raise ValueError("Expected the current unpadded TP2 vocabulary")
        sampler = SamplingGenerator(args=args, mesh_device=mesh, tt_ccl=TT_CCL(mesh))
        sampler.set_trace_bucket(1)
        if not sampler.tt_sampling.force_argmax_sampling:
            raise AssertionError("Force argmax did not engage")
        if sampler.seed_manager.has_active_request_seed():
            raise AssertionError("Explicit seed would disable internal sampling trace")
        tensor = ttnn.from_torch(logits_case(args.vocab_size, 32, "random"), device=mesh,
                                 layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                                 mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
        def verify(result, expected, kind, execution):
            output = result[0] if isinstance(result, tuple) else result
            shards = ttnn.get_device_tensors(output)
            if len(shards) != 2:
                raise AssertionError("Missing chip output")
            for chip, shard in enumerate(shards):
                actual = ttnn.to_torch(shard).reshape(-1).to(torch.long)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                report["cases"].append(dict(kind=kind, execution=execution, chip=chip, passed=True))

        for kind in ("random", "boundaries", "cross-shard-tie", "near-tie", "all-equal"):
            values = logits_case(args.vocab_size, 32, kind)
            expected = values.argmax(dim=-1).reshape(-1)
            host = ttnn.from_torch(values, device=None, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                                   mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3))
            ttnn.copy_host_to_device_tensor(host, tensor)
            ttnn.synchronize_device(mesh)
            eager = sampler.sample(tensor, enable_trace=False)
            verify(eager, expected, kind, "eager")
            ttnn.deallocate(eager[0] if isinstance(eager, tuple) else eager)
            for repeat in range(2):
                verify(sampler.sample(tensor, enable_trace=True), expected, kind, f"trace-{repeat}")
        timings = []
        for repeat in range(20):
            started = time.perf_counter()
            sampler.sample(tensor, enable_trace=True)
            ttnn.synchronize_device(mesh)
            timings.append((time.perf_counter() - started) * 1000)
        report.update(passed=True, force_argmax=True, trace_slots=len(sampler._trace_states),
                      synchronized_sampler_ms=timings, median_sampler_ms=statistics.median(timings))
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (root / "sampling-kernel.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        if sampler is not None:
            sampler.reset_trace()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
