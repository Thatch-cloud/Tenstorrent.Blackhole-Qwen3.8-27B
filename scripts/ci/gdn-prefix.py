"""E3a: real-weight GDN prefix/rollback gate; not a full-model verifier."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from gdn_prefix import decode_projected, gated_decode


def main():
    if os.environ.get("QWEN_HARDWARE_TESTS") != "1" or os.environ.get("QWEN_CARDS_ALLOCATED") != "1":
        raise RuntimeError("Explicit hardware allocation required")
    if os.environ.get("TT_METAL_SIMULATOR") or os.environ.get("TT_METAL_SLOW_DISPATCH_MODE"):
        raise RuntimeError("Fast-dispatch hardware required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-output", action="store_true")
    options = parser.parse_args()
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tests.test_factory import load_gdn_layer
    from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
    from models.tt_transformers.tt.ccl import TT_CCL, tt_all_reduce

    report = dict(passed=False, checks=[], timings=[], negative_controls=[],
                  scope="One real-weight GDN layer, B1 in Bmax8, TP2; no attention KV, logits, drafter or serving throughput")
    root = Path("/experiment/results")
    report["batched_output_projection"] = options.batch_output
    output_path = root / ("gdn-block.json" if options.batch_output else "gdn-prefix.json")
    mesh = None
    traces = []
    try:
        source = Path("/opt/tt-metal/models/demos/blackhole/qwen36/tt/gdn/tp.py")
        report["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        if report["source_sha256"] != "f767d0648ae01b0b1c0bb7bf601f5490661707b845c11ce6dbebb86ba0f84dc9":
            raise ValueError("Native GDN differs from audited run 33960203974")
        flags = ("QWEN_GDN_CONV_GATES", "QWEN_GDN_PROJ_DIRECT", "QWEN_GDN_PACKED_QKV", "QWEN_GDN_NORM_GATE",
                 "QWEN35_GDN_STATE_BF16", "QWEN35_GDN_DECODE_BF16", "QWEN_GDN_FUSED_DECODE", "QWEN_GDN_FUSED_INPLACE")
        report["flags"] = {name: os.environ.get(name) for name in flags}
        if any(value != "1" for value in report["flags"].values()):
            raise ValueError("Pinned fused GDN flags required")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        args = Qwen36ModelArgs(mesh, max_batch_size=8, max_seq_len=256)
        layer = next(index for index, kind in enumerate(args.attention_type_list) if kind == "linear_attention")
        gdn = TPGatedDeltaNet(mesh, args, load_gdn_weights_tp(mesh, load_gdn_layer(args.CKPT_DIR, layer), args), TT_CCL(mesh))
        gdn.reset_state()
        gdn._stable_state = True
        native_gated = gated_decode(gdn) if options.batch_output else None
        if not gdn._fuse_ab or not gdn._conv_gates_enabled() or gdn.rec_state.dtype != ttnn.bfloat16:
            raise ValueError("Expected fused projection and unchanged K-image BF16 persistent recurrence")
        live = [gdn.rec_state, *gdn.conv_states]
        addresses = [tensor.buffer_address() for tensor in live]
        report.update(layer=layer, state_shapes=[list(tensor.shape) for tensor in live])

        def upload(host):
            return ttnn.from_torch(host, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def host(tensor):
            shards = ttnn.get_device_tensors(tensor)
            if len(shards) != 2:
                raise AssertionError("Both chips required")
            return [ttnn.to_torch(shard).clone() for shard in shards]

        def snapshot():
            return [ttnn.clone(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG) for tensor in live]

        def copy(source_state, destination_state):
            for source_tensor, destination in zip(source_state, destination_state, strict=True):
                ttnn.copy(source_tensor, destination)

        def state_host(state):
            return [host(tensor) for tensor in state]

        def exact(left, right):
            return len(left) == len(right) and all(torch.equal(first, second) for first, second in zip(left, right))

        def state_exact(left, right):
            return len(left) == len(right) and all(exact(first, second) for first, second in zip(left, right))

        def release(tensors):
            for tensor in tensors:
                ttnn.deallocate(tensor)

        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            gdn.reset_state_inplace()
            prime = upload(torch.randn(1, 8, args.dim).bfloat16())
            release([gdn.forward_decode(prime), prime])
            for _ in range(4):
                prime = upload(torch.randn(1, 1, args.dim).bfloat16())
                release([gdn.forward_decode(prime), prime])
            initial = snapshot()
            for rows in (1, 2, 4, 8, 16):
                values = torch.randn(1, rows, args.dim).bfloat16()
                packed = upload(values)
                tokens = [upload(values[:, index:index + 1]) for index in range(rows)]
                correction = [upload(torch.randn(1, 1, args.dim).bfloat16()) for _ in range(2)]
                reference = [snapshot() for _ in range(rows + 1)]
                staging = [snapshot() for _ in range(rows + 1)]
                copy(initial, live)
                copy(live, reference[0])
                expected_outputs = []
                for index, token in enumerate(tokens):
                    output = gdn.forward_decode(token)
                    expected_outputs.append(host(output))
                    release([output])
                    copy(live, reference[index + 1])
                expected_states = [state_host(state) for state in reference]
                continuations = []
                for prefix in range(rows + 1):
                    copy(reference[prefix], live)
                    continuation = []
                    for token in correction:
                        output = gdn.forward_decode(token)
                        continuation.append(host(output))
                        release([output])
                    continuations.append((continuation, state_host(live)))

                def operation():
                    copy(live, staging[0])
                    outputs = decode_projected(gdn, packed, tokens, lambda prefix: copy(live, staging[prefix]),
                                               ttnn, forward=native_gated)
                    if not options.batch_output:
                        return outputs
                    gated = outputs[0] if rows == 1 else ttnn.concat(outputs, dim=1)
                    partial = gdn._row_proj(gated, gdn.tw["out"])
                    if rows != 1:
                        release([gated])
                    release(outputs)
                    partial = ttnn.reshape(partial, (1, 1, rows, partial.shape[-1]))
                    output = tt_all_reduce(partial, mesh, gdn.tt_ccl, cluster_axis=0, dim=3,
                                           topology=args.ccl_topology(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    return [output]

                def read_outputs(outputs):
                    if not options.batch_output:
                        return [host(output) for output in outputs]
                    shards = host(outputs[0])
                    return [[shard[:, :, index:index + 1, :] for shard in shards] for index in range(rows)]

                copy(initial, live)
                release(operation())
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                captured_outputs = operation()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                traces.append(trace)
                for mode in ("eager", "trace"):
                    copy(initial, live)
                    ttnn.synchronize_device(mesh)
                    started = time.perf_counter()
                    if mode == "trace":
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                        outputs = captured_outputs
                    else:
                        outputs = operation()
                        ttnn.synchronize_device(mesh)
                    report["timings"].append(dict(seed=seed, rows=rows, mode=mode,
                                                   layer_with_snapshots_ms=1000 * (time.perf_counter() - started)))
                    if not all(exact(output, expected) for output, expected in zip(read_outputs(outputs), expected_outputs, strict=True)):
                        raise AssertionError(f"GDN output divergence: {seed=} {rows=} {mode=}")
                    for prefix in range(rows + 1):
                        if not state_exact(state_host(staging[prefix]), expected_states[prefix]):
                            raise AssertionError(f"Prefix state divergence: {seed=} {rows=} {mode=} {prefix=}")
                        copy(staging[prefix], live)
                        if not state_exact(state_host(live), expected_states[prefix]):
                            raise AssertionError("Restore did not reproduce every state tensor on both chips")
                        expected_continuation, expected_final = continuations[prefix]
                        for token, expected in zip(correction, expected_continuation, strict=True):
                            output = gdn.forward_decode(token)
                            matched = exact(host(output), expected)
                            release([output])
                            if not matched:
                                raise AssertionError(f"Correction continuation divergence: {prefix=}")
                        if not state_exact(state_host(live), expected_final):
                            raise AssertionError("Post-correction recurrent/conv state divergence")
                        report["checks"].append(dict(seed=seed, rows=rows, mode=mode, prefix=prefix,
                                                     both_chips_exact=True, correction_steps=2))
                    if mode == "eager":
                        release(outputs)
                for family, indices in (("recurrence", [0]), ("convolution", list(range(1, len(live))))):
                    copy(reference[rows], live)
                    for index in indices:
                        ttnn.copy(reference[0][index], live[index])
                    detected = not state_exact(state_host(live), expected_states[rows])
                    output = gdn.forward_decode(correction[0])
                    continuation_detected = not exact(host(output), continuations[rows][0][0])
                    release([output])
                    report["negative_controls"].append(dict(seed=seed, rows=rows, family=family,
                                                             state_detected=detected, continuation_detected=continuation_detected))
                    if not detected or not continuation_detected:
                        raise AssertionError("Stale-state negative control was not detected")
                if addresses != [tensor.buffer_address() for tensor in [gdn.rec_state, *gdn.conv_states]]:
                    raise AssertionError("Persistent state addresses changed")
                ttnn.release_trace(mesh, trace)
                traces.remove(trace)
                release(captured_outputs + tokens + correction + [packed])
                for state in reference + staging:
                    release(state)
                output_path.write_text(json.dumps(report, indent=2))
                print(json.dumps(dict(seed=seed, rows=rows, exact=True)), flush=True)
            release(initial)
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        output_path.write_text(json.dumps(report, indent=2))
        if mesh is not None:
            for trace in traces:
                ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
