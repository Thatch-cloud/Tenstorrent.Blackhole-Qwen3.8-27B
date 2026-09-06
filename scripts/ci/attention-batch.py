"""Real-weight attention batching gate; static fixture positions, not a serving verifier."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from attention_batch import OrderedCacheWriter, SerialAttentionReader, SerialCacheWriter, serial_tail


def main():
    if os.environ.get("QWEN_HARDWARE_TESTS") != "1" or os.environ.get("QWEN_CARDS_ALLOCATED") != "1":
        raise RuntimeError("Explicit hardware allocation required")
    if os.environ.get("TT_METAL_SIMULATOR") or os.environ.get("TT_METAL_SLOW_DISPATCH_MODE"):
        raise RuntimeError("Fast-dispatch hardware required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--ordered-cache", action="store_true")
    options = parser.parse_args()
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tests.test_factory import load_attn_layer
    from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp
    from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
    from models.tt_transformers.tt.ccl import TT_CCL

    report = dict(passed=False, checks=[], negative_controls=[], timings=[], eligible_for_serving_gate=False,
                  scope="One real-weight attention layer, TP2, static positions, serialized shared-page writes; no full-model speed claim")
    report['ordered_cache'] = options.ordered_cache
    report['timing_control'] = 'Batched attention with serial cache writes and B1 SDPA' if options.ordered_cache else 'Native serial B1 attention'
    output_path = Path("/experiment/results/attention-timing.json" if options.timing else "/experiment/results/attention-batch.json")
    mesh = None
    traces = []
    try:
        source = Path("/opt/tt-metal/models/demos/blackhole/qwen36/tt/attention/tp.py")
        report["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        if report["source_sha256"] != "e0c685a43796f6f8a0ba42fd70a9533b502461b50fdda15e51c8753340f3dc3a":
            raise ValueError("Attention source differs from audited run 34000023393")
        report["flags"] = {name: os.environ.get(name) for name in ("QWEN_ATTN_PREP", "QWEN_SDPA_BF8")}
        if any(value != "1" for value in report["flags"].values()):
            raise ValueError("Pinned fused attention and BF8 cache flags required")
        kernels = None
        if options.ordered_cache:
            from ordered_cache import HASHES, load_kernels
            kernels = load_kernels('/opt/tt-metal')
            report['cache_native_hashes'] = HASHES
            report['cache_generated_hashes'] = {role: hashlib.sha256(value.encode()).hexdigest() for role, value in kernels.items()}
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        args = Qwen36ModelArgs(mesh, max_batch_size=8, max_seq_len=256)
        layer = next(index for index, kind in enumerate(args.attention_type_list) if kind == "full_attention")
        attention = TPAttention(mesh, args, load_attention_weights_tp(mesh, load_attn_layer(args.CKPT_DIR, layer), args), TT_CCL(mesh))
        if not attention._fused_qkv or (attention.NKV, attention.HD) != (2, 256):
            raise ValueError("Expected native fused TP2 attention geometry")
        report["layer"] = layer

        def upload(value, dtype=None):
            dtype = ttnn.bfloat16 if dtype is None else dtype
            return ttnn.from_torch(value, device=mesh, dtype=dtype,
                                  layout=ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def host(value):
            shards = ttnn.get_device_tensors(value)
            if len(shards) != 2:
                raise AssertionError("Both chips required")
            return [ttnn.to_torch(shard).clone() for shard in shards]

        def release(values):
            for value in values:
                ttnn.deallocate(value)

        def difference(actual, expected):
            if len(actual) != len(expected):
                raise AssertionError("Shard count mismatch")
            return [dict(chip=chip, unequal=int((value != reference).sum()),
                         max_abs=float((value.float() - reference.float()).abs().max()))
                    for chip, (value, reference) in enumerate(zip(actual, expected, strict=True))]

        def equal(diagnostics):
            return all(item["unequal"] == 0 for item in diagnostics)

        for seed in (17, 41):
            torch.manual_seed(seed)
            initial = [upload(torch.randn(8, 2, 64, 256, dtype=torch.bfloat16) * 0.1, ttnn.bfloat8_b) for _ in range(2)]
            caches = [ttnn.clone(value, memory_config=ttnn.DRAM_MEMORY_CONFIG) for value in initial]
            attention.set_paged_kv_cache(*caches)
            addresses = [value.buffer_address() for value in caches]
            pages_host = torch.tensor([[3, 1, 6, 2]], dtype=torch.int32)
            page_single = upload(pages_host, ttnn.int32)

            def reset():
                for value, cache in zip(initial, caches, strict=True):
                    ttnn.copy(value, cache)

            for start in (31, 63, 65):
                for rows in (1, 2, 4, 8, 16):
                    values = torch.randn(1, 1, rows, 5120, dtype=torch.bfloat16)
                    packed = upload(values)
                    tokens = [upload(values[:, :, index:index + 1].contiguous()) for index in range(rows)]
                    positions = torch.arange(start, start + rows, dtype=torch.int32)
                    packed_position = upload(positions, ttnn.int32)
                    singleton_positions = [upload(position.reshape(1), ttnn.int32) for position in positions]
                    packed_pages = upload(pages_host.repeat(rows, 1), ttnn.int32)
                    cos, sin = rot_mats_decode(mesh, args.rope_head_dim, args.max_seq_len, args.rope_theta, positions)
                    singleton_rope = [rot_mats_decode(mesh, args.rope_head_dim, args.max_seq_len, args.rope_theta,
                                                     position.reshape(1)) for position in positions]
                    serial_writer = SerialCacheWriter(ttnn, singleton_positions, [page_single] * rows, attention._kv_shard_cfg(1))
                    writer = OrderedCacheWriter(mesh, ttnn, kernels) if options.ordered_cache else serial_writer
                    reader = SerialAttentionReader(ttnn, singleton_positions, [page_single] * rows) if options.ordered_cache else None
                    tail = serial_tail(attention, writer, ttnn, reader)

                    def reference():
                        return [attention.forward_decode(token, position, *rope, page_table=page_single)
                                for token, position, rope in zip(tokens, singleton_positions, singleton_rope, strict=True)]

                    def candidate(control=False):
                        selected_writer = serial_writer if control else writer
                        before = selected_writer.calls
                        attention._decode_from_prep = serial_tail(attention, serial_writer, ttnn, reader) if control else tail
                        try:
                            result = attention.forward_decode(packed, packed_position, cos, sin, page_table=packed_pages)
                            if selected_writer.calls - before != 2:
                                raise AssertionError("Both native K/V writes must use the selected adapter")
                            return [result]
                        finally:
                            del attention._decode_from_prep

                    reset()
                    expected_device = reference()
                    expected_outputs = [torch.cat([host(value)[chip] for value in expected_device], dim=-2) for chip in range(2)]
                    expected_caches = [host(cache) for cache in caches]
                    release(expected_device)
                    reset()
                    warm = candidate()
                    release(warm)
                    for mode in ("eager", "trace"):
                        reset()
                        trace = None
                        if mode == "trace":
                            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                        outputs = candidate()
                        if trace is not None:
                            ttnn.end_trace_capture(mesh, trace, cq_id=0)
                            traces.append(trace)
                            reset()
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                        output_difference = difference(host(outputs[0]), expected_outputs)
                        cache_differences = [difference(host(cache), expected) for cache, expected in zip(caches, expected_caches, strict=True)]
                        check = dict(seed=seed, start=start, rows=rows, mode=mode,
                                     output=output_difference, caches=cache_differences,
                                     exact=equal(output_difference) and all(equal(value) for value in cache_differences))
                        report["checks"].append(check)
                        print(json.dumps(check), flush=True)
                        if trace is not None:
                            ttnn.release_trace(mesh, trace)
                            traces.remove(trace)
                        release(outputs)
                        output_path.write_text(json.dumps(report, indent=2))
                    if options.timing and all(check["exact"] for check in report["checks"][-2:]):
                        timed_traces = {}
                        timed_outputs = {}
                        for arm, operation in (("serial", (lambda: candidate(control=True)) if options.ordered_cache else reference), ("batch", candidate)):
                            reset()
                            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                            outputs = operation()
                            ttnn.end_trace_capture(mesh, trace, cq_id=0)
                            traces.append(trace)
                            timed_traces[arm] = trace
                            timed_outputs[arm] = outputs
                            reset()
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                            traced_outputs = [torch.cat([host(value)[chip] for value in outputs], dim=-2) for chip in range(2)]
                            if not equal(difference(traced_outputs, expected_outputs)) or any(
                                not equal(difference(host(cache), expected))
                                for cache, expected in zip(caches, expected_caches, strict=True)
                            ):
                                raise AssertionError(f"Timing trace {arm} differs from native B1 oracle")
                        for block in range(3):
                            samples = []
                            for arm in ("serial", "batch", "batch", "serial"):
                                reset()
                                ttnn.execute_trace(mesh, timed_traces[arm], cq_id=0, blocking=True)
                                ttnn.synchronize_device(mesh)
                                started = time.perf_counter()
                                for _ in range(30):
                                    ttnn.execute_trace(mesh, timed_traces[arm], cq_id=0, blocking=False)
                                ttnn.synchronize_device(mesh)
                                elapsed_ms = (time.perf_counter() - started) * 1000 / 30
                                samples.append(dict(arm=arm, milliseconds_per_block=elapsed_ms))
                                traced_outputs = [torch.cat([host(value)[chip] for value in timed_outputs[arm]], dim=-2)
                                                  for chip in range(2)]
                                if not equal(difference(traced_outputs, expected_outputs)) or any(
                                    not equal(difference(host(cache), expected))
                                    for cache, expected in zip(caches, expected_caches, strict=True)
                                ):
                                    raise AssertionError("Repeated timing replay changed outputs or KV")
                            serial_ms = sum(sample["milliseconds_per_block"] for sample in samples if sample["arm"] == "serial") / 2
                            batch_ms = sum(sample["milliseconds_per_block"] for sample in samples if sample["arm"] == "batch") / 2
                            report["timings"].append(dict(seed=seed, start=start, rows=rows, block=block, repeats=30,
                                                         samples=samples, serial_ms=serial_ms, batch_ms=batch_ms,
                                                         speedup=serial_ms / batch_ms))
                        for arm, trace in timed_traces.items():
                            ttnn.release_trace(mesh, trace)
                            traces.remove(trace)
                            release(timed_outputs[arm])
                    reset()
                    missing_write_detected = any(not equal(difference(host(cache), expected))
                                                 for cache, expected in zip(caches, expected_caches, strict=True))
                    wrong_pages = upload(torch.tensor([[7, 5, 4, 0]], dtype=torch.int32), ttnn.int32)
                    wrong_outputs = [attention.forward_decode(token, position, *rope, page_table=wrong_pages)
                                     for token, position, rope in zip(tokens, singleton_positions, singleton_rope, strict=True)]
                    wrong_page_detected = any(not equal(difference(host(cache), expected))
                                              for cache, expected in zip(caches, expected_caches, strict=True))
                    report["negative_controls"].append(dict(seed=seed, start=start, rows=rows,
                                                             missing_write_detected=missing_write_detected,
                                                             wrong_page_detected=wrong_page_detected))
                    if not missing_write_detected or not wrong_page_detected:
                        raise AssertionError("KV negative control was not detected")
                    if addresses != [value.buffer_address() for value in caches]:
                        raise AssertionError("Persistent cache addresses changed")
                    release(wrong_outputs + [wrong_pages, packed, packed_position, packed_pages, cos, sin] + tokens + singleton_positions)
                    release([value for pair in singleton_rope for value in pair])
            release(initial + caches + [page_single])
        if len(report["checks"]) != 60 or not all(check["exact"] for check in report["checks"]):
            raise AssertionError("Batched attention does not satisfy the exact native B1 gate")
        if options.timing and len(report["timings"]) != 90:
            raise AssertionError("Missing paired timing blocks")
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
