"""Bounded in-situ stage attribution with exact serial/state/KV gates."""

import statistics
import time

from model_batch import ModelBatch
from stage_profile import StageProfile


def aggregate(records):
    totals = {}
    for record in records:
        total = totals.setdefault(record["category"], dict(category=record["category"], calls=0,
                                                           inclusive_ms=0.0, exclusive_ms=0.0))
        for name in ("calls", "inclusive_ms", "exclusive_ms"):
            total[name] += record[name]
    roots = [value for value in totals.values() if value["category"] == "model.block"]
    if len(roots) != 1 or roots[0]["calls"] != 1:
        raise ValueError("Expected one full-model root per profiling pass")
    duration = roots[0]["inclusive_ms"]
    if abs(sum(value["exclusive_ms"] for value in totals.values()) - duration) > 0.01:
        raise ValueError("Exclusive category times do not reconcile with the root")
    return sorted(totals.values(), key=lambda value: value["exclusive_ms"], reverse=True)


def measure(model, tokens, length, pages, helpers, checkpoints, *, prefill, save_initial,
            restore_initial, state_digest, kv_digest, local_host):
    import torch
    import ttnn

    mesh = model.mesh_device
    rows = len(tokens)
    synchronize = lambda: ttnn.synchronize_device(mesh)
    profiler = StageProfile(synchronize)
    prefill()
    save_initial()
    fixture = ModelBatch(model, tokens, length, pages, helpers, checkpoints, rows,
                         serial_sdpa=True, profiler=profiler)
    singleton = [ModelBatch(model, [token], length + index, pages, helpers, checkpoints, 1)
                 for index, token in enumerate(tokens)]
    report = dict(length=length, rows=rows, exact=False, passes=[], trace_ms=[], eager_ms=[],
                  scope="Fenced eager host+device stage intervals; not a traced device critical-path decomposition",
                  instrumentation_version=2,
                  decoder_types=[dict(index=index, full_attention=layer.is_full_attention) for index, layer in enumerate(model.layers)])
    trace = None
    trace_output = None

    def read(values):
        shards = [local_host(value) for value in values]
        return [torch.cat([value[chip].reshape(-1, model.args.vocab_size) for value in shards], dim=0)
                for chip in range(2)]

    def validate(values, expected, expected_state, expected_kv):
        if any(not torch.equal(actual, reference) for actual, reference in zip(read(values), expected, strict=True)):
            raise AssertionError("Profiled or traced logits differ from native serial B1")
        if state_digest() != expected_state or kv_digest(length + rows) != expected_kv:
            raise AssertionError("Profiled or traced state/KV differs from native serial B1")

    try:
        expected_outputs = [model._forward_decode(value.tokens, value.cos, value.sin, value.positions, value.pages)
                            for value in singleton]
        expected = read(expected_outputs)
        expected_state, expected_kv = state_digest(), kv_digest(length + rows)
        for output in expected_outputs:
            ttnn.deallocate(output)
        restore_initial()
        output = fixture.run()
        validate([output], expected, expected_state, expected_kv)
        ttnn.deallocate(output)
        restore_initial()
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        trace_output = fixture.run()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        for _ in range(3):
            restore_initial()
            synchronize()
            started = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            report["trace_ms"].append((time.perf_counter() - started) * 1000)
            validate([trace_output], expected, expected_state, expected_kv)

        for iteration in range(3):
            restore_initial()
            synchronize()
            started = time.perf_counter()
            output = fixture.run()
            synchronize()
            report["eager_ms"].append((time.perf_counter() - started) * 1000)
            validate([output], expected, expected_state, expected_kv)
            ttnn.deallocate(output)
            restore_initial()
            profiler.begin()
            output = profiler.wrap("model.block", fixture.run)()
            records = profiler.finish()
            validate([output], expected, expected_state, expected_kv)
            ttnn.deallocate(output)
            totals = aggregate(records)
            counts = {value["category"]: value["calls"] for value in totals}
            required = {"gdn.native_row": 48 * rows, "gdn.input_projection": 48,
                        "gdn.projected_row_copy": 48 * rows, "gdn.conv_gates": 48 * rows,
                        "gdn.recurrence_norm_gate": 48 * rows, "gdn.active_state_slice": 48 * rows,
                        "gdn.active_state_write": 48 * rows,
                        "gdn.output_projection": 48, "gdn.checkpoint": 48,
                        "attention.input_projection": 16, "attention.output_projection": 16,
                        "attention.kv_write": 32, "attention.sdpa_and_row_packing": 16,
                        "decoder.block": 64, "lm_head": 1}
            if any(counts.get(name) != count for name, count in required.items()):
                raise AssertionError(f"Missing attribution coverage: {counts}")
            report["passes"].append(dict(iteration=iteration, totals=totals, layers=records, exact=True))
        idle = []
        for _ in range(30):
            started = time.perf_counter()
            synchronize()
            synchronize()
            idle.append((time.perf_counter() - started) * 1000)
        report["idle_fence_pair_median_ms"] = statistics.median(idle)
        report["trace_median_ms"] = statistics.median(report["trace_ms"])
        report["eager_median_ms"] = statistics.median(report["eager_ms"])
        report["exact"] = True
        return report
    finally:
        profiler.enabled = False
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        if trace_output is not None:
            ttnn.deallocate(trace_output)
        fixture.close()
        for value in singleton:
            value.close()
