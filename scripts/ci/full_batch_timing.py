"""Paired static full-logit block timings, not end-to-end speculative throughput."""

import time

from model_batch import ModelBatch


def summarize(samples):
    if [sample["arm"] for sample in samples] != ["serial", "batch", "batch", "serial"]:
        raise ValueError("Expected one complete ABBA block")
    if any(sample["milliseconds"] <= 0 for sample in samples):
        raise ValueError("Expected positive block latency")
    serial = (samples[0]["milliseconds"] + samples[3]["milliseconds"]) / 2
    batch = (samples[1]["milliseconds"] + samples[2]["milliseconds"]) / 2
    return dict(serial_ms=serial, batch_ms=batch, speedup=serial / batch)


def measure(model, tokens, length, pages, helpers, checkpoints, *, prefill, save_initial,
            restore_initial, state_digest, kv_digest, local_host, serial_sdpa=False):
    import torch
    import ttnn

    rows = len(tokens)
    mesh = model.mesh_device
    candidate = ModelBatch(model, tokens, length, pages, helpers, checkpoints, rows, serial_sdpa=serial_sdpa)
    singleton = [ModelBatch(model, [token], length + index, pages, helpers, checkpoints, 1)
                 for index, token in enumerate(tokens)]
    traces = {}
    outputs = {}
    report = dict(length=length, rows=rows, blocks=[], restore_samples_ms=[], exact=False, serial_sdpa=serial_sdpa,
                  checkpoint_policy="One preselected end-prefix snapshot set, not all-prefix staging")

    def serial():
        return [model._forward_decode(fixture.tokens, fixture.cos, fixture.sin, fixture.positions, fixture.pages)
                for fixture in singleton]

    def batch():
        return [candidate.run()]

    def read(values):
        shards = [local_host(value) for value in values]
        return [torch.cat([value[chip].reshape(-1, model.args.vocab_size) for value in shards], dim=0)
                for chip in range(2)]

    def exact_logits(actual, expected):
        return len(actual) == len(expected) == 2 and all(torch.equal(value, reference)
            for value, reference in zip(actual, expected, strict=True))

    try:
        prefill()
        save_initial()
        initial_state = state_digest()
        expected_outputs = serial()
        expected = read(expected_outputs)
        expected_state = state_digest()
        expected_kv = kv_digest(length + rows)
        for value in expected_outputs:
            ttnn.deallocate(value)

        for arm, operation in (("serial", serial), ("batch", batch)):
            restore_initial()
            warm = operation()
            for value in warm:
                ttnn.deallocate(value)
            restore_initial()
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            outputs[arm] = operation()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            traces[arm] = trace
            restore_initial()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            if not exact_logits(read(outputs[arm]), expected) or state_digest() != expected_state or kv_digest(length + rows) != expected_kv:
                raise AssertionError(f"{arm} timing trace differs from native serial logits/state/KV")

        for block in range(3):
            samples = []
            for arm in ("serial", "batch", "batch", "serial"):
                elapsed = []
                for _ in range(10):
                    restore_initial()
                    ttnn.synchronize_device(mesh)
                    started = time.perf_counter()
                    ttnn.execute_trace(mesh, traces[arm], cq_id=0, blocking=True)
                    elapsed.append((time.perf_counter() - started) * 1000)
                if not exact_logits(read(outputs[arm]), expected) or state_digest() != expected_state or kv_digest(length + rows) != expected_kv:
                    raise AssertionError("Timed replay changed full logits/state/KV")
                samples.append(dict(arm=arm, milliseconds=sum(elapsed) / len(elapsed), repeats=10, replay_ms=elapsed))
            report["blocks"].append(dict(block=block, samples=samples, **summarize(samples)))

        restore_initial()
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        restore_initial()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        traces["restore"] = trace
        for _ in range(3):
            ttnn.execute_trace(mesh, traces["batch"], cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            started = time.perf_counter()
            for _ in range(30):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            report["restore_samples_ms"].append((time.perf_counter() - started) * 1000 / 30)
            if state_digest() != initial_state:
                raise AssertionError("Measured restore did not recover the initial GDN state")
        report["exact"] = True
        return report
    finally:
        for trace in traces.values():
            ttnn.release_trace(mesh, trace)
        for values in outputs.values():
            for value in values:
                ttnn.deallocate(value)
        candidate.close()
        for fixture in singleton:
            fixture.close()
