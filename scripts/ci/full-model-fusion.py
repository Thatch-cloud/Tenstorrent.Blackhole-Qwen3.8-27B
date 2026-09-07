"""Full-model B1 fusion gate through the real Generator, not HTTP serving throughput."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

from full_model_fusion import FusionArm, paired_block


def main():
    if os.environ.get("QWEN_HARDWARE_TESTS") != "1" or os.environ.get("QWEN_CARDS_ALLOCATED") != "1":
        raise RuntimeError("Explicit hardware allocation required")
    import torch
    import ttnn
    from transformers import AutoConfig, AutoTokenizer
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

    root = Path("/experiment/results")
    report = dict(passed=False, checks=[], blocks=[], memory=[], weight_checks=[],
        scope="Full 64-layer B1 TP2 Generator, host argmax/readback included; no HTTP scheduler or engine-commit timing",
        max_batch=8, max_context=65536, generated_tokens=128, prerequisite_run=33964993645,
        weight_policy="Reuse existing packed prefill tensors; native gate/up remain for fallback; no additional weights")
    mesh = None
    arm = None
    stores = {"control": {}, "fused": {}}
    try:
        revision = subprocess.check_output(["git", "-C", "/opt/tt-metal", "rev-parse", "HEAD"], text=True).strip()
        if revision != "9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9":
            raise ValueError("Unreviewed runtime revision")
        report["source"] = {name: hashlib.sha256((Path("/opt/tt-metal") / name).read_bytes()).hexdigest()
            for name in ("models/demos/blackhole/qwen36/tt/mlp.py", "models/demos/blackhole/qwen36/tt/tp_common.py",
                         "models/demos/blackhole/qwen36/tt/qwen36_vllm.py", "models/tt_transformers/tt/generator.py")}
        spec = importlib.util.spec_from_file_location("baseline", Path(__file__).with_name("baseline-client.py"))
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=1073741824)
        mesh.enable_program_cache()
        weights = os.environ["MODEL_WEIGHTS_DIR"]
        config = AutoConfig.from_pretrained(weights, local_files_only=True, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True, trust_remote_code=False)
        generator = Qwen36ForCausalLM.initialize_vllm_model(config, mesh, max_batch_size=8, max_seq_len=65536)
        model = generator.model[0]
        if model.args.vocab_size != 248320 or len(model.layers) != 64:
            raise ValueError("Unexpected full model")
        devices = [tensor.device() for tensor in ttnn.get_device_tensors(model.layers[0].feed_forward.weights.w1)]

        def memory(label):
            ttnn.synchronize_device(mesh)
            for chip, device in enumerate(devices):
                for kind in ("DRAM", "L1", "TRACE"):
                    view = ttnn.get_memory_view(device, getattr(ttnn.BufferType, kind))
                    fields = ("num_banks", "total_bytes_per_bank", "total_bytes_allocated_per_bank",
                              "total_bytes_free_per_bank", "largest_contiguous_bytes_free_per_bank")
                    report["memory"].append(dict(label=label, chip=chip, kind=kind,
                                                  **{field: getattr(view, field) for field in fields}))

        def addresses():
            return [[tensor.buffer_address() for tensor in (layer.feed_forward.weights.w1,
                layer.feed_forward.weights.w2, layer.feed_forward.weights.w3, layer.feed_forward.weights.w_gate_up)]
                for layer in model.layers]

        memory("loaded")
        arm = FusionArm(model)
        original_addresses = addresses()
        report["kernel"] = arm.kernels[0]
        for index, layer in enumerate(model.layers):
            native = layer.feed_forward.weights
            packed_shards = ttnn.get_device_tensors(native.w_gate_up)
            gate_shards, up_shards = ttnn.get_device_tensors(native.w1), ttnn.get_device_tensors(native.w3)
            if not all(len(shards) == 2 for shards in (packed_shards, gate_shards, up_shards)):
                raise ValueError("Missing weight shard")
            for chip, (packed, gate, up) in enumerate(zip(packed_shards, gate_shards, up_shards)):
                decoded = ttnn.to_torch(packed).reshape(5120, 272, 2, 32)
                for position, reference in enumerate((gate, up)):
                    if not torch.equal(decoded[:, :, position, :].reshape(5120, 8704),
                                       ttnn.to_torch(reference).reshape(5120, 8704)):
                        raise ValueError(f"Packed weight mismatch: layer={index}, chip={chip}, projection={position}")
                del decoded
                report["weight_checks"].append(dict(layer=index, chip=chip, exact=True))
            print(json.dumps(dict(weight_layer_verified=index)), flush=True)
        memory("verified_existing_weights")
        kv_cache = generator.allocate_kv_cache((8200, model.args.n_local_kv_heads, 64, model.args.head_dim),
                                               ttnn.bfloat16, len(model.layers))
        page_table = torch.arange(1024, dtype=torch.int32).reshape(1, 1024)
        prompts = {length: baseline.make_prompt(tokenizer, length, 0) for length in (128, 64512)}

        def select(name):
            arm.enabled = name == "fused"
            generator._bucket_trace_store = stores[name]
            generator.prev_page_table = None

        def warm(name, trace):
            select(name)
            before = list(arm.hits)
            generator.warmup_model_decode(kv_cache=kv_cache, enable_trace=trace, max_batch_size=1,
                num_blocks=1024, can_sample_on_device=False, skip_trace_precompile=trace)
            stores[name] = generator._bucket_trace_store
            if name == "fused" and any(after <= prior for after, prior in zip(arm.hits, before)):
                raise AssertionError("Fused warmup did not engage all 64 MLP layers")
            if name == "control" and before != arm.hits:
                raise AssertionError("Control warmup engaged fused code")

        for name in stores:
            warm(name, False)
        arm.enabled = False
        generator.warmup_model_prefill(kv_cache=kv_cache, enable_trace=True)
        memory("compiled")

        def generate(name, prompt, count, trace, label, capture_logits=False):
            arm.enabled = False
            started = time.perf_counter()
            logits, _ = generator.prefill_forward(torch.tensor([prompt], dtype=torch.int32), page_table,
                kv_cache, [len(prompt)], empty_slots=[0], enable_trace=True)
            generated = [int(logits.reshape(-1, model.args.vocab_size)[0].float().argmax())]
            reference_logits = [logits.clone()] if capture_logits else []
            prefill_seconds = time.perf_counter() - started
            select(name)
            started = time.perf_counter()
            for step in range(count - 1):
                result = generator.decode_forward(tokens=torch.tensor([[generated[-1]]], dtype=torch.int32),
                    start_pos=torch.tensor([len(prompt) + step], dtype=torch.int32), page_table=page_table,
                    kv_cache=kv_cache, enable_trace=trace, read_from_device=True)
                logits = result[0] if isinstance(result, tuple) else result
                generated.append(int(logits.reshape(-1, model.args.vocab_size)[0].float().argmax()))
                if capture_logits:
                    reference_logits.append(logits.clone())
            elapsed = time.perf_counter() - started
            result = dict(label=label, arm=name, trace=trace, prompt_tokens=len(prompt), token_ids=generated,
                          prefill_seconds=prefill_seconds, decode_seconds=elapsed,
                          host_decode_tok_s=(count - 1) / elapsed)
            (root / f"{label}.json").write_text(json.dumps(result, indent=2))
            print(json.dumps({key: value for key, value in result.items() if key != "token_ids"}), flush=True)
            return result, reference_logits

        eager = {}
        for length, prompt in prompts.items():
            for name in stores:
                record, logits = generate(name, prompt, 16, False, f"fusion-eager-{name}-{length}", True)
                if name == "control":
                    eager[length] = record, logits
                else:
                    control, reference = eager[length]
                    if record["token_ids"] != control["token_ids"] or not all(torch.equal(left, right) for left, right in zip(logits, reference)):
                        raise AssertionError(f"Eager full-model divergence at {length}")
                    report["checks"].append(dict(mode="eager", length=length, tokens_exact=True, logits_exact=True))
        for name in stores:
            warm(name, True)
        report["trace_ids"] = {name: {str(chip): int(trace) for chip, trace in store[1][0][False].items()}
                               for name, store in stores.items()}
        if report["trace_ids"]["control"] == report["trace_ids"]["fused"]:
            raise AssertionError("Arms share a decode trace")
        memory("both_traces")
        for length, prompt in prompts.items():
            for name in stores:
                record, logits = generate(name, prompt, 16, True, f"fusion-trace-check-{name}-{length}", True)
                control, reference = eager[length]
                if record["token_ids"] != control["token_ids"] or not all(torch.equal(left, right) for left, right in zip(logits, reference)):
                    raise AssertionError(f"Traced full-model divergence: {name}, {length}")
                report["checks"].append(dict(mode="trace", arm=name, length=length, tokens_exact=True, logits_exact=True))
            token_reference = None
            for block in range(3):
                records = []
                for order, name in enumerate(("control", "fused", "fused", "control")):
                    record, _ = generate(name, prompt, 128, True, f"fusion-{length}-{block}-{order}-{name}")
                    if token_reference is None:
                        token_reference = record["token_ids"]
                    if record["token_ids"] != token_reference:
                        raise AssertionError("Token divergence between paired blocks")
                    records.append(record)
                report["blocks"].append(dict(length=length, block=block, **paired_block(records)))
            memory(f"after_length_{length}")
        if addresses() != original_addresses:
            raise AssertionError("Weight allocation addresses changed")
        report.update(passed=True, fused_python_calls=arm.hits, native_python_calls=arm.fallbacks,
                      additional_weight_bytes=0, weight_addresses_unchanged=True,
                      eligible_for_serving_gate=all(block["latency_change"] < -.02 for block in report["blocks"]))
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (root / "full-model-fusion.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        if arm is not None:
            arm.restore()
        if mesh is not None:
            released = set()
            for store in stores.values():
                for trace_ids, _, _ in store.values():
                    for ids in trace_ids.values():
                        for trace in (ids or {}).values():
                            if trace not in released:
                                ttnn.release_trace(mesh, trace)
                                released.add(trace)
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
