"""Full-model serial rollback oracle across attention page boundaries, not batched verification speed."""

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

from gdn_snapshot import ActiveSnapshot


def logical_kv_prefix(host, valid_tokens):
    if type(valid_tokens) is not int or not 0 < valid_tokens <= 128:
        raise ValueError("Expected a valid prefix of the first two KV pages")
    if len(host.shape) != 4 or host.shape[0] != 2 or host.shape[2] != 64:
        raise ValueError("Expected two 64-token KV pages")
    heads, width = host.shape[1], host.shape[3]
    return host.permute(1, 0, 2, 3).reshape(heads, 128, width)[:, :valid_tokens]


def main():
    if os.environ.get("QWEN_HARDWARE_TESTS") != "1" or os.environ.get("QWEN_CARDS_ALLOCATED") != "1":
        raise RuntimeError("Explicit hardware allocation required")
    if os.environ.get("TT_METAL_SIMULATOR") or os.environ.get("TT_METAL_SLOW_DISPATCH_MODE"):
        raise RuntimeError("Fast-dispatch hardware required")
    import torch
    import ttnn
    from transformers import AutoConfig, AutoTokenizer
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

    root = Path("/experiment/results")
    report = dict(passed=False, checks=[], negative_controls=[], rows=16,
                  scope="Native sequential 64-layer target; active GDN restore and logical KV rollback, no drafter or speed claim")
    mesh = None
    generator = None
    try:
        source = Path("/opt/tt-metal")
        expected_source = {
            "models/demos/blackhole/qwen36/tt/gdn/tp.py": "f767d0648ae01b0b1c0bb7bf601f5490661707b845c11ce6dbebb86ba0f84dc9",
            "models/demos/blackhole/qwen36/tt/qwen36_vllm.py": "cda38c3121b7a61417885469c224c0c69189fda899fbf8361565f4d93125c2fe",
            "models/tt_transformers/tt/generator.py": "4c2633ba8e5e6b0430550ef99409e9a6f0e0a901b4c6627540c579eb9b7d5a3e",
        }
        report["source"] = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in expected_source}
        if report["source"] != expected_source:
            raise ValueError("Unreviewed native GDN/Generator source")
        report["model_source_sha256"] = hashlib.sha256((source / "models/demos/blackhole/qwen36/tt/model.py").read_bytes()).hexdigest()
        report["copy_kernel_sha256"] = hashlib.sha256(Path(__file__).with_name("gdn_state_copy.cpp").read_bytes()).hexdigest()
        if report["copy_kernel_sha256"] != "4f921500b63817a5f26f288725a9da1fee9223841a68d058d96fac0f67a23428":
            raise ValueError("Direct copy kernel must pass the layer gate first")
        report["prerequisite_run"] = 33999114362
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
        if len(model.layers) != 64 or model.args.vocab_size != 248320:
            raise ValueError("Expected frozen 64-layer target")
        kv_cache = generator.allocate_kv_cache((8200, model.args.n_local_kv_heads, 64, model.args.head_dim),
                                               ttnn.bfloat16, len(model.layers))
        page_table = torch.arange(1024, dtype=torch.int32).reshape(1, 1024)
        layers = [layer.attention for layer in model.layers if not layer.is_full_attention]
        caches = [tensor for pair in model._paged_kv_caches for tensor in pair]
        if len(layers) != 48 or len(caches) != 32:
            raise ValueError("Expected 48 GDN layers and 16 attention K/V pairs")
        helpers = [ActiveSnapshot(layer, ttnn, direct=True) for layer in layers]
        saved = [helper.allocate() for helper in helpers]
        scratch = [helper.allocate() for helper in helpers]
        report["snapshot_bytes_per_chip"] = sum(math.prod(tensor.padded_shape) * 2
                                                 for state in saved + scratch for tensor in state)
        report["kv_dtypes"] = [str(tensor.dtype) for tensor in caches]

        def addresses():
            return [tensor.buffer_address() for layer in layers for tensor in [layer.rec_state, *layer.conv_states]]

        original_addresses = addresses()

        def save(destination):
            for helper, state in zip(helpers, destination, strict=True):
                helper.save(state)

        def restore():
            if addresses() != original_addresses:
                raise AssertionError("Native state addresses changed")
            for helper, state in zip(helpers, saved, strict=True):
                helper.restore(state)

        def digest(tensor):
            return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

        def local_host(tensor):
            shards = ttnn.get_device_tensors(tensor)
            if len(shards) != 2:
                raise AssertionError("Both chips required")
            return [ttnn.to_torch(shard) for shard in shards]

        def state_digest(state):
            return [digest(host) for layer in state for tensor in layer for host in local_host(tensor)]

        def live_digest():
            save(scratch)
            return state_digest(scratch)

        def kv_digest(valid_tokens):
            result = []
            if not 0 < valid_tokens <= 128:
                raise ValueError("Oracle covers the first two physical pages only")
            for tensor in caches:
                heads, block_size, width = tensor.shape[1:]
                if block_size != 64:
                    raise ValueError("Expected 64-token KV pages")
                first_pages = ttnn.slice(tensor, (0, 0, 0, 0), (2, heads, 64, width))
                for host in local_host(first_pages):
                    valid = logical_kv_prefix(host, valid_tokens)
                    result.append(digest(valid))
                ttnn.deallocate(first_pages)
            return result

        def decode(token, position, trace, pages=None):
            result = generator.decode_forward(tokens=torch.tensor([[token]], dtype=torch.int32),
                start_pos=torch.tensor([position], dtype=torch.int32), page_table=page_table if pages is None else pages,
                kv_cache=kv_cache, enable_trace=trace, read_from_device=True)
            logits = result[0] if isinstance(result, tuple) else result
            return logits.clone()

        def argmax(logits):
            return int(logits.reshape(-1, model.args.vocab_size)[0].float().argmax())

        def prefill(prompt):
            generator.prev_page_table = None
            logits, _ = generator.prefill_forward(torch.tensor([prompt], dtype=torch.int32), page_table,
                kv_cache, [len(prompt)], empty_slots=[0], enable_trace=True)
            if addresses() != original_addresses:
                raise AssertionError("Prefill replaced persistent decode state")
            return argmax(logits)

        generator.warmup_model_decode(kv_cache=kv_cache, enable_trace=False, max_batch_size=1,
                                      num_blocks=1024, can_sample_on_device=False)
        save(saved)
        save(scratch)
        restore()
        kv_digest(128)
        generator.warmup_model_prefill(kv_cache=kv_cache, enable_trace=True)
        generator.warmup_model_decode(kv_cache=kv_cache, enable_trace=True, max_batch_size=1,
                                      num_blocks=1024, can_sample_on_device=False, skip_trace_precompile=True)
        base_prompt = baseline.make_prompt(tokenizer, 128, 0)
        for length in (63, 64, 65):
            prompt = base_prompt[:length]
            if len(prompt) != length:
                raise AssertionError("Insufficient fixed prompt tokens")
            oracle = [prefill(prompt)]
            oracle_logits = []
            for position in range(18):
                logits = decode(oracle[-1], length + position, False)
                oracle_logits.append(logits)
                oracle.append(argmax(logits))
            for trace in (False, True):
                prefill(prompt)
                for index, expected in enumerate(oracle_logits):
                    if not torch.equal(decode(oracle[index], length + index, trace), expected):
                        raise AssertionError("Native eager/trace baseline differs")
                report.setdefault("mode_checks", []).append(dict(length=length, trace=trace, logits_exact=True))
                for prefix in range(17):
                    if prefill(prompt) != oracle[0]:
                        raise AssertionError("Reference prefill seed changed")
                    for index in range(prefix):
                        decode(oracle[index], length + index, trace)
                    save(saved)
                    expected_state = state_digest(saved)
                    expected_kv = kv_digest(length + prefix)
                    expected_logits = [decode(oracle[index], length + index, trace) for index in range(prefix, prefix + 2)]
                    expected_final_state = live_digest()
                    expected_final_kv = kv_digest(length + prefix + 2)

                    if prefill(prompt) != oracle[0]:
                        raise AssertionError("Candidate prefill seed changed")
                    for index in range(prefix):
                        decode(oracle[index], length + index, trace)
                    for index in range(prefix, 16):
                        decode((oracle[index] + 137) % model.args.vocab_size, length + index, trace)
                    restore()
                    if live_digest() != expected_state or kv_digest(length + prefix) != expected_kv:
                        raise AssertionError(f"Rollback state/KV prefix mismatch: {length=} {prefix=} {trace=}")
                    for index, expected in zip(range(prefix, prefix + 2), expected_logits, strict=True):
                        actual = decode(oracle[index], length + index, trace)
                        if not torch.equal(actual, expected):
                            raise AssertionError(f"Full-logit continuation mismatch: {length=} {prefix=} {trace=}")
                    if live_digest() != expected_final_state or kv_digest(length + prefix + 2) != expected_final_kv:
                        raise AssertionError("Corrected continuation state/KV mismatch")
                    report["checks"].append(dict(length=length, prefix=prefix, trace=trace,
                        logits_exact=True, all_gdn_states_exact=True, valid_kv_exact=True, correction_steps=2))
                    if prefix == 0:
                        prefill(prompt)
                        for index in range(16):
                            decode((oracle[index] + 137) % model.args.vocab_size, length + index, trace)
                        stale = decode(oracle[0], length, trace)
                        stale_detected = not torch.equal(stale, expected_logits[0])
                        prefill(prompt)
                        wrong_pages = page_table.clone()
                        wrong_pages[0, 0] = 8199
                        wrong = decode(oracle[0], length, trace, wrong_pages)
                        page_detected = not torch.equal(wrong, expected_logits[0])
                        report["negative_controls"].append(dict(length=length, trace=trace,
                            stale_gdn_detected=stale_detected, wrong_page_detected=page_detected))
                        if not stale_detected or not page_detected:
                            raise AssertionError("Full-model negative control was not detected")
                    (root / "full-prefix.json").write_text(json.dumps(report, indent=2))
                    print(json.dumps(dict(length=length, prefix=prefix, trace=trace, exact=True)), flush=True)
        if addresses() != original_addresses:
            raise AssertionError("Persistent state addresses changed")
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (root / "full-prefix.json").write_text(json.dumps(report, indent=2))
        if mesh is not None:
            if generator is not None:
                for store in getattr(generator, "_bucket_trace_store", {}).values():
                    for per_device in store[0].values():
                        if per_device:
                            for trace in per_device.values():
                                ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
