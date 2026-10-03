"""Projection-free prefill continuity gate using upstream Qwen GDN and paged KV ops."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, choices=(64, 65, 79, 96), default=79)
    parser.add_argument("--suite", choices=("gdn", "kv", "all"), default="all")
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--device-index", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    source = Path(os.environ["TT_METAL_HOME"]).resolve()
    if args.hardware:
        if os.environ.get("QWEN_HARDWARE_TESTS") != "1" or os.environ.get("TT_METAL_SIMULATOR") or os.environ.get("TT_METAL_SLOW_DISPATCH_MODE"):
            raise RuntimeError("Hardware requires explicit authorization, no simulator and fast dispatch")
    else:
        if not Path(os.environ.get("TT_METAL_SIMULATOR", "")).is_file():
            raise RuntimeError("Simulator required; no physical-device fallback")
        if os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") != "1":
            raise RuntimeError("Slow dispatch required")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != "9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9":
        raise RuntimeError(f"Unreviewed TT-Metal revision: {revision}")

    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import chunk_gated_delta_rule_fused_adapter
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir

    if not Path(ttnn.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Expected the source-built TTNN module")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    key_heads, value_heads, dimension, taps_count, block = 8, 24, 128, 4, 32
    channels = (2 * key_heads + value_heads) * dimension
    padded = ((args.length + block - 1) // block) * block
    results = []
    device = ttnn.open_device(device_id=args.device_index, l1_small_size=24576)
    passed = False

    def upload(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def host(tensor):
        return ttnn.to_torch(tensor).float().clone()

    def compare(name, actual, expected, exact=False):
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError(f"Nonfinite values: {name}")
        torch.testing.assert_close(actual, expected, rtol=0 if exact else 1e-4,
                                   atol=0 if exact else 1e-5, msg=name)
        results.append({"check": name, "passed": True,
                        "max_abs_error": (actual - expected).abs().max().item()})

    try:
        device.enable_program_cache()
        if args.suite in ("gdn", "all"):
            inputs = torch.randn(1, padded, channels).bfloat16().float()
            beta = torch.sigmoid(torch.randn(1, padded, value_heads)).bfloat16().float()
            decay = (-0.01 - 0.02 * torch.rand(1, padded, value_heads)).bfloat16().float()
            taps = [upload(torch.randn(1, 1, channels).bfloat16() * 0.25) for _ in range(taps_count)]

            def run_gdn(chunked, fault=None):
                state = upload(torch.zeros(1, value_heads, dimension, dimension), ttnn.float32)
                carry = upload(torch.zeros(1, taps_count - 1, channels))
                addresses = (state.buffer_address(), carry.buffer_address())
                outputs, convolved = [], []
                step = block if chunked else padded
                for start in range(0, padded, step):
                    if start == block and fault:
                        target = state if fault == "reset_recurrent" else carry
                        zeros = upload(torch.zeros(tuple(target.shape)), target.dtype)
                        ttnn.copy(zeros, target)
                        ttnn.deallocate(zeros)
                    valid = min(step, args.length - start)
                    valid_len = valid if valid < step else None
                    projected = upload(inputs[:, start:start + step])
                    conv, next_carry = _causal_conv1d_fir(
                        projected, None, None, taps_count, device,
                        memory_config=ttnn.L1_MEMORY_CONFIG, conv_state=carry,
                        weight_taps=taps, valid_len=valid_len)
                    key_width = key_heads * dimension
                    query = ttnn.slice(conv, (0, 0, 0), (1, step, key_width))
                    key = ttnn.slice(conv, (0, 0, key_width), (1, step, 2 * key_width))
                    value = ttnn.slice(conv, (0, 0, 2 * key_width), (1, step, channels))
                    beta_tt = upload(beta[:, start:start + step], ttnn.float32)
                    decay_tt = upload(decay[:, start:start + step], ttnn.float32)
                    output, next_state = chunk_gated_delta_rule_fused_adapter(
                        query, key, value, beta_tt, decay_tt, initial_state=state,
                        device=device, valid_len=valid_len, scale=dimension ** -0.5,
                        qkv_head_dims=(key_heads, dimension, value_heads, dimension))
                    outputs.append(host(output)[:, :valid])
                    convolved.append(host(conv)[:, :valid])
                    ttnn.copy(next_state, state)
                    ttnn.copy(next_carry, carry)
                    if addresses != (state.buffer_address(), carry.buffer_address()):
                        raise AssertionError("Persistent state addresses changed")
                    for tensor in (projected, conv, next_carry, query, key, value, beta_tt,
                                   decay_tt, output, next_state):
                        ttnn.deallocate(tensor)
                answer = {"output": torch.cat(outputs, dim=1), "conv": torch.cat(convolved, dim=1),
                          "state": host(state), "carry": host(carry)}
                ttnn.deallocate(state)
                ttnn.deallocate(carry)
                return answer

            uninterrupted = run_gdn(False)
            chunked = run_gdn(True)
            for field in ("conv", "carry", "state", "output"):
                compare(f"gdn_chunked_{field}", chunked[field], uninterrupted[field], exact=field in ("conv", "carry"))
            compare("gdn_carry_real_tail", chunked["carry"], inputs[:, args.length - 3:args.length], exact=True)

            conv = uninterrupted["conv"].reshape(args.length, channels)
            key_width = key_heads * dimension
            query = torch.nn.functional.normalize(conv[:, :key_width].reshape(args.length, key_heads, dimension), dim=-1)
            key = torch.nn.functional.normalize(conv[:, key_width:2 * key_width].reshape(args.length, key_heads, dimension), dim=-1)
            query = query.repeat_interleave(value_heads // key_heads, dim=1) * dimension ** -0.5
            key = key.repeat_interleave(value_heads // key_heads, dim=1)
            value = conv[:, 2 * key_width:].reshape(args.length, value_heads, dimension)
            state_ref = torch.zeros(value_heads, dimension, dimension)
            output_ref = []
            for position in range(args.length):
                state_ref *= decay[0, position].exp()[:, None, None]
                residual = value[position] - torch.einsum("hk,hkv->hv", key[position], state_ref)
                state_ref += torch.einsum("hk,hv->hkv", key[position], residual * beta[0, position, :, None])
                output_ref.append(torch.einsum("hk,hkv->hv", query[position], state_ref))
            for field, reference in (("state", state_ref.unsqueeze(0)), ("output", torch.stack(output_ref).unsqueeze(0))):
                actual = uninterrupted[field]
                relative = (actual - reference).norm() / reference.norm().clamp_min(1e-8)
                correlation = torch.corrcoef(torch.stack((actual.flatten(), reference.flatten())))[0, 1]
                if not torch.isfinite(relative) or not torch.isfinite(correlation) or relative > 0.02 or correlation < 0.999:
                    raise AssertionError(f"Independent recurrent oracle failed: {field}, {relative=}, {correlation=}")
                results.append({"check": f"gdn_cpu_oracle_{field}", "passed": True,
                                "relative_l2": relative.item(), "pcc": correlation.item()})
            for fault in ("reset_recurrent", "reset_conv"):
                broken = run_gdn(True, fault)
                if torch.allclose(broken["state"], uninterrupted["state"], rtol=1e-4, atol=1e-5):
                    raise AssertionError(f"Negative control was not detected: {fault}")
                results.append({"check": f"detect_{fault}", "passed": True})
            for tensor in taps:
                ttnn.deallocate(tensor)

        if args.suite in ("kv", "all"):
            heads, head_dim, physical_blocks = 2, 256, 8
            page_ids = torch.tensor([[5, 1, 6]], dtype=torch.int32)[:, :padded // block]
            kv_inputs = [torch.randn(1, heads, padded, head_dim).bfloat16().float() for _ in range(2)]
            cache_initial = torch.full((physical_blocks, heads, block, head_dim), -7.0)

            def run_kv(chunked, wrong_offset=False):
                caches = [upload(cache_initial) for _ in range(2)]
                step = block if chunked else padded
                for start in range(0, padded, step):
                    first = 0 if wrong_offset else start // block
                    table = upload(page_ids[:, first:first + step // block], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                    for cache, values in zip(caches, kv_inputs):
                        chunk = upload(values[:, :, start:start + step])
                        ttnn.experimental.paged_fill_cache(cache, chunk, table, batch_idx=0)
                        ttnn.deallocate(chunk)
                    ttnn.deallocate(table)
                answer = [host(cache) for cache in caches]
                for cache in caches:
                    ttnn.deallocate(cache)
                return answer

            uninterrupted_kv, chunked_kv = run_kv(False), run_kv(True)
            for index, label in enumerate(("key", "value")):
                expected = cache_initial.clone()
                for logical, physical in enumerate(page_ids[0].tolist()):
                    expected[physical] = kv_inputs[index][0, :, logical * block:(logical + 1) * block]
                compare(f"paged_{label}_chunked", chunked_kv[index], uninterrupted_kv[index], exact=True)
                compare(f"paged_{label}_mapping_and_untouched_blocks", chunked_kv[index], expected, exact=True)
            broken_kv = run_kv(True, wrong_offset=True)
            if torch.equal(broken_kv[0], uninterrupted_kv[0]):
                raise AssertionError("Wrong page-table offset was not detected")
            results.append({"check": "detect_wrong_page_offset", "passed": True})
        passed = True
    finally:
        ttnn.close_device(device)
        upstream_files = (
            "models/demos/blackhole/qwen36/tt/gdn/fused_chunk.py",
            "models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py",
        )
        report = {"passed": passed, "seed": args.seed, "length": args.length, "suite": args.suite,
                          "backend": "hardware" if args.hardware else "simulator", "device_index": args.device_index,
                          "tt_metal_revision": revision,
                          "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "upstream_python_sha256": {
                              name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in upstream_files},
                          "extension_sha256": hashlib.sha256(Path(ttnn._ttnn.__file__).read_bytes()).hexdigest(),
                          "scope": "projection-free single-chip TP2-local-shape operator gate; not full model, scheduler, or speed validation",
                          "results": results}
        serialized = json.dumps(report, indent=2) + "\n"
        if os.environ.get("PREFILL_RESULT_PATH"):
            Path(os.environ["PREFILL_RESULT_PATH"]).write_text(serialized)
        print(serialized, flush=True)


if __name__ == "__main__":
    main()
