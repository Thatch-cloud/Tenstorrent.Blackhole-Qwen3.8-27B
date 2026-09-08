"""Synthetic draft GQA/mask correctness; not learned attention or throughput."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from draft_attention import draft_attention_mask, draft_sdpa, composed_draft_attention
from feature_projection import require_projection_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--timing', action='store_true')
    parser.add_argument('--uniform-query', action='store_true')
    parser.add_argument('--streaming', action='store_true')
    parser.add_argument('--composed', action='store_true')
    parser.add_argument('--precise-native', action='store_true')
    options = parser.parse_args()
    if options.composed and options.streaming:
        parser.error('Composed and native streaming are separate controls')
    if options.precise_native and (options.composed or options.streaming):
        parser.error('Precise native SDPA uses its isolated FP32 accumulation path')
    if options.hardware and not options.composed:
        parser.error('Only the simulator-validated composed path is enabled for hardware')
    if options.timing and not (options.hardware and options.composed):
        parser.error('Timing requires hardware and the composed path')
    require_projection_environment(os.environ, options.hardware)
    from native_draft_sdpa import run_precise_probe

    run(options, run_precise_probe(__file__) if options.precise_native else None)


def run(options, kernel_audit):
    import torch
    import ttnn

    report = dict(passed=False, scope=__doc__, checks=[], uniform_query=options.uniform_query,
        streaming=options.streaming, tolerance=dict(rtol=.01, atol=.01),
        composed=options.composed, native_kernel=kernel_audit,
        backend='hardware' if options.hardware else 'simulator', timings=[],
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('draft-attention-probe.py', 'draft_attention.py')})
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D if options.hardware else ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        for context in (0, 31, 2048):
            mask = draft_attention_mask(context)
            key_rows = mask.shape[-1]
            generator = torch.Generator().manual_seed(8712 + context)
            query = (torch.randn((1, 32, 32, 128), generator=generator) * .125).bfloat16()
            if options.uniform_query:
                query.zero_()
            key = (torch.randn((1, 8, key_rows, 128), generator=generator) * .125).bfloat16()
            value = torch.randn((1, 8, key_rows, 128), generator=generator).bfloat16()
            value[..., context + 7, :] += 8
            value[..., context + 8:, :] = 8192
            if context == 2048:
                value[..., 0, :] = -8192
            tensors = []
            output = None
            try:
                for host in (query, key, value):
                    tensors.append(ttnn.from_torch(host, device=mesh, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1)))
                device_mask = ttnn.from_torch(mask, device=mesh, dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                tensors.append(device_mask)
                output = composed_draft_attention(ttnn, mesh, *tensors) if options.composed else draft_sdpa(ttnn, *tensors, streaming=options.streaming)
                ttnn.synchronize_device(mesh)
                shards = ttnn.get_device_tensors(output)
                if len(shards) != 2:
                    raise AssertionError('Both draft attention ranks required')
                for chip, shard in enumerate(shards):
                    local_query = query[:, chip * 16:(chip + 1) * 16, :8].float()
                    local_key = key[:, chip * 4:(chip + 1) * 4].float().repeat_interleave(4, dim=1)
                    local_value = value[:, chip * 4:(chip + 1) * 4].float().repeat_interleave(4, dim=1)
                    expected = torch.nn.functional.scaled_dot_product_attention(local_query, local_key, local_value,
                        attn_mask=mask[..., :8, :].float(), is_causal=False)
                    causal_mask = mask[..., :8, :].float().clone()
                    causal_mask.masked_fill_(torch.arange(key_rows)[None, :] > context + torch.arange(8)[:, None], float('-inf'))
                    wrong = torch.nn.functional.scaled_dot_product_attention(local_query, local_key, local_value,
                        attn_mask=causal_mask, is_causal=False)
                    if torch.allclose(wrong, expected, rtol=1e-5, atol=1e-5):
                        raise AssertionError('Fixture does not distinguish causal target and noncausal draft masks')
                    actual = ttnn.to_torch(shard)[..., :8, :].float()
                    check = dict(context=context, chip=chip, max_abs_error=float((actual - expected).abs().max()),
                        causal_negative_detected=True, passed=bool(torch.allclose(actual, expected, rtol=.01, atol=.01)))
                    check['first_values'] = actual[0, 0, 0, :16].tolist()
                    check['expected_first_values'] = expected[0, 0, 0, :16].tolist()
                    check['mean_abs_error'] = float((actual - expected).abs().mean())
                    report['checks'].append(check)
                    if not check['passed']:
                        raise AssertionError('Draft GQA numerical or mask gate failed')
                if options.timing:
                    samples = []
                    for repeat in range(5):
                        repeated = None
                        started = time.perf_counter()
                        try:
                            repeated = composed_draft_attention(ttnn, mesh, *tensors)
                            elapsed = time.perf_counter() - started
                            for chip, shard in enumerate(ttnn.get_device_tensors(repeated)):
                                if not torch.equal(ttnn.to_torch(shard), ttnn.to_torch(shards[chip])):
                                    raise AssertionError('Repeated composed attention differs from validated output')
                            samples.append(elapsed)
                        finally:
                            if repeated is not None:
                                ttnn.deallocate(repeated)
                    report['timings'].append(dict(context=context, block_rows=8, seconds=samples,
                        scope='Warm composed attention only: includes operation dispatch, temporary allocation and synchronization; excludes uploads, output readback, final output release and all other draft/target work'))
            finally:
                if output is not None:
                    ttnn.deallocate(output)
                for tensor in reversed(tensors):
                    ttnn.deallocate(tensor)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))
    report['passed'] = True
    options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
