"""Replay hash-pinned learned attention operands without repeating weight projection."""

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

from draft_attention import draft_sdpa
from feature_projection import require_projection_environment
from native_draft_sdpa import run_precise_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--key-chunk-size', type=int, choices=(32, 64), default=32)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if hashlib.sha256(options.fixture.read_bytes()).hexdigest() != options.sha256:
        parser.error('Captured operand integrity mismatch')
    run(options, run_precise_probe(__file__))


def run(options, kernel_audit):
    import torch
    import ttnn

    data = options.fixture.read_bytes()
    if hashlib.sha256(data).hexdigest() != options.sha256:
        raise ValueError('Captured operands changed before tensor conversion')
    fixture = torch.load(BytesIO(data), map_location='cpu', weights_only=True)
    values = [fixture[name] for name in ('query', 'key', 'value', 'mask')]
    shapes = ((1, 16, 32, 128), (1, 4, 64, 128), (1, 4, 64, 128), (1, 1, 32, 64))
    if any(not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != torch.bfloat16
            for value, shape in zip(values, shapes, strict=True)):
        raise ValueError('Layer-zero CTX31 captured TP2 operand geometry required')
    if any(not torch.isfinite(value).all() for value in values[:3]) or torch.isnan(values[3]).any():
        raise ValueError('Finite learned operands and non-NaN mask required')
    query, key, value, mask = [value.float() for value in values]
    expected = torch.nn.functional.scaled_dot_product_attention(query, key.repeat_interleave(4, dim=1),
        value.repeat_interleave(4, dim=1), attn_mask=mask, is_causal=False)[..., :8, :]
    report = dict(passed=False, scope=__doc__, fixture_sha256=options.sha256,
        native_kernel=kernel_audit, key_chunk_size=options.key_chunk_size, checks=[],
        source_rank=0, scope_limit='One captured rank replicated on two simulator chips; not original rank-one validation')
    mesh, output = None, None
    inputs = []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        for value in values:
            inputs.append(ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)))
        output = draft_sdpa(ttnn, *inputs, key_chunk_size=options.key_chunk_size)
        ttnn.synchronize_device(mesh)
        shards = ttnn.get_device_tensors(output)
        if len(shards) != 2:
            raise AssertionError('Both simulated chips required')
        for chip, shard in enumerate(shards):
            actual = ttnn.to_torch(shard).float()[..., :8, :]
            error = (actual - expected).abs()
            report['checks'].append(dict(chip=chip, max_abs_error=float(error.max()),
                rms_error=float(error.square().mean().sqrt()),
                failed_elements=int((error > (.01 + .01 * expected.abs())).sum()),
                passed=bool(torch.allclose(actual, expected, rtol=.01, atol=.01))))
        if not all(check['passed'] for check in report['checks']):
            raise AssertionError('Captured learned operand numerical gate failed')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            if output is not None:
                ttnn.deallocate(output)
            for value in reversed(inputs):
                ttnn.deallocate(value)
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
