"""Simulator-only prefill feature oracle using synthetic layers, not Qwen numerical parity."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from full_prefill_features import verify_prefill_features
from gdn_multitoken_conv import addresses, release_owned


def main():
    if not os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_SLOW_DISPATCH_MODE'):
        raise RuntimeError('Fast-dispatch simulator required')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    import torch
    import ttnn

    report = dict(passed=False, scope=__doc__, checks=[], sources={name:
        hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('prefill-feature-probe.py', 'full_prefill_features.py', 'target_features.py')})
    mesh = None
    owned = []
    state = dict(position=0)
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()

        def forward(hidden):
            result = ttnn.add(hidden, 1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            owned.append(result)
            return result

        model = SimpleNamespace(args=SimpleNamespace(vocab_size=5120),
            layers=[SimpleNamespace(forward=forward) for index in range(64)])

        def local_host(value):
            return [ttnn.to_torch(part) for part in ttnn.get_device_tensors(value)]

        def prefill_logits(prompt):
            bucket = ((len(prompt) + 127) // 128) * 128
            host = torch.arange(bucket).reshape(1, 1, bucket, 1).expand(1, 1, bucket, 5120).clone().bfloat16() / 8
            host[..., 2560:] += 16
            hidden = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))
            owned.append(hidden)
            for layer in model.layers:
                hidden = layer.forward(hidden)
            result = torch.cat(local_host(hidden), dim=-1)[:, :, len(prompt) - 1].clone()
            release_owned(ttnn, owned)
            owned.clear()
            state['position'] = len(prompt)
            return result

        def decode(token, position, trace):
            if trace:
                raise ValueError('Eager synthetic continuation only')
            state['position'] = position + 1
            return torch.full((1, 5120), float(position))

        for length in (63, 129):
            report['checks'].append(verify_prefill_features(model, [1] * length, (5, 19, 33, 47, 61),
                prefill_logits=prefill_logits, decode=decode, live_digest=lambda: state['position'],
                kv_digest=lambda count: (count, state['position']), inactive_digest=lambda: 'synthetic-inactive',
                snapshot=lambda value: ttnn.clone(value, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                release=ttnn.deallocate, storage_ids=lambda value: tuple(enumerate(addresses(ttnn, value))),
                local_host=local_host))
    finally:
        if mesh is not None:
            release_owned(ttnn, owned)
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))
    report['passed'] = True
    options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
