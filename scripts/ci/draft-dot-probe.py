"""Simulator-only fused SFPU dot-product correctness prerequisite."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from draft_dot import fused_dot
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--keys', type=int, default=32)
    parser.add_argument('--width', type=int, default=128)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    report = dict(passed=False, checks=[], scope=__doc__, keys=options.keys, width=options.width,
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('draft_dot.py', 'draft_dot_io.cpp', 'draft_dot_compute.cpp')})
    mesh = None
    tensors = []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        generator = torch.Generator().manual_seed(319)
        left = torch.randn((1, 16, 32, options.width), generator=generator)
        right = torch.randn((1, 16, options.keys, options.width), generator=generator)
        for host in (left, right):
            tensors.append(ttnn.from_torch(host, device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)))
        output = fused_dot(mesh, tensors[0], tensors[1], tensors)
        expected = (left.double() @ right.double().transpose(-1, -2)).float()
        for chip, tensor in enumerate(ttnn.get_device_tensors(output)):
            actual = ttnn.to_torch(tensor)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)
            report['checks'].append(dict(chip=chip, max_error=float((actual - expected).abs().max())))
    finally:
        if mesh is not None:
            ttnn.synchronize_device(mesh)
            release_owned(ttnn, tensors)
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))
    report['passed'] = True
    options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
