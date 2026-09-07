"""Simulator-only fused row reduction prerequisite, not attention or throughput."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from draft_row_sum import row_sum
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    report = dict(passed=False, checks=[], scope=__doc__, sources={name:
        hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('draft_row_sum.py', 'draft_row_sum_io.cpp', 'draft_row_sum_compute.cpp')})
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        for width in (32, 64, 2080):
            tensors = []
            try:
                host = torch.rand((1, 16, 32, width), generator=torch.Generator().manual_seed(91 + width))
                source = ttnn.from_torch(host, device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                tensors.append(source)
                output = row_sum(mesh, source, tensors)
                expected = host.double().sum(-1, keepdim=True).float()
                for chip, tensor in enumerate(ttnn.get_device_tensors(output)):
                    actual = ttnn.to_torch(tensor)
                    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-6)
                    report['checks'].append(dict(width=width, chip=chip, max_error=float((actual - expected).abs().max())))
            finally:
                ttnn.synchronize_device(mesh)
                release_owned(ttnn, tensors)
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))
    report['passed'] = True
    options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
