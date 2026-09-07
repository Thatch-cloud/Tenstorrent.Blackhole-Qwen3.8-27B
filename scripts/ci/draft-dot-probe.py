"""Fused SFPU dot-product correctness and isolated hardware latency."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from draft_dot import fused_dot
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--keys', type=int, default=32)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--timing', action='store_true')
    parser.add_argument('--cache-tiles', action='store_true')
    options = parser.parse_args()
    require_projection_environment(os.environ, options.hardware)
    if options.hardware and (options.keys, options.width) not in ((32, 128), (2080, 128), (128, 2080)):
        parser.error('Hardware requires a simulator-validated dot shape')
    if options.timing and not options.hardware:
        parser.error('Timing requires allocated hardware')
    import torch
    import ttnn

    report = dict(passed=False, checks=[], scope=__doc__, keys=options.keys, width=options.width,
        cache_tiles=options.cache_tiles,
        timings_ms=[], backend='hardware' if options.hardware else 'simulator',
        timing_scope='Warm dispatch, output allocation and synchronization; excludes input uploads and output readback/release',
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('draft-dot-probe.py', 'draft_dot.py', 'draft_dot_io.cpp', 'draft_dot_compute.cpp')})
    mesh = None
    tensors = []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D if options.hardware else ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        generator = torch.Generator().manual_seed(319)
        left = torch.randn((1, 16, 32, options.width), generator=generator)
        right = torch.randn((1, 16, options.keys, options.width), generator=generator)
        for host in (left, right):
            tensors.append(ttnn.from_torch(host, device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)))
        output = fused_dot(mesh, tensors[0], tensors[1], tensors, cache_tiles=options.cache_tiles)
        control = fused_dot(mesh, tensors[0], tensors[1], tensors) if options.cache_tiles else None
        expected = (left.double() @ right.double().transpose(-1, -2)).float()
        for chip, tensor in enumerate(ttnn.get_device_tensors(output)):
            actual = ttnn.to_torch(tensor)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)
            if control is not None and not torch.equal(actual, ttnn.to_torch(ttnn.get_device_tensors(control)[chip])):
                raise AssertionError('L1 cached dot differs from streaming control')
            report['checks'].append(dict(chip=chip, max_error=float((actual - expected).abs().max())))
        if options.timing:
            baselines = [ttnn.to_torch(tensor).clone() for tensor in ttnn.get_device_tensors(output)]
            for repetition in range(5):
                temporary = []
                try:
                    ttnn.synchronize_device(mesh)
                    started = time.perf_counter()
                    repeated = fused_dot(mesh, tensors[0], tensors[1], temporary, cache_tiles=options.cache_tiles)
                    ttnn.synchronize_device(mesh)
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    for chip, tensor in enumerate(ttnn.get_device_tensors(repeated)):
                        if not torch.equal(ttnn.to_torch(tensor), baselines[chip]):
                            raise AssertionError('Repeated dot output changed')
                    report['timings_ms'].append(elapsed_ms)
                finally:
                    ttnn.synchronize_device(mesh)
                    release_owned(ttnn, temporary)
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
