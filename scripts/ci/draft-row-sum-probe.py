"""Fused row reduction correctness and isolated hardware cost, not attention throughput."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from draft_row_sum import row_sum
from draft_attention import pairwise_column_sum
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--timing', action='store_true')
    options = parser.parse_args()
    require_projection_environment(os.environ, options.hardware)
    if options.timing and not options.hardware:
        parser.error('Timing requires allocated hardware')
    import torch
    import ttnn

    report = dict(passed=False, checks=[], timings=[], scope=__doc__,
        backend='hardware' if options.hardware else 'simulator',
        timing_scope='Warm dispatch, allocation, synchronization and temporary cleanup; excludes uploads, readback and final output release', sources={name:
        hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('draft-row-sum-probe.py', 'draft_attention.py', 'draft_row_sum.py', 'draft_row_sum_io.cpp', 'draft_row_sum_compute.cpp')})
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D if options.hardware else ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        for width in (32, 64, 2080):
            tensors = []
            try:
                host = torch.rand((1, 16, 32, width), generator=torch.Generator().manual_seed(91 + width))
                source = ttnn.from_torch(host, device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                tensors.append(source)
                expected = host.double().sum(-1, keepdim=True).float()
                baselines = {}
                for repetition in range(6 if options.timing else 1):
                    for mode in (('fused', 'pairwise') if repetition % 2 == 0 else ('pairwise', 'fused')):
                        owned = []

                        def retain(value):
                            owned.append(value)
                            return value

                        try:
                            ttnn.synchronize_device(mesh)
                            started = time.perf_counter()
                            output = row_sum(mesh, source, owned) if mode == 'fused' else pairwise_column_sum(ttnn, source, retain)
                            ttnn.synchronize_device(mesh)
                            output_addresses = addresses(ttnn, output)
                            temporary = [tensor for tensor in owned if addresses(ttnn, tensor) != output_addresses]
                            release_owned(ttnn, temporary)
                            owned = [output]
                            elapsed_ms = (time.perf_counter() - started) * 1000
                            for chip, tensor in enumerate(ttnn.get_device_tensors(output)):
                                actual = ttnn.to_torch(tensor)
                                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-6)
                                if repetition == 0:
                                    baselines[mode, chip] = actual.clone()
                                    report['checks'].append(dict(width=width, mode=mode, chip=chip,
                                        max_error=float((actual - expected).abs().max())))
                                elif not torch.equal(actual, baselines[mode, chip]):
                                    raise AssertionError('Repeated reduction output changed')
                            if repetition:
                                report['timings'].append(dict(width=width, mode=mode, repetition=repetition, ms=elapsed_ms))
                        finally:
                            ttnn.synchronize_device(mesh)
                            release_owned(ttnn, owned)
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
