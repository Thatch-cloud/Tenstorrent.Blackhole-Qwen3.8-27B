"""Synthetic exact arithmetic gate, not learned convolution or draft-model quality."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from draft_convolution import convolution_reference, grouped_causal_convolution
from feature_projection import require_projection_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    report = dict(passed=False, scope=__doc__, backend='simulator', checks=[], sources={name:
        hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('draft-convolution-probe.py', 'draft_convolution.py')})
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        for rows in (1, 8):
            generator = torch.Generator().manual_seed(4821 + rows)
            host = [torch.randint(-2, 3, shape, generator=generator).bfloat16() / 4
                for shape in ((2, 1, rows, 5120), (2, 1, rows, 320), (2, 1, rows, 320),
                              (2, 1, 1, 5120), (2, 1, 1, 5120))]
            tensors = []
            output = None
            try:
                for value in host:
                    tensors.append(ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)))
                output = grouped_causal_convolution(ttnn, mesh, tensors[0], tensors[1:3], tensors[3:5])
                shards = ttnn.get_device_tensors(output)
                if len(shards) != 2:
                    raise AssertionError('Both ranks required')
                for chip, shard in enumerate(shards):
                    local = [value[chip:chip + 1] for value in host]
                    expected = convolution_reference(local[0], local[1:3], local[3:5])
                    actual = ttnn.to_torch(shard)
                    check = dict(rows=rows, chip=chip, shape=list(actual.shape), exact=bool(torch.equal(actual, expected)))
                    report['checks'].append(check)
                    if not check['exact']:
                        raise AssertionError('Dyadic convolution arithmetic or causal/group layout differs')
                for device_value, expected in zip(tensors, host, strict=True):
                    for chip, shard in enumerate(ttnn.get_device_tensors(device_value)):
                        if not torch.equal(ttnn.to_torch(shard), expected[chip:chip + 1]):
                            raise AssertionError('Borrowed convolution input changed')
            finally:
                if output is not None:
                    ttnn.deallocate(output)
                for tensor in tensors:
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
