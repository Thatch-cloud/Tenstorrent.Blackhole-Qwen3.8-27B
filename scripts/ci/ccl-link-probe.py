"""Exact TP2 projection collective gate; explicit links, not a throughput benchmark."""

import argparse
import json
import os
from pathlib import Path

from feature_collective import gather_add_projection
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned
from projection_link_policy import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, options.hardware)
    report = dict(passed=False, policy=validate(os.environ), checks=[], scope=__doc__)
    import torch
    import ttnn
    from models.tt_transformers.tt.ccl import TT_CCL

    mesh, owned = None, []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        collectives = TT_CCL(mesh)
        for rows in (1, 8, 32):
            first = (torch.arange(rows * 5120).reshape(1, 1, rows, 5120) % 17).float()
            second = first + 1
            tensor = ttnn.from_torch(torch.cat((first, second), dim=0), device=mesh,
                dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
            owned.append(tensor)
            output = gather_add_projection(ttnn, mesh, collectives, tensor)
            owned.append(output)
            shards = ttnn.get_device_tensors(output)
            if len(shards) != 2:
                raise AssertionError('Both chip outputs required')
            for chip, shard in enumerate(shards):
                if not torch.equal(ttnn.to_torch(shard), first + second):
                    raise AssertionError(f'Projection collective mismatch rows={rows} chip={chip}')
                original = ttnn.to_torch(ttnn.get_device_tensors(tensor)[chip])
                if not torch.equal(original, (first, second)[chip]):
                    raise AssertionError('Projection input mutated')
                report['checks'].append(dict(rows=rows, chip=chip, exact=True))
            release_owned(ttnn, owned)
            owned.clear()
        report['passed'] = len(report['checks']) == 6
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            release_owned(ttnn, owned)
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
