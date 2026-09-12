"""Exact TP2 projection collective gate; explicit links, not a throughput benchmark."""

import argparse
import json
import os
import statistics
import time
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
    report['timings'] = []
    report['timing_scope'] = 'Blocking eager collective including temporary handling; excludes upload and validation; not model TG'
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
            if options.hardware:
                original_links = os.environ.get('QWEN_PROJECTION_LINKS')
                try:
                    for links in (1, 4):
                        os.environ['QWEN_PROJECTION_LINKS'] = str(links)
                        warm = gather_add_projection(ttnn, mesh, collectives, tensor)
                        ttnn.synchronize_device(mesh)
                        ttnn.deallocate(warm)
                    for block in range(3):
                        for arm, links in enumerate((1, 4, 4, 1)):
                            os.environ['QWEN_PROJECTION_LINKS'] = str(links)
                            ttnn.synchronize_device(mesh)
                            started = time.perf_counter()
                            timed = gather_add_projection(ttnn, mesh, collectives, tensor)
                            elapsed_ms = (time.perf_counter() - started) * 1000
                            try:
                                parts = ttnn.get_device_tensors(timed)
                                if len(parts) != 2 or any(not torch.equal(ttnn.to_torch(part), first + second) for part in parts):
                                    raise AssertionError('Timed collective output differs')
                                report['timings'].append(dict(rows=rows, block=block, arm=arm,
                                                              links=links, elapsed_ms=elapsed_ms, exact=True))
                            finally:
                                ttnn.deallocate(timed)
                finally:
                    if original_links is None:
                        os.environ.pop('QWEN_PROJECTION_LINKS', None)
                    else:
                        os.environ['QWEN_PROJECTION_LINKS'] = original_links
            release_owned(ttnn, owned)
            owned.clear()
        report['passed'] = len(report['checks']) == 6
        if options.hardware:
            report['passed'] = report['passed'] and len(report['timings']) == 36
            report['medians_ms'] = [dict(rows=rows, links=links,
                median=statistics.median(record['elapsed_ms'] for record in report['timings']
                                         if record['rows'] == rows and record['links'] == links))
                for rows in (1, 8, 32) for links in (1, 4)]
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
