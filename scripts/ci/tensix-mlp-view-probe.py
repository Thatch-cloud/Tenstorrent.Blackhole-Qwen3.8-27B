"""Simulate full native BF4/BF4/BF8 2D shards and verify borrowed 4D metadata aliases on both chips."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from tensix_mlp_gate import hashes
from tensix_mlp_view_gate import NATIVE_SOURCES, SOURCES, qualify
from tensix_mlp_weight_views import WEIGHTS, weight_views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args()
    if not os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('QWEN_SIM_PACKER_ZERO_GRAFT'):
        raise ValueError('Simulator only, with original native packer')
    report = dict(backend='simulator', passed=False, closed_cleanly=False, checks=[],
        shard_axes=dict(gate=1, up=1, down=0),
        sources=hashes(Path(__file__).parent, SOURCES),
        native_sources=hashes(Path(os.environ['TT_METAL_HOME']), NATIVE_SOURCES))
    if report['native_sources'] != NATIVE_SOURCES:
        raise ValueError('Reviewed original native view sources required')
    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2))
        print(stage, flush=True)
    import torch
    import ttnn
    mesh, weights, views, canonical = None, {}, {}, {}
    def digest(value):
        return hashlib.sha256(value.view(torch.int16).contiguous().numpy().tobytes()).hexdigest()
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        for index, (name, (inner, width, dtype)) in enumerate(WEIGHTS.items()):
            axis = report['shard_axes'][name]
            shape = (inner * 2, width) if axis == 0 else (inner, width * 2)
            host = torch.randn(shape, generator=torch.Generator().manual_seed(1659 + index)).bfloat16()
            weights[name] = ttnn.from_torch(host, device=mesh, dtype=getattr(ttnn, dtype),
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=axis))
            del host
            progress(f'{name}_uploaded')
        ttnn.synchronize_device(mesh)
        before = mesh.num_program_cache_entries()
        views, report['weight_views'] = weight_views(ttnn, weights)
        canonical, unused_checks = weight_views(ttnn, views)
        report['program_cache'] = [before, mesh.num_program_cache_entries()]
        reference_hashes = {}
        for name in WEIGHTS:
            reference_hashes[name] = []
            for chip in range(2):
                native = ttnn.to_torch(ttnn.get_device_tensors(weights[name])[chip])
                candidate = ttnn.to_torch(ttnn.get_device_tensors(views[name])[chip])
                if not torch.equal(native.view(torch.int16), candidate.reshape(native.shape).view(torch.int16)):
                    raise AssertionError('Rank expansion changed chip-local dequantized weight contents')
                reference_hashes[name].append(digest(native))
                report['checks'].append(dict(weight=name, chip=chip, contents_exact=True,
                    canonical_identity=canonical[name] is views[name]))
                del native, candidate
            if reference_hashes[name][0] == reference_hashes[name][1]:
                raise AssertionError('Shard fixtures must distinguish accidental replication or chip substitution')
            progress(f'{name}_contents_exact')
        canonical.clear()
        views.clear()
        for check in report['checks']:
            current = ttnn.to_torch(ttnn.get_device_tensors(weights[check['weight']])[check['chip']])
            check['native_unchanged_after_view_release'] = digest(current) == reference_hashes[check['weight']][check['chip']]
            check['distinct_shards'] = True
            del current
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        canonical.clear()
        views.clear()
        if mesh is not None:
            ttnn.synchronize_device(mesh)
            weights.clear()
            ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
        progress('complete' if report['passed'] else 'failed')
    print(json.dumps(qualify(report, report['sources'], report['native_sources'])), flush=True)


if __name__ == '__main__':
    main()
