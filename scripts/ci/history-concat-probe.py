"""Weight-free TP2 concat setup lifetime checks; no model or throughput qualification."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from dspark_history import TensorScope, join_rows as original
from history_concat_lifetime import join_rows as candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or Path('/dev/tenstorrent').exists() or options.output.exists()):
        raise ValueError('Fresh device-free simulator result required')
    import torch
    import ttnn

    names = ('history-concat-probe.py', 'history_concat_lifetime.py', 'dspark_history.py', 'gdn_multitoken_conv.py')
    fingerprints = lambda: {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in names}
    report = dict(passed=False, closed_cleanly=False, backend='simulator', checks=[],
        performance_qualified=False, model_integrated=False, sources=fingerprints())
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=1048576)
        mesh.enable_program_cache()
        for count in (65, 128):
            for seed in (0, 1):
                generator = torch.Generator().manual_seed(seed)
                pieces = [torch.randint(-64, 64, (1, 4, 32, 128), generator=generator).to(torch.bfloat16)
                    for unused in range(count)]
                expected = torch.cat(pieces, dim=2)
                for name, concatenate in (('original', original), ('candidate', candidate)):
                    print(json.dumps(dict(stage='concat', arm=name, count=count, seed=seed)), flush=True)
                    borrowed = ttnn.from_torch(pieces[0], device=mesh, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                    scope = TensorScope(ttnn, [borrowed])
                    try:
                        values = [borrowed]
                        for piece in pieces[1:]:
                            values.append(scope.retain(ttnn.from_torch(piece, device=mesh,
                                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))))
                        output = concatenate(ttnn, values, scope.retain)
                        ttnn.synchronize_device(mesh)
                        for label, value, reference in (('output', output, expected), ('borrowed', borrowed, pieces[0])):
                            shards = ttnn.get_device_tensors(value)
                            if len(shards) != 2:
                                raise AssertionError('Both chips required')
                            for chip, shard in enumerate(shards):
                                exact = torch.equal(ttnn.to_torch(shard), reference)
                                report['checks'].append(dict(arm=name, count=count, seed=seed,
                                    tensor=label, chip=chip, exact=exact))
                                if not exact:
                                    raise AssertionError('Concat lifetime changed tensor contents')
                    finally:
                        scope.release()
                        ttnn.deallocate(borrowed)
                    options.output.write_text(json.dumps(report, indent=2) + '\n')
        report['passed'] = len(report['checks']) == 32 and all(check['exact'] for check in report['checks'])
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
        finally:
            report['sources_after'] = fingerprints()
            options.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
