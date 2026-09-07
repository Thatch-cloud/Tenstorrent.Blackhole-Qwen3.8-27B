"""Simulator-only prefix-copy comparison including every physical padding element."""

import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/ci'))
from gdn_conv_prefix_copy import copy_prefix
from gdn_multitoken_conv import release_owned


def main():
    if not os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_SLOW_DISPATCH_MODE') != '1':
        raise RuntimeError('Dedicated slow-dispatch simulator required')
    import torch
    import ttnn

    root = Path(__file__).resolve().parents[2] / 'scripts/ci'
    path = Path(os.environ['QWEN_SIM_REPORT'])
    report = dict(passed=False, scope=__doc__, checks=[], hashes={suffix:
        hashlib.sha256((root / f'gdn_conv_prefix_copy.{suffix}').read_bytes()).hexdigest() for suffix in ('py', 'cpp')})
    mesh = None
    owned = []
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        generator = torch.Generator().manual_seed(38148)

        def upload(value):
            result = ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            owned.append(result)
            return result

        for rows in (1, 2, 4, 8, 16, 32):
            host_sources = [torch.randn((1, rows, 5120), generator=generator).bfloat16() for _ in range(4)]
            sources = [upload(value) for value in host_sources]
            destinations = [[upload(torch.full((1, 32, 5120), 17., dtype=torch.bfloat16)).reshape((1, 1, 5120))
                             for _ in range(4)] for _ in range(2)]
            for prefix in range(1, rows + 1):
                for candidate, targets in zip((False, True), destinations, strict=True):
                    copy_prefix(mesh, sources, targets, prefix, reuse_zero_tile=candidate)
                    for slot, target in enumerate(targets):
                        expected = torch.zeros((1, 32, 5120), dtype=torch.bfloat16)
                        expected[:, :1] = host_sources[slot][:, prefix - 1:prefix]
                        shards = ttnn.get_device_tensors(target)
                        if len(shards) != 2:
                            raise AssertionError('Both simulated chips required')
                        for shard in shards:
                            actual = ttnn.to_torch(shard.reshape(shard.padded_shape))
                            if not torch.equal(actual, expected):
                                raise AssertionError(f'Prefix or physical padding mismatch: rows={rows}, prefix={prefix}, candidate={candidate}')
                report['checks'].append(dict(rows=rows, prefix=prefix, both_chips=True, physical_padding_exact=True))
                path.write_text(json.dumps(report, indent=2))
            for source, expected in zip(sources, host_sources, strict=True):
                if not all(torch.equal(ttnn.to_torch(shard), expected) for shard in ttnn.get_device_tensors(source)):
                    raise AssertionError('Read-only packed source changed')
            release_owned(ttnn, owned)
            owned.clear()
            print(json.dumps(dict(rows=rows, all_prefixes_exact=True)), flush=True)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            release_owned(ttnn, owned)
            ttnn.close_mesh_device(mesh)
        path.write_text(json.dumps(report, indent=2))
    if len(report['checks']) != 63:
        raise AssertionError('All supported prefixes required')
    report['passed'] = True
    path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
