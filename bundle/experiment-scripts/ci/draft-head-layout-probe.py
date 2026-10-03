"""Simulator-only exact native DFlash2 head split/merge and changed-input trace checks."""

import argparse
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from draft_head_layout import split_projected_heads, concatenate_query_heads
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    report = dict(passed=False, scope=__doc__, checks=[])
    mesh = trace = None
    owned = []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        def retain(value):
            owned.append(value)
            return value
        def upload(value, device=True):
            return ttnn.from_torch(value, device=mesh if device else None, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
        def host(rows, width, pattern):
            base = (torch.arange(rows * width).reshape(1, 1, rows, width) % 127 + pattern).bfloat16()
            return torch.cat((base, base + 128), dim=0)
        for case, rows in enumerate((192, 192, 32, 2080, 192)):
            gold = [host(32, 2048, case), host(rows, 512, case + 7), host(rows, 512, case + 19)]
            inputs = [retain(upload(value)) for value in gold]
            def operation():
                heads = split_projected_heads(ttnn, *inputs, retain)
                merged = concatenate_query_heads(ttnn, heads['q'], retain)
                return heads, merged
            heads, merged = operation()
            if case == 4:
                trace, (heads, merged) = capture_operation(ttnn, mesh, operation)
                gold = [host(32, 2048, 29), host(rows, 512, 37), host(rows, 512, 47)]
                for source, destination in zip(gold, inputs, strict=True):
                    ttnn.copy_host_to_device_tensor(upload(source, device=False), destination)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            for name, source, count in zip(('q', 'k', 'v'), gold, (16, 4, 4), strict=True):
                for chip, shard in enumerate(ttnn.get_device_tensors(heads[name])):
                    sequence = source.shape[2]
                    expected = source[chip:chip + 1].reshape(1, sequence, count, 128).transpose(1, 2)
                    if not torch.equal(ttnn.to_torch(shard), expected):
                        raise AssertionError(f'Native {name} layout differs: rows={rows}, chip={chip}')
                    report['checks'].append(dict(case=case, rows=rows, head=name, chip=chip, exact=True, traced=case == 4))
            for chip, shard in enumerate(ttnn.get_device_tensors(merged)):
                if not torch.equal(ttnn.to_torch(shard), gold[0][chip:chip + 1]):
                    raise AssertionError('Native head concatenation changes query elements')
                report['checks'].append(dict(case=case, rows=rows, head='merged', chip=chip, exact=True, traced=case == 4))
            print(json.dumps(dict(case=case, rows=rows, exact=True, traced=case == 4)), flush=True)
            if trace is not None:
                ttnn.release_trace(mesh, trace)
                trace = None
            release_owned(ttnn, owned)
            owned.clear()
        report['passed'] = len(report['checks']) == 40
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        release_owned(ttnn, owned)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
