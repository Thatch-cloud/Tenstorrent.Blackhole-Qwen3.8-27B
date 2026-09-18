"""Weight-free two-chip sliding K/V transport qualification; no throughput claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from draft_kv_slide import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or Path('/dev/tenstorrent').exists() or options.output.exists()):
        raise ValueError('Fresh device-free simulator run required')
    import torch
    import ttnn

    names = (Path(__file__).name, 'draft_kv_slide.py', 'draft_kv_slide.cpp')
    report = dict(passed=False, closed_cleanly=False, checks=[], backend='simulator',
        scope=__doc__, model_integrated=False, performance_qualified=False,
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names})
    mesh, owned, traces = None, [], []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=1048576)
        mesh.enable_program_cache()

        def upload(value, device=True):
            options = dict(device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG) if device else {}
            tensor = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0), **options)
            if device:
                owned.append(tensor)
            return tensor

        def compare(tensor, expected, **metadata):
            shards = ttnn.get_device_tensors(tensor)
            if len(shards) != 2:
                raise AssertionError('Both device shards required')
            for chip, shard in enumerate(shards):
                actual = ttnn.to_torch(shard).contiguous().view(torch.int16)
                golden = expected[chip:chip + 1].contiguous().view(torch.int16)
                exact = torch.equal(actual, golden)
                report['checks'].append(dict(metadata, chip=chip, exact=exact))
                if not exact:
                    raise AssertionError('Sliding cache mismatch: ' + str(metadata))

        base = torch.arange(2 * 4 * 2048 * 128).reshape(2, 4, 2048, 128)
        small = torch.arange(2 * 4 * 32 * 128).reshape(2, 4, 32, 128)
        for history, prefix in ((31, 2), (2047, 2), (2048, 1), (2048, 16), (2048, 32)):
            print(json.dumps(dict(stage='case', history=history, prefix=prefix)), flush=True)
            active_host = (base % 127 + 1).bfloat16()
            delta_host = (small % 113 + 129).bfloat16()
            active, delta = upload(active_host), upload(delta_host)
            spare = upload(torch.full_like(active_host, -257))
            operation = prepare(mesh, active, delta, spare, history_rows=history, prefix=prefix)
            operation()
            ttnn.synchronize_device(mesh)
            eager = torch.cat((active_host[:, :, :history], delta_host[:, :, :prefix]), dim=2)[:, :, -2048:]
            eager = torch.cat((eager, torch.zeros((2, 4, 2048 - eager.shape[2], 128), dtype=torch.bfloat16)), dim=2)
            for tensor, host, name in ((active, active_host, 'active_unchanged'),
                    (delta, delta_host, 'delta_unchanged'), (spare, eager, 'sliding_output')):
                compare(tensor, host, history=history, prefix=prefix, ordinal=-1, name=name)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            traces.append(trace)
            try:
                operation()
            finally:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
            previous = None
            for ordinal, seed in enumerate((0, 1, 0)):
                active_host = ((base + seed * 17) % 127 + 1).bfloat16()
                delta_host = ((small + seed * 23) % 113 + 129).bfloat16()
                delta_host[:, :, prefix:] = 511
                for host, target in ((active_host, active), (delta_host, delta),
                        (torch.full_like(active_host, -257), spare)):
                    ttnn.copy_host_to_device_tensor(upload(host, device=False), target)
                expected = torch.cat((active_host[:, :, :history], delta_host[:, :, :prefix]), dim=2)[:, :, -2048:]
                expected = torch.cat((expected, torch.zeros((2, 4, 2048 - expected.shape[2], 128), dtype=torch.bfloat16)), dim=2)
                if previous is not None and torch.equal(previous, expected):
                    raise AssertionError('Changed-input negative control is insensitive')
                previous = expected.clone()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                for tensor, host, name in ((active, active_host, 'active_unchanged'),
                        (delta, delta_host, 'delta_unchanged'), (spare, expected, 'sliding_output')):
                    compare(tensor, host, history=history, prefix=prefix, ordinal=ordinal, name=name)
            ttnn.release_trace(mesh, trace)
            traces.remove(trace)
            for value in reversed(owned):
                ttnn.deallocate(value)
            owned.clear()
        report['passed'] = len(report['checks']) == 120 and all(check['exact'] for check in report['checks'])
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                for trace in traces:
                    ttnn.release_trace(mesh, trace)
                for value in reversed(owned):
                    ttnn.deallocate(value)
                ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
        finally:
            options.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
