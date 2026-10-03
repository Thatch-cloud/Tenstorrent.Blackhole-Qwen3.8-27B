"""Weight-free tiled draft-tail correctness and replay; not model or speed acceptance."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from dspark_native_cached_layer import append_queries as reference
from frozen_draft_tail import append_queries as candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_SIM_PACKER_ZERO_GRAFT') != '1'
            or Path('/dev/tenstorrent').exists() or options.output.exists()):
        raise ValueError('Fresh device-free simulator with pinned packer compatibility required')
    import torch
    import ttnn

    sources = ('draft-tail-probe.py', 'frozen_draft_tail.py', 'dspark_native_cached_layer.py',
        'dspark_full_attention.py', 'dspark_projection.py', 'attention_batch.py')
    checks, owned, traces = [], [], []
    report = dict(passed=False, closed_cleanly=False, backend='simulator', checks=checks,
        performance_qualified=False, model_integrated=False, scope=__doc__,
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources})
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=1048576)
        mesh.enable_program_cache()

        def retain(value):
            owned.append(value)
            return value

        def host(value):
            return ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))

        def upload(value):
            return retain(ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)))

        def compare(value, expected, name, position, proposals, seed):
            shards = ttnn.get_device_tensors(value)
            if len(shards) != 2:
                raise AssertionError('Both chips required')
            for chip, shard in enumerate(shards):
                actual = ttnn.to_torch(shard).contiguous().view(torch.int16)
                golden = expected[chip:chip + 1].contiguous().view(torch.int16)
                exact = torch.equal(actual, golden)
                checks.append(dict(name=name, position=position, proposals=proposals, seed=seed,
                    chip=chip, exact=exact))
                if not exact:
                    raise AssertionError('Bitwise draft-tail mismatch: ' + name)

        for position in (64, 96):
            for proposals in (7, 15):
                def inputs(seed):
                    history = ((torch.arange(2 * 4 * position * 128).reshape(2, 4, position, 128)
                        + seed * 17) % 127 - 63).bfloat16()
                    queries = ((torch.arange(2 * 4 * 32 * 128).reshape(2, 4, 32, 128)
                        + seed * 23) % 113 - 56).bfloat16()
                    queries[:, :, proposals:] = 999
                    padded = ((position + proposals + 63) // 64) * 64
                    expected = torch.zeros(2, 4, padded, 128, dtype=torch.bfloat16)
                    expected[:, :, :position] = history
                    expected[:, :, position:position + proposals] = queries[:, :, :proposals]
                    return history, queries, expected

                history, queries, expected = inputs(0)
                device_history, device_queries = upload(history), upload(queries)
                outputs = []
                for name, operation in (('reference', reference), ('candidate', candidate)):
                    def invoke(operation=operation):
                        return operation(ttnn, device_history, device_queries, retain,
                            position=position, proposals=proposals)
                    warm = invoke()
                    compare(warm, expected, name + '_eager', position, proposals, 0)
                    trace, output = capture_operation(ttnn, mesh, invoke)
                    traces.append(trace)
                    outputs.append((name, trace, output))
                for seed in (1, 0, 2):
                    history, queries, expected = inputs(seed)
                    ttnn.copy_host_to_device_tensor(host(history), device_history)
                    ttnn.copy_host_to_device_tensor(host(queries), device_queries)
                    for name, trace, output in outputs:
                        ttnn.execute_trace(mesh, trace, blocking=True)
                        compare(output, expected, name + '_replay', position, proposals, seed)
                    compare(device_history, history, 'history_unchanged', position, proposals, seed)
                    compare(device_queries, queries, 'queries_unchanged', position, proposals, seed)
                ttnn.synchronize_device(mesh)
                for trace in traces:
                    ttnn.release_trace(mesh, trace)
                traces.clear()
                for value in reversed(owned):
                    ttnn.deallocate(value)
                owned.clear()
                print(json.dumps(dict(stage='tail', position=position, proposals=proposals, checks=len(checks))), flush=True)
        report['passed'] = len(checks) == 112 and all(check['exact'] for check in checks)
        if not report['passed']:
            raise AssertionError('Complete eager and changed-input replay matrix required')
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
