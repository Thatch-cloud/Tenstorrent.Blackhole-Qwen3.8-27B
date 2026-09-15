"""Weight-free two-chip simulator test; not trace, model or throughput acceptance."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from history_append_dma import prepare
from history_append_plan import BankAppendPlanner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args()
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or Path('/dev/tenstorrent').exists() or options.output.exists()):
        raise ValueError('Fresh device-free simulator run required')
    import torch
    import ttnn

    capacity, initial = 160, 31
    actions = ((3, True), (15, True), (32, False), (1, True), (32, True), (16, True))
    sources = ('history-append-probe.py', 'history_append_plan.py', 'history_append_dma.py', 'history_append_dma.cpp')
    report = dict(passed=False, closed_cleanly=False, backend='simulator', scope=__doc__, checks=[],
        performance_qualified=False, model_integrated=False, trace_qualified=False,
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources})
    mesh, owned = None, []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=1048576)
        mesh.enable_program_cache()

        def upload(value):
            result = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
            owned.append(result)
            return result

        def compare(value, expected, name, ordinal):
            for chip, shard in enumerate(ttnn.get_device_tensors(value)):
                actual = ttnn.to_torch(shard).contiguous().view(torch.int16)
                golden = expected[chip:chip + 1].contiguous().view(torch.int16)
                exact = torch.equal(actual, golden)
                differences = torch.nonzero(actual != golden)
                mismatches = [dict(index=index.tolist(), actual_bits=int(actual[tuple(index)]),
                    expected_bits=int(golden[tuple(index)])) for index in differences[:16]]
                report['checks'].append(dict(ordinal=ordinal, chip=chip, name=name, exact=exact,
                    mismatch_count=len(differences), mismatches=mismatches))
                if not exact:
                    raise AssertionError('Entire bank differs bitwise: ' + name)

        expected = (torch.arange(2 * 4 * capacity * 128).reshape(2, 4, capacity, 128) % 127 - 63).bfloat16()
        expected[:, :, initial:] = 0
        active, spare = upload(expected), upload(expected.clone())
        bindings = sorted(shard.buffer_address() for tensor in (active, spare) for shard in ttnn.get_device_tensors(tensor))
        planner = BankAppendPlanner(initial, capacity)
        for ordinal, (prefix, commit) in enumerate(actions):
            delta = ((torch.arange(2 * 4 * 32 * 128).reshape(2, 4, 32, 128) + ordinal * 17) % 239 - 119).bfloat16()
            delta[1] = -delta[1]
            delta[:, :, prefix:] = 257
            device_delta = upload(delta)
            uploaded = torch.cat([ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(device_delta)], dim=0)
            changed = uploaded.contiguous().view(torch.int16) != delta.contiguous().view(torch.int16)
            allowed = (uploaded == 0) & (delta == 0)
            if torch.any(changed & ~allowed):
                raise AssertionError('Upload changed nonzero input bits')
            report.setdefault('upload_zero_sign_changes', []).append(int(changed.sum()))
            delta = uploaded.clone()
            compare(device_delta, delta, 'delta_device_baseline_exact', ordinal)
            plan = planner.prepare(prefix)
            prepared = expected.clone()
            prepared[:, :, plan.position:plan.position + prefix] = delta[:, :, :prefix]
            prepared[:, :, plan.position + prefix:] = 0
            prepare(mesh, active, device_delta, spare, plan)()
            ttnn.synchronize_device(mesh)
            compare(active, expected, 'active_unchanged', ordinal)
            compare(spare, prepared, 'prepared_spare', ordinal)
            compare(device_delta, delta, 'delta_unchanged', ordinal)
            planner.resolve(plan, commit=commit)
            if commit:
                active, spare = spare, active
                expected = prepared
            if sorted(shard.buffer_address() for tensor in (active, spare)
                    for shard in ttnn.get_device_tensors(tensor)) != bindings:
                raise AssertionError('Persistent bank addresses changed')
            print(json.dumps(dict(stage='append', ordinal=ordinal, prefix=prefix, commit=commit)), flush=True)
        report['passed'] = len(report['checks']) == 48 and all(check['exact'] for check in report['checks'])
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                for value in reversed(owned):
                    ttnn.deallocate(value)
                ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
        finally:
            options.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
