"""Weight-free explicit-slot state DMA audit; no model or throughput qualification."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from gdn_slot_copy import copy_slot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args()
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_HARDWARE_TESTS') == '1' or os.environ.get('QWEN_CARDS_ALLOCATED') == '1'
            or Path('/dev/tenstorrent').exists() or options.output.exists()):
        raise ValueError('Fresh device-free simulator evidence required')
    import torch
    import ttnn

    names = ('gdn-slot-copy-probe.py', 'gdn_slot_copy.py', 'gdn_slot_copy.cpp',
             'gdn_state_copy.py', 'attention_batch.py')

    def fingerprints():
        return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}

    report = dict(passed=False, closed_cleanly=False, backend='simulator', checks=[],
        sources=fingerprints(), performance_qualified=False, model_integrated=False)
    mesh, trace, owned = None, None, []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=1048576)
        mesh.enable_program_cache()

        def values(full, seed):
            shapes = [(8 if full else 1, 24, 128, 128)] + [(1, 8 if full else 1, 5120)] * 4
            result = []
            for index, shape in enumerate(shapes):
                size = 1
                for dimension in shape:
                    size *= dimension
                positive = ((torch.arange(size).reshape(shape) + seed + index * 11) % 119 + 1).bfloat16()
                result.append(torch.cat((positive, -positive), dim=0))
            return result

        def upload(values):
            result = []
            for value in values:
                tensor = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
                owned.append(tensor)
                result.append(tensor)
            return result

        full, compact = upload(values(True, 0)), upload(values(False, 1))

        def bindings():
            return [[shard.buffer_address() for shard in ttnn.get_device_tensors(tensor)]
                    for tensor in owned]

        initial_bindings = bindings()

        def write(device, host):
            for tensor, value in zip(device, host, strict=True):
                payload = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
                ttnn.copy_host_to_device_tensor(payload, tensor)

        def compare(device, expected, slot, direction, phase, role):
            ttnn.synchronize_device(mesh)
            for index, (tensor, value) in enumerate(zip(device, expected, strict=True)):
                shards = ttnn.get_device_tensors(tensor)
                if len(shards) != 2:
                    raise AssertionError('Both chips required')
                for chip, (shard, golden) in enumerate(zip(shards, value.chunk(2, dim=0), strict=True)):
                    actual = ttnn.to_torch(shard).contiguous().view(torch.int16)
                    exact = torch.equal(actual, golden.contiguous().view(torch.int16))
                    report['checks'].append(dict(slot=slot, direction=direction, phase=phase,
                        role=role, tensor=index, chip=chip, exact=exact))
                    if not exact:
                        raise AssertionError('State DMA changed the wrong row or failed exact copy')

        for slot in range(8):
            for direction in ('load', 'store'):
                source, destination = (full, compact) if direction == 'load' else (compact, full)

                def operation():
                    copy_slot(source, destination, slot)

                operation()
                ttnn.synchronize_device(mesh)
                trace, _ = capture_operation(ttnn, mesh, operation)
                for ordinal, phase in enumerate(('eager', 'replay', 'changed-replay')):
                    source_host = values(direction == 'load', slot * 7 + ordinal * 23)
                    destination_host = values(direction != 'load', 87 + ordinal)
                    expected = [value.clone() for value in destination_host]
                    for index in range(5):
                        for chip in range(2):
                            source_local = source_host[index].chunk(2, dim=0)[chip]
                            target_local = expected[index].chunk(2, dim=0)[chip]
                            if index == 0:
                                if direction == 'load':
                                    target_local[:] = source_local[slot:slot + 1]
                                else:
                                    target_local[slot:slot + 1] = source_local
                            elif direction == 'load':
                                target_local[:] = source_local[:, slot:slot + 1]
                            else:
                                target_local[:, slot:slot + 1] = source_local
                    write(source, source_host)
                    write(destination, destination_host)
                    if phase == 'eager':
                        operation()
                    else:
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    compare(source, source_host, slot, direction, phase, 'source')
                    compare(destination, expected, slot, direction, phase, 'destination')
                    if bindings() != initial_bindings:
                        raise AssertionError('Persistent state addresses changed')
                ttnn.release_trace(mesh, trace)
                trace = None
                print(json.dumps(dict(slot=slot, direction=direction, passed=True)), flush=True)
        report['sources_after'] = fingerprints()
        if report['sources_after'] != report['sources']:
            raise AssertionError('Probe sources changed')
        report['passed'] = True
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                for tensor in reversed(owned):
                    ttnn.deallocate(tensor)
                ttnn.close_mesh_device(mesh)
                report['closed_cleanly'] = True
        finally:
            options.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
