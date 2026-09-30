"""Weight-free two-chip GDN slot isolation; no batching or performance qualification."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from gdn_slot_copy import copy_slot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or Path('/dev/tenstorrent').exists() or output.exists()):
        raise ValueError('Fresh device-free simulator run required')
    import torch
    import ttnn

    names = (Path(__file__).name, 'gdn_slot_copy.py', 'gdn_slot_copy.cpp', 'gdn_state_copy.py')
    report = dict(passed=False, closed_cleanly=False, backend='simulator', checks=[],
        scope=__doc__, model_integrated=False, performance_qualified=False,
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names})
    mesh, owned, traces = None, [], []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=1048576)
        mesh.enable_program_cache()

        def upload(host, device=True):
            options = dict(device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG) if device else {}
            tensor = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0), **options)
            if device:
                owned.append(tensor)
            return tensor

        def fixture(full, seed):
            shapes = [(16 if full else 2, 24, 128, 128)] + [(2, 8 if full else 1, 5120)] * 4
            return [((torch.arange(torch.Size(shape).numel()).reshape(shape) + seed * 19 + index * 7)
                % 127 + 1).bfloat16() for index, shape in enumerate(shapes)]

        for slot in (0, 1, 7):
            for direction in ('save', 'restore'):
                saving = direction == 'save'
                source_host, destination_host = fixture(saving, 0), fixture(not saving, 8)
                source = [upload(host) for host in source_host]
                destination = [upload(host) for host in destination_host]

                def check(ordinal):
                    for operand, (source_tensor, destination_tensor) in enumerate(zip(source, destination, strict=True)):
                        golden = destination_host[operand].clone()
                        for source_chip, destination_chip in zip(source_host[operand].chunk(2), golden.chunk(2), strict=True):
                            if operand == 0:
                                if saving:
                                    destination_chip.copy_(source_chip[slot:slot + 1])
                                else:
                                    destination_chip[slot:slot + 1].copy_(source_chip)
                            elif saving:
                                destination_chip.copy_(source_chip[:, slot:slot + 1])
                            else:
                                destination_chip[:, slot:slot + 1].copy_(source_chip)
                        for name, tensor, expected in (('source_unchanged', source_tensor, source_host[operand]),
                                ('destination_all_slots', destination_tensor, golden)):
                            parts = ttnn.get_device_tensors(tensor)
                            if len(parts) != 2:
                                raise AssertionError('Both device shards required')
                            for chip, (part, reference) in enumerate(zip(parts, expected.chunk(2), strict=True)):
                                actual = ttnn.to_torch(part).contiguous().view(torch.int16)
                                exact = torch.equal(actual, reference.contiguous().view(torch.int16))
                                report['checks'].append(dict(slot=slot, direction=direction, ordinal=ordinal,
                                    operand=operand, name=name, chip=chip, exact=exact))
                                if not exact:
                                    raise AssertionError('Slot copy mismatch: ' + str(report['checks'][-1]))
                    print(json.dumps(dict(slot=slot, direction=direction, ordinal=ordinal,
                        checks=len(report['checks']))), flush=True)

                copy_slot(mesh, source, destination, slot=slot)
                ttnn.synchronize_device(mesh)
                check(-1)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                traces.append(trace)
                try:
                    copy_slot(mesh, source, destination, slot=slot)
                finally:
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                previous = None
                for ordinal, seed in enumerate((0, 1, 0)):
                    source_host, destination_host = fixture(saving, seed), fixture(not saving, seed + 8)
                    if previous is not None and torch.equal(previous, source_host[0]):
                        raise AssertionError('Changed-input negative control is insensitive')
                    previous = source_host[0].clone()
                    for hosts, tensors in ((source_host, source), (destination_host, destination)):
                        for host, tensor in zip(hosts, tensors, strict=True):
                            ttnn.copy_host_to_device_tensor(upload(host, device=False), tensor)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    check(ordinal)
                ttnn.release_trace(mesh, trace)
                traces.remove(trace)
                for tensor in reversed(owned):
                    ttnn.deallocate(tensor)
                owned.clear()
        report['passed'] = len(report['checks']) == 480 and all(check['exact'] for check in report['checks'])
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                for trace in traces:
                    ttnn.release_trace(mesh, trace)
                for tensor in reversed(owned):
                    ttnn.deallocate(tensor)
                ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
        finally:
            output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
