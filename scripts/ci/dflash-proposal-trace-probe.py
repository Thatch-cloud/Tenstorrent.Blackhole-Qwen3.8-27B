"""Simulator-only proposal buffer ownership and interleaved replay, not learned-model throughput."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace, MethodType
from itertools import product

from attention_batch import capture_operation
from dflash_device import DFlashDevice
from dflash_proposal_inputs import proposal_inputs
from dflash_proposal_trace import PreparedDFlashProposal
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    report = dict(passed=False, scope=__doc__, checks=[], hashes={name:
        hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('dflash_proposal_inputs.py', 'dflash_proposal_trace.py', 'dflash-proposal-trace-probe.py')})
    mesh = prepared = other_trace = None
    owned = []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=268435456)
        mesh.enable_program_cache()
        mapper = ttnn.ReplicateTensorToMesh(mesh)
        for block, initial_position in product((8, 32), (170, 4093)):
            def history_pattern(position):
                rows = min(position, 2048)
                value = torch.zeros((1, 1, 2048, 5120), dtype=torch.bfloat16)
                value[..., :rows, :] = (torch.arange(position - rows, position).reshape(1, 1, rows, 1) % 64 + position % 17).bfloat16()
                return value
            def upload(value):
                tensor = ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
                owned.append(tensor)
                return tensor
            device = SimpleNamespace(operations=ttnn, mesh=mesh, position=initial_position,
                history_rows=min(initial_position, 2048), block_rows=block,
                owned=[], history=upload(history_pattern(initial_position)), spare_history=upload(history_pattern(initial_position)),
                progress=lambda *args, **kwargs: None)
            device.temporaries = MethodType(DFlashDevice.temporaries, device)
            def execute(identifiers, history, mask, rope, *, retain, **kwargs):
                return tuple(retain(ttnn.clone(value, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                    for value in (identifiers, history, mask, *rope['q'], *rope['k']))
            device.execute_proposal = execute
            device.proposal_snapshot = lambda outputs: tuple(ttnn.to_torch(shard).clone() for output in outputs
                for shard in ttnn.get_device_tensors(output))
            def select(outputs, seed, count):
                context = outputs[1].shape[2] - 32
                host = proposal_inputs(seed, device.position, device.history_rows, block, context)
                history = torch.cat((history_pattern(device.position)[..., :context, :],
                    torch.zeros((1, 1, 32, 5120), dtype=torch.bfloat16)), dim=2)
                expected = (host['identifiers'], history, host['mask'], *host['rope']['q'], *host['rope']['k'])
                for output, reference in zip(outputs, expected, strict=True):
                    for chip, shard in enumerate(ttnn.get_device_tensors(output)):
                        if not torch.equal(ttnn.to_torch(shard).to(reference.dtype), reference):
                            raise AssertionError('Captured proposal buffer contains stale or misplaced input')
                        report['checks'].append(dict(block_rows=block, position=device.position, context=context,
                            chip=chip, exact=True))
                return tuple((seed + offset + 1) % 248320 for offset in range(count))
            device.select_proposal = select
            prepared = PreparedDFlashProposal(device, max_new_tokens=200)
            other_input = device.history
            def operation():
                return ttnn.add(other_input, 1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            warm = operation()
            ttnn.deallocate(warm)
            other_trace, other_output = capture_operation(ttnn, mesh, operation)
            owned.append(other_output)
            positions = (170, 178, 256, 257, 300) if initial_position == 170 else (4093, 4096, 4101, 4125, 4133)
            for repetition, position in enumerate(positions):
                device.position, device.history_rows = position, min(position, 2048)
                device.history, device.spare_history = device.spare_history, device.history
                payload = ttnn.from_torch(history_pattern(position), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
                ttnn.copy_host_to_device_tensor(payload, device.history)
                ttnn.execute_trace(mesh, other_trace, cq_id=0, blocking=True)
                other_before = device.proposal_snapshot((other_output,))
                result = prepared.propose(17 + repetition, block - 1)
                if len(result) != block - 1:
                    raise AssertionError('Incomplete proposal fixture')
                other_after = device.proposal_snapshot((other_output,))
                if any(not torch.equal(left, right) for left, right in zip(other_before, other_after, strict=True)):
                    raise AssertionError('Proposal replay corrupted a separately captured live output')
            if len(prepared.checks) != 5:
                raise AssertionError('Every updated input requires an eager-versus-replay audit')
            ttnn.release_trace(mesh, other_trace)
            other_trace = None
            prepared.close()
            prepared = None
            release_owned(ttnn, owned)
            owned.clear()
        report['passed'] = len(report['checks']) == 280
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            if other_trace is not None:
                ttnn.release_trace(mesh, other_trace)
            if prepared is not None:
                prepared.close()
            release_owned(ttnn, owned)
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
