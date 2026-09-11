"""Synthetic complete projected T16 GDN direct-state versus compact control; no throughput claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from gdn_native_slot_projected import execute as native_slot
from gdn_batched_conv import run_batched_projected as compact


SOURCES = ('gdn-native-projected-probe.py', 'gdn_native_slot_projected.py', 'gdn_native_slot_recurrence.py',
    'gdn_native_slot_windows.py', 'gdn_conv_windows.py', 'gdn_conv_windows.cpp', 'gdn_batched_conv.py', 'gdn_vsplit.py',
    'gdn_vsplit_norm_batch.py', 'gdn_vsplit_prefetch.py', 'gdn_multitoken.py',
    'gdn_multitoken_conv.py', 'attention_batch.py', 'feature_projection.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if options.output.exists():
        raise ValueError('Fresh simulator report required')
    import torch
    import ttnn

    directory = Path(__file__).parent
    root = directory.resolve().parents[1] / 'hardware-evidence.local/34009341359/qwen-hardware-inventory-34009341359/gdn-source'
    from gdn_vsplit_norm_batch import load_kernels
    load_kernels(root)
    def hashes():
        return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in SOURCES}
    report = dict(passed=False, closed_cleanly=False, backend='simulator', scope=__doc__,
        sources=hashes(), checks=[], immutable_checks=[], hardware_qualified=False)
    mesh, owned, traces = None, [], []
    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)
    def fixture(seed):
        generator = torch.Generator().manual_seed(384000 + seed)
        state = torch.randn((16, 24, 128, 128), generator=generator).to(torch.bfloat16)
        for chip in range(2):
            state[chip * 8] *= 0.01
            for slot in range(1, 8):
                state[chip * 8 + slot].fill_(32 * (seed + 1) + 8 * chip + slot)
        history = [torch.randn((2, 8, 5120), generator=generator).to(torch.bfloat16) * 0.1 for slot in range(4)]
        for slot, value in enumerate(history):
            for chip in range(2):
                for inactive in range(1, 8):
                    value[chip, inactive].fill_(32 * seed + 8 * chip + inactive + slot)
        projected = torch.randn((2, 16, 8256), generator=generator).to(torch.bfloat16) * 0.1
        taps = [torch.randn((2, 1, 5120), generator=generator).to(torch.bfloat16) * 0.1 for slot in range(4)]
        return [projected, state, *history, *taps,
            torch.zeros((2, 1, 24), dtype=torch.bfloat16),
            -torch.ones((2, 1, 24), dtype=torch.bfloat16),
            torch.ones((2, 1, 128), dtype=torch.bfloat16),
            state[[0, 8]].clone(), *(value[:, :1].clone() for value in history)]
    def upload(value):
        tensor = ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
        owned.append(tensor)
        return tensor
    def read(value):
        return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(value)]
    def compare_outputs(control, candidate, seed, mode):
        for operand, (expected, actual) in enumerate(zip(control, candidate, strict=True)):
            for chip, (reference, observed) in enumerate(zip(read(expected), read(actual), strict=True)):
                if not torch.equal(reference, observed):
                    raise AssertionError(f'Output mismatch {seed=} {mode=} {operand=} {chip=}')
                report['checks'].append(dict(seed=seed, mode=mode, operand=operand, chip=chip, exact=True))
    def immutable(inputs, host, seed, mode):
        for operand, (tensor, expected) in enumerate(zip(inputs, host, strict=True)):
            for chip, (actual, reference) in enumerate(zip(read(tensor), expected.chunk(2, dim=0), strict=True)):
                if not torch.equal(actual, reference):
                    raise AssertionError(f'Input mutation {seed=} {mode=} {operand=} {chip=}')
                report['immutable_checks'].append(dict(seed=seed, mode=mode, operand=operand, chip=chip, exact=True))
    try:
        save('open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        host = fixture(0)
        inputs = [upload(value) for value in host]
        bindings = [addresses(ttnn, value) for value in inputs]
        def run(candidate):
            if candidate:
                result = native_slot(ttnn, mesh, inputs[0], inputs[1:6], inputs[6:10],
                    inputs[10], inputs[11], inputs[12], root=root, experimental=True)
            else:
                result = compact(mesh, inputs[0], inputs[13], inputs[14:18], inputs[6:10],
                    inputs[10], inputs[11], inputs[12], None, ttnn, dma_windows=True,
                    packed_checkpoints=True, norm_batch=True, norm_source_root=root,
                    defer_conv_publication=True)
            owned.extend(result['owned'])
            return [result['output'], result['states'], *result['packed_conv_states']]
        save('warm_both_paths')
        control, candidate = run(False), run(True)
        ttnn.synchronize_device(mesh)
        compare_outputs(control, candidate, 0, 'eager')
        immutable(inputs, host, 0, 'eager')
        save('capture_both_paths')
        control_trace, control = capture_operation(ttnn, mesh, lambda: run(False))
        traces.append(control_trace)
        candidate_trace, candidate = capture_operation(ttnn, mesh, lambda: run(True))
        traces.append(candidate_trace)
        for seed in (0, 1, 2):
            save('replay_seed_' + str(seed))
            host = fixture(seed)
            for value, destination in zip(host, inputs, strict=True):
                source = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
                ttnn.copy_host_to_device_tensor(source, destination)
            reference = run(False)
            ttnn.synchronize_device(mesh)
            for trace in traces:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            compare_outputs(reference, control, seed, 'control_replay')
            compare_outputs(reference, candidate, seed, 'native_replay')
            immutable(inputs, host, seed, 'replay')
            if [addresses(ttnn, value) for value in inputs] != bindings:
                raise AssertionError('Persistent input bindings changed')
        if len(report['checks']) != 84 or len(report['immutable_checks']) != 144:
            raise AssertionError('Incomplete two-chip eager/replay matrix')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                for trace in reversed(traces):
                    ttnn.release_trace(mesh, trace)
                release_owned(ttnn, owned)
                ttnn.close_mesh_device(mesh)
                report['closed_cleanly'] = True
        finally:
            report['sources_after'] = hashes()
            report['passed'] = report['passed'] and report['closed_cleanly'] and report['sources'] == report['sources_after']
            save('complete' if report['passed'] else 'failed')


if __name__ == '__main__':
    main()
