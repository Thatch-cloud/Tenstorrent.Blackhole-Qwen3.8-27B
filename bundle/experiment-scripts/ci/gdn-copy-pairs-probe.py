"""Synthetic T16 recurrence copy scheduling; no model weights or timing claims."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_CARDS_ALLOCATED') == '1' or options.output.exists()):
        raise ValueError('Simulator-only execution and a fresh report required')
    import torch
    import ttnn
    import gdn_copy_pairs
    import gdn_vsplit_norm_batch as baseline
    from attention_batch import capture_operation
    from gdn_multitoken import HASHES, KERNEL_ROOT
    from gdn_vsplit_prepared import PreparedVSplit

    root = Path(os.environ['TT_METAL_HOME'])
    kernels = gdn_copy_pairs.load_kernels(root)
    native = baseline.load_kernels(root)
    if kernels['norm_gate'] != native['norm_gate']:
        raise AssertionError('Norm must not change')
    for role in ('reader', 'writer'):
        if kernels['recurrence'][role] != native['recurrence'][role]:
            raise AssertionError('Dataflow must not change')
    directory = Path(__file__).parent
    def hashes():
        paths = list(directory.glob('*.py')) + [root / KERNEL_ROOT / name for name in HASHES]
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report = dict(passed=False, closed_cleanly=False, backend='simulator', sources=hashes(),
        checks=[], immutable_checks=[], scope=__doc__, hardware_qualified=False, timing_qualified=False,
        rows=16, norm_unchanged=True, dataflow_unchanged=True)
    mesh, inputs, prepared, traces = None, [], [], []
    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)
    def fixture(seed):
        generator = torch.Generator().manual_seed(389100 + seed)
        return [(torch.randn((2, 16, 5120), generator=generator) * .1).bfloat16(),
            torch.sigmoid(torch.randn((2, 16, 24), generator=generator)).bfloat16(),
            (-torch.rand((2, 16, 24), generator=generator)).bfloat16(),
            (torch.randn((2, 24, 128, 128), generator=generator) * .01).bfloat16(),
            torch.randn((2, 16, 3072), generator=generator).bfloat16(),
            (1 + torch.randn((2, 1, 128), generator=generator) * .1).bfloat16()]
    def read(tensor):
        shards = ttnn.get_device_tensors(tensor)
        if len(shards) != 2:
            raise AssertionError('Both chips required')
        return [ttnn.to_torch(shard).clone() for shard in shards]
    def check(control, candidate, host, mode):
        for operand, (reference, observed) in enumerate(zip(control, candidate, strict=True)):
            for chip, (expected, actual) in enumerate(zip(read(reference), read(observed), strict=True)):
                if not torch.isfinite(expected).all() or not torch.equal(expected, actual):
                    raise AssertionError(f'Output mismatch {mode=} {operand=} {chip=}')
                report['checks'].append(dict(mode=mode, operand=operand, chip=chip, exact=True))
        for operand, (tensor, expected) in enumerate(zip(inputs, host, strict=True)):
            for chip, (actual, reference) in enumerate(zip(read(tensor), expected.chunk(2, dim=0), strict=True)):
                if not torch.equal(actual, reference):
                    raise AssertionError(f'Input mutation {mode=} {operand=} {chip=}')
                report['immutable_checks'].append(dict(mode=mode, operand=operand, chip=chip, exact=True))
    try:
        save('open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576,
            trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        host = fixture(0)
        for value in host:
            inputs.append(ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper))
        def prepare():
            operation = PreparedVSplit(mesh, *inputs[:4], z=inputs[4], norm_w=inputs[5],
                root=root, experimental=True, batch_norm=True, output_memory=ttnn.L1_MEMORY_CONFIG)
            prepared.append(operation)
            return operation
        control = prepare()
        with patch.object(baseline, 'load_kernels', return_value=kernels):
            candidate = prepare()
        save('eager')
        reference, actual = control.run(), candidate.run()
        ttnn.synchronize_device(mesh)
        check(reference, actual, host, 'eager')
        save('capture')
        for operation in prepared:
            trace, unused = capture_operation(ttnn, mesh, operation.run)
            traces.append(trace)
        for seed in (1, 2, 0):
            save('replay_' + str(seed))
            host = fixture(seed)
            for value, destination in zip(host, inputs, strict=True):
                payload = ttnn.from_torch(value, dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
                ttnn.copy_host_to_device_tensor(payload, destination)
            ttnn.synchronize_device(mesh)
            for trace in traces:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            check(reference, actual, host, 'replay_' + str(seed))
        if len(report['checks']) != 24 or len(report['immutable_checks']) != 48:
            raise AssertionError('Incomplete comparison matrix')
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
                for operation in reversed(prepared):
                    operation.close()
                for tensor in reversed(inputs):
                    ttnn.deallocate(tensor)
                ttnn.close_mesh_device(mesh)
                report['closed_cleanly'] = True
        finally:
            report['sources_after'] = hashes()
            report['passed'] = report['passed'] and report['closed_cleanly'] and report['sources'] == report['sources_after']
            save('complete' if report['passed'] else 'failed')


if __name__ == '__main__':
    main()
