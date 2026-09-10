"""Simulator controls for exact physical BF16 comparison; not model or throughput evidence."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from tensor_bit_compare import prepare


SOURCES = ('tensor-bit-compare-probe.py', 'tensor_bit_compare.py', 'tensor_bit_compare.cpp',
    'attention_batch.py', 'feature_projection.py', 'gdn_multitoken_conv.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if options.output.exists():
        raise ValueError('Fresh comparator evidence required')
    import torch
    import ttnn
    root = Path(__file__).parent
    def hashes():
        return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}
    report = dict(passed=False, closed_cleanly=False, backend='simulator', checks=[], sources=hashes())
    mesh, owned, traces = None, [], []
    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(stage=stage)), flush=True)
    try:
        save('open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        poison = ttnn.from_torch(torch.full((1, 1, 32), 0xffffffff, dtype=torch.uint32),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        fixtures = []
        for width in (32, 2048):
            base = (torch.arange(2 * width).reshape(2, 1, width).float() / 100).bfloat16()
            changed = base.clone()
            for chip in range(2):
                for worker in range(min(32, width // 32)):
                    page = worker + (32 if width == 2048 and chip == 0 else 0)
                    column = page * 32 + (31 if chip == 0 else 0)
                    changed.view(torch.int16)[chip, 0, column] ^= 1
            staged = [ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                pad_value=padding, mesh_mapper=mapper) for value, padding in
                ((base, float('nan')), (base, 0), (changed, float('nan')))]
            left, right = [ttnn.from_torch(base, device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, pad_value=float('nan'), memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper) for unused in range(2)]
            output = ttnn.from_torch(torch.zeros((1, 1, 32), dtype=torch.uint32), device=mesh,
                dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            owned.extend((left, right, output))
            expected_inputs = []
            for value, padding in ((base, float('nan')), (base, 0), (changed, float('nan'))):
                physical = torch.full((2, 32, width), padding, dtype=torch.bfloat16)
                physical[:, :1] = value
                expected_inputs.append(physical)
            fixtures.append((width, staged, left, right, output, expected_inputs,
                prepare(ttnn, mesh, left, right, output)))
        bindings = [addresses(ttnn, value) for value in owned]
        def run_case(fixture, case, mode, repetition, trace=None):
            width, staged, left, right, output, expected_inputs, operation = fixture
            save(f'{width}_{mode}_{case}_{repetition}')
            ttnn.copy_host_to_device_tensor(staged[0], left)
            ttnn.copy_host_to_device_tensor(staged[case], right)
            ttnn.copy_host_to_device_tensor(poison, output)
            for shard in ttnn.get_device_tensors(output):
                if not torch.all(ttnn.to_torch(shard).to(torch.int64) == 0xffffffff):
                    raise AssertionError('All worker counters must be poisoned before execution')
            if trace is None:
                operation()
                ttnn.synchronize_device(mesh)
            else:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            expected = torch.zeros(32, dtype=torch.int64)
            for page in range(width // 32):
                if case == 1:
                    expected[page % 32] += 496
            if case == 2:
                expected[:min(32, width // 32)] = 1
            for chip, shard in enumerate(ttnn.get_device_tensors(output)):
                actual = ttnn.to_torch(shard).reshape(-1).to(torch.int64)
                if not torch.equal(actual, expected):
                    raise AssertionError(f'Incorrect exact mismatch counts: {width=} {case=} {chip=} {actual.tolist()=}')
                for tensor, source in ((left, expected_inputs[0]), (right, expected_inputs[case])):
                    host = ttnn.from_device(ttnn.get_device_tensors(tensor)[chip])
                    actual_bits = ttnn.to_torch(host.reshape(host.padded_shape)).view(torch.int16)
                    expected_bits = source[chip:chip + 1].view(torch.int16)
                    if not torch.equal(actual_bits, expected_bits):
                        raise AssertionError('Comparator modified physical input bits')
                report['checks'].append(dict(width=width, case=case, mode=mode, repetition=repetition,
                    chip=chip, exact=True, inputs_unchanged=True, poisoned_counters_replaced=True))
            if bindings != [addresses(ttnn, value) for value in owned]:
                raise AssertionError('Comparator bindings changed')
        for fixture in fixtures:
            for case in range(3):
                run_case(fixture, case, 'eager', None)
        for fixture in fixtures:
            trace, unused = capture_operation(ttnn, mesh, fixture[-1])
            traces.append(trace)
            for repetition, case in enumerate((0, 1, 2, 0)):
                run_case(fixture, case, 'replay', repetition, trace)
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                for trace in traces:
                    ttnn.release_trace(mesh, trace)
                release_owned(ttnn, owned)
                ttnn.close_mesh_device(mesh)
                report['closed_cleanly'] = True
        finally:
            report['sources_after'] = hashes()
            if report['sources_after'] != report['sources'] or not report['closed_cleanly']:
                report['passed'] = False
            save('complete' if report['passed'] else 'failed')


if __name__ == '__main__':
    main()
