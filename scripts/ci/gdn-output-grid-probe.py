"""Simulator comparison of native versus wider T16 GDN output projection grids."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from contextlib import nullcontext


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_CARDS_ALLOCATED') == '1' or options.output.exists()):
        raise ValueError('Simulator-only execution and a fresh report required')
    import torch
    import ttnn
    from attention_batch import capture_operation
    from output_projection_grid import widen
    from models.demos.blackhole.qwen36.tt import tp_common as native

    paths = [Path(__file__), Path(__file__).with_name('output_projection_grid.py'), Path(__file__).with_name('attention_batch.py'),
        Path(native.__file__)]
    def hashes():
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report = dict(passed=False, closed_cleanly=False, backend='simulator', sources=hashes(),
        checks=[], scope='One synthetic local projection on each chip; excludes fabric reduction and full model',
        hardware_qualified=False, timing_qualified=False)
    mesh, owned, traces = None, [], []
    def retain(value):
        owned.append(value)
        return value
    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)
    try:
        save('open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576,
            trace_region_size=134217728)
        mesh.enable_program_cache()
        generator = torch.Generator().manual_seed(20260913)
        mapper = ttnn.ReplicateTensorToMesh(mesh)
        def upload(value, dtype=ttnn.bfloat16):
            return retain(ttnn.from_torch(value, device=mesh, dtype=dtype,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper))
        save('synthetic_operands')
        weight = upload(torch.randn((3072, 5120), generator=generator) * .02, ttnn.bfloat8_b)
        hosts = [(torch.randn((1, 16, 3072), generator=generator) * .125).bfloat16() for index in range(3)]
        inputs = [upload(value) for value in hosts]
        program = native.create_matmul_1d_decode_progcfg(16, 3072, 5120, num_cores=33, grid_w=11)
        def control(value, operand):
            return native.matmul_1d_decode(value, operand, program, native.COMPUTE_HIFI2,
                out_memory_config=ttnn.DRAM_MEMORY_CONFIG)
        candidate_program = widen(ttnn, program)
        report['grids'] = dict(control=[11, 3], candidate=[11, 6])
        def candidate(value):
            return native.matmul_1d_decode(value, weight, candidate_program, native.COMPUTE_HIFI2,
                out_memory_config=ttnn.DRAM_MEMORY_CONFIG)
        def compare(expected, actual, label):
            expected_shards, actual_shards = ttnn.get_device_tensors(expected), ttnn.get_device_tensors(actual)
            if len(expected_shards) != 2 or len(actual_shards) != 2:
                raise AssertionError('Both chips required')
            for chip, (reference, candidate) in enumerate(zip(expected_shards, actual_shards, strict=True)):
                reference, candidate = ttnn.to_torch(reference), ttnn.to_torch(candidate)
                if not torch.isfinite(reference).all() or not torch.equal(reference, candidate):
                    raise AssertionError('Wider grid changed the native projection')
                report['checks'].append(dict(label=label, chip=chip, exact=True))
        with nullcontext():
            save('eager')
            references = [retain(control(value, weight)) for value in inputs]
            for index, value in enumerate(inputs):
                actual = retain(candidate(value))
                if actual.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                    raise AssertionError('Candidate changed output placement')
                compare(references[index], actual, 'eager_' + str(index))
            save('capture')
            trace, output = capture_operation(ttnn, mesh,
                lambda: candidate(inputs[0]))
            traces.append(trace)
            retain(output)
            for index in (1, 2, 0):
                payload = ttnn.from_torch(hosts[index], dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
                ttnn.copy_host_to_device_tensor(payload, inputs[0])
                ttnn.synchronize_device(mesh)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                compare(references[index], output, 'replay_' + str(index))
        report['native_configuration_unchanged'] = tuple(program.compute_with_storage_grid_size) == (11, 3)
        report['sources_after'] = hashes()
        if report['sources_after'] != report['sources'] or len(report['checks']) != 12:
            raise AssertionError('Complete unchanged-source matrix required')
    finally:
        if mesh is not None:
            ttnn.synchronize_device(mesh)
            for trace in traces:
                ttnn.release_trace(mesh, trace)
            for value in reversed(owned):
                ttnn.deallocate(value)
            ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
        save('closed')
    report['passed'] = True
    save('passed')


if __name__ == '__main__':
    main()
