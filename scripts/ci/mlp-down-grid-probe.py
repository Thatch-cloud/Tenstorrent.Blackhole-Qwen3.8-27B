"""Synthetic native T16 MLP-down grids; no throughput or full-model claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_CARDS_ALLOCATED') == '1' or options.output.exists()):
        raise ValueError('Simulator-only fresh output required')
    import torch
    import ttnn
    from attention_batch import capture_operation
    from mlp_down_grid import widen
    from models.demos.blackhole.qwen36.tt import tp_common as native

    paths = [Path(__file__), Path(__file__).with_name('mlp_down_grid.py'),
        Path(__file__).with_name('attention_batch.py'), Path(native.__file__)]
    def hashes():
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report = dict(passed=False, closed_cleanly=False, backend='simulator', sources=hashes(),
        checks=[], immutable_checks=[], negative_controls=[], hardware_qualified=False,
        timing_qualified=False, scope=__doc__, projection='mlp_down', rows=16, k=8704, n=5120,
        grids=dict(control=[11, 3], candidate=[11, 8]), output_placement='L1', weights='bfloat8_b')
    mesh, trace, owned = None, None, []
    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)
    def retain(value):
        owned.append(value)
        return value
    def read(value):
        return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(value)]
    def digest(value):
        return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    def compare(expected, actual, label):
        if len(expected) != 2 or len(actual) != 2:
            raise AssertionError('Both chips required')
        for chip, (reference, observed) in enumerate(zip(expected, actual, strict=True)):
            if not torch.isfinite(observed).all() or not torch.equal(reference, observed):
                raise AssertionError('Native MLP-down numerical mismatch')
            report['checks'].append(dict(label=label, chip=chip, exact=True))
    try:
        save('open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        generator = torch.Generator().manual_seed(20260918)
        def upload(value, dtype, memory, device=mesh):
            return ttnn.from_torch(value, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                memory_config=memory, mesh_mapper=mapper)
        save('synthetic_operands')
        weight = retain(upload(torch.randn((2, 1, 8704, 5120), generator=generator) * .01,
            ttnn.bfloat8_b, ttnn.DRAM_MEMORY_CONFIG))
        hosts = [(torch.randn((2, 1, 16, 8704), generator=generator) * .125).bfloat16() for seed in range(3)]
        inputs = [retain(upload(value, ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG)) for value in hosts]
        immutable_weights = [digest(value) for value in read(weight)]
        program = native.create_matmul_1d_decode_progcfg(16, 8704, 5120, num_cores=33, grid_w=11)
        candidate_program = widen(ttnn, program)
        compute = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        def project(value, selected):
            return ttnn.linear(value, weight, program_config=selected, compute_kernel_config=compute,
                memory_config=ttnn.L1_MEMORY_CONFIG)
        references = []
        def audit_input(index, label):
            for chip, observed in enumerate(read(inputs[0] if label.startswith('replay') else inputs[index])):
                if not torch.equal(observed, hosts[index][chip:chip + 1]):
                    raise AssertionError('Projection mutated its activation')
                report['immutable_checks'].append(dict(label=label, operand='activation', chip=chip, exact=True))
        save('eager')
        for index, value in enumerate(inputs):
            reference = project(value, program)
            references.append(read(reference))
            ttnn.deallocate(reference)
            actual = project(value, candidate_program)
            compare(references[index], read(actual), 'eager_' + str(index))
            ttnn.deallocate(actual)
            audit_input(index, 'eager_' + str(index))
        poison = ttnn.from_torch(torch.full((2, 1, 16, 5120), float('nan'), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        payloads = [ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            for value in hosts]
        save('capture')
        trace, output = capture_operation(ttnn, mesh, lambda: project(inputs[0], candidate_program))
        retain(output)
        for index in (1, 2, 0):
            label = 'replay_' + str(index)
            ttnn.copy_host_to_device_tensor(payloads[index], inputs[0])
            ttnn.copy_host_to_device_tensor(poison, output)
            ttnn.synchronize_device(mesh)
            if not all(torch.isnan(value).all() for value in read(output)):
                raise AssertionError('Negative control must poison every output')
            report['negative_controls'].append(dict(label=label, poisoned=True))
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            compare(references[index], read(output), label)
            audit_input(index, label)
        for chip, value in enumerate(read(weight)):
            if digest(value) != immutable_weights[chip]:
                raise AssertionError('Compressed weight values changed')
            report['immutable_checks'].append(dict(label='complete', operand='weight', chip=chip, exact=True))
        report['sources_after'] = hashes()
        if report['sources'] != report['sources_after'] or len(report['checks']) != 12:
            raise AssertionError('Complete unchanged-source comparison required')
    finally:
        if mesh is not None:
            ttnn.synchronize_device(mesh)
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            for value in reversed(owned):
                ttnn.deallocate(value)
            ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
        save('closed')
    report['passed'] = True
    save('complete')


if __name__ == '__main__':
    main()
