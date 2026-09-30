"""Simulator-only real-weight HiFi2 execution screen, not quality or throughput acceptance."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from dspark_layer import linear as reference, pack_weight
from dspark_projection import tensor_digest
from dspark_projection_precision import linear as candidate
from dspark_weights import VerifiedWeights
from gdn_multitoken_conv import addresses, release_owned


PROJECTIONS = ('self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
    'self_attn.o_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight')
SOURCES = ('dspark-projection-precision-probe.py', 'dspark_projection_precision.py', 'dspark_layer.py',
    'dspark_projection.py', 'dspark_weights.py', 'dspark_checkpoint.py', 'dspark_intake.py',
    'attention_batch.py', 'gdn_multitoken_conv.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--projection', choices=PROJECTIONS, required=True)
    options = parser.parse_args()
    if not os.environ.get('TT_METAL_SIMULATOR') or options.output.exists():
        raise ValueError('Fresh simulator-only output required')
    import torch
    import ttnn

    sources = lambda: {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCES}
    report = dict(passed=False, closed_cleanly=False, backend='simulator', projection=options.projection,
        sources=sources(), differences=[], replay_checks=[], integrity_checks=[],
        component_execution_only=True, target_correctness_qualified=False, committed_tg=None)
    mesh = trace = None
    owned, transient = [], []
    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)
    def retain(value):
        transient.append(value)
        return value
    try:
        progress('load_one_verified_projection')
        with VerifiedWeights(options.checkpoint) as reader:
            name = 'layers.0.' + options.projection
            packed, sharded = pack_weight(options.projection, reader.tensor(name))
            if not sharded:
                raise ValueError('TP2 matrix required')
            report['weight_sha256'] = reader.fingerprints()[name]
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=67108864)
        mesh.enable_program_cache()
        weight = ttnn.from_torch(packed, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
        owned.append(weight)
        generator = torch.Generator().manual_seed(383827)
        patterns = [torch.randn((1, 1, 32, packed.shape[-2]), generator=generator).bfloat16() for unused in range(2)]
        payloads = [ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)) for value in patterns]
        value = ttnn.from_torch(patterns[0], device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        owned.append(value)
        bindings = addresses(ttnn, value), addresses(ttnn, weight)
        def read(tensor):
            ttnn.synchronize_device(mesh)
            return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tensor)]
        original_weights = [tensor_digest(part) for part in read(weight)]
        def update(case):
            ttnn.synchronize_device(mesh)
            ttnn.copy_host_to_device_tensor(payloads[case], value)
            ttnn.synchronize_device(mesh)
        def integrity(case, mode):
            exact = (bindings == (addresses(ttnn, value), addresses(ttnn, weight))
                and [tensor_digest(part) for part in read(weight)] == original_weights
                and all(torch.equal(part, patterns[case]) for part in read(value)))
            report['integrity_checks'].append(dict(case=case, mode=mode, exact=exact))
            if not exact:
                raise AssertionError('Projection modified inputs, weights or bindings')
        eager = {}
        for case in range(2):
            update(case)
            progress(f'eager_{case}')
            baseline = read(reference(ttnn, value, weight, retain, rounded=False))
            release_owned(ttnn, transient)
            transient.clear()
            eager[case] = read(candidate(ttnn, value, weight, retain, rounded=False))
            for chip, (expected, actual) in enumerate(zip(baseline, eager[case], strict=True)):
                if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
                    raise AssertionError('Nonfinite projection output')
                difference = actual.float() - expected.float()
                report['differences'].append(dict(case=case, chip=chip, finite=True,
                    exact=torch.equal(actual, expected), max_abs=float(difference.abs().max()),
                    rms_error=float(difference.square().mean().sqrt()),
                    reference_rms=float(expected.float().square().mean().sqrt())))
            integrity(case, 'eager')
            release_owned(ttnn, transient)
            transient.clear()
        update(0)
        progress('capture')
        trace, output = capture_operation(ttnn, mesh, lambda: candidate(ttnn, value, weight, retain, rounded=False))
        poison = ttnn.from_torch(torch.full(tuple(output.shape), float('nan')), dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        output_bindings = addresses(ttnn, output)
        for ordinal, case in enumerate((1, 0, 1)):
            progress(f'replay_{ordinal}')
            update(case)
            ttnn.copy_host_to_device_tensor(poison, output)
            ttnn.synchronize_device(mesh)
            if not all(torch.isnan(part).all() for part in read(output)):
                raise AssertionError('Output poisoning failed')
            ttnn.execute_trace(mesh, trace, blocking=True)
            for chip, actual in enumerate(read(output)):
                exact = torch.equal(actual, eager[case][chip])
                report['replay_checks'].append(dict(ordinal=ordinal, case=case, chip=chip, exact=exact,
                    poison_replaced=bool(torch.isfinite(actual).all())))
                if not exact:
                    raise AssertionError('Candidate replay differs from same-policy eager output')
            if addresses(ttnn, output) != output_bindings:
                raise AssertionError('Trace output binding changed')
            integrity(case, 'replay')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                release_owned(ttnn, transient)
                release_owned(ttnn, owned)
                ttnn.close_mesh_device(mesh)
            report['sources_after'] = sources()
            if report['sources_after'] != report['sources']:
                raise ValueError('Probe sources changed')
            report['closed_cleanly'] = True
        finally:
            progress('complete' if report['passed'] and report['closed_cleanly'] else 'failed')


if __name__ == '__main__':
    main()
