"""Learned RMSNorm with explicit host or fabric reduction diagnostics."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from draft_projection_full_fixture import load_projection
from feature_normalization import bf16_ulp_distance, projection_row, rms_reference
from feature_projection import require_projection_environment
from feature_collective import reduce_projection, gather_add_projection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--fabric-reduce', action='store_true')
    parser.add_argument('--gather-add', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--projection-report', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, options.hardware)
    if options.gather_add and not options.fabric_reduce:
        parser.error('--gather-add requires --fabric-reduce')
    if options.fabric_reduce and not options.hardware and os.environ.get('QWEN_SIM_SHARED_BDF') != '1':
        raise RuntimeError('Fabric simulator requires shared BDF loading')
    import torch
    import ttnn

    manifest, projection_weight, norm_weight = load_projection(options.fixture)
    del projection_weight
    projection_bytes = options.projection_report.read_bytes()
    projection = json.loads(projection_bytes)
    if projection['checkpoint'] != manifest:
        raise ValueError('Projection must use the same audited learned checkpoint')
    value = projection_row(projection)
    expected = rms_reference(value, norm_weight)
    report = dict(passed=False, scope=__doc__, backend='hardware' if options.hardware else 'simulator',
        projection_sha256=hashlib.sha256(projection_bytes).hexdigest(),
        checkpoint=manifest, fabric_reduce=options.fabric_reduce, gather_add=options.gather_add,
        epsilon=1e-6, max_bf16_ulps=2, checks=[], sources={name:
            hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('feature-norm-probe.py', 'feature_normalization.py', 'draft_projection_full_fixture.py', 'feature_collective.py')})
    mesh = None
    tensors = []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D if options.hardware or options.fabric_reduce else ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        if options.fabric_reduce:
            from models.tt_transformers.tt.ccl import TT_CCL

            partials = torch.tensor([check['actual_first_row'] for check in sorted(projection['checks'], key=lambda check: check['chip'])],
                dtype=torch.float32).reshape(2, 1, 1, 5120)
            device_partials = ttnn.from_torch(partials, device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
            tensors.append(device_partials)
            operation = gather_add_projection if options.gather_add else reduce_projection
            reduced = operation(ttnn, mesh, TT_CCL(mesh), device_partials)
            tensors.append(reduced)
            reference_sum = partials.sum(dim=0, keepdim=True)
            report['reduction_checks'] = []
            for chip, shard in enumerate(ttnn.get_device_tensors(reduced)):
                actual_sum = ttnn.to_torch(shard)
                report['reduction_checks'].append(dict(chip=chip,
                    max_abs_error=float((actual_sum - reference_sum).abs().max()), exact=bool(torch.equal(actual_sum, reference_sum))))
                torch.testing.assert_close(actual_sum, reference_sum, rtol=0, atol=0)
            device_value = ttnn.typecast(reduced, ttnn.bfloat16)
        else:
            device_value = ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        tensors.append(device_value)
        device_weight = ttnn.from_torch(norm_weight.reshape(1, 1, 160, 32), device=mesh,
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        tensors.append(device_weight)
        kernel = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        output = ttnn.rms_norm(device_value, epsilon=1e-6, weight=device_weight,
            compute_kernel_config=kernel, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tensors.append(output)
        ttnn.synchronize_device(mesh)
        for chip, shard in enumerate(ttnn.get_device_tensors(output)):
            actual = ttnn.to_torch(shard)
            distance = bf16_ulp_distance(actual, expected)
            check = dict(chip=chip, shape=list(actual.shape), max_bf16_ulps=int(distance.max()),
                max_abs_error=float((actual.float() - expected.float()).abs().max()),
                passed=bool((distance <= 2).all()))
            report['checks'].append(check)
        if len(report['checks']) != 2 or not all(check['passed'] for check in report['checks']):
            raise AssertionError('Learned normalization exceeds two BF16 ULPs')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        for tensor in reversed(tensors):
            ttnn.deallocate(tensor)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))
    report['passed'] = True
    options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
